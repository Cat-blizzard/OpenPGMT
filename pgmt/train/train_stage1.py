"""Stage 1 PPO entry point.

The ``--mock``/``--dry-run`` path is intentionally small but complete: it
exercises observation shapes, three-headed rewards, timeout bootstrapping,
PPO updates, and checkpoint save/resume without requiring Isaac Lab.  The
normal path uses the batched G1 PD/reference adapter; the optional Isaac Lab
shell is kept at the simulator boundary and is probed separately before a
physics run.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from pgmt.cfg.assumptions import dump as dump_assumptions, get
from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.train.stage1 import BatchRewardAdapter


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _zeros_observations(num_envs: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros(num_envs, OBS_DIM, device=device),
        "history": torch.zeros(num_envs, HISTORY_LEN, OBS_DIM, device=device),
        "future": torch.zeros(num_envs, get("A2").value.K, REF_FRAME_DIM, device=device),
        "privileged": torch.zeros(num_envs, PRIV_DIM, device=device),
    }


class MockStage1Env:
    """A deterministic protocol test double, not a physics environment."""

    def __init__(self, num_envs: int, device: torch.device, horizon: int = 8):
        if num_envs <= 0 or horizon <= 0:
            raise ValueError("num_envs and horizon must be positive")
        self.num_envs, self.device, self.horizon = num_envs, device, int(horizon)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.observations = _zeros_observations(num_envs, device)
        self.reward = BatchRewardAdapter(device=device, strict=True)
        self.reset()

    def reset(self):
        self.steps.zero_()
        for value in self.observations.values():
            value.zero_()
        return self.observations

    def _reward(self, action: torch.Tensor) -> torch.Tensor:
        # The mock supplies complete term values with the same semantics as the
        # real environment: tracking residuals for upper/lower, pre-kernel
        # auxiliary values for the negative-cost terms, and kernel values for
        # the positive auxiliary terms.
        n = action.shape[0]
        q_error = action.square().mean(-1).sqrt()
        values = {
            "link_pos": q_error, "link_ori": q_error, "link_lin_vel": q_error,
            "link_ang_vel": q_error, "joint_pos": q_error, "joint_vel": q_error,
            "ta_link_pos": q_error, "ta_link_ori": q_error, "ta_joint_pos": q_error,
            "root_ori": torch.ones(n, device=self.device),
            "corrected_root_vel": torch.ones(n, device=self.device),
            "floating_anchor_pos": torch.ones(n, device=self.device),
            "recovery_upward_vel": torch.ones(n, device=self.device),
            "pelvis_vert_accel": torch.zeros(n, device=self.device),
            "ee_accel_mismatch": torch.zeros(n, device=self.device),
            "action_rate": action.square().mean(-1),
            "joint_limit": torch.zeros(n, device=self.device),
            "undesired_contact": torch.zeros(n, device=self.device),
            "head_torso_impact": torch.zeros(n, device=self.device),
        }
        return self.reward(values)

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        if action.shape != (self.num_envs, ACT_DIM):
            raise ValueError(f"action must have shape ({self.num_envs},{ACT_DIM})")
        rewards = self._reward(action)
        self.steps += 1
        # The terminal observation must be captured before reset.  This is the
        # same contract needed by a reset-before-step Isaac Lab adapter.
        truncated = self.steps >= self.horizon
        terminated = torch.zeros_like(truncated)
        terminal = {name: value.clone() for name, value in self.observations.items()}
        if truncated.any():
            ids = truncated.nonzero(as_tuple=False).flatten()
            self.steps[ids] = 0
            for value in self.observations.values():
                value[ids] = 0
        info = {"terminal_observation": terminal} if truncated.any() else {}
        return self.observations, rewards, terminated, truncated, info


def build_env(num_envs: int, device: torch.device, *, mock: bool = False,
              reference_data: str | None = None, asset_path: str | None = None):
    if mock:
        return MockStage1Env(num_envs, device)
    if asset_path:
        path = Path(asset_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
    # This is the same batched PD/reference/reward contract used by the
    # Isaac-Lab shell, with deterministic torch integration when no simulator
    # articulation is supplied.  It is useful for starting Stage 1 and for
    # validating the policy/rollout path before paying Isaac Sim startup cost.
    database = MotionDatabase(reference_data) if reference_data else None
    return G1Env(G1EnvConfig(num_envs=num_envs, device=str(device), asset_path=asset_path),
                 reference_database=database)


def run(args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if args.dry_run:
        args.updates = 1
        args.num_envs = min(args.num_envs, 2)
    env = build_env(args.num_envs, device, mock=args.mock or args.dry_run,
                    reference_data=args.reference_data, asset_path=args.asset)
    config = replace(get("A6").value,
                     num_steps_per_env=args.steps_per_env,
                     num_learning_epochs=args.learning_epochs,
                     num_mini_batches=min(args.mini_batches, args.steps_per_env * args.num_envs))
    policy = Stage1Policy().to(device)
    ppo = PPO(policy, config=config, total_updates=args.updates)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        ppo.load_state_dict(state["ppo"] if "ppo" in state else state)
    observations = env.reset()
    all_metrics = []
    for _ in range(args.updates - ppo.update_count):
        observations, collected = ppo.collect_rollout(env, observations)
        metrics = ppo.update()
        metrics.update({"reward_mean": collected["reward_mean"], "timeouts": collected["timeouts"]})
        all_metrics.append(metrics)
        if args.checkpoint:
            _save_checkpoint(Path(args.checkpoint), ppo, args)
    if args.checkpoint and not all_metrics:
        _save_checkpoint(Path(args.checkpoint), ppo, args)
    return {"updates": all_metrics, "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "device": str(device)}


def _save_checkpoint(path: Path, ppo: PPO, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ppo": ppo.state_dict(), "assumptions": dump_assumptions(),
               "runner": {"seed": args.seed, "mock": bool(args.mock or args.dry_run)}}
    torch.save(payload, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="run the simulator-independent protocol test double")
    parser.add_argument("--dry-run", action="store_true", help="one tiny mock update and exit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--learning-epochs", type=int, default=1)
    parser.add_argument("--mini-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-data", type=str, help="directory of retargeted G1 NPZ reference motions")
    parser.add_argument("--asset", type=str, help="licensed local G1 USD/URDF path (validated, not copied)")
    args = parser.parse_args(argv)
    if args.updates <= 0 or args.steps_per_env <= 0:
        parser.error("--updates and --steps-per-env must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
