"""Stage 2 perception-injection PPO entry point.

The default CPU backend is a deterministic fake environment used to verify
the complete four-head rollout/update path.  A physical backend is supplied
by the Isaac Lab launcher in the project-level runner; this module refuses to
silently substitute the fake environment when ``--backend isaaclab`` is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pgmt.cfg.assumptions import dump as dump_assumptions, get
from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.recovery import FallRecoveryPool
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.train.policy import Stage1Policy, Stage2Policy
from pgmt.train.ppo import PPO


def _observations(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros(n, OBS_DIM, device=device),
        "history": torch.zeros(n, HISTORY_LEN, OBS_DIM, device=device),
        "future": torch.zeros(n, get("A2").value.K, REF_FRAME_DIM, device=device),
        "privileged": torch.zeros(n, PRIV_DIM, device=device),
        "elevation": torch.zeros(n, get("A8").value.map_size,
                                  get("A8").value.map_size, device=device),
    }


class MockStage2Env:
    """Small protocol environment with a real elevation-map observation."""

    def __init__(self, num_envs: int, device: torch.device, horizon: int = 8):
        if num_envs <= 0 or horizon <= 0:
            raise ValueError("num_envs and horizon must be positive")
        self.num_envs, self.device, self.horizon = num_envs, device, horizon
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.observations = _observations(num_envs, device)

    def reset(self):
        self.steps.zero_()
        for value in self.observations.values():
            value.zero_()
        return self.observations

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, ACT_DIM):
            raise ValueError(f"actions must have shape ({self.num_envs},{ACT_DIM})")
        # Four non-negative split rewards, including the terrain head.
        cost = actions.square().mean(-1)
        rewards = torch.stack((1.0 - cost, 1.0 - cost, 0.5 - cost, 1.0 - 0.1 * cost), -1)
        self.steps += 1
        truncated = self.steps >= self.horizon
        terminated = torch.zeros_like(truncated)
        terminal = {key: value.clone() for key, value in self.observations.items()}
        if truncated.any():
            ids = truncated.nonzero(as_tuple=False).flatten()
            self.steps[ids] = 0
            for value in self.observations.values():
                value[ids] = 0
        info = {"terminal_observation": terminal} if truncated.any() else {}
        return self.observations, rewards, terminated, truncated, info


def _load_stage1(path: Path, device: torch.device) -> Stage1Policy:
    source = Stage1Policy().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("ppo", payload)
    source.load_state_dict(state["policy"] if "policy" in state else state, strict=True)
    return source


def _load_fall_pool(path: Path | None):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, FallRecoveryPool):
        raise TypeError("--fall-pool must point to a serialized FallRecoveryPool")
    return payload


def _build_torch_env(*, device: torch.device, num_envs: int, reference_data: str | None,
                     urdf: str | None, fall_pool=None):
    if not reference_data or not urdf:
        raise ValueError("--backend torch requires both --reference-data and --urdf")
    database = MotionDatabase(reference_data)
    return G1Env(G1EnvConfig(num_envs=num_envs, device=str(device), stage=2,
                             reference_data_dir=reference_data,
                             reference_urdf_path=urdf),
                 reference_database=database, recovery_pool=fall_pool)


def run(*, device: str = "cpu", num_envs: int = 2, steps_per_env: int = 8,
        updates: int = 1, learning_epochs: int = 1, mini_batches: int = 1,
        stage1_checkpoint: Path | None = None, checkpoint: Path | None = None,
        resume: Path | None = None, backend: str = "mock",
        reference_data: str | None = None, asset: str | None = None,
        urdf: str | None = None, fall_pool: Path | None = None) -> dict:
    if stage1_checkpoint is not None and resume is not None:
        raise ValueError("--stage1-checkpoint and --resume are mutually exclusive")
    if backend not in {"mock", "torch", "isaaclab"}:
        raise ValueError(f"unknown Stage 2 backend: {backend}")
    dev = torch.device(device)
    policy = Stage2Policy().to(dev)
    env = None
    app = None
    try:
        if stage1_checkpoint is not None:
            policy.load_stage1_state_dict(_load_stage1(stage1_checkpoint, dev).state_dict())
        if backend == "mock":
            env = MockStage2Env(num_envs, dev)
        elif backend == "torch":
            env = _build_torch_env(device=dev, num_envs=num_envs,
                                   reference_data=reference_data, urdf=urdf,
                                   fall_pool=_load_fall_pool(fall_pool))
        else:
            if not asset or not urdf or not reference_data:
                raise ValueError("--backend isaaclab requires --asset, --urdf, and --reference-data")
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True).app
            import importlib
            import pgmt.envs.g1_env as g1_module
            g1_module = importlib.reload(g1_module)
            pgmt_cfg = g1_module.G1EnvConfig(num_envs=num_envs, device=str(dev), stage=2,
                                              asset_path=asset, reference_data_dir=reference_data,
                                              reference_urdf_path=urdf)
            robot_cfg = g1_module.make_g1_articulation_cfg(asset, pgmt_cfg)
            cfg_env = g1_module.G1DirectRLEnvCfg(robot_cfg=robot_cfg, pgmt_cfg=pgmt_cfg)
            cfg_env.scene.num_envs = num_envs
            cfg_env.scene.env_spacing = 2.5
            env = g1_module.IsaacLabPPOAdapter(g1_module.IsaacLabG1Env(cfg_env))
        cfg = get("A6").value
        from dataclasses import replace
        cfg = replace(cfg, num_steps_per_env=steps_per_env,
                      num_learning_epochs=learning_epochs,
                      num_mini_batches=min(mini_batches, steps_per_env * num_envs))
        ppo = PPO(policy, config=cfg, total_updates=updates)
        obs = env.reset()
        if backend != "mock" and "elevation" not in obs:
            raise RuntimeError("Stage 2 environment must provide an 'elevation' observation; terrain provider was not connected")
        if resume is not None:
            payload = torch.load(resume, map_location=dev, weights_only=False)
            ppo.load_state_dict(payload["ppo"] if "ppo" in payload else payload)
            if "env" in payload and hasattr(env, "load_state_dict"):
                env.load_state_dict(payload["env"])
        metrics = []
        for _ in range(updates):
            obs, collected = ppo.collect_rollout(env, obs)
            update = ppo.update()
            update.update({"reward_mean": collected["reward_mean"], "timeouts": collected["timeouts"]})
            metrics.append(update)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ppo": ppo.state_dict(), "assumptions": dump_assumptions()}
            if hasattr(env, "state_dict"):
                payload["env"] = env.state_dict()
            torch.save(payload, checkpoint)
        return {"updates": metrics, "heads": list(policy.head_names), "device": str(dev),
                "backend": backend, "checkpoint": None if checkpoint is None else str(checkpoint)}
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        if app is not None:
            try:
                app.close(skip_cleanup=True)
            except TypeError:
                app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "torch", "isaaclab"), default="mock")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps-per-env", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--learning-epochs", type=int, default=1)
    parser.add_argument("--mini-batches", type=int, default=1)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-data")
    parser.add_argument("--asset")
    parser.add_argument("--urdf")
    parser.add_argument("--fall-pool", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps_per_env, args.updates) <= 0:
        parser.error("num-envs, steps-per-env, and updates must be positive")
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
