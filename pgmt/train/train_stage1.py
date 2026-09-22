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
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import replace, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from pgmt.cfg.assumptions import dump as dump_assumptions, get
from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.actuators import ACTUATOR_PROFILES, actuator_options, validate_actuator_resume
from pgmt.train.diagnostics import attach_diagnostics
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.recovery import FallRecoveryPool
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


def _json_default(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize metrics value of type {type(value).__name__}")


@contextmanager
def _atomic_output(path: Path):
    """Keep the last complete artifact if serialization is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_metrics(path: Path | None, result: dict) -> None:
    if path is not None:
        encoded = (json.dumps(result, ensure_ascii=False, indent=2,
                              default=_json_default) + "\n").encode("utf-8")
        with _atomic_output(Path(path)) as stream:
            stream.write(encoded)


def _lr_schedule_horizon(stop_update, requested=None, checkpoint=None):
    saved = None if checkpoint is None else checkpoint.get("ppo", checkpoint)
    if requested is None:
        requested = saved["total_updates"] if saved is not None else get("A6").value.lr_schedule_updates
    if requested is None or requested <= 0 or stop_update > requested:
        raise ValueError("lr schedule horizon must be positive and cover --updates")
    if saved is not None and requested != saved["total_updates"]:
        raise ValueError("resume must preserve the checkpoint learning-rate schedule")
    return requested


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
            "recovery_upward_vel": torch.zeros(n, device=self.device),
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
              reference_data: str | None = None, asset_path: str | None = None,
              reference_urdf: str | None = None,
              fall_pool: FallRecoveryPool | None = None,
              backend: str = "torch", app=None, env_options=None):
    env_options = dict(env_options or {})
    if backend not in {"mock", "torch", "isaaclab"}:
        raise ValueError(f"unknown backend: {backend}")
    if backend == "mock" or mock:
        return MockStage1Env(num_envs, device)
    if backend == "isaaclab":
        if not asset_path:
            raise ValueError("--backend isaaclab requires --asset/--urdf; refusing implicit torch fallback")
        if app is None:
            raise RuntimeError("Isaac Lab backend requires an initialized AppLauncher")
        import importlib
        import pgmt.envs.g1_env as g1_module
        g1_module = importlib.reload(g1_module)
        if g1_module.IsaacLabG1Env is None:
            raise RuntimeError("Isaac Lab modules were not available after AppLauncher startup")
        pgmt_cfg = G1EnvConfig(num_envs=num_envs, device=str(device), asset_path=asset_path,
                               reference_data_dir=reference_data,
                               reference_urdf_path=reference_urdf, **env_options)
        robot_cfg = g1_module.make_g1_articulation_cfg(asset_path, pgmt_cfg)
        cfg = g1_module.G1DirectRLEnvCfg(robot_cfg=robot_cfg, pgmt_cfg=pgmt_cfg)
        cfg.scene.num_envs = num_envs
        cfg.scene.env_spacing = 2.5
        return g1_module.IsaacLabPPOAdapter(g1_module.IsaacLabG1Env(cfg, recovery_pool=fall_pool))
    if asset_path:
        path = Path(asset_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
    if backend == "torch" and (not reference_data or not reference_urdf):
        raise ValueError("--backend torch requires both --reference-data and --urdf")
    # This is the same batched PD/reference/reward contract used by the
    # Isaac-Lab shell, with deterministic torch integration when no simulator
    # articulation is supplied.  It is useful for starting Stage 1 and for
    # validating the policy/rollout path before paying Isaac Sim startup cost.
    database = MotionDatabase(reference_data) if reference_data else None
    return G1Env(G1EnvConfig(num_envs=num_envs, device=str(device), asset_path=asset_path,
                             reference_data_dir=reference_data,
                             reference_urdf_path=reference_urdf, **env_options),
                 reference_database=database, recovery_pool=fall_pool)


def run(args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if args.dry_run:
        args.updates = 1
        args.num_envs = min(args.num_envs, 2)
    app = None
    env = None
    backend = "mock" if args.mock or args.dry_run else args.backend
    pool = _load_fall_pool(args.fall_pool) if args.fall_pool else None
    try:
        if backend == "isaaclab" and pool is None:
            raise ValueError("physical Stage 1 requires --fall-pool; collect an offline pool first")
        if backend == "isaaclab":
            if not args.asset or not args.urdf or not args.reference_data:
                raise ValueError("--backend isaaclab requires --asset, --urdf, and --reference-data")
            from pgmt.envs.isaac_app import launch_isaac_app
            app = launch_isaac_app(device)
        env = build_env(args.num_envs, device, mock=False, backend=backend,
                        reference_data=args.reference_data, asset_path=args.asset,
                        reference_urdf=args.urdf, fall_pool=pool, app=app,
                        env_options={"seed": args.seed, "randomize_dynamics": True,
                                     "corrupt_observations": True, "max_action_delay": 2,
                                     **actuator_options(getattr(args, "actuator_profile", "asset_effort_v1"))})
        diagnostics = attach_diagnostics(env, getattr(args, "diagnostics_dir", None))
        if (args.learning_epochs is not None and args.learning_epochs <= 0) or (
                args.mini_batches is not None and args.mini_batches <= 0):
            raise ValueError("--learning-epochs and --mini-batches must be positive")
        base = get("A6").value
        state = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
        if state is not None and backend != "mock":
            core = getattr(getattr(env, "env", env), "core", env)
            validate_actuator_resume(state, core.cfg)
        schedule = _lr_schedule_horizon(args.updates, getattr(args, "lr_schedule_updates", None), state)
        complete_critic = getattr(args, "critic_completion", None)
        if complete_critic is None and state is not None:
            complete_critic = state.get("ppo", state)["config"]["complete_critic_epochs"]
        learning_epochs = base.num_learning_epochs if args.learning_epochs is None else args.learning_epochs
        mini_batches = base.num_mini_batches if args.mini_batches is None else args.mini_batches
        config = replace(base,
                         num_steps_per_env=args.steps_per_env,
                         num_learning_epochs=learning_epochs,
                         lr_schedule_updates=schedule,
                         complete_critic_epochs=base.complete_critic_epochs if complete_critic is None else complete_critic,
                         num_mini_batches=min(mini_batches, args.steps_per_env * args.num_envs))
        policy = Stage1Policy().to(device)
        ppo = PPO(policy, config=config, total_updates=schedule, diagnostics=diagnostics)
        restored_env = False
        if args.resume:
            ppo.load_state_dict(state["ppo"] if "ppo" in state else state)
            if ppo.update_count > args.updates:
                raise ValueError(
                    f"checkpoint has {ppo.update_count} updates, but --updates={args.updates} "
                    "is a lower total target"
                )
            if "env" in state and hasattr(env, "load_state_dict"):
                env.load_state_dict(state["env"])
                restored_env = True
        if restored_env:
            # A stateful environment must continue from the checkpointed
            # observation.  ``get_observations`` is side-effect free when
            # provided; the legacy private fallback is kept for older envs.
            if hasattr(env, "get_observations"):
                observations = env.get_observations()
            elif hasattr(env, "_build_observations"):
                observations = env._build_observations()
            elif hasattr(env, "observations"):
                observations = env.observations
            else:
                observations = env.reset()
        else:
            observations = env.reset()
        all_metrics = []
        result = {
            "updates": all_metrics,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "device": str(device), "backend": backend,
            "status": "running", "start_update": ppo.update_count,
            "completed_updates": ppo.update_count, "target_updates": args.updates,
            "resume": None if args.resume is None else str(args.resume),
            "environment_restored": restored_env,
            "lr_schedule_updates": schedule, "critic_completion": config.complete_critic_epochs,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path = getattr(args, "metrics", None)
        _write_metrics(metrics_path, result)
        print(json.dumps({key: value for key, value in result.items() if key != "updates"}),
              flush=True)
        initial_checkpoint = getattr(args, "initial_checkpoint", None)
        if initial_checkpoint is not None:
            if args.resume or ppo.update_count:
                raise ValueError("--initial-checkpoint is only valid for a fresh run")
            if Path(initial_checkpoint).exists():
                raise FileExistsError(initial_checkpoint)
            _save_checkpoint(Path(initial_checkpoint), ppo, args, env=env)
        for _ in range(args.updates - ppo.update_count):
            observations, collected = ppo.collect_rollout(env, observations)
            metrics = ppo.update()
            metrics.update(collected)
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
            all_metrics.append(metrics)
            if args.checkpoint:
                _save_checkpoint(Path(args.checkpoint), ppo, args, env=env)
            result["completed_updates"] = ppo.update_count
            _write_metrics(metrics_path, result)
            print(json.dumps({"event": "update", **metrics}, default=_json_default), flush=True)
        if args.checkpoint and not all_metrics:
            _save_checkpoint(Path(args.checkpoint), ppo, args, env=env)
        result["status"] = "completed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_metrics(metrics_path, result)
        return result
    except BaseException:
        # Print before Kit shutdown: Isaac Sim can swallow an otherwise
        # unhandled traceback while its application is being closed.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        if app is not None:
            # ``skip_cleanup=True`` maps to Kit's immediate-exit path.  It can
            # terminate the interpreter before an exception traceback or the
            # runner's JSON result is emitted, so training uses the normal
            # shutdown path and keeps failures observable.
            app.close()


def _load_fall_pool(path: str) -> FallRecoveryPool:
    from pgmt.envs.recovery import load_fall_pool
    return load_fall_pool(path)


def _save_checkpoint(path: Path, ppo: PPO, args: argparse.Namespace, *, env=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ppo": ppo.state_dict(), "assumptions": dump_assumptions(),
               "runner": {"seed": args.seed, "stage": 1,
                          "backend": "mock" if args.mock or args.dry_run else getattr(args, "backend", "torch"),
                          "mock": bool(args.mock or args.dry_run or getattr(args, "backend", "") == "mock")}}
    core = getattr(getattr(env, "env", env), "core", env)
    if core is not None and hasattr(core, "cfg"):
        from pgmt.envs.randomization import RANDOMIZATION_SPEC
        payload["environment_config"] = asdict(core.cfg)
        payload["randomization_spec"] = RANDOMIZATION_SPEC
    if env is not None and hasattr(env, "state_dict"):
        payload["env"] = env.state_dict()
    with _atomic_output(path) as stream:
        torch.save(payload, stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="compatibility alias for --backend mock")
    parser.add_argument("--dry-run", action="store_true", help="one tiny mock update and exit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("torch", "mock", "isaaclab"), default="torch")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--updates", type=int, default=1, help="absolute stopping update for this invocation")
    parser.add_argument("--lr-schedule-updates", type=int,
                        help="full LR schedule horizon (default A6=1000; on resume inherit checkpoint)")
    parser.add_argument("--critic-completion", action=argparse.BooleanOptionalAction, default=None,
                        help="complete remaining value steps with only private critic parameters")
    parser.add_argument("--learning-epochs", type=int, default=None,
                        help="PPO learning epochs per update (default: A6 assumption)")
    parser.add_argument("--mini-batches", type=int, default=None,
                        help="PPO minibatches per epoch (default: A6 assumption)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path, help="optional fresh policy snapshot for matched learning diagnostics")
    parser.add_argument("--metrics", type=Path, help="save this invocation's training metrics after every update")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-data", type=str, help="directory of retargeted G1 NPZ reference motions")
    parser.add_argument("--asset", type=str, help="licensed local G1 USD/URDF path (validated, not copied)")
    parser.add_argument("--urdf", help="licensed local G1 URDF used for reference FK; may be paired with --asset USD")
    parser.add_argument("--fall-pool", help="serialized FallRecoveryPool; omitted means recovery is disabled")
    parser.add_argument("--actuator-profile", choices=ACTUATOR_PROFILES, default="asset_effort_v1",
                        help="effort-only correction, or explicit v4 limit control; kp/kd unchanged")
    parser.add_argument("--diagnostics-dir", type=Path, help="opt-in bounded anomaly snapshots for CPU replay")
    args = parser.parse_args(argv)
    if args.updates <= 0 or args.steps_per_env <= 0:
        parser.error("--updates and --steps-per-env must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
