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
from pgmt.train.policy import Stage1Policy, Stage2Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import build_env, _seed_everything, _atomic_output, _lr_schedule_horizon, _write_metrics as write_progress


def _json_default(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize metrics value of type {type(value).__name__}")


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


def _load_stage1(path: Path, device: torch.device, *, require_physics=False) -> Stage1Policy:
    source = Stage1Policy().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("ppo", payload)
    if require_physics and (payload.get("runner", {}).get("backend") != "isaaclab" or payload.get("runner", {}).get("stage") != 1):
        raise ValueError("formal Stage 2 requires a physical Stage 1 checkpoint, not a protocol smoke test")
    if state.get("action_contract") != source.action_contract:
        raise ValueError("Stage 1 checkpoint uses an obsolete action contract")
    if state.get("training_contract") != PPO.training_contract:
        raise ValueError("Stage 1 checkpoint uses an obsolete training/reward contract")
    source.load_state_dict(state["policy"] if "policy" in state else state, strict=True)
    return source


def _load_fall_pool(path: Path | None):
    if path is None:
        return None
    from pgmt.envs.recovery import load_fall_pool
    return load_fall_pool(path)


def run(*, device: str = "cpu", num_envs: int = 2, steps_per_env: int = 8,
        updates: int = 1, learning_epochs: int | None = None,
        mini_batches: int | None = None,
        lr_schedule_updates: int | None = None, critic_completion: bool | None = None,
        stage1_checkpoint: Path | None = None, checkpoint: Path | None = None,
        resume: Path | None = None, backend: str = "mock",
        reference_data: str | None = None, asset: str | None = None,
        urdf: str | None = None, fall_pool: Path | None = None,
        metrics: Path | None = None, seed: int = 0,
        allow_scratch_ablation: bool = False, actuator_profile: str = "asset_effort_v1",
        diagnostics_dir: Path | None = None) -> dict:
    from pgmt.envs.actuators import actuator_options, validate_actuator_resume
    from pgmt.train.diagnostics import attach_diagnostics
    if stage1_checkpoint is not None and resume is not None:
        raise ValueError("--stage1-checkpoint and --resume are mutually exclusive")
    if backend not in {"mock", "torch", "isaaclab"}:
        raise ValueError(f"unknown Stage 2 backend: {backend}")
    if backend != "mock" and stage1_checkpoint is None and resume is None and not allow_scratch_ablation:
        raise ValueError("Stage 2 requires --stage1-checkpoint (or --resume); scratch training requires --allow-scratch-ablation")
    _seed_everything(seed)
    dev = torch.device(device)
    policy = Stage2Policy().to(dev)
    env = None
    app = None
    try:
        if stage1_checkpoint is not None:
            policy.load_stage1_state_dict(_load_stage1(stage1_checkpoint, dev, require_physics=backend == "isaaclab").state_dict())
        if backend == "mock":
            env = MockStage2Env(num_envs, dev)
        else:
            if not reference_data or not urdf or (backend == "isaaclab" and not asset):
                raise ValueError("physical Stage 2 requires --asset, --urdf and --reference-data")
            if backend == "isaaclab":
                from pgmt.envs.isaac_app import launch_isaac_app
                app = launch_isaac_app(dev)
            env = build_env(num_envs, dev, backend=backend, reference_data=reference_data,
                            asset_path=asset, reference_urdf=urdf, app=app,
                            fall_pool=_load_fall_pool(fall_pool),
                            env_options={"stage": 2, "seed": seed, "terrain_curriculum": True,
                                         "randomize_dynamics": True, "corrupt_observations": True,
                                         "max_action_delay": 2, **actuator_options(actuator_profile)})
        diagnostics = attach_diagnostics(env, diagnostics_dir)
        cfg = get("A6").value
        payload = torch.load(resume, map_location=dev, weights_only=False) if resume is not None else None
        if payload is not None and backend != "mock":
            core = getattr(getattr(env, "env", env), "core", env)
            validate_actuator_resume(payload, core.cfg)
        schedule = _lr_schedule_horizon(updates, lr_schedule_updates, payload)
        if critic_completion is None and payload is not None:
            critic_completion = payload.get("ppo", payload)["config"]["complete_critic_epochs"]
        if learning_epochs is None:
            learning_epochs = cfg.num_learning_epochs
        if mini_batches is None:
            mini_batches = cfg.num_mini_batches
        if learning_epochs <= 0 or mini_batches <= 0:
            raise ValueError("learning_epochs and mini_batches must be positive")
        from dataclasses import replace
        cfg = replace(cfg, num_steps_per_env=steps_per_env,
                      num_learning_epochs=learning_epochs,
                      lr_schedule_updates=schedule,
                      complete_critic_epochs=cfg.complete_critic_epochs if critic_completion is None else critic_completion,
                      num_mini_batches=min(mini_batches, steps_per_env * num_envs))
        # The stopping point does not shorten the full LR schedule.
        ppo = PPO(policy, config=cfg, total_updates=schedule, diagnostics=diagnostics)
        restored_env = False
        if resume is not None:
            ppo.load_state_dict(payload["ppo"] if "ppo" in payload else payload)
            if ppo.update_count > updates:
                raise ValueError(
                    f"checkpoint has {ppo.update_count} updates, but --updates={updates} "
                    "is a lower total target"
                )
            if "env" in payload and hasattr(env, "load_state_dict"):
                env.load_state_dict(payload["env"])
                restored_env = True
        if restored_env:
            if hasattr(env, "get_observations"):
                obs = env.get_observations()
            elif hasattr(env, "_build_observations"):
                obs = env._build_observations()
            elif hasattr(env, "observations"):
                obs = env.observations
            else:
                obs = env.reset()
        else:
            obs = env.reset()
        if backend != "mock" and "elevation" not in obs:
            raise RuntimeError("Stage 2 environment must provide an 'elevation' observation; terrain provider was not connected")
        all_metrics = []
        result = {"updates": all_metrics, "heads": list(policy.head_names), "device": str(dev),
                  "backend": backend, "checkpoint": None if checkpoint is None else str(checkpoint),
                  "status": "running", "start_update": ppo.update_count,
                  "completed_updates": ppo.update_count, "target_updates": updates,
                  "stage1_checkpoint": None if stage1_checkpoint is None else str(stage1_checkpoint),
                  "scratch_ablation": bool(allow_scratch_ablation), "seed": seed}
        result.update(lr_schedule_updates=schedule, critic_completion=cfg.complete_critic_epochs)
        write_progress(metrics, result)
        for _ in range(updates - ppo.update_count):
            obs, collected = ppo.collect_rollout(env, obs)
            update = ppo.update()
            update.update(collected)
            all_metrics.append(update)
            if checkpoint is not None:
                from dataclasses import asdict
                core = getattr(getattr(env, "env", env), "core", env)
                from pgmt.envs.randomization import RANDOMIZATION_SPEC
                payload = {"ppo": ppo.state_dict(), "assumptions": dump_assumptions(),
                           "runner": {"backend": backend, "stage": 2, "seed": seed},
                           "environment_config": asdict(core.cfg) if hasattr(core, "cfg") else None,
                           "randomization_spec": RANDOMIZATION_SPEC,
                           "stage1_checkpoint": result["stage1_checkpoint"],
                           "scratch_ablation": bool(allow_scratch_ablation)}
                if hasattr(env, "state_dict"):
                    payload["env"] = env.state_dict()
                with _atomic_output(checkpoint) as stream:
                    torch.save(payload, stream)
            result["completed_updates"] = ppo.update_count
            write_progress(metrics, result)
            print(json.dumps({"event": "update", **update}), flush=True)
        result["status"] = "completed"
        write_progress(metrics, result)
        return result
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        if app is not None:
            # Keep Kit failures and the final runner result observable; the
            # immediate ``skip_cleanup`` path can exit Python before either is
            # printed.
            app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "torch", "isaaclab"), default="mock")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps-per-env", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--lr-schedule-updates", type=int)
    parser.add_argument("--critic-completion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--learning-epochs", type=int, default=None,
                        help="PPO learning epochs per update (default: A6 assumption)")
    parser.add_argument("--mini-batches", type=int, default=None,
                        help="PPO minibatches per epoch (default: A6 assumption)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-scratch-ablation", action="store_true")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reference-data")
    parser.add_argument("--asset")
    parser.add_argument("--urdf")
    parser.add_argument("--fall-pool", type=Path)
    from pgmt.envs.actuators import ACTUATOR_PROFILES
    parser.add_argument("--actuator-profile", choices=ACTUATOR_PROFILES, default="asset_effort_v1")
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics", type=Path, help="write final training metrics as JSON")
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps_per_env, args.updates) <= 0:
        parser.error("num-envs, steps-per-env, and updates must be positive")
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
