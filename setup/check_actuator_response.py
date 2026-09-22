"""Manual GPU-only neutral/reference PD probe, with no policy learning.

Each invocation tests one effort profile and exits at the requested step count.
Free-standing reference playback is a controller diagnostic, not a PPO score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from pgmt.envs.actuators import ACTUATOR_PROFILES, actuator_options
from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.train_stage1 import _seed_everything, _write_metrics, build_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("asset", "urdf", "reference-data"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--target-mode", choices=("neutral", "reference"), default="neutral")
    p.add_argument("--actuator-profile", choices=ACTUATOR_PROFILES, required=True)
    p.add_argument("--reset-mode", choices=("nominal", "reference_state"), default="nominal")
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".diagnostics").exists():
        p.error("use a fresh output path; refusing to overwrite probe evidence")
    if a.steps <= 0 or a.num_envs <= 0:
        p.error("steps and num-envs must be positive")
    _seed_everything(a.seed)
    from pgmt.envs.isaac_app import launch_isaac_app
    app, env = launch_isaac_app(a.device), None
    report = {"status": "running", "scope": "PD response diagnostic, no learning, ordinary resets only",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "steps": []}
    try:
        env = build_env(a.num_envs, torch.device(a.device), backend="isaaclab", app=app,
                        reference_data=a.reference_data, asset_path=a.asset, reference_urdf=a.urdf,
                        env_options={"seed": a.seed, "enable_adaptive_sampling": False,
                                     "randomize_dynamics": False, "corrupt_observations": False,
                                     "max_action_delay": 0, "reset_mode": a.reset_mode,
                                     **actuator_options(a.actuator_profile)})
        recorder = attach_diagnostics(env, a.output.with_suffix(".diagnostics"))
        env.reset()
        core = env.env.core
        report["environment_config"] = asdict(core.cfg)
        report["initial"] = recorder.context(core, torch.arange(a.num_envs, device=core.device))
        from pgmt.envs.reset_geometry import CollisionFloor
        geometry = core._reset_collision_floor or CollisionFloor(a.urdf, core._reference_motion.body_names, core.device)
        reference = core._reference_batch(core.reference_seq_idx, core.reference_frame)
        desired = {name: reference[key] for name, key in (("qpos", "joint_pos"), ("qvel", "joint_vel"),
            ("root_pos", "root_pos"), ("root_quat", "root_quat"),
            ("root_lin_vel", "root_lin_vel"), ("root_ang_vel", "root_ang_vel"))}
        errors = {name: (getattr(core, name) - value).abs().max().item() for name, value in desired.items()}
        clearance = geometry.min_height(core.body_pos, core.body_quat)
        report["reset_audit"] = cpu_copy({"expected_reference_state": desired, "state_max_abs_error": errors,
            "collision_min_height_m": clearance, "height_lift_m": core._reset_height_lift,
            "body_lin_vel": core.body_lin_vel, "previous_body_lin_vel": core._previous_body_lin_vel,
            "reference_root_pos": reference["root_pos"], "reference_root_quat": reference["root_quat"],
            "reference_root_lin_vel": reference["root_lin_vel"], "reference_root_ang_vel": reference["root_ang_vel"],
            "contact_forces": core.contact_forces,
            "geometric_floor_source": "URDF collisions; excludes PhysX contact/rest offsets"})
        _write_metrics(a.output, report)
        if a.reset_mode == "reference_state":
            if max(errors.values()) > 2e-4:
                raise RuntimeError(f"reference reset physical readback mismatch: {errors}")
            if clearance.min() < -2e-4:
                raise RuntimeError("reference reset collision geometry penetrates the floor")
            for name in ("action", "_prev_action", "target"):
                torch.testing.assert_close(getattr(core, name), core.reference_qpos, atol=2e-5, rtol=0)
            torch.testing.assert_close(core._action_queue, core.reference_qpos[:, None].expand_as(core._action_queue))
            if core.history[:, :-1].abs().max() != 0:
                raise RuntimeError("reference reset retains stale observation history")
        report["first_episode_end"] = [None] * a.num_envs
        # Read before auto-reset. This hook observes the reward input state;
        # it leaves targets, reward, termination, and dynamics unchanged.
        reward_impl = core._reward
        evidence = {}
        ee_names = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
        ee_ids = [core._reward_computer.bi[name] for name in ee_names]
        report["ee_names"] = list(ee_names)

        def observed_reward():
            actual = (core.body_lin_vel[:, ee_ids] - core._previous_body_lin_vel[:, ee_ids]) / core.cfg.control_dt
            reference = core.reference_body["body_accel"][:, ee_ids]
            tau = core.articulation.data.applied_torque[:, core.joint_ids]
            evidence.clear()
            evidence.update(cpu_copy({"qpos": core.qpos, "qvel": core.qvel, "target": core.target,
                "estimated_torque_nm": tau, "root_pos": core.root_pos, "root_quat": core.root_quat,
                "episode_steps": core.episode_length_buf, "reference_seq_idx": core.reference_seq_idx,
                "reference_frame": core.reference_frame, "state_ee_accel": actual,
                "reference_ee_accel": reference}))
            return reward_impl()

        core._reward = observed_reward
        _write_metrics(a.output, report)
        for step in range(a.steps):
            desired = core.default_q if a.target_mode == "neutral" else core.reference_qpos
            # There is no PPO here. Explicitly record any reference target
            # projected into the policy's rounded legal joint interval.
            targets = desired.clamp(core.joint_low, core.joint_high)
            projected = (desired - targets).abs().max().item()
            _, reward, terminated, timeout, info = env.step(targets)
            if not torch.isfinite(reward).all():
                raise RuntimeError("non-finite probe reward")
            if not all(torch.isfinite(v).all() for v in evidence.values()):
                raise RuntimeError("non-finite physical probe state")
            reasons = info.get("termination_reasons", {})
            for eid in (terminated | timeout).nonzero().flatten().tolist():
                if report["first_episode_end"][eid] is None:
                    report["first_episode_end"][eid] = {"step": step + 1,
                        "duration_s": float(info["episode_elapsed_s"][eid]),
                        "terminated": bool(terminated[eid]), "timeout": bool(timeout[eid]),
                        "reasons": [name for name, value in reasons.items() if bool(value[eid])]}
            report["steps"].append({"step": step + 1, "target_projection_max_rad": projected,
                                    "terminated": terminated.sum().item(), "timeout": timeout.sum().item(),
                                    "tracking": {k: v.float().mean().item() for k, v in info.get("tracking", {}).items()},
                                    "termination_reasons": {k: v.sum().item() for k, v in reasons.items()},
                                    "per_env_tracking": cpu_copy(info.get("tracking", {})),
                                    "pre_reset": dict(evidence)})
            if (step + 1) % 25 == 0:
                _write_metrics(a.output, report)
        report["status"] = "completed"
        _write_metrics(a.output, report)
    except BaseException:
        import traceback
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc()
        _write_metrics(a.output, report)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
