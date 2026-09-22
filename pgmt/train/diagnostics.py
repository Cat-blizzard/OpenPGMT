"""Opt-in, bounded anomaly evidence. No RNG draws or policy/dynamics changes.

Snapshots are detached CPU tensors. Limits apply per event kind across restarts
in the same directory; intervals apply within each process. Implicit actuator
torques are estimates, not measured solver torques. All thresholds here are
recording triggers only, never PPO guards, rewards, or termination settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile

import torch
from torch.distributions import kl_divergence

from data.retarget_lafan1 import G1_JOINT_NAMES


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


@dataclass(frozen=True)
class DiagnosticConfig:
    max_events: int = 4  # per kind, including KL model snapshot pairs
    max_samples: int = 4  # states per event
    min_interval: int = 25  # control steps (physics) or updates (KL)
    kl_threshold: float = .1
    ee_error_threshold: float = 100.  # m/s^2, Euclidean norm per end effector

    def __post_init__(self):
        if self.max_events <= 0 or self.max_samples <= 0 or self.min_interval < 0:
            raise ValueError("invalid diagnostic limits")
        if not (0 < self.kl_threshold < float("inf") and 0 < self.ee_error_threshold < float("inf")):
            raise ValueError("diagnostic thresholds must be positive and finite")


class BoundedDiagnostics:
    kinds = ("first_action", "joint_speed", "ee_acceleration", "kl", "actuators")

    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config or DiagnosticConfig()
        self.counts = {kind: len(list(self.directory.glob(kind + "_*.pt"))) for kind in self.kinds}
        self.last = {}

    def ready(self, kind, step):
        return (self.counts[kind] < self.config.max_events
                and step - self.last.get(kind, -self.config.min_interval) >= self.config.min_interval)

    def emit(self, kind, step, payload):
        if not self.ready(kind, step):
            return
        record = cpu_copy({"schema": "pgmt.diagnostic.v1", "kind": kind, "step": step,
                           "recording_config": asdict(self.config), **payload})
        # Link a complete temporary file exclusively: never overwrite evidence.
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
        try:
            torch.save(record, temporary)
            index = self.counts[kind]
            while True:
                path = self.directory / f"{kind}_{index:03d}.pt"
                try:
                    os.link(temporary, path)
                    break
                except FileExistsError:
                    index += 1
                    if index >= self.config.max_events:
                        self.counts[kind] = index
                        return
            self.counts[kind] = index + 1
            self.last[kind] = step
        finally:
            temporary.unlink(missing_ok=True)

    def context(self, core, ids):
        result = {"env_ids": ids, "joint_names": list(G1_JOINT_NAMES), "control_dt": core.cfg.control_dt,
                  "backend": "isaaclab" if core.articulation is not None else "torch_protocol_only"}
        for key in ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
                    "target", "action", "_prev_action", "_action_queue", "_action_delay", "history",
                    "episode_length_buf", "_recovery_active", "reference_seq_idx", "reference_frame",
                    "reference_qpos", "reference_qvel", "priv_motor_strength"):
            value = getattr(core, key, None)
            result[key] = None if value is None else value[ids]
        if core.reference_database is not None and core.reference_seq_idx is not None:
            result["sequence_names"] = [core.reference_database.seqs[i]["name"]
                                         for i in core.reference_seq_idx[ids].tolist()]
        return cpu_copy(result)

    def before_action(self, core):
        if not self.ready("first_action", core._diagnostic_step):
            return None
        ids = (core.episode_length_buf == 0).nonzero().flatten()[:self.config.max_samples]
        return None if not len(ids) else (ids, self.context(core, ids))

    def after_action(self, core, first):
        ids, before = first
        self.emit("first_action", core._diagnostic_step, {"before": before, "after": self.context(core, ids)})

    def _joint_evidence(self, core, ids):
        result = self.context(core, ids)
        result["effort_limit_nm"] = core.pd.torque_limit
        data = None if core.articulation is None else core.articulation.data
        tau = getattr(data, "applied_torque", None)
        if tau is not None:
            tau = tau[ids][:, core.joint_ids]
            result["torque_source"] = "isaaclab_implicit_pd_estimate_last_substep"
        else:
            tau = core.pd(core.qpos[ids], core.qvel[ids], core.target[ids])
            result["torque_source"] = "nominal_pd_estimate_at_snapshot"
        result["estimated_torque_nm"] = tau
        result["estimated_saturation_fraction"] = (tau.abs() >= .99 * core.pd.torque_limit).float().mean(-1)
        result["overspeed_mask"] = core.qvel[ids].abs() > core.cfg.max_joint_velocity * 1.5
        return result

    def physics(self, core, state, reference):
        step = core._diagnostic_step
        if self.ready("joint_speed", step):
            peak = core.qvel.abs().amax(-1)
            ids = (peak > core.cfg.max_joint_velocity * 1.5).nonzero().flatten()
            if len(ids):
                ids = ids[peak[ids].argsort(descending=True)[:self.config.max_samples]]
                self.emit("joint_speed", step, self._joint_evidence(core, ids))
        if self.ready("ee_acceleration", step) and "body_accel" in reference:
            names = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
            indices = [core._reward_computer.bi[n] for n in names]
            actual, ref = state["body_accel"][:, indices], reference["body_accel"][:, indices]
            error = (actual - ref).norm(dim=-1)
            peak = error.amax(-1)
            ids = (peak > self.config.ee_error_threshold).nonzero().flatten()
            if len(ids):
                ids = ids[peak[ids].argsort(descending=True)[:self.config.max_samples]]
                self.emit("ee_acceleration", step, {**self._joint_evidence(core, ids),
                    "body_names": names, "state_accel_m_s2": actual[ids], "reference_accel_m_s2": ref[ids],
                    "error_m_s2": error[ids], "previous_body_velocity_m_s": core._previous_body_lin_vel[ids][:, indices],
                    "body_velocity_m_s": core.body_lin_vel[ids][:, indices]})

    @torch.no_grad()
    def capture_kl(self, ppo, behavior, before):
        step = ppo.update_count
        if before is None or not self.ready("kl", step):
            return
        selected, offset = None, 0
        for obs, old in behavior:
            new = ppo.policy.latent_distribution(obs)
            per_joint = kl_divergence(old, new)
            scores = per_joint.sum(-1)
            local = (scores > self.config.kl_threshold).nonzero().flatten()
            if len(local):
                local = local[scores[local].argsort(descending=True)[:self.config.max_samples]]
                values = {"indices": local + offset, "old_loc": old.loc[local], "old_scale": old.scale[local],
                          "new_loc": new.loc[local], "new_scale": new.scale[local], "kl_per_joint": per_joint[local],
                          **{"obs/" + k: v[local] for k, v in obs.items()}}
                if selected is not None:
                    values = {k: torch.cat((selected[k], v)) for k, v in values.items()}
                keep = values["kl_per_joint"].sum(-1).argsort(descending=True)[:self.config.max_samples]
                selected = {k: v[keep] for k, v in values.items()}
            offset += old.loc.shape[0]
        if selected is None:
            return
        ids = selected.pop("indices")
        observations = {k[4:]: selected.pop(k) for k in list(selected) if k.startswith("obs/")}
        labels = ppo._rollout_recovery
        self.emit("kl", step, {
            "heads": list(ppo.policy.head_names), "training_contract": ppo.training_contract,
            "ppo_config": asdict(ppo.config), "rollout_flat_indices": ids,
            "rollout_step": ids // ppo.storage.num_envs, "env_ids": ids % ppo.storage.num_envs,
            "recovery": None if labels is None else labels[ids],
            "observations": observations, **selected,
            "policy_before": before, "policy_after": ppo.policy.state_dict(),
            "snapshot_scope": "entire accepted update including private critic completion",
        })


def attach_diagnostics(env, directory):
    if directory is None:
        return None
    recorder = BoundedDiagnostics(directory)
    core = getattr(getattr(env, "env", env), "core", env)
    if hasattr(core, "cfg"):
        core.diagnostics = recorder
        if core.articulation is not None:
            view = core.articulation.root_physx_view
            ids = core.joint_ids.cpu()
            effort = view.get_dof_max_forces().cpu()[:, ids]
            expected = core.pd.torque_limit.cpu().expand_as(effort)
            if not torch.allclose(effort, expected, atol=1e-4, rtol=1e-5):
                raise RuntimeError("PhysX effort limits differ from the configured canonical joint limits")
            recorder.emit("actuators", 0, {"joint_names": list(G1_JOINT_NAMES), "effort_verified": True,
                "environment_config": asdict(core.cfg), "runtime_joint_ids": ids,
                "effort_nm": effort[:recorder.config.max_samples],
                "velocity_rad_s": view.get_dof_max_velocities().cpu()[:recorder.config.max_samples, ids],
                "stiffness_nm_rad": view.get_dof_stiffnesses().cpu()[:recorder.config.max_samples, ids],
                "damping_nm_s_rad": view.get_dof_dampings().cpu()[:recorder.config.max_samples, ids]})
    return recorder
