"""G1 tracking environment adapter.

This module keeps the simulator boundary deliberately small.  ``G1Env`` is a
torch-only batch adapter which owns the action/PD/reference/observation
contract and is usable in unit tests without Isaac Lab.  ``IsaacLabG1Env`` is
an optional :class:`isaaclab.envs.DirectRLEnv` shell.  The latter is imported
only after Isaac Sim has been bootstrapped (importing it in a normal Python
process would otherwise import ``omni`` and fail).

The adapter never downloads or copies robot files.  ``asset_path`` must point
to a user-provided, licensed USD/URDF.  Runtime name resolution is strict for
the 29 actuated joints and explicit for the bodies used by the reward code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.train.stage1 import BatchRewardAdapter, TorchMotionDatabase
from pgmt.rewards.spec import SIGMAS

try:  # importing this file must stay possible without Isaac Sim
    from data.retarget_lafan1 import G1_JOINT_LIMITS, G1_JOINT_NAMES
except Exception:  # pragma: no cover - only for deliberately minimal installs
    G1_JOINT_NAMES = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
        "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
        "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
        "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
        "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
        "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
        "right_wrist_pitch", "right_wrist_yaw",
    ]
    G1_JOINT_LIMITS = {name: (-np.pi, np.pi) for name in G1_JOINT_NAMES}


REQUIRED_BODY_NAMES = (
    "pelvis", "torso_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link",
)


def _tuple_floats(value: float | Sequence[float], n: int, name: str) -> tuple[float, ...]:
    if np.isscalar(value):
        out = (float(value),) * n
    else:
        out = tuple(float(v) for v in value)
        if len(out) != n:
            raise ValueError(f"{name} must have length {n}, got {len(out)}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return out


def _canonical_name(name: str) -> str:
    """Map asset naming variants to the project's canonical G1 names.

    Unitree URDF/USD assets conventionally expose ``*_joint`` names, while the
    retargeting contract stores the same actuators without that suffix.  The
    suffix is the only accepted alias; silently stripping arbitrary prefixes
    would make a wrong asset appear valid.
    """
    name = str(name)
    return name[:-6] if name.endswith("_joint") else name


def resolve_name_indices(actual_names: Sequence[str], expected_names: Sequence[str], *, kind: str = "name") -> np.ndarray:
    """Resolve canonical names to runtime indices, preserving canonical order.

    The G1 asset is allowed to expose fixed bodies in addition to the reward
    bodies, but every actuated joint is required exactly once.  Failing here is
    much safer than silently training a policy against a permuted robot.
    """
    actual = list(actual_names)
    canonical = [_canonical_name(name) for name in actual]
    if len(set(canonical)) != len(canonical):
        duplicates = sorted({n for n in canonical if canonical.count(n) > 1})
        raise ValueError(f"runtime {kind} names contain duplicates: {duplicates}")
    index = {name: i for i, name in enumerate(canonical)}
    missing = [name for name in expected_names if name not in index]
    if missing:
        raise ValueError(f"runtime asset is missing {kind}s: {missing}")
    return np.asarray([index[name] for name in expected_names], dtype=np.int64)


def validate_g1_mapping(
    joint_names: Sequence[str], body_names: Sequence[str], *,
    required_body_names: Sequence[str] = REQUIRED_BODY_NAMES,
) -> dict[str, np.ndarray]:
    """Validate and return canonical-to-runtime maps for an instantiated asset."""
    if len(joint_names) != ACT_DIM:
        raise ValueError(f"G1 requires exactly {ACT_DIM} actuated joints, got {len(joint_names)}")
    joint_ids = resolve_name_indices(joint_names, G1_JOINT_NAMES, kind="joint")
    body_ids = resolve_name_indices(body_names, required_body_names, kind="body")
    return {"joint_ids": joint_ids, "body_ids": body_ids}


@dataclass(frozen=True)
class AssetPreflight:
    """Read-only asset check; no conversion or copying is performed."""

    path: str
    exists: bool
    suffix: str
    parsed_joint_names: tuple[str, ...] = ()
    missing_joints: tuple[str, ...] = ()
    extra_joints: tuple[str, ...] = ()
    needs_runtime_mapping: bool = False
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.exists and not self.missing_joints and not self.extra_joints


def preflight_asset(asset_path: str | os.PathLike[str]) -> AssetPreflight:
    """Inspect a local USD/URDF/MJCF/XML path without touching external assets.

    USD joint names are intentionally deferred to runtime because parsing a USD
    stage requires Isaac Sim.  URDF/XML files are parsed conservatively by
    looking for ``<joint name=... type=...>`` entries and checking the 29 names.
    """
    path = Path(asset_path).expanduser()
    suffix = path.suffix.lower()
    if not path.exists():
        return AssetPreflight(str(path), False, suffix, message="asset path does not exist")
    if suffix not in {".usd", ".usda", ".usdc", ".urdf", ".xml", ".mjcf"}:
        return AssetPreflight(str(path), True, suffix, message="unsupported asset suffix")
    if suffix in {".usd", ".usda", ".usdc"}:
        return AssetPreflight(str(path), True, suffix, needs_runtime_mapping=True,
                              message="USD mapping is checked after Isaac Lab instantiates the articulation")
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = tuple(dict.fromkeys(re.findall(r"<joint\b[^>]*\bname\s*=\s*[\"']([^\"']+)", text)))
    canonical_names = tuple(_canonical_name(name) for name in names)
    expected = set(G1_JOINT_NAMES)
    found = set(canonical_names)
    missing = tuple(name for name in G1_JOINT_NAMES if name not in found)
    # A URDF can include fixed joints; only extra *actuated* names matter.
    actuated = set(re.findall(r"<joint\b(?=[^>]*\btype\s*=\s*[\"'](?:revolute|continuous|prismatic)[\"'])[^>]*\bname\s*=\s*[\"']([^\"']+)", text))
    extra = tuple(sorted(_canonical_name(name) for name in actuated - {f"{n}_joint" for n in expected}))
    message = "URDF/XML joint names parsed; runtime body and actuator mapping remains required"
    return AssetPreflight(str(path), True, suffix, canonical_names, missing, extra,
                          needs_runtime_mapping=False, message=message)


@dataclass
class G1EnvConfig:
    """Simulator-independent part of the Stage 1 environment contract."""

    num_envs: int = 1
    device: str = "cpu"
    control_dt: float = 0.02
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 30.0
    asset_path: str | None = None
    action_scale: float = 0.5
    stiffness: float | Sequence[float] = 80.0
    damping: float | Sequence[float] = 2.0
    torque_limit: float | Sequence[float] = 120.0
    default_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.793)
    default_root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    default_joint_pos: Sequence[float] | None = None
    reference_data_dir: str | None = None
    max_joint_velocity: float = 30.0
    auto_reset: bool = True

    def __post_init__(self):
        if self.num_envs <= 0 or self.decimation <= 0:
            raise ValueError("num_envs and decimation must be positive")
        if self.control_dt <= 0 or self.sim_dt <= 0 or self.episode_length_s <= 0:
            raise ValueError("dt and episode_length_s must be positive")
        expected_dt = self.sim_dt * self.decimation
        if not np.isclose(expected_dt, self.control_dt, rtol=0, atol=1e-8):
            raise ValueError(f"control_dt ({self.control_dt}) must equal sim_dt*decimation ({expected_dt})")
        if self.action_scale <= 0 or self.max_joint_velocity <= 0:
            raise ValueError("action_scale and max_joint_velocity must be positive")
        self.stiffness = _tuple_floats(self.stiffness, ACT_DIM, "stiffness")
        self.damping = _tuple_floats(self.damping, ACT_DIM, "damping")
        self.torque_limit = _tuple_floats(self.torque_limit, ACT_DIM, "torque_limit")
        if self.default_joint_pos is None:
            self.default_joint_pos = tuple(0.0 for _ in range(ACT_DIM))
        self.default_joint_pos = _tuple_floats(self.default_joint_pos, ACT_DIM, "default_joint_pos")
        if len(self.default_root_pos) != 3 or len(self.default_root_quat) != 4:
            raise ValueError("default root pose must be (3,) and (4,)")
        qnorm = np.linalg.norm(self.default_root_quat)
        if not np.isfinite(qnorm) or qnorm < 1e-8:
            raise ValueError("default_root_quat must be finite and non-zero")

    @property
    def max_episode_steps(self) -> int:
        return int(np.ceil(self.episode_length_s / self.control_dt))

    @property
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        return tuple(tuple(float(x) for x in G1_JOINT_LIMITS[name]) for name in G1_JOINT_NAMES)


class PDController:
    """Batched position PD controller with explicit torque saturation."""

    def __init__(self, stiffness: Sequence[float], damping: Sequence[float], torque_limit: Sequence[float], device="cpu"):
        self.kp = torch.as_tensor(stiffness, dtype=torch.float32, device=device)
        self.kd = torch.as_tensor(damping, dtype=torch.float32, device=device)
        self.torque_limit = torch.as_tensor(torque_limit, dtype=torch.float32, device=device)

    def __call__(self, q: torch.Tensor, qd: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if q.shape != qd.shape or q.shape != target.shape or q.shape[-1] != ACT_DIM:
            raise ValueError(f"q, qd and target must have shape (B,{ACT_DIM})")
        tau = self.kp * (target - q) - self.kd * qd
        return torch.clamp(tau, -self.torque_limit, self.torque_limit)


def _quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Quaternion ``[w,x,y,z]`` to the first two rotation-matrix columns."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    # Isaac Lab's ``root_link_quat_w`` and the project reference quaternions
    # are both wxyz.  Keep the first two *columns* in the conventional 6-D
    # order (r00,r10,r20,r01,r11,r21).
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + z * w),
                        2 * (x * z - y * w), 2 * (x * y - z * w),
                        1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)


def _relative_rot6d(robot_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """Return ``R_reference.T @ R_robot`` as the contract's 6D encoding."""
    robot_quat = robot_quat / robot_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reference_quat = reference_quat / reference_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # q_rel = conjugate(q_reference) * q_robot, both in wxyz order.
    rw, rx, ry, rz = robot_quat.unbind(-1)
    sw, sx, sy, sz = reference_quat.unbind(-1)
    # conjugate(reference) = (sw,-sx,-sy,-sz)
    qw = sw * rw + sx * rx + sy * ry + sz * rz
    qx = sw * rx - sx * rw - sy * rz + sz * ry
    qy = sw * ry + sx * rz - sy * rw - sz * rx
    qz = sw * rz - sx * ry + sy * rx - sz * rw
    return _quat_to_rot6d(torch.stack((qw, qx, qy, qz), dim=-1))


class G1Env:
    """Batched Stage 1 contract, with a kinematic torch fallback.

    When ``articulation`` is supplied, the class only reads/writes its tensor
    interface and Isaac Lab performs the actual integration.  Without one it
    runs a deterministic PD integration useful for protocol tests and dry-run
    PPO; this fallback is not a physics result.
    """

    def __init__(self, config: G1EnvConfig | None = None, *, articulation: Any | None = None,
                 reference_database: Any | None = None):
        self.cfg = config or G1EnvConfig()
        self.articulation = articulation
        self.reference_database = reference_database
        if articulation is not None:
            actual_joint_names = list(getattr(articulation, "joint_names", getattr(getattr(articulation, "data", None), "joint_names", [])))
            actual_body_names = list(getattr(articulation, "body_names", getattr(getattr(articulation, "data", None), "body_names", [])))
            if not actual_joint_names or not actual_body_names:
                raise ValueError("articulation must expose joint_names and body_names")
            maps = validate_g1_mapping(actual_joint_names, actual_body_names)
            self.joint_ids = torch.as_tensor(maps["joint_ids"], dtype=torch.long)
            self.body_ids = torch.as_tensor(maps["body_ids"], dtype=torch.long)
            self.num_envs = int(getattr(articulation, "num_instances", getattr(articulation, "num_envs", self.cfg.num_envs)))
            self.device = torch.device(getattr(articulation, "device", self.cfg.device))
        else:
            self.num_envs = self.cfg.num_envs
            self.device = torch.device(self.cfg.device)
            self.joint_ids = torch.arange(ACT_DIM, dtype=torch.long, device=self.device)
            self.body_ids = torch.arange(len(REQUIRED_BODY_NAMES), dtype=torch.long, device=self.device)
        self.joint_ids = self.joint_ids.to(self.device)
        self.body_ids = self.body_ids.to(self.device)
        self.qpos = torch.zeros((self.num_envs, ACT_DIM), device=self.device)
        self.qvel = torch.zeros_like(self.qpos)
        self.root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.root_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.root_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.action = torch.zeros_like(self.qpos)
        self.target = torch.zeros_like(self.qpos)
        self.reference_qpos = torch.zeros_like(self.qpos)
        self.reference_qvel = torch.zeros_like(self.qvel)
        self.reference_root_pos = torch.zeros_like(self.root_pos)
        self.reference_root_quat = torch.zeros_like(self.root_quat)
        self.reference_root_lin_vel = torch.zeros_like(self.root_lin_vel)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pd = PDController(self.cfg.stiffness, self.cfg.damping, self.cfg.torque_limit, device=self.device)
        low, high = zip(*self.cfg.joint_limits)
        self.joint_low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.joint_high = torch.tensor(high, dtype=torch.float32, device=self.device)
        self.default_q = torch.tensor(self.cfg.default_joint_pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()
        self.default_root_pos = torch.tensor(self.cfg.default_root_pos, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()
        self.default_root_quat = torch.tensor(self.cfg.default_root_quat, dtype=torch.float32, device=self.device).expand(self.num_envs, -1).clone()
        self.history = torch.zeros((self.num_envs, HISTORY_LEN, OBS_DIM), device=self.device)
        self.reward_adapter = BatchRewardAdapter(device=self.device)
        self._prev_action = torch.zeros_like(self.action)
        self.reference_seq_idx = None
        self.reference_frame = None
        self._torch_reference = None
        if self.reference_database is None and self.cfg.reference_data_dir:
            from pgmt.envs.reference_sampler import MotionDatabase
            self.reference_database = MotionDatabase(self.cfg.reference_data_dir)
        if self.reference_database is not None:
            if isinstance(self.reference_database, TorchMotionDatabase):
                if self.reference_database.device != self.device:
                    raise ValueError(
                        f"TorchMotionDatabase is on {self.reference_database.device}, "
                        f"but G1Env is on {self.device}; construct both on the same device")
                self._torch_reference = self.reference_database
            else:
                self._torch_reference = TorchMotionDatabase(self.reference_database, device=self.device)
        self._last_obs = None
        self.reset()

    def _read_articulation(self):
        if self.articulation is None:
            return
        data = self.articulation.data
        self.qpos.copy_(data.joint_pos[:, self.joint_ids])
        self.qvel.copy_(data.joint_vel[:, self.joint_ids])
        self.root_pos.copy_(data.root_link_pos_w)
        self.root_quat.copy_(data.root_link_quat_w)
        self.root_lin_vel.copy_(data.root_link_lin_vel_w)
        self.root_ang_vel.copy_(data.root_link_ang_vel_w)

    def _obs(self) -> dict[str, torch.Tensor]:
        # The root/reference pose starts aligned in Stage 1.  Keeping the
        # quaternion conversion here makes a future world-to-anchor extension
        # explicit rather than hiding NumPy work in the rollout hot path.
        e6 = _relative_rot6d(self.root_quat, self.reference_root_quat)
        obs = torch.cat((e6, self.root_ang_vel, self.qpos, self.qvel, self.action), dim=-1)
        self._last_obs = {"obs": obs}
        return self._last_obs

    def _build_observations(self) -> dict[str, torch.Tensor]:
        current = self._obs()
        # history is the ten frames *before* o_t; update after making current.
        history = self.history.clone()
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = current["obs"]
        if hasattr(self, "reference_future"):
            future = self.reference_future
        else:
            future = torch.cat((self.reference_qpos.unsqueeze(1).expand(-1, 6, -1),
                                self.reference_qvel.unsqueeze(1).expand(-1, 6, -1),
                                self.reference_root_lin_vel[:, None, :].expand(-1, 6, -1)), dim=-1)
        privileged = torch.zeros((self.num_envs, PRIV_DIM), device=self.device)
        # A9 base velocity and angular velocity occupy the first six slots.
        privileged[:, :3] = self.root_lin_vel
        privileged[:, 3:6] = self.root_ang_vel
        return {"obs": current["obs"], "history": history, "future": future, "privileged": privileged}

    def _set_reference_defaults(self, env_ids: torch.Tensor):
        self.reference_qpos[env_ids] = self.default_q[env_ids]
        self.reference_qvel[env_ids] = 0.0
        self.reference_root_pos[env_ids] = self.default_root_pos[env_ids]
        self.reference_root_quat[env_ids] = self.default_root_quat[env_ids]
        self.reference_root_lin_vel[env_ids] = 0.0

    def _sample_reference(self, env_ids: torch.Tensor) -> None:
        """Sample/update reference motion for a subset of environments."""
        if self._torch_reference is None:
            self._set_reference_defaults(env_ids)
            if not hasattr(self, "reference_future"):
                self.reference_future = torch.zeros((self.num_envs, 6, REF_FRAME_DIM), device=self.device)
            self.reference_future[env_ids, :, :ACT_DIM] = self.reference_qpos[env_ids, None]
            self.reference_future[env_ids, :, ACT_DIM:2 * ACT_DIM] = self.reference_qvel[env_ids, None]
            self.reference_future[env_ids, :, 2 * ACT_DIM:] = self.reference_root_lin_vel[env_ids, None]
            return
        seq, frame = self._torch_reference.sample_segments(int(env_ids.numel()), seg_len=1)
        if self.reference_seq_idx is None:
            self.reference_seq_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.reference_frame = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reference_seq_idx[env_ids] = seq
        self.reference_frame[env_ids] = frame.to(torch.float32)
        batch = self._torch_reference.batch(seq, self.reference_frame[env_ids])
        self.reference_qpos[env_ids] = batch.qpos
        self.reference_qvel[env_ids] = batch.qvel
        self.reference_root_pos[env_ids] = batch.root_pos
        self.reference_root_quat[env_ids] = batch.root_rot
        self.reference_root_lin_vel[env_ids] = batch.future[:, 0, -3:]
        if not hasattr(self, "reference_future"):
            self.reference_future = torch.zeros((self.num_envs, 6, REF_FRAME_DIM), device=self.device)
        self.reference_future[env_ids] = batch.future

    def _advance_reference(self) -> None:
        if self._torch_reference is None:
            return
        self.reference_frame += self.cfg.control_dt / self._torch_reference.frame_time[self.reference_seq_idx]
        batch = self._torch_reference.batch(self.reference_seq_idx, self.reference_frame)
        self.reference_qpos.copy_(batch.qpos)
        self.reference_qvel.copy_(batch.qvel)
        self.reference_root_pos.copy_(batch.root_pos)
        self.reference_root_quat.copy_(batch.root_rot)
        self.reference_root_lin_vel.copy_(batch.future[:, 0, -3:])
        self.reference_future.copy_(batch.future)

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        ids = torch.arange(self.num_envs, device=self.device)
        self.qpos.copy_(self.default_q)
        self.qvel.zero_()
        self.root_pos.copy_(self.default_root_pos)
        self.root_quat.copy_(self.default_root_quat)
        self.root_lin_vel.zero_()
        self.root_ang_vel.zero_()
        self.action.zero_()
        self._prev_action.zero_()
        self.target.copy_(self.default_q)
        self.episode_length_buf.zero_()
        self.history.zero_()
        self._sample_reference(ids)
        if self.articulation is not None:
            root_state = torch.cat((self.default_root_pos, self.default_root_quat,
                                    torch.zeros((self.num_envs, 6), device=self.device)), dim=-1)
            self.articulation.write_root_state_to_sim(root_state)
            self.articulation.write_joint_state_to_sim(self.default_q, torch.zeros_like(self.default_q), joint_ids=self.joint_ids)
            self.articulation.set_joint_position_target(self.target[:, :], joint_ids=self.joint_ids)
        return self._build_observations()

    def _apply_action(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.shape != (self.num_envs, ACT_DIM):
            raise ValueError(f"actions must have shape ({self.num_envs},{ACT_DIM}), got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("actions contain NaN/Inf")
        self.action.copy_(actions.clamp(-1.0, 1.0))
        self.target.copy_(torch.clamp(self.default_q + self.cfg.action_scale * self.action,
                                      self.joint_low, self.joint_high))
        if self.articulation is not None:
            self.articulation.set_joint_position_target(self.target, joint_ids=self.joint_ids)

    def _fallback_integrate(self):
        # Deliberately simple and deterministic dry-run dynamics.  Isaac Lab's
        # implicit actuators provide the actual rigid-body dynamics.
        for _ in range(self.cfg.decimation):
            tau = self.pd(self.qpos, self.qvel, self.target)
            self.qvel.add_(tau * self.cfg.sim_dt)
            self.qvel.clamp_(-self.cfg.max_joint_velocity, self.cfg.max_joint_velocity)
            self.qpos.add_(self.qvel * self.cfg.sim_dt)
        self.qpos.clamp_(self.joint_low, self.joint_high)

    def _reward(self) -> torch.Tensor:
        q_abs = (self.qpos - self.reference_qpos).abs()
        v_abs = (self.qvel - self.reference_qvel).abs()
        upper = torch.arange(12, ACT_DIM, device=self.device)
        lower = torch.arange(0, 12, device=self.device)
        root_vel_err = (self.root_lin_vel - self.reference_root_lin_vel).square().mean(-1).sqrt()
        anchor_err = (self.root_pos - self.reference_root_pos).square().mean(-1).sqrt()
        values = {
            "link_pos": q_abs[:, upper].mean(-1),
            "link_ori": q_abs[:, upper].mean(-1),
            "link_lin_vel": v_abs[:, upper].mean(-1),
            "link_ang_vel": v_abs[:, upper].mean(-1),
            "joint_pos": q_abs[:, upper].mean(-1),
            "joint_vel": v_abs[:, upper].mean(-1),
            "ta_link_pos": q_abs[:, lower].mean(-1),
            "ta_link_ori": q_abs[:, lower].mean(-1),
            "ta_joint_pos": q_abs[:, lower].mean(-1),
            "root_ori": torch.ones(self.num_envs, device=self.device),
            "corrected_root_vel": torch.exp(-root_vel_err.square() / SIGMAS["link_lin_vel"]),
            "floating_anchor_pos": torch.exp(-anchor_err.square() / 1.0),
            "recovery_upward_vel": torch.ones(self.num_envs, device=self.device),
            "pelvis_vert_accel": torch.zeros(self.num_envs, device=self.device),
            "ee_accel_mismatch": torch.zeros(self.num_envs, device=self.device),
            "action_rate": (self.action - self._prev_action).square().mean(-1),
            "joint_limit": (torch.relu(self.joint_low - self.qpos).square()
                             + torch.relu(self.qpos - self.joint_high).square()).sum(-1),
            "undesired_contact": torch.zeros(self.num_envs, device=self.device),
            "head_torso_impact": torch.zeros(self.num_envs, device=self.device),
        }
        rewards = self.reward_adapter(values)
        self._prev_action.copy_(self.action)
        return rewards

    def _dones(self):
        timeout = self.episode_length_buf + 1 >= self.cfg.max_episode_steps
        # Conservative physics termination.  The fallback never falls unless
        # a caller writes an invalid root state; Isaac Lab gets real contacts.
        terminated = (self.root_pos[:, 2] < 0.25) | (self.qvel.abs().amax(-1) > self.cfg.max_joint_velocity * 1.5)
        return terminated, timeout & ~terminated

    def step(self, actions: torch.Tensor):
        self._read_articulation()
        self._apply_action(actions)
        if self.articulation is None:
            self._fallback_integrate()
        self.episode_length_buf += 1
        self._read_articulation()
        self._advance_reference()
        rewards = self._reward()
        terminated, truncated = self._dones()
        terminal_observation = None
        if truncated.any():
            # Preserve the pre-reset state for PPO timeout bootstrapping.
            terminal_observation = {key: value.clone() for key, value in self._build_observations().items()}
        done = terminated | truncated
        if self.cfg.auto_reset and done.any():
            ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            # The current implementation resets all fallback buffers; this is
            # safe and deterministic, and a per-env path is used for tensors.
            self.qpos[ids] = self.default_q[ids]
            self.qvel[ids] = 0.0
            self.root_pos[ids] = self.default_root_pos[ids]
            self.root_quat[ids] = self.default_root_quat[ids]
            self.root_lin_vel[ids] = 0.0
            self.root_ang_vel[ids] = 0.0
            self.action[ids] = 0.0
            self.target[ids] = self.default_q[ids]
            self.episode_length_buf[ids] = 0
            self.history[ids] = 0.0
            self._sample_reference(ids)
        observations = self._build_observations()
        info: dict[str, Any] = {"episode_length": self.episode_length_buf.clone()}
        if terminal_observation is not None:
            info["terminal_observation"] = terminal_observation
        return observations, rewards, terminated, truncated, info


# ---------------------------------------------------------------------------
# Optional Isaac Lab shell
# ---------------------------------------------------------------------------

try:  # do not import omni in normal unit-test / data-processing processes
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import Articulation, ArticulationCfg
    import isaaclab.sim as _sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    _ISAACLAB_IMPORTABLE = True
except Exception:  # pragma: no cover - exercised only on a non-Isaac process
    _ISAACLAB_IMPORTABLE = False


if _ISAACLAB_IMPORTABLE:

    def make_g1_articulation_cfg(asset_path: str, cfg: G1EnvConfig | None = None) -> Any:
        """Create an Isaac Lab ArticulationCfg for a user-provided USD/URDF."""
        c = cfg or G1EnvConfig(asset_path=asset_path)
        path = str(Path(asset_path).expanduser().resolve())
        if not Path(path).exists():
            raise FileNotFoundError(path)
        suffix = Path(path).suffix.lower()
        if suffix in {".urdf"}:
            spawn = _sim_utils.UrdfFileCfg(asset_path=path, fix_base=False)
        elif suffix in {".usd", ".usda", ".usdc"}:
            spawn = _sim_utils.UsdFileCfg(usd_path=path)
        else:
            raise ValueError("Isaac Lab G1 asset must be .usd/.usda/.usdc/.urdf")
        actuator = ImplicitActuatorCfg(
            # Both Unitree URDF-derived USD and the retargeting contract are
            # supported; Isaac Lab expressions accept the optional suffix.
            joint_names_expr=[f"{name}(_joint)?" for name in G1_JOINT_NAMES],
            stiffness=list(c.stiffness), damping=list(c.damping),
            effort_limit_sim=list(c.torque_limit),
        )
        return ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=c.default_root_pos, rot=c.default_root_quat,
                joint_pos={name: value for name, value in zip(G1_JOINT_NAMES, c.default_joint_pos)},
                joint_vel={name: 0.0 for name in G1_JOINT_NAMES},
            ),
            actuators={"g1_pd": actuator},
        )


    @configclass
    class G1DirectRLEnvCfg(DirectRLEnvCfg):
        """Isaac Lab 2.3 DirectRLEnv config for the G1 adapter."""

        decimation: int = 4
        episode_length_s: float = 30.0
        observation_space: int = OBS_DIM
        action_space: int = ACT_DIM
        state_space: int = PRIV_DIM
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)
        robot_cfg: Any = None
        pgmt_cfg: G1EnvConfig = field(default_factory=G1EnvConfig)


    class IsaacLabG1Env(DirectRLEnv):
        """Isaac Lab 2.3 DirectRLEnv implementation.

        Construct with ``cfg.robot_cfg = make_g1_articulation_cfg(path)`` after
        AppLauncher has initialized Isaac Sim.  Runtime mapping is checked in
        ``_setup_scene`` before any rollout starts.
        """

        cfg: G1DirectRLEnvCfg

        def __init__(self, cfg: G1DirectRLEnvCfg, **kwargs):
            if cfg.robot_cfg is None:
                raise ValueError("G1DirectRLEnvCfg.robot_cfg must be set to a licensed USD/URDF")
            self._pgmt_cfg = cfg.pgmt_cfg
            super().__init__(cfg, **kwargs)
            self.core = G1Env(self._pgmt_cfg, articulation=self.robot)
            self.last_adapter_obs = None
            self.last_split_reward = torch.zeros((self.num_envs, 3), device=self.device)
            self.last_terminal_obs = None

        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot_cfg)
            self.scene.articulations["robot"] = self.robot
            _sim_utils.spawn_ground_plane("/World/ground", _sim_utils.GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

        def _pre_physics_step(self, actions: torch.Tensor):
            self.core._apply_action(actions)

        def _apply_action(self):
            self.robot.set_joint_position_target(self.core.target, joint_ids=self.core.joint_ids)

        def _get_observations(self):
            self.core._read_articulation()
            obs = self.core._build_observations()
            self.last_adapter_obs = {key: value.clone() for key, value in obs.items()}
            # DirectRLEnv expects a single policy tensor; training adapters can
            # use ``core`` directly when they need history/future/privileged.
            return {"policy": obs["obs"]}

        def _get_rewards(self):
            self.core._read_articulation()
            self.core.episode_length_buf.copy_(self.episode_length_buf)
            self.core._advance_reference()
            self.last_terminal_obs = None
            if self.reset_time_outs.any():
                self.last_terminal_obs = {
                    key: value.clone() for key, value in self.core._build_observations().items()
                }
            self.last_split_reward = self.core._reward()
            return self.last_split_reward.sum(-1)

        def _get_dones(self):
            self.core._read_articulation()
            self.core.episode_length_buf.copy_(self.episode_length_buf)
            terminated, timeout = self.core._dones()
            return terminated, timeout

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            ids = torch.as_tensor(env_ids, device=self.core.device, dtype=torch.long)
            self.core.qpos[ids] = self.core.default_q[ids]
            self.core.qvel[ids] = 0.0
            self.core.root_pos[ids] = self.core.default_root_pos[ids]
            self.core.root_quat[ids] = self.core.default_root_quat[ids]
            self.core.root_lin_vel[ids] = 0.0
            self.core.root_ang_vel[ids] = 0.0
            self.core.action[ids] = 0.0
            self.core._prev_action[ids] = 0.0
            self.core.target[ids] = self.core.default_q[ids]
            self.core.history[ids] = 0.0
            self.core.episode_length_buf[ids] = 0
            self.core._sample_reference(ids)

        def _set_debug_vis_impl(self, debug_vis: bool):
            raise NotImplementedError

else:
    make_g1_articulation_cfg = None
    G1DirectRLEnvCfg = None
    IsaacLabG1Env = None


class IsaacLabPPOAdapter:
    """Expose ``IsaacLabG1Env`` through the repository's PPO batch protocol."""

    def __init__(self, env):
        if IsaacLabG1Env is None or not isinstance(env, IsaacLabG1Env):
            raise TypeError("env must be an initialized IsaacLabG1Env")
        self.env = env

    def reset(self):
        self.env.reset()
        return self.env.last_adapter_obs

    def step(self, actions):
        _, _, terminated, truncated, extras = self.env.step(actions)
        info = dict(extras)
        if self.env.last_terminal_obs is not None:
            info["terminal_observation"] = self.env.last_terminal_obs
        return self.env.last_adapter_obs, self.env.last_split_reward, terminated, truncated, info


__all__ = [
    "AssetPreflight", "G1Env", "G1EnvConfig", "G1DirectRLEnvCfg", "IsaacLabG1Env",
    "PDController", "REQUIRED_BODY_NAMES", "G1_JOINT_NAMES", "G1_JOINT_LIMITS",
    "make_g1_articulation_cfg", "preflight_asset", "resolve_name_indices", "validate_g1_mapping",
    "IsaacLabPPOAdapter",
]
