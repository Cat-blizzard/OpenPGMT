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
from pgmt.envs.observations import PRIV_DIM, PRIV_LAYOUT
from pgmt.train.stage1 import BatchRewardAdapter, TorchMotionDatabase
from pgmt.rewards.spec import SIGMAS
from pgmt.rewards.batched import BatchedRewardComputer
from pgmt.envs.recovery import FallRecoveryPool
from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.batched import TerrainAtlas
from pgmt.envs.termination import TerminationReason, termination_delay, tolerance_budget

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
    extra = tuple(sorted({_canonical_name(name) for name in actuated} - expected))
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
    # A URDF is required for the reference FK/body-level reward path.  The
    # simulator asset may be USD; keeping this path separate makes the
    # kinematic reference auditable and avoids parsing USD in the rollout.
    reference_urdf_path: str | None = None
    stage: int = 1
    enable_global_position_correction: bool = True
    enable_body_tracking: bool = True
    terrain_family: str = "flat"
    terrain_level: int = 0
    enable_adaptive_sampling: bool = True
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
        if self.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if self.terrain_level < 0 or self.terrain_level > 9:
            raise ValueError("terrain_level must be in [0,9]")
        if self.terrain_family not in {"flat", "slopes", "stairs", "boxes", "rough"}:
            raise ValueError(f"unknown terrain_family: {self.terrain_family}")
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


def _yaw(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    return torch.stack((torch.cos(half), torch.zeros_like(half),
                        torch.zeros_like(half), torch.sin(half)), dim=-1)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), -1).reshape(q.shape[:-1] + (3, 3))


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw), dim=-1)


class G1Env:
    """Batched Stage 1 contract, with a kinematic torch fallback.

    When ``articulation`` is supplied, the class only reads/writes its tensor
    interface and Isaac Lab performs the actual integration.  Without one it
    runs a deterministic PD integration useful for protocol tests and dry-run
    PPO; this fallback is not a physics result.
    """

    def __init__(self, config: G1EnvConfig | None = None, *, articulation: Any | None = None,
                 reference_database: Any | None = None,
                 recovery_pool: FallRecoveryPool | None = None):
        self.cfg = config or G1EnvConfig()
        self.articulation = articulation
        self.reference_database = reference_database
        self.recovery_pool = recovery_pool
        self._adaptive_sampler = None
        self._rng = np.random.default_rng()
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
        self._terrain = TerrainAtlas(self.device, stage=self.cfg.stage) if self.cfg.stage == 2 else None
        self._terrain_family_id = {"flat": 0, "slopes": 1, "stairs": 2, "boxes": 3, "rough": 4}[self.cfg.terrain_family]
        self._terrain_families = torch.full((self.num_envs,), self._terrain_family_id, dtype=torch.long, device=self.device)
        self._terrain_levels = torch.full((self.num_envs,), self.cfg.terrain_level, dtype=torch.long, device=self.device)
        _families = ("flat", "slopes", "stairs", "boxes", "rough")
        _term_cfg = get("A20").value
        self._termination_ref_threshold = torch.full(
            (self.num_envs,), float(_term_cfg.ref_deviation_base), device=self.device)
        self._termination_ref_delay = torch.full(
            (self.num_envs,), float(_term_cfg.delay_s["ref_deviation"]), device=self.device)
        for family_id, family in enumerate(_families):
            mask = self._terrain_families == family_id
            if mask.any():
                level = int(self.cfg.terrain_level)
                self._termination_ref_threshold[mask] = tolerance_budget(family, level, _term_cfg)
                self._termination_ref_delay[mask] = termination_delay(
                    TerminationReason.REF_DEVIATION, family, level, _term_cfg)
        self.qpos = torch.zeros((self.num_envs, ACT_DIM), device=self.device)
        self.qvel = torch.zeros_like(self.qpos)
        self.root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.root_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.root_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.body_pos = torch.zeros((self.num_envs, len(REQUIRED_BODY_NAMES), 3), device=self.device)
        self.body_quat = torch.zeros((self.num_envs, len(REQUIRED_BODY_NAMES), 4), device=self.device)
        self.body_lin_vel = torch.zeros_like(self.body_pos)
        self.body_ang_vel = torch.zeros_like(self.body_pos)
        self.contact_forces = torch.zeros_like(self.body_pos)
        # Keep sensor history separate from the current sample.  Stage 2
        # needs it to identify touchdown and rapid contact switching.
        self._previous_contact_forces = torch.zeros_like(self.contact_forces)
        self.foot_contact = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self._previous_foot_contact = torch.zeros_like(self.foot_contact)
        self._contact_age = torch.zeros((self.num_envs, 2), dtype=torch.float32,
                                        device=self.device)
        # A9 privileged dynamics.  These tensors are intentionally public so
        # an Isaac Lab randomization callback can write the actual values each
        # reset; defaults are the neutral nominal robot, never fake noise.
        self.priv_friction = torch.ones((self.num_envs, 2), device=self.device)
        self.priv_terrain_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.priv_mass = torch.ones((self.num_envs, 1), device=self.device)
        self.priv_com = torch.zeros((self.num_envs, 3), device=self.device)
        self.priv_push = torch.zeros((self.num_envs, 3), device=self.device)
        self.priv_motor_strength = torch.ones((self.num_envs, ACT_DIM), device=self.device)
        self._previous_body_lin_vel = torch.zeros_like(self.body_lin_vel)
        self._previous_root_lin_vel = torch.zeros_like(self.root_lin_vel)
        self._body_state_available = False
        self._contact_sensor_available = articulation is None
        self.action = torch.zeros_like(self.qpos)
        self.target = torch.zeros_like(self.qpos)
        self.reference_qpos = torch.zeros_like(self.qpos)
        self.reference_qvel = torch.zeros_like(self.qvel)
        self.reference_root_pos = torch.zeros_like(self.root_pos)
        self.reference_root_quat = torch.zeros_like(self.root_quat)
        self.reference_root_lin_vel = torch.zeros_like(self.root_lin_vel)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._termination_elapsed = {
            "ref_deviation": torch.zeros(self.num_envs, device=self.device),
            "root_low": torch.zeros(self.num_envs, device=self.device),
            "tilted": torch.zeros(self.num_envs, device=self.device),
        }
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
        self.reference_start_frame = None
        self._recovery_active = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reference_body = None
        self._torch_reference = None
        self._reference_motion = None
        self._reference_placement_pos = torch.zeros_like(self.root_pos)
        self._reference_placement_quat = self.default_root_quat.clone()
        self._reference_raw_root_pos = torch.zeros_like(self.root_pos)
        self._reference_raw_root_quat = self.default_root_quat.clone()
        self._reference_raw_root_lin_vel = torch.zeros_like(self.root_lin_vel)
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
            if self.cfg.enable_adaptive_sampling and not isinstance(self.reference_database, TorchMotionDatabase):
                from pgmt.envs.reference_sampler import AdaptiveSampler
                self._adaptive_sampler = AdaptiveSampler(self.reference_database, seg_len=1)
            urdf = self.cfg.reference_urdf_path
            if urdf is None and self.cfg.asset_path and str(self.cfg.asset_path).lower().endswith(".urdf"):
                urdf = self.cfg.asset_path
            if urdf and self.cfg.enable_body_tracking:
                from pgmt.envs.reference_motion import ReferenceMotion
                self._reference_motion = ReferenceMotion(
                    self._torch_reference, urdf, device=self.device,
                    control_dt=self.cfg.control_dt)
        if (self.articulation is not None and self.reference_database is not None
                and self.cfg.enable_body_tracking and self._reference_motion is None):
            raise ValueError(
                "physics G1Env requires reference_urdf_path (or an URDF asset) "
                "to compute body-level reference tracking")
        self._reward_computer = BatchedRewardComputer(
            REQUIRED_BODY_NAMES, G1_JOINT_NAMES, device=self.device,
            dt=self.cfg.control_dt)
        self._last_obs = None
        self._last_reward_metrics: dict[str, Any] = {}
        self.reset()

    def _read_articulation(self):
        if self.articulation is None:
            if self._reference_motion is not None:
                # Match the articulation path's finite-difference history so
                # fallback FK does not report a spurious first-step accel.
                self._previous_root_lin_vel.copy_(self.root_lin_vel)
                self._previous_body_lin_vel.copy_(self.body_lin_vel)
                kin = self._reference_motion.kinematics(
                    self.qpos, self.qvel, self.root_pos, self.root_quat,
                    self.root_lin_vel, self.root_ang_vel)
                self.body_pos.copy_(kin["body_pos"])
                self.body_quat.copy_(kin["body_quat"])
                self.body_lin_vel.copy_(kin["body_lin_vel"])
                self.body_ang_vel.copy_(kin["body_ang_vel"])
                self._body_state_available = True
            return
        data = self.articulation.data
        self._previous_root_lin_vel.copy_(self.root_lin_vel)
        self._previous_body_lin_vel.copy_(self.body_lin_vel)
        self.qpos.copy_(data.joint_pos[:, self.joint_ids])
        self.qvel.copy_(data.joint_vel[:, self.joint_ids])
        self.root_pos.copy_(data.root_link_pos_w)
        self.root_quat.copy_(data.root_link_quat_w)
        self.root_lin_vel.copy_(data.root_link_lin_vel_w)
        self.root_ang_vel.copy_(data.root_link_ang_vel_w)
        # Isaac Lab exposes body state tensors on ArticulationData.  Keep the
        # access explicit and fail closed: the fallback reward remains a dry
        # run, while a real articulation never silently trains on joint-error
        # proxies when body data are unavailable.
        body_pos = getattr(data, "body_link_pos_w", getattr(data, "body_pos_w", None))
        body_quat = getattr(data, "body_link_quat_w", getattr(data, "body_quat_w", None))
        body_lin = getattr(data, "body_link_lin_vel_w", getattr(data, "body_lin_vel_w", None))
        body_ang = getattr(data, "body_link_ang_vel_w", getattr(data, "body_ang_vel_w", None))
        if all(x is not None for x in (body_pos, body_quat, body_lin, body_ang)):
            self.body_pos.copy_(body_pos[:, self.body_ids])
            self.body_quat.copy_(body_quat[:, self.body_ids])
            self.body_lin_vel.copy_(body_lin[:, self.body_ids])
            self.body_ang_vel.copy_(body_ang[:, self.body_ids])
            net_force = getattr(data, "net_contact_forces_w", None)
            if net_force is None:
                net_force = getattr(data, "net_forces_w", None)
            if net_force is not None:
                self.contact_forces.copy_(net_force[:, self.body_ids])
            else:
                self.contact_forces.zero_()
            feet = torch.tensor([
                REQUIRED_BODY_NAMES.index("left_ankle_roll_link"),
                REQUIRED_BODY_NAMES.index("right_ankle_roll_link")], device=self.device)
            # A contact sensor is preferred; the force threshold is only a
            # compatibility fallback for articulations without one.
            self.foot_contact.copy_(self.contact_forces[:, feet].norm(dim=-1) > 1.0)
            self._body_state_available = True
        else:
            self._body_state_available = False

    def _obs(self) -> dict[str, torch.Tensor]:
        # The root/reference pose starts aligned in Stage 1.  Keeping the
        # quaternion conversion here makes a future world-to-anchor extension
        # explicit rather than hiding NumPy work in the rollout hot path.
        e6 = _relative_rot6d(self.root_quat, self.reference_root_quat)
        # The observation contract is expressed in the base frame.  Isaac Lab
        # exposes root-link velocities in world coordinates, so rotate them
        # explicitly before they reach the actor (and keep world velocities
        # available to the reward path).
        root_rot = _quat_to_mat(self.root_quat)
        omega_base = torch.bmm(root_rot.transpose(1, 2), self.root_ang_vel.unsqueeze(-1)).squeeze(-1)
        obs = torch.cat((e6, omega_base, self.qpos, self.qvel, self.action), dim=-1)
        self._last_obs = {"obs": obs}
        return self._last_obs

    def _build_observations(self, *, update_history: bool = True) -> dict[str, torch.Tensor]:
        current = self._obs()
        # history is the ten frames *before* o_t; update after making current.
        history = self.history.clone()
        if update_history:
            self.history[:, :-1] = self.history[:, 1:].clone()
            self.history[:, -1] = current["obs"]
        if hasattr(self, "reference_future"):
            future = self.reference_future
        else:
            future = torch.cat((self.reference_qpos.unsqueeze(1).expand(-1, 6, -1),
                                self.reference_qvel.unsqueeze(1).expand(-1, 6, -1),
                                self.reference_root_lin_vel[:, None, :].expand(-1, 6, -1)), dim=-1)
        privileged = torch.zeros((self.num_envs, PRIV_DIM), device=self.device)
        # A9 layout is explicit; do not leave the critic's extra inputs as an
        # indistinguishable block of zeros.  Nominal defaults above are neutral
        # and can be overwritten by the simulator's randomization callback.
        root_rot = _quat_to_mat(self.root_quat)
        root_rot_t = root_rot.transpose(1, 2)
        base_lin_vel = torch.bmm(root_rot_t, self.root_lin_vel.unsqueeze(-1)).squeeze(-1)
        base_ang_vel = torch.bmm(root_rot_t, self.root_ang_vel.unsqueeze(-1)).squeeze(-1)
        privileged[:, PRIV_LAYOUT.index("base_lin_vel")] = base_lin_vel
        privileged[:, PRIV_LAYOUT.index("base_ang_vel")] = base_ang_vel
        privileged[:, PRIV_LAYOUT.index("foot_contact_states")] = self.foot_contact.float()
        privileged[:, PRIV_LAYOUT.index("friction_coefficients")] = self.priv_friction
        privileged[:, PRIV_LAYOUT.index("terrain_height_at_feet")] = self.priv_terrain_height
        privileged[:, PRIV_LAYOUT.index("base_mass_perturbation")] = self.priv_mass
        privileged[:, PRIV_LAYOUT.index("com_perturbation")] = self.priv_com
        privileged[:, PRIV_LAYOUT.index("push_perturbation")] = self.priv_push
        privileged[:, PRIV_LAYOUT.index("motor_strength_scale")] = self.priv_motor_strength
        out = {"obs": current["obs"], "history": history, "future": future, "privileged": privileged}
        if self._terrain is not None:
            out["elevation"] = self._terrain.elevation(
                self.root_pos, self.root_quat,
                families=self._terrain_families, levels=self._terrain_levels)
        return out

    def get_observations(self) -> dict[str, torch.Tensor]:
        """Return the current observation without advancing history.

        Checkpoint restore and timeout bootstrapping need a read-only view.  A
        separate method prevents those paths from silently appending the same
        frame a second time.
        """
        return self._build_observations(update_history=False)

    def _set_reference_defaults(self, env_ids: torch.Tensor):
        self.reference_qpos[env_ids] = self.default_q[env_ids]
        self.reference_qvel[env_ids] = 0.0
        self.reference_root_pos[env_ids] = self.default_root_pos[env_ids]
        self.reference_root_quat[env_ids] = self.default_root_quat[env_ids]
        self.reference_root_lin_vel[env_ids] = 0.0

    def _reference_position_error(self) -> torch.Tensor:
        """A13 ``e_p`` in the current reference-anchor frame."""
        delta = self.reference_root_pos - self.root_pos
        local = torch.bmm(_quat_to_mat(self.reference_root_quat).transpose(1, 2),
                          delta.unsqueeze(-1)).squeeze(-1)
        return local[:, :2]

    def _reference_batch(self, seq: torch.Tensor, frame: torch.Tensor,
                         *, e_p: torch.Tensor | None = None,
                         placement_pos: torch.Tensor | None = None,
                         placement_quat: torch.Tensor | None = None):
        """Fetch a reference frame, optionally including body FK and A13."""
        if placement_pos is None:
            placement_pos = self._reference_placement_pos
        if placement_quat is None:
            placement_quat = self._reference_placement_quat
        if self._reference_motion is not None:
            out = self._reference_motion.sample(
                seq, frame,
                placement_pos=(self._reference_placement_pos if placement_pos is None else placement_pos),
                placement_quat=(self._reference_placement_quat if placement_quat is None else placement_quat))
            out["future"] = self._torch_reference.future_refs(
                seq, frame,
                e_p=e_p if self.cfg.enable_global_position_correction else None)
            return out
        batch = self._torch_reference.batch(
            seq, frame,
            e_p=e_p if self.cfg.enable_global_position_correction else None)
        if placement_pos is not None or placement_quat is not None:
            pp = torch.zeros_like(batch.root_pos) if placement_pos is None else placement_pos
            pq = torch.zeros_like(batch.root_rot)
            pq[:, 0] = 1.0
            if placement_quat is not None:
                pq = placement_quat
            R = _quat_to_mat(pq)
            batch.root_pos = torch.bmm(R, (batch.root_pos + pp).unsqueeze(-1)).squeeze(-1)
            batch.root_rot = _quat_mul(pq, batch.root_rot)
        return batch

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
        recovery_mask = np.zeros(int(env_ids.numel()), dtype=bool)
        if self.recovery_pool is not None and len(self.recovery_pool):
            recovery_mask = self._rng.random(int(env_ids.numel())) < self.recovery_pool.probability
        recovery_count = int(recovery_mask.sum())
        normal_count = int(env_ids.numel()) - recovery_count
        if self._adaptive_sampler is None:
            seq, frame = self._torch_reference.sample_segments(max(normal_count, 1), seg_len=1)
        else:
            pairs = [self._adaptive_sampler.sample(self._rng) for _ in range(max(normal_count, 1))]
            seq = torch.tensor([p[0] for p in pairs], device=self.device, dtype=torch.long)
            frame = torch.tensor([p[1] for p in pairs], device=self.device, dtype=torch.long)
        if recovery_count:
            rec = self.recovery_pool.sample(recovery_count, device=self.device)
            seq = torch.cat((seq[:normal_count], rec["seq_idx"]), dim=0)
            frame = torch.cat((frame[:normal_count], rec["frame"]), dim=0)
        if self.reference_seq_idx is None:
            self.reference_seq_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.reference_frame = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.reference_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Recovery samples are appended after ordinary samples; reorder them
        # back to the caller's env-id order before writing state tensors.
        if recovery_count:
            normal_ids = env_ids[torch.as_tensor(~recovery_mask, device=self.device)]
            recovery_ids = env_ids[torch.as_tensor(recovery_mask, device=self.device)]
            ordered_ids = torch.cat((normal_ids, recovery_ids))
            self._recovery_active[normal_ids] = 0.0
            self._recovery_active[recovery_ids] = 1.0
        else:
            ordered_ids = env_ids
            self._recovery_active[ordered_ids] = 0.0
        self.reference_seq_idx[ordered_ids] = seq
        self.reference_frame[ordered_ids] = frame.to(torch.float32)
        self.reference_start_frame[ordered_ids] = frame
        if recovery_count:
            self.qpos[recovery_ids] = rec["qpos"]
            self.qvel[recovery_ids] = rec["qvel"]
            self.root_pos[recovery_ids] = rec["root_pos"]
            self.root_quat[recovery_ids] = rec["root_quat"]
            self.root_lin_vel[recovery_ids] = rec["root_lin_vel"]
            self.root_ang_vel[recovery_ids] = rec["root_ang_vel"]
        env_ids = ordered_ids
        # Place each sampled sequence at the simulator reset pose.  This keeps
        # the raw dataset's arbitrary world translation/heading out of the
        # root-centric tracking problem while preserving its motion.
        _, _, raw_pos, raw_quat = self._torch_reference.ref_at(seq, self.reference_frame[env_ids])
        self._reference_raw_root_pos[env_ids] = raw_pos
        self._reference_raw_root_quat[env_ids] = raw_quat
        yaw_delta = _yaw(self.root_quat[env_ids]) - _yaw(raw_quat)
        self._reference_placement_quat[env_ids] = _yaw_quat(yaw_delta)
        inv_place = _quat_to_mat(self._reference_placement_quat[env_ids]).transpose(1, 2)
        desired = torch.bmm(inv_place, self.root_pos[env_ids].unsqueeze(-1)).squeeze(-1)
        self._reference_placement_pos[env_ids] = desired - raw_pos
        batch = self._reference_batch(
            seq, self.reference_frame[env_ids],
            placement_pos=self._reference_placement_pos[env_ids],
            placement_quat=self._reference_placement_quat[env_ids])
        self.reference_qpos[env_ids] = batch["joint_pos"] if isinstance(batch, dict) else batch.qpos
        self.reference_qvel[env_ids] = batch["joint_vel"] if isinstance(batch, dict) else batch.qvel
        self.reference_root_pos[env_ids] = batch["root_pos"] if isinstance(batch, dict) else batch.root_pos
        self.reference_root_quat[env_ids] = batch["root_quat"] if isinstance(batch, dict) else batch.root_rot
        local_vel = (batch["future"] if isinstance(batch, dict) else batch.future)[:, 0, -3:]
        self.reference_root_lin_vel[env_ids] = torch.bmm(
            _quat_to_mat(self.reference_root_quat[env_ids]), local_vel.unsqueeze(-1)).squeeze(-1)
        if not hasattr(self, "reference_future"):
            self.reference_future = torch.zeros((self.num_envs, 6, REF_FRAME_DIM), device=self.device)
        self.reference_future[env_ids] = batch["future"] if isinstance(batch, dict) else batch.future
        if isinstance(batch, dict):
            # Store a full body reference for the reward boundary.  The dict is
            # batch-sized; subsequent reference advances replace it in full.
            if self.reference_body is None:
                self.reference_body = {k: torch.zeros_like(v) for k, v in batch.items()
                                       if isinstance(v, torch.Tensor)}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    self.reference_body[key][env_ids] = value

    def _advance_reference(self) -> None:
        if self._torch_reference is None:
            return
        self.reference_frame += self.cfg.control_dt / self._torch_reference.frame_time[self.reference_seq_idx]
        # A13 is evaluated using the placed current reference anchor.
        raw = self._reference_batch(self.reference_seq_idx, self.reference_frame)
        self.reference_root_pos.copy_(raw["root_pos"] if isinstance(raw, dict) else raw.root_pos)
        self.reference_root_quat.copy_(raw["root_quat"] if isinstance(raw, dict) else raw.root_rot)
        ep = self._reference_position_error()
        batch = self._reference_batch(self.reference_seq_idx, self.reference_frame, e_p=ep)
        self.reference_qpos.copy_(batch["joint_pos"] if isinstance(batch, dict) else batch.qpos)
        self.reference_qvel.copy_(batch["joint_vel"] if isinstance(batch, dict) else batch.qvel)
        self.reference_root_pos.copy_(batch["root_pos"] if isinstance(batch, dict) else batch.root_pos)
        self.reference_root_quat.copy_(batch["root_quat"] if isinstance(batch, dict) else batch.root_rot)
        future = batch["future"] if isinstance(batch, dict) else batch.future
        self.reference_root_lin_vel.copy_(torch.bmm(
            _quat_to_mat(self.reference_root_quat), future[:, 0, -3:].unsqueeze(-1)).squeeze(-1))
        self.reference_future.copy_(future)
        if isinstance(batch, dict):
            self.reference_body = batch

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            self._rng = np.random.default_rng(seed)
        ids = torch.arange(self.num_envs, device=self.device)
        self.qpos.copy_(self.default_q)
        self.qvel.zero_()
        self.root_pos.copy_(self.default_root_pos)
        self.root_quat.copy_(self.default_root_quat)
        self.root_lin_vel.zero_()
        self.root_ang_vel.zero_()
        self.body_pos.zero_()
        self.body_quat.zero_()
        self.body_lin_vel.zero_()
        self.body_ang_vel.zero_()
        self.contact_forces.zero_()
        self._previous_contact_forces.zero_()
        self.foot_contact.zero_()
        self._previous_foot_contact.zero_()
        self._contact_age.zero_()
        self._body_state_available = False
        self.action.zero_()
        self._prev_action.zero_()
        self.target.copy_(self.default_q)
        self.episode_length_buf.zero_()
        for value in self._termination_elapsed.values():
            value.zero_()
        self.history.zero_()
        self._sample_reference(ids)
        if self.articulation is not None:
            # Recovery sampling may have selected a post-fall state.  Write
            # the state that was actually sampled instead of unconditionally
            # teleporting every episode back to the nominal pose.
            root_state = torch.cat((self.root_pos, self.root_quat,
                                    self.root_lin_vel, self.root_ang_vel), dim=-1)
            self.articulation.write_root_state_to_sim(root_state)
            self.articulation.write_joint_state_to_sim(self.qpos, self.qvel, joint_ids=self.joint_ids)
            self.articulation.set_joint_position_target(self.target[:, :], joint_ids=self.joint_ids)
        return self._build_observations()

    def state_dict(self) -> dict[str, Any]:
        """Serializable rollout state for deterministic PPO resume."""
        state = {
            "qpos": self.qpos.detach().cpu(), "qvel": self.qvel.detach().cpu(),
            "root_pos": self.root_pos.detach().cpu(), "root_quat": self.root_quat.detach().cpu(),
            "root_lin_vel": self.root_lin_vel.detach().cpu(), "root_ang_vel": self.root_ang_vel.detach().cpu(),
            "action": self.action.detach().cpu(), "prev_action": self._prev_action.detach().cpu(),
            "episode_length_buf": self.episode_length_buf.detach().cpu(),
            "reference_seq_idx": None if self.reference_seq_idx is None else self.reference_seq_idx.detach().cpu(),
            "reference_frame": None if self.reference_frame is None else self.reference_frame.detach().cpu(),
            "reference_start_frame": None if self.reference_start_frame is None else self.reference_start_frame.detach().cpu(),
            "history": self.history.detach().cpu(),
            "body_pos": self.body_pos.detach().cpu(), "body_quat": self.body_quat.detach().cpu(),
            "body_lin_vel": self.body_lin_vel.detach().cpu(), "body_ang_vel": self.body_ang_vel.detach().cpu(),
            "contact_forces": self.contact_forces.detach().cpu(),
            "previous_body_lin_vel": self._previous_body_lin_vel.detach().cpu(),
            "previous_root_lin_vel": self._previous_root_lin_vel.detach().cpu(),
            "previous_contact_forces": self._previous_contact_forces.detach().cpu(),
            "foot_contact": self.foot_contact.detach().cpu(),
            "previous_foot_contact": self._previous_foot_contact.detach().cpu(),
            "contact_age": self._contact_age.detach().cpu(),
            "recovery_active": self._recovery_active.detach().cpu(),
            "reference_qpos": self.reference_qpos.detach().cpu(),
            "reference_qvel": self.reference_qvel.detach().cpu(),
            "reference_root_pos": self.reference_root_pos.detach().cpu(),
            "reference_root_quat": self.reference_root_quat.detach().cpu(),
            "reference_root_lin_vel": self.reference_root_lin_vel.detach().cpu(),
            "reference_future": (None if not hasattr(self, "reference_future")
                                  else self.reference_future.detach().cpu()),
            "reference_placement_pos": self._reference_placement_pos.detach().cpu(),
            "reference_placement_quat": self._reference_placement_quat.detach().cpu(),
            "reference_raw_root_pos": self._reference_raw_root_pos.detach().cpu(),
            "reference_raw_root_quat": self._reference_raw_root_quat.detach().cpu(),
            "rng_state": self._rng.bit_generator.state,
        }
        if self.reference_body is not None:
            state["reference_body"] = {
                key: value.detach().cpu() for key, value in self.reference_body.items()
                if isinstance(value, torch.Tensor)
            }
        if self.recovery_pool is not None:
            state["recovery_pool"] = self.recovery_pool.state_dict()
        if self._adaptive_sampler is not None:
            state["adaptive_fails"] = dict(self._adaptive_sampler.fails)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` after construction."""
        for key, target in (("qpos", self.qpos), ("qvel", self.qvel),
                            ("root_pos", self.root_pos), ("root_quat", self.root_quat),
                            ("root_lin_vel", self.root_lin_vel), ("root_ang_vel", self.root_ang_vel),
                            ("action", self.action), ("prev_action", self._prev_action),
                            ("episode_length_buf", self.episode_length_buf), ("history", self.history),
                            ("body_pos", self.body_pos), ("body_quat", self.body_quat),
                            ("body_lin_vel", self.body_lin_vel), ("body_ang_vel", self.body_ang_vel),
                            ("contact_forces", self.contact_forces),
                            ("previous_body_lin_vel", self._previous_body_lin_vel),
                            ("previous_root_lin_vel", self._previous_root_lin_vel),
                            ("previous_contact_forces", self._previous_contact_forces),
                            ("foot_contact", self.foot_contact),
                            ("previous_foot_contact", self._previous_foot_contact),
                            ("contact_age", self._contact_age),
                            ("recovery_active", self._recovery_active),
                            ("reference_qpos", self.reference_qpos), ("reference_qvel", self.reference_qvel),
                            ("reference_root_pos", self.reference_root_pos),
                            ("reference_root_quat", self.reference_root_quat),
                            ("reference_root_lin_vel", self.reference_root_lin_vel),
                            ("reference_placement_pos", self._reference_placement_pos),
                            ("reference_placement_quat", self._reference_placement_quat),
                            ("reference_raw_root_pos", self._reference_raw_root_pos),
                            ("reference_raw_root_quat", self._reference_raw_root_quat)):
            if key in state:
                target.copy_(torch.as_tensor(state[key], device=self.device, dtype=target.dtype))
        for key, attr in (("reference_seq_idx", "reference_seq_idx"),
                          ("reference_frame", "reference_frame"),
                          ("reference_start_frame", "reference_start_frame")):
            value = state.get(key)
            if value is not None and getattr(self, attr) is not None:
                getattr(self, attr).copy_(torch.as_tensor(value, device=self.device,
                                                          dtype=getattr(self, attr).dtype))
        if self._adaptive_sampler is not None and "adaptive_fails" in state:
            self._adaptive_sampler.fails = dict(state["adaptive_fails"])
            self._adaptive_sampler._invalidate_weights()
        if state.get("reference_future") is not None:
            self.reference_future = torch.as_tensor(
                state["reference_future"], device=self.device, dtype=self.history.dtype).clone()
        if state.get("reference_body") is not None:
            self.reference_body = {
                key: torch.as_tensor(value, device=self.device).clone()
                for key, value in state["reference_body"].items()
            }
        if state.get("rng_state") is not None:
            self._rng.bit_generator.state = state["rng_state"]
        if self.recovery_pool is not None and state.get("recovery_pool") is not None:
            self.recovery_pool.load_state_dict(state["recovery_pool"])

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
        if (self.cfg.enable_body_tracking and self._body_state_available
                and self.reference_body is not None):
            if self.articulation is not None and not self._contact_sensor_available:
                raise RuntimeError(
                    "Isaac Lab G1 reward requires ContactSensor force data; "
                    "refusing to train with zero contact placeholders")
            state = {
                "joint_pos": self.qpos,
                "joint_vel": self.qvel,
                "root_pos": self.root_pos,
                "root_quat": self.root_quat,
                "root_lin_vel": self.root_lin_vel,
                "root_ang_vel": self.root_ang_vel,
                "body_pos": self.body_pos,
                "body_quat": self.body_quat,
                "body_lin_vel": self.body_lin_vel,
                "body_ang_vel": self.body_ang_vel,
                "body_accel": (self.body_lin_vel - self._previous_body_lin_vel) / self.cfg.control_dt,
                "contact_forces": self.contact_forces,
                "joint_low": self.joint_low,
                "joint_high": self.joint_high,
            }
            reference = {key: value for key, value in self.reference_body.items()
                         if isinstance(value, torch.Tensor)}
            terrain = None
            if self.cfg.stage == 2:
                feet = torch.tensor([
                    REQUIRED_BODY_NAMES.index("left_ankle_roll_link"),
                    REQUIRED_BODY_NAMES.index("right_ankle_roll_link")], device=self.device)
                terrain = {
                    "family_ids": self._terrain_families,
                    "level": self._terrain_levels,
                    "foot_height_samples": self._terrain.foot_samples(self.body_pos[:, feet]),
                    # Age of the state before this step's transition.  It is
                    # advanced only after the reward below is computed.
                    "contact_age": self._contact_age,
                }
            rewards, metrics = self._reward_computer.compute(
                state, reference,
                previous_state={
                    "root_lin_vel": self._previous_root_lin_vel,
                    "body_lin_vel": self._previous_body_lin_vel,
                    "contact_forces": self._previous_contact_forces,
                },
                action=self.action, previous_action=self._prev_action,
                corrected_velocity=self.reference_root_lin_vel,
                recovery_mask=getattr(self, "_recovery_active", None),
                terrain=terrain, stage2=self.cfg.stage == 2)
            current_contact = self.foot_contact
            self._contact_age.copy_(torch.where(
                current_contact != self._previous_foot_contact,
                torch.ones_like(self._contact_age), self._contact_age + 1.0))
            self._previous_foot_contact.copy_(current_contact)
            self._previous_contact_forces.copy_(self.contact_forces)
            self._last_reward_metrics = metrics
            self._prev_action.copy_(self.action)
            return rewards

        # This branch is deliberately a protocol-only fallback.  It allows
        # CPU tests without Isaac Sim, but is not a paper-equivalent reward.
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
        self._last_reward_metrics = {}
        return rewards

    def _dones(self):
        timeout = self.episode_length_buf >= self.cfg.max_episode_steps
        term_cfg = get("A20").value
        deviation = (self.root_pos - self.reference_root_pos).norm(dim=-1)
        # z component of the robot's local up axis in world coordinates; this
        # is the same projected-gravity tilt signal used by the simulator
        # termination state machine, without requiring an Isaac-only tensor.
        r = _quat_to_mat(self.root_quat)
        gravity_z = r[:, 2, 2]
        if (self._body_state_available and self.reference_body is not None
                and isinstance(self.reference_body.get("body_pos"), torch.Tensor)):
            body_error = (self.body_pos - self.reference_body["body_pos"]).norm(dim=-1).mean(-1)
            ref_delay = self._termination_ref_delay
            ref_condition = body_error > self._termination_ref_threshold
        elif "body_pos_error" in self._last_reward_metrics:
            body_error = self._last_reward_metrics["body_pos_error"]
            ref_delay = self._termination_ref_delay
            ref_condition = body_error > self._termination_ref_threshold
        else:
            ref_delay = torch.full((self.num_envs,), float(term_cfg.delay_s["ref_deviation"]), device=self.device)
            ref_condition = deviation > float(term_cfg.ref_deviation_base)
        active = {
            "ref_deviation": ref_condition,
            "root_low": self.root_pos[:, 2] < float(term_cfg.root_height_min),
            "tilted": gravity_z < float(np.cos(np.deg2rad(term_cfg.tilt_max_deg))),
        }
        delays = {
            "ref_deviation": ref_delay,
            "root_low": float(term_cfg.delay_s["root_low"]),
            "tilted": float(term_cfg.delay_s["tilted"]),
        }
        fired = []
        for name, condition in active.items():
            elapsed = self._termination_elapsed[name]
            elapsed.add_(condition.to(elapsed.dtype) * self.cfg.control_dt)
            elapsed.masked_fill_(~condition, 0.0)
            delay = delays[name] if isinstance(delays[name], torch.Tensor) else float(delays[name])
            fired.append(condition & (elapsed >= delay - 1e-7))
        # Retain a conservative high-speed guard from the fallback adapter.
        fired.append(self.qvel.abs().amax(-1) > self.cfg.max_joint_velocity * 1.5)
        terminated = torch.stack(fired, dim=-1).any(-1)
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
        if self._adaptive_sampler is not None and terminated.any():
            for env_id in torch.nonzero(terminated, as_tuple=False).flatten().tolist():
                self._adaptive_sampler.update(
                    int(self.reference_seq_idx[env_id]),
                    int(self.reference_start_frame[env_id]),
                    failed=True)
        if self.recovery_pool is not None and (terminated | truncated).any():
            finished = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
            for env_id in finished.tolist():
                if not bool(self._recovery_active[env_id]):
                    continue
                self.recovery_pool.record_outcome(not bool(terminated[env_id]))
                if bool(terminated[env_id]) and self.reference_seq_idx is not None:
                    self.recovery_pool.add(
                        {"qpos": self.qpos[env_id], "qvel": self.qvel[env_id],
                         "root_pos": self.root_pos[env_id], "root_quat": self.root_quat[env_id],
                         "root_lin_vel": self.root_lin_vel[env_id], "root_ang_vel": self.root_ang_vel[env_id]},
                        seq_idx=int(self.reference_seq_idx[env_id]),
                        frame=float(self.reference_frame[env_id]))
        terminal_observation = None
        if truncated.any():
            # Preserve the pre-reset state for PPO timeout bootstrapping.
            terminal_observation = {
                key: value.clone() for key, value in self._build_observations(update_history=False).items()
            }
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
            self.body_pos[ids] = 0.0
            self.body_quat[ids] = 0.0
            self.body_lin_vel[ids] = 0.0
            self.body_ang_vel[ids] = 0.0
            self.action[ids] = 0.0
            self._previous_root_lin_vel[ids] = 0.0
            self._previous_body_lin_vel[ids] = 0.0
            self.contact_forces[ids] = 0.0
            self._previous_contact_forces[ids] = 0.0
            self.foot_contact[ids] = False
            self._previous_foot_contact[ids] = False
            self._contact_age[ids] = 0.0
            self.target[ids] = self.default_q[ids]
            self.episode_length_buf[ids] = 0
            self.history[ids] = 0.0
            for value in self._termination_elapsed.values():
                value[ids] = 0.0
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
    from isaaclab.sensors import ContactSensorCfg
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
            spawn = _sim_utils.UrdfFileCfg(
                asset_path=path, fix_base=False, activate_contact_sensors=True)
        elif suffix in {".usd", ".usda", ".usdc"}:
            spawn = _sim_utils.UsdFileCfg(usd_path=path, activate_contact_sensors=True)
        else:
            raise ValueError("Isaac Lab G1 asset must be .usd/.usda/.usdc/.urdf")
        def _joint_values(value: float | Sequence[float], name: str) -> dict[str, float]:
            vals = _tuple_floats(value, ACT_DIM, name)
            return {f"{joint}(_joint)?": val for joint, val in zip(G1_JOINT_NAMES, vals)}

        actuator = ImplicitActuatorCfg(
            # Both Unitree URDF-derived USD and the retargeting contract are
            # supported; Isaac Lab expressions accept the optional suffix.
            joint_names_expr=[f"{name}(_joint)?" for name in G1_JOINT_NAMES],
            stiffness=_joint_values(c.stiffness, "stiffness"),
            damping=_joint_values(c.damping, "damping"),
            effort_limit_sim=_joint_values(c.torque_limit, "torque_limit"),
        )
        return ArticulationCfg(
            # DirectRLEnv builds articulations in ``_setup_scene`` before the
            # manager-based scene resolver expands ``{ENV_REGEX_NS}``.
            prim_path="/World/envs/env_.*/Robot", spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=c.default_root_pos, rot=c.default_root_quat,
                joint_pos={f"{name}(_joint)?": value for name, value in zip(G1_JOINT_NAMES, c.default_joint_pos)},
                joint_vel={f"{name}(_joint)?": 0.0 for name in G1_JOINT_NAMES},
            ),
            actuators={"g1_pd": actuator},
        )


    @configclass
    class G1SceneCfg(InteractiveSceneCfg):
        """G1 scene with per-body contact forces for auxiliary/terrain terms."""

        contact_forces: Any = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            track_air_time=True,
        )


    @configclass
    class G1DirectRLEnvCfg(DirectRLEnvCfg):
        """Isaac Lab 2.3 DirectRLEnv config for the G1 adapter."""

        decimation: int = 4
        episode_length_s: float = 30.0
        observation_space: int = OBS_DIM
        action_space: int = ACT_DIM
        state_space: int = PRIV_DIM
        scene: G1SceneCfg = G1SceneCfg(num_envs=1, env_spacing=2.5)
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
            # Isaac Lab otherwise writes to /tmp/isaaclab/logs, which is often
            # owned by another user on shared servers.  Keep per-checkout Kit
            # logs writable and colocated with the smoke-test logs.
            log_dir = os.environ.get(
                "PGMT_ISAACLAB_LOG_DIR",
                os.path.join(os.getcwd(), "data", "processed", "isaaclab_logs"),
            )
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            cfg.sim.log_dir = log_dir
            # Keep Isaac's physics clock identical to the simulator-independent
            # contract.  Isaac Lab defaults to 1/60 s, which would otherwise
            # turn the declared 20 ms control step into 66.7 ms at decimation 4.
            cfg.sim.dt = self._pgmt_cfg.sim_dt
            cfg.sim.device = self._pgmt_cfg.device
            cfg.decimation = self._pgmt_cfg.decimation
            super().__init__(cfg, **kwargs)
            self.core = G1Env(self._pgmt_cfg, articulation=self.robot)
            self._sync_contact_forces()
            self.last_adapter_obs = None
            reward_heads = 4 if self._pgmt_cfg.stage == 2 else 3
            self.last_split_reward = torch.zeros(
                (self.num_envs, reward_heads), device=self.device)
            self.last_terminal_obs = None

        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot_cfg)
            self.scene.articulations["robot"] = self.robot
            _sim_utils.spawn_ground_plane("/World/ground", _sim_utils.GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

        def _sync_contact_forces(self):
            sensor = self.scene.sensors.get("contact_forces")
            # ``force_matrix_w`` is only allocated for filtered contacts and
            # is None for the default unfiltered ContactSensor.  The latter's
            # per-body net force is the signal needed by the reward.
            if sensor is None:
                self.core.contact_forces.zero_()
                self.core.foot_contact.zero_()
                self.core._contact_sensor_available = False
                return
            forces = getattr(sensor.data, "net_forces_w", None)
            if forces is None:
                self.core.contact_forces.zero_()
                self.core.foot_contact.zero_()
                self.core._contact_sensor_available = False
                return
            if forces.ndim != 3 or forces.shape[-1] != 3:
                raise RuntimeError(
                    "ContactSensor.net_forces_w must have shape (N, bodies, 3), "
                    f"got {tuple(forces.shape)}")
            names = list(getattr(sensor, "body_names", []))
            if not names:
                names = list(getattr(sensor.data, "body_names", []))
            # Sensor body names may be prim paths; retain the final token for
            # matching the articulation mapping.  A 28-body sensor without
            # names is accepted only when its order is the declared contract.
            names = [str(name).split("/")[-1] for name in names]
            ids = {name: i for i, name in enumerate(names)}
            self.core.contact_forces.zero_()
            mapped = 0
            for out_i, body in enumerate(REQUIRED_BODY_NAMES):
                candidates = (body, body.removesuffix("_link"), body + "_joint")
                src = next((ids[x] for x in candidates if x in ids), None)
                if src is not None:
                    self.core.contact_forces[:, out_i] = forces[:, src]
                    mapped += 1
            if not names and forces.shape[1] == len(REQUIRED_BODY_NAMES):
                self.core.contact_forces.copy_(forces)
                mapped = len(REQUIRED_BODY_NAMES)
            elif mapped != len(REQUIRED_BODY_NAMES):
                self.core.contact_forces.zero_()
                self.core.foot_contact.zero_()
                self.core._contact_sensor_available = False
                return
            self.core.foot_contact.copy_(self.core.contact_forces[:, [
                REQUIRED_BODY_NAMES.index("left_ankle_roll_link"),
                REQUIRED_BODY_NAMES.index("right_ankle_roll_link")
            ]].norm(dim=-1) > 1.0)
            self.core._contact_sensor_available = True

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
            self._sync_contact_forces()
            self.core.episode_length_buf.copy_(self.episode_length_buf)
            self.core._advance_reference()
            self.last_terminal_obs = None
            if self.reset_time_outs.any():
                self.last_terminal_obs = {
                    key: value.clone() for key, value in self.core._build_observations(update_history=False).items()
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
            # ``DirectRLEnv._reset_idx`` resets bookkeeping and actuator
            # buffers, but it does not write a fallen articulation pose.  Use
            # the sampled nominal/recovery state for the actual simulator too.
            self.robot.write_root_state_to_sim(torch.cat((
                self.core.root_pos[ids], self.core.root_quat[ids],
                self.core.root_lin_vel[ids], self.core.root_ang_vel[ids]), dim=-1), ids)
            self.robot.write_joint_state_to_sim(
                self.core.qpos[ids], self.core.qvel[ids],
                joint_ids=self.core.joint_ids, env_ids=ids)
            self.robot.set_joint_position_target(
                self.core.target[ids], joint_ids=self.core.joint_ids, env_ids=ids)

        def get_observations(self):
            """Read the adapter observation without appending history."""
            return self.core.get_observations()

        def _set_debug_vis_impl(self, debug_vis: bool):
            raise NotImplementedError

else:
    make_g1_articulation_cfg = None
    G1DirectRLEnvCfg = None
    G1SceneCfg = None
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
    "AssetPreflight", "G1Env", "G1EnvConfig", "G1SceneCfg", "G1DirectRLEnvCfg", "IsaacLabG1Env",
    "PDController", "REQUIRED_BODY_NAMES", "G1_JOINT_NAMES", "G1_JOINT_LIMITS",
    "make_g1_articulation_cfg", "preflight_asset", "resolve_name_indices", "validate_g1_mapping",
    "IsaacLabPPOAdapter",
]
