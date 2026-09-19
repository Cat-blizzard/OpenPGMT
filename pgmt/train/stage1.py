"""Torch adapters shared by the Stage 1 environment and PPO runner.

The simulator owns contact queries and body kinematics.  This module keeps the
parts that do not need a simulator on the torch side: padded reference motion
lookup, PD target/torque conversion, and the three headed Table-I reward.  The
arrays are copied to the selected device once at construction time; calls used
inside an environment step do not convert through NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from pgmt.cfg.assumptions import get
from pgmt.contracts import ACT_DIM, REF_FRAME_DIM
from pgmt.rewards.spec import AUX, LOWER, SIGMAS, UPPER


def _as_device_tensor(value, *, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    out = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return out.to(device=device, dtype=dtype)


def _finite_nonnegative(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a batched vector, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if (value < 0).any():
        raise ValueError(f"{name} contains a negative value; residuals/costs are >= 0")


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Shortest-arc [w,x,y,z] quaternion interpolation, batched."""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    w = w.unsqueeze(-1)
    regular = (torch.sin((1.0 - w) * theta) * q0 + torch.sin(w * theta) * q1) / sin_theta.clamp_min(1e-6)
    linear = _quat_normalize((1.0 - w) * q0 + w * q1)
    return torch.where(sin_theta.abs() < 1e-5, linear, regular)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


@dataclass
class TorchReferenceBatch:
    """A batch of current/future reference values in the G1 joint order."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    root_pos: torch.Tensor
    root_rot: torch.Tensor
    future: torch.Tensor


class TorchMotionDatabase:
    """Padded, device-resident view of :class:`MotionDatabase`.

    ``MotionDatabase`` deliberately remains a NumPy/data-loading API.  This
    class is the one-time bridge used by a batched simulator environment.  It
    preserves the source-frame interpolation and shortest-arc root quaternion
    semantics while avoiding per-step Python/NumPy copies.
    """

    def __init__(self, database, device: str | torch.device = "cpu"):
        seqs = getattr(database, "seqs", None)
        if not seqs:
            raise ValueError("database must expose a non-empty seqs list")
        self.device = torch.device(device)
        self.num_sequences = len(seqs)
        self.lengths = torch.tensor([len(s["qpos"]) for s in seqs], device=self.device, dtype=torch.long)
        self.max_length = int(self.lengths.max().item())
        self.frame_time = torch.tensor([float(s["frame_time"]) for s in seqs], device=self.device)
        self.qpos = self._pad(seqs, "qpos", ACT_DIM)
        self.qvel = self._pad(seqs, "qvel", ACT_DIM)
        self.root_pos = self._pad(seqs, "root_pos", 3)
        self.root_rot = _quat_normalize(self._pad(seqs, "root_rot", 4))
        self.offsets = torch.tensor(get("A2").value.offsets, device=self.device, dtype=torch.float32)

    def _pad(self, seqs, key: str, dim: int) -> torch.Tensor:
        out = torch.zeros((len(seqs), self.max_length, dim), device=self.device)
        for i, sequence in enumerate(seqs):
            value = _as_device_tensor(np.asarray(sequence[key]), device=self.device)
            if value.ndim != 2 or value.shape[1] != dim:
                raise ValueError(f"sequence {i} {key} must have shape (T,{dim})")
            out[i, : value.shape[0]] = value
            if value.shape[0] < self.max_length:
                out[i, value.shape[0] :] = value[-1]
        return out

    def _indices(self, seq_idx: torch.Tensor, frame: torch.Tensor):
        seq_idx = seq_idx.to(device=self.device, dtype=torch.long).reshape(-1)
        frame = frame.to(device=self.device, dtype=torch.float32).reshape(-1)
        if seq_idx.shape != frame.shape:
            raise ValueError("seq_idx and frame must have the same shape")
        if (seq_idx < 0).any() or (seq_idx >= self.num_sequences).any():
            raise IndexError("sequence index out of range")
        max_frame = (self.lengths[seq_idx] - 1).to(torch.float32)
        frame = frame.clamp_min(0).minimum(max_frame)
        i0 = frame.floor().to(torch.long)
        i1 = (i0 + 1).minimum(self.lengths[seq_idx] - 1)
        return seq_idx, i0, i1, frame - i0.to(frame.dtype)

    def _lerp(self, values: torch.Tensor, seq_idx, i0, i1, weight):
        return values[seq_idx, i0] * (1 - weight[:, None]) + values[seq_idx, i1] * weight[:, None]

    def ref_at(self, seq_idx: torch.Tensor, frame: torch.Tensor):
        seq_idx, i0, i1, w = self._indices(seq_idx, frame)
        q = self._lerp(self.qpos, seq_idx, i0, i1, w)
        qd = self._lerp(self.qvel, seq_idx, i0, i1, w)
        root_pos = self._lerp(self.root_pos, seq_idx, i0, i1, w)
        root_rot = _quat_slerp(self.root_rot[seq_idx, i0], self.root_rot[seq_idx, i1], w)
        return q, qd, root_pos, root_rot

    def anchor_velocity(self, seq_idx: torch.Tensor, frame: torch.Tensor,
                        anchor_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_idx, _, _, _ = self._indices(seq_idx, frame)
        # MotionDatabase uses a forward difference and, at the final source
        # frame, keeps the final interval (rather than returning an artificial
        # zero velocity).  Preserve that convention for future references.
        max_frame = (self.lengths[seq_idx] - 1).to(torch.float32)
        frame = frame.to(device=self.device, dtype=torch.float32).reshape(-1).clamp_min(0).minimum(max_frame)
        i0 = frame.floor().to(torch.long).minimum((self.lengths[seq_idx] - 2).clamp_min(0))
        i1 = (i0 + 1).minimum(self.lengths[seq_idx] - 1)
        velocity_world = (self.root_pos[seq_idx, i1] - self.root_pos[seq_idx, i0]) / self.frame_time[seq_idx, None]
        anchor_frame = frame if anchor_frame is None else anchor_frame
        _, _, _, anchor_rot = self.ref_at(seq_idx, anchor_frame)
        # R^T v, matching MotionDatabase.anchor_velocity; z is intentionally zero.
        velocity_anchor = torch.bmm(_quat_to_mat(anchor_rot).transpose(1, 2), velocity_world.unsqueeze(-1)).squeeze(-1)
        return torch.cat((velocity_anchor[:, :2], torch.zeros_like(velocity_anchor[:, :1])), dim=-1)

    def future_refs(self, seq_idx: torch.Tensor, frame: torch.Tensor,
                    e_p: Optional[torch.Tensor] = None,
                    lambda_pos: Optional[float] = None) -> torch.Tensor:
        seq_idx = seq_idx.to(device=self.device, dtype=torch.long).reshape(-1)
        frame = frame.to(device=self.device, dtype=torch.float32).reshape(-1)
        batch = seq_idx.shape[0]
        source_step = get("A1").value.dt / self.frame_time[seq_idx]
        query = frame[:, None] + self.offsets[None] * source_step[:, None]
        flat_seq = seq_idx[:, None].expand(-1, self.offsets.numel()).reshape(-1)
        flat_frame = query.reshape(-1)
        q, qd, _, _ = self.ref_at(flat_seq, flat_frame)
        velocity = self.anchor_velocity(flat_seq, flat_frame, frame[:, None].expand(-1, self.offsets.numel()).reshape(-1))
        velocity = velocity.reshape(batch, -1, 3)
        if e_p is not None:
            e_p = e_p.to(device=self.device, dtype=torch.float32)
            if e_p.shape != (batch, 2):
                raise ValueError(f"e_p must have shape ({batch},2)")
            lam = get("A13").value.lambda_pos if lambda_pos is None else float(lambda_pos)
            speed = velocity[:, :, :2].norm(dim=-1)
            t = ((speed - get("A13").value.gate_v0) /
                 max(get("A13").value.gate_v1 - get("A13").value.gate_v0, 1e-9)).clamp(0, 1)
            gate = t * t * (3 - 2 * t)
            correction = gate[..., None] * lam * e_p[:, None, :]
            correction = correction.clamp(-get("A13").value.clip_v, get("A13").value.clip_v)
            velocity[:, :, :2] += correction
        return torch.cat((q.reshape(batch, -1, ACT_DIM), qd.reshape(batch, -1, ACT_DIM), velocity), dim=-1)

    def sample_segments(self, batch_size: int, seg_len: int = 1,
                        generator: Optional[torch.Generator] = None):
        if batch_size <= 0 or seg_len <= 0:
            raise ValueError("batch_size and seg_len must be positive")
        usable = (self.lengths - int(seg_len) + 1).clamp_min(1).to(torch.float32)
        seq = torch.multinomial(usable / usable.sum(), batch_size, replacement=True, generator=generator)
        starts = torch.floor(torch.rand(batch_size, device=self.device, generator=generator) * usable[seq]).to(torch.long)
        return seq, starts

    def batch(self, seq_idx: torch.Tensor, frame: torch.Tensor,
              e_p: Optional[torch.Tensor] = None) -> TorchReferenceBatch:
        q, qd, root_pos, root_rot = self.ref_at(seq_idx, frame)
        return TorchReferenceBatch(q, qd, root_pos, root_rot,
                                   self.future_refs(seq_idx, frame, e_p=e_p))


@dataclass
class PDConfig:
    """Action-to-PD contract; values are explicit to avoid hidden robot assumptions."""

    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    default_q: torch.Tensor
    torque_limits: Optional[torch.Tensor] = None
    action_clip: float = 1.0


class PDBatchController:
    """Vectorized position PD controller for ``(N,29)`` actions/states."""

    def __init__(self, config: PDConfig, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.action_scale = _as_device_tensor(config.action_scale, device=self.device)
        self.kp = _as_device_tensor(config.kp, device=self.device)
        self.kd = _as_device_tensor(config.kd, device=self.device)
        self.default_q = _as_device_tensor(config.default_q, device=self.device)
        self.torque_limits = None if config.torque_limits is None else _as_device_tensor(config.torque_limits, device=self.device).abs()
        self.action_clip = float(config.action_clip)
        if self.action_scale.shape != self.kp.shape or self.kp.shape != self.kd.shape or self.kd.shape != self.default_q.shape or self.kp.shape != (ACT_DIM,):
            raise ValueError("PD vectors must all have shape (29,)")
        if self.action_clip <= 0 or not torch.isfinite(torch.cat((self.action_scale, self.kp, self.kd, self.default_q))).all():
            raise ValueError("PD parameters must be finite and action_clip positive")

    def target(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim != 2 or action.shape[1] != ACT_DIM:
            raise ValueError(f"action must have shape (N,{ACT_DIM})")
        return self.default_q.unsqueeze(0) + action.clamp(-self.action_clip, self.action_clip) * self.action_scale.unsqueeze(0)

    def torque(self, action: torch.Tensor, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        target = self.target(action)
        q, qd = q.to(self.device), qd.to(self.device)
        if q.shape != target.shape or qd.shape != target.shape:
            raise ValueError("q and qd must match action shape")
        tau = self.kp * (target - q) - self.kd * qd
        if self.torque_limits is not None:
            lim = self.torque_limits.unsqueeze(0)
            tau = tau.clamp(-lim, lim)
        return tau


class BatchRewardAdapter:
    """Compute Stage 1 split returns from batched non-negative term values.

    Tracking terms are interpreted as residuals and mapped with the repository's
    exponential kernels.  Auxiliary terms follow ``compute_aux_values``: the
    four positive entries are already kernels, and the six negative-weight
    entries are non-negative costs.  This keeps the adapter compatible with the
    simulator-side auxiliary implementation while making the three-head output
    explicit and GPU resident.
    """

    _AUX_POSITIVE = {"root_ori", "corrected_root_vel", "floating_anchor_pos", "recovery_upward_vel"}
    _TRACKING = {name for name, _ in UPPER.terms + LOWER.terms}

    def __init__(self, device: str | torch.device = "cpu", *, strict: bool = True):
        self.device = torch.device(device)
        self.strict = bool(strict)

    def __call__(self, values: Mapping[str, torch.Tensor], *, terrain_chi: Optional[torch.Tensor] = None,
                 tau_m: float = 0.0, tau_rad: float = 0.0, alpha: float = 1.0) -> torch.Tensor:
        if not values:
            raise ValueError("reward values cannot be empty")
        tensors = {name: _as_device_tensor(value, device=self.device) for name, value in values.items()}
        batch = next(iter(tensors.values())).shape
        if len(batch) != 1:
            raise ValueError("reward values must be vectors with shape (N,)")
        for name, value in tensors.items():
            _finite_nonnegative(name, value)
        # A shared ``link_pos`` key is accepted for the common/simple case;
        # production environments can provide ``upper_link_pos`` and
        # ``lower_link_pos`` (and corresponding ``upper_*``/``lower_*`` keys)
        # to keep the two body partitions independent.
        def lookup(group: str, term: str):
            aliases = [f"{group}_{term}", term]
            if group == "lower" and term.startswith("ta_"):
                aliases.insert(1, f"lower_{term[3:]}")
            for alias in aliases:
                if alias in tensors:
                    return tensors[alias]
            return None
        missing = set()
        for group, terms in (("upper", UPPER.names), ("lower", LOWER.names), ("aux", AUX.names)):
            for term in terms:
                if lookup(group, term) is None:
                    missing.add(f"{group}_{term}")
        if self.strict and missing:
            raise KeyError(f"Stage 1 reward values missing {sorted(missing)}")
        if terrain_chi is not None:
            terrain_chi = _as_device_tensor(terrain_chi, device=self.device)
            if terrain_chi.shape != batch:
                raise ValueError(f"terrain_chi must have shape {batch}")
            if (terrain_chi < 0).any() or (terrain_chi > 1).any():
                raise ValueError("terrain_chi must be in [0,1]")
        def term_value(group: str, term: str, *, tau: float = 0.0):
            value = lookup(group, term)
            if value is None:
                return torch.zeros(batch, device=self.device)
            if term.startswith("ta_"):
                if terrain_chi is not None:
                    value = (value - float(alpha) * terrain_chi * float(tau)).clamp_min(0.0)
                sigma_term = term
            else:
                sigma_term = term
            return torch.exp(-(value.square()) / float(SIGMAS[sigma_term]))
        upper = sum(weight * term_value("upper", name) for name, weight in UPPER.terms)
        lower = sum(weight * term_value("lower", name, tau=(tau_m if name == "ta_link_pos" else tau_rad)) for name, weight in LOWER.terms)
        aux = torch.zeros(batch, device=self.device)
        for name, weight in AUX.terms:
            value = tensors.get(name, torch.zeros(batch, device=self.device))
            # Positive auxiliary entries are already exp(-e^2/sigma); costs are
            # raw non-negative values.  We deliberately do not infer this from
            # the sign of weight so a future positive cost cannot silently change semantics.
            aux = aux + float(weight) * value
        rewards = torch.stack((upper, lower, aux), dim=-1)
        if not torch.isfinite(rewards).all():
            raise ValueError("non-finite Stage 1 reward")
        return rewards
