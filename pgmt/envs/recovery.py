"""Fall-state storage and recovery curriculum for Stage 1.

The paper requires recovery episodes to start from post-fall states.  This
module is simulator agnostic: the environment owns the tensors and records a
snapshot when a physical termination occurs; the pool only controls bounded
storage, sampling, and the survival-rate curriculum.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class RecoveryState:
    qpos: torch.Tensor
    qvel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    seq_idx: int
    frame: float


class FallRecoveryPool:
    """Bounded post-fall state pool with survival-aware sampling probability."""

    def __init__(self, capacity: int = 2048, init_prob: float = .1,
                 prob_max: float = .5, survival_window: int = 100):
        if capacity <= 0 or survival_window <= 0:
            raise ValueError("capacity and survival_window must be positive")
        if not 0 <= init_prob <= prob_max <= 1:
            raise ValueError("require 0 <= init_prob <= prob_max <= 1")
        self.capacity = int(capacity)
        self.init_prob = float(init_prob)
        self.prob_max = float(prob_max)
        self.outcomes = deque(maxlen=int(survival_window))
        self._states: list[RecoveryState] = []

    def __len__(self) -> int:
        return len(self._states)

    @property
    def probability(self) -> float:
        if not self.outcomes:
            return self.init_prob
        survival = sum(self.outcomes) / len(self.outcomes)
        return self.init_prob + (1.0 - survival) * (self.prob_max - self.init_prob)

    def record_outcome(self, survived: bool) -> None:
        self.outcomes.append(bool(survived))

    def add(self, state: Mapping[str, torch.Tensor], *, seq_idx: int, frame: float) -> None:
        required = ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel")
        if any(k not in state for k in required):
            raise KeyError(f"recovery state missing {[k for k in required if k not in state]}")
        values = {k: torch.as_tensor(state[k]).detach().cpu().flatten().clone() for k in required}
        if values["qpos"].numel() != 29 or values["qvel"].numel() != 29:
            raise ValueError("recovery qpos/qvel must contain 29 values")
        if values["root_pos"].numel() != 3 or values["root_quat"].numel() != 4:
            raise ValueError("recovery root pose must be (3,) and (4,)")
        item = RecoveryState(**values, seq_idx=int(seq_idx), frame=float(frame))
        if len(self._states) >= self.capacity:
            self._states.pop(0)
        self._states.append(item)

    def sample(self, batch_size: int, *, device="cpu", generator=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._states:
            raise RuntimeError("cannot sample an empty recovery pool")
        idx = torch.randint(len(self._states), (batch_size,), generator=generator)
        out = {}
        for key in ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel"):
            out[key] = torch.stack([getattr(self._states[int(i)], key) for i in idx]).to(device)
        out["seq_idx"] = torch.tensor([self._states[int(i)].seq_idx for i in idx], device=device, dtype=torch.long)
        out["frame"] = torch.tensor([self._states[int(i)].frame for i in idx], device=device)
        return out

    def state_dict(self) -> dict:
        """Return the bounded curriculum state for checkpointed training."""
        return {
            "capacity": self.capacity,
            "init_prob": self.init_prob,
            "prob_max": self.prob_max,
            "survival_window": self.outcomes.maxlen,
            "outcomes": list(self.outcomes),
            "states": [
                {
                    key: getattr(item, key).clone() if isinstance(getattr(item, key), torch.Tensor)
                    else getattr(item, key)
                    for key in ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "seq_idx", "frame")
                }
                for item in self._states
            ],
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore a state previously returned by :meth:`state_dict`."""
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError("recovery checkpoint capacity does not match the configured pool")
        self.outcomes = deque((bool(x) for x in state.get("outcomes", ())),
                              maxlen=self.outcomes.maxlen)
        self._states = []
        for item in state.get("states", ()):
            self.add(item, seq_idx=int(item["seq_idx"]), frame=float(item["frame"]))
