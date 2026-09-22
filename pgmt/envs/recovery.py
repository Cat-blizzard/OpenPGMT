"""Fall-state storage and recovery curriculum for Stage 1.

The paper requires recovery episodes to start from post-fall states.  This
module is simulator agnostic: the environment owns the tensors and records a
snapshot when a physical termination occurs; the pool only controls bounded
storage, sampling, and the survival-rate curriculum.
"""

from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass
from typing import Mapping

import torch
from pgmt.cfg.assumptions import get


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
        self.recovery_episode_count = 0
        cfg = get("A10").value
        self.min_ordinary_episodes = cfg.min_ordinary_episodes
        self.curriculum_interval = cfg.curriculum_interval
        self.probability_max_step = cfg.probability_max_step
        self.ordinary_survival = deque(maxlen=int(survival_window))
        self.ordinary_episode_count = 0
        self._probability = self.init_prob
        self._states: list[RecoveryState] = []
        self.provenance: dict = {}

    def __len__(self) -> int:
        return len(self._states)

    @property
    def probability(self) -> float:
        return self._probability

    def record_outcome(self, survived: bool) -> None:
        """Recovery completion is diagnostic; failure must not raise difficulty."""
        self.outcomes.append(bool(survived))
        self.recovery_episode_count += 1

    def record_ordinary_episode(self, duration_s: float, horizon_s: float) -> None:
        """A10 engineering choice: ordinary normalized survival gates recovery.

        Smooth over a window, require sufficient ordinary episodes, and bound
        changes at each review. PGMT does not publish this exact schedule.
        """
        if not math.isfinite(duration_s) or not math.isfinite(horizon_s) or duration_s < 0 or horizon_s <= 0:
            raise ValueError("invalid ordinary episode duration/horizon")
        self.ordinary_survival.append(min(duration_s / horizon_s, 1.))
        self.ordinary_episode_count += 1
        if (self.ordinary_episode_count < self.min_ordinary_episodes
                or self.ordinary_episode_count % self.curriculum_interval):
            return
        survival = sum(self.ordinary_survival) / len(self.ordinary_survival)
        desired = self.init_prob + survival * (self.prob_max - self.init_prob)
        change = max(-self.probability_max_step, min(self.probability_max_step, desired - self._probability))
        self._probability = max(self.init_prob, min(self.prob_max, self._probability + change))

    def curriculum_metrics(self) -> dict:
        return {"reset_probability": self.probability,
                "ordinary_episodes": self.ordinary_episode_count,
                "ordinary_survival_fraction": (sum(self.ordinary_survival) / len(self.ordinary_survival)
                                                 if self.ordinary_survival else 0.),
                "recovery_episodes": self.recovery_episode_count,
                "recovery_window_episodes": len(self.outcomes),
                "recovery_completion": sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.}

    def add(self, state: Mapping[str, torch.Tensor], *, seq_idx: int, frame: float) -> None:
        required = ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel")
        if any(k not in state for k in required):
            raise KeyError(f"recovery state missing {[k for k in required if k not in state]}")
        values = {k: torch.as_tensor(state[k]).detach().cpu().flatten().clone() for k in required}
        if values["qpos"].numel() != 29 or values["qvel"].numel() != 29:
            raise ValueError("recovery qpos/qvel must contain 29 values")
        if values["root_pos"].numel() != 3 or values["root_quat"].numel() != 4:
            raise ValueError("recovery root pose must be (3,) and (4,)")
        if any(not torch.isfinite(v).all() for v in values.values()):
            raise ValueError("recovery state contains NaN/Inf")
        if any(values[k].numel() != 3 for k in ("root_lin_vel", "root_ang_vel")):
            raise ValueError("recovery velocities must contain 3 values")
        if not torch.isclose(values["root_quat"].norm(), torch.tensor(1.), atol=1e-3):
            raise ValueError("recovery quaternion must be normalized")
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
            "provenance": self.provenance,
            "capacity": self.capacity,
            "init_prob": self.init_prob,
            "prob_max": self.prob_max,
            "survival_window": self.outcomes.maxlen,
            "outcomes": list(self.outcomes),
            "recovery_episode_count": self.recovery_episode_count,
            "curriculum_version": 4,
            "ordinary_survival": list(self.ordinary_survival),
            "ordinary_episode_count": self.ordinary_episode_count,
            "probability": self._probability,
            "min_ordinary_episodes": self.min_ordinary_episodes,
            "curriculum_interval": self.curriculum_interval,
            "probability_max_step": self.probability_max_step,
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
        if state.get("curriculum_version") != 4 and state.get("outcomes"):
            raise ValueError("obsolete recovery curriculum; only untrained offline pools may be migrated")
        self.outcomes = deque((bool(x) for x in state.get("outcomes", ())),
                              maxlen=self.outcomes.maxlen)
        self.recovery_episode_count = int(state.get("recovery_episode_count", len(self.outcomes)))
        self.ordinary_survival = deque(state.get("ordinary_survival", ()), maxlen=self.outcomes.maxlen)
        self.ordinary_episode_count = int(state.get("ordinary_episode_count", 0))
        self._probability = float(state.get("probability", self.init_prob))
        for key in ("min_ordinary_episodes", "curriculum_interval", "probability_max_step"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(f"recovery curriculum setting differs: {key}")
        self._states = []
        self.provenance = dict(state.get("provenance", {}))
        for item in state.get("states", ()):
            self.add(item, seq_idx=int(item["seq_idx"]), frame=float(item["frame"]))


def load_fall_pool(path, *, require_physics=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, FallRecoveryPool):
        pool = FallRecoveryPool(payload.capacity, payload.init_prob, payload.prob_max, payload.outcomes.maxlen)
        # Pickled legacy objects are accepted only for the historical empty
        # pool error; populated artifacts use the explicit schema below.
        if len(payload):
            raise ValueError("use a versioned offline pool artifact")
    elif isinstance(payload, dict) and payload.get("schema") == "pgmt_fall_pool_v2":
        state = payload["pool"]
        pool = FallRecoveryPool(state["capacity"], state["init_prob"], state["prob_max"], state["survival_window"])
        pool.load_state_dict(state)
    else:
        raise ValueError("expected pgmt_fall_pool_v2 artifact")
    if not len(pool):
        raise ValueError("fall pool is empty")
    if require_physics and getattr(pool, "provenance", {}).get("backend") != "isaaclab":
        raise ValueError("formal training requires an offline physical fall pool with provenance")
    return pool
