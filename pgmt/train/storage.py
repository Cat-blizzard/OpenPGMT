"""分头 rollout 与 GAE；显式存下一状态价值以隔离自动 reset 边界。"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch

from pgmt.cfg.assumptions import get
from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import PRIV_DIM


class RolloutStorage:
    """``next_values`` 必须来自 reset 前的下一状态。

    真正 termination 不 bootstrap；time limit 使用终态价值 bootstrap，
    但两者都截断 GAE 的跨步递推，禁止把下个 episode 的优势传回来。
    """

    def __init__(self, num_steps: int, num_envs: int,
                 observation_shapes: Optional[Dict[str, tuple]] = None,
                 action_dim: int = ACT_DIM, num_heads: int = 3,
                 device: str = "cpu"):
        if num_steps <= 0 or num_envs <= 0 or num_heads <= 0:
            raise ValueError("rollout dimensions must be positive")
        self.num_steps, self.num_envs = int(num_steps), int(num_envs)
        self.num_heads = int(num_heads)
        self.device = torch.device(device)
        shapes = observation_shapes or {
            "obs": (OBS_DIM,), "history": (HISTORY_LEN, OBS_DIM),
            "future": (get("A2").value.K, REF_FRAME_DIM), "privileged": (PRIV_DIM,),
        }
        prefix = (self.num_steps, self.num_envs)
        self.observations = {
            key: torch.zeros(prefix + tuple(shape), device=self.device)
            for key, shape in shapes.items()
        }
        self.actions = torch.zeros(prefix + (action_dim,), device=self.device)
        self.log_probs = torch.zeros(prefix, device=self.device)
        self.values = torch.zeros(prefix + (num_heads,), device=self.device)
        self.rewards = torch.zeros_like(self.values)
        self.next_values = torch.zeros_like(self.values)
        self.terminated = torch.zeros(prefix, dtype=torch.bool, device=self.device)
        self.timeouts = torch.zeros_like(self.terminated)
        self.advantages = torch.zeros_like(self.values)
        self.returns = torch.zeros_like(self.values)
        self.policy_advantages = torch.zeros(prefix, device=self.device)
        self.step = 0
        self.ready = False

    @torch.no_grad()
    def add(self, observations, actions, log_probs, values, rewards,
            terminated, timeouts, next_values):
        if self.step >= self.num_steps:
            raise RuntimeError("rollout is full; clear it after updating")
        if set(observations) != set(self.observations):
            raise ValueError("observation keys differ from rollout layout")
        targets = [(self.observations[k][self.step], v, k)
                   for k, v in observations.items()]
        targets += [(getattr(self, key)[self.step], value, key) for key, value in (
            ("actions", actions), ("log_probs", log_probs), ("values", values),
            ("rewards", rewards), ("terminated", terminated), ("timeouts", timeouts),
            ("next_values", next_values),
        )]
        for target, value, key in targets:
            if target.shape != value.shape:
                raise ValueError("%s shape %s != %s" % (key, value.shape, target.shape))
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("%s contains non-finite values" % key)
        for target, value, _ in targets:
            target.copy_(value.detach())
        self.step += 1
        self.ready = False

    @torch.no_grad()
    def compute_returns(self, gamma: float, lam: float):
        if self.step != self.num_steps:
            raise RuntimeError("compute_returns requires a full rollout")
        if not (0 <= gamma <= 1 and 0 <= lam <= 1):
            raise ValueError("gamma and lambda must be in [0, 1]")
        advantage = torch.zeros_like(self.values[0])
        for index in reversed(range(self.num_steps)):
            bootstrap = (~self.terminated[index]).unsqueeze(-1)
            continuation = (~(self.terminated[index] | self.timeouts[index])).unsqueeze(-1)
            delta = (self.rewards[index] + gamma * bootstrap * self.next_values[index]
                     - self.values[index])
            advantage = delta + gamma * lam * continuation * advantage
            self.advantages[index].copy_(advantage)
        self.returns.copy_(self.advantages + self.values)
        # 先逐奖励头归一化，再等权平均；不让大尺度组支配策略更新。
        flat = self.advantages.flatten(0, 1)
        normalized = (flat - flat.mean(0)) / (flat.std(0, unbiased=False) + 1e-8)
        self.policy_advantages.copy_(normalized.mean(-1).view(self.num_steps, self.num_envs))
        self.ready = True

    def minibatches(self, num_mini_batches: int, num_epochs: int) -> Iterator[dict]:
        if not self.ready:
            raise RuntimeError("compute_returns must precede minibatches")
        count = self.num_steps * self.num_envs
        if not (1 <= num_mini_batches <= count) or num_epochs <= 0:
            raise ValueError("invalid minibatch count or epoch count")
        flat_obs = {key: value.flatten(0, 1) for key, value in self.observations.items()}
        fields = {key: getattr(self, key).flatten(0, 1) for key in (
            "actions", "log_probs", "values", "returns", "policy_advantages",
        )}
        for _ in range(num_epochs):
            permutation = torch.randperm(count, device=self.device)
            for indices in torch.tensor_split(permutation, num_mini_batches):
                batch = {key: value[indices] for key, value in fields.items()}
                batch["observations"] = {key: value[indices] for key, value in flat_obs.items()}
                yield batch

    def clear(self):
        self.step = 0
        self.ready = False
