"""Stage 1 的完整高斯策略：共享历史/意图编码器、Actor 和三头价值网络。

策略采样的是未缩放的关节动作，log probability 也始终对应这个随机变量。
动作到 PD 目标的确定性变换由环境负责，不能把缩放/裁剪后的动作写回 PPO。
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from pgmt.contracts import ACT_DIM
from pgmt.envs.observations import PRIV_DIM
from pgmt.policy.actor import Actor
from pgmt.policy.history_encoder import HistoryEncoder
from pgmt.policy.ifm import IFM
from pgmt.policy.multi_head_critic import HEAD_ORDER_STAGE1, MultiHeadCritic


class PolicyOutput(NamedTuple):
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    entropy: torch.Tensor


class Stage1Policy(nn.Module):
    """接受 ``obs/history/future/privileged`` 字典，价值列序为 upper/lower/aux。

    ``log_std`` 是每个关节独立、与状态无关的可训练探索尺度。
    ``forward`` 返回确定性均值；``act`` 采样；``evaluate_actions`` 用于 PPO。
    """

    head_names = HEAD_ORDER_STAGE1

    def __init__(self, priv_dim: int = PRIV_DIM, init_noise_std: float = 1.0):
        super().__init__()
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")
        self.history_encoder = HistoryEncoder()
        self.ifm = IFM()
        self.actor = Actor()
        self.critic = MultiHeadCritic(priv_dim=priv_dim, head_names=self.head_names)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(init_noise_std)))

    def _intent(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        history = self.history_encoder(observations["obs"], observations["history"])
        return self.ifm(history, observations["future"])

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(observations["obs"], self._intent(observations))

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(observations["privileged"], self._intent(observations))

    def _distribution_and_values(self, observations):
        intent = self._intent(observations)
        mean = self.actor(observations["obs"], intent)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        values = self.critic(observations["privileged"], intent)
        return distribution, values

    def act(self, observations: Dict[str, torch.Tensor],
            deterministic: bool = False) -> PolicyOutput:
        distribution, values = self._distribution_and_values(observations)
        actions = distribution.mean if deterministic else distribution.sample()
        return PolicyOutput(actions, distribution.log_prob(actions).sum(-1), values,
                            distribution.entropy().sum(-1))

    def evaluate_actions(self, observations: Dict[str, torch.Tensor],
                         actions: torch.Tensor) -> PolicyOutput:
        distribution, values = self._distribution_and_values(observations)
        return PolicyOutput(actions, distribution.log_prob(actions).sum(-1), values,
                            distribution.entropy().sum(-1))
