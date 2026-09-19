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
from pgmt.policy.multi_head_critic import HEAD_ORDER_STAGE2, stage1_to_stage2_head
from pgmt.policy.glimpse_encoder import TerrainGlimpseEncoder


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


class Stage2Policy(nn.Module):
    """Perception-injected policy from Sec. IV-B of the paper.

    The history encoder, IFM, actor and critic backbone are initialized from a
    Stage 1 policy.  ``observations`` must additionally contain an elevation
    map under ``elevation`` (``terrain`` is accepted as an explicit alias).
    The critic has the paper's four split-return heads:
    ``upper/lower/terrain/aux``.
    """

    head_names = HEAD_ORDER_STAGE2

    def __init__(self, priv_dim: int = PRIV_DIM, init_noise_std: float = 1.0):
        super().__init__()
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")
        self.history_encoder = HistoryEncoder()
        self.glimpse_encoder = TerrainGlimpseEncoder()
        self.ifm = IFM()
        self.actor = Actor()
        self.critic = MultiHeadCritic(priv_dim=priv_dim, head_names=self.head_names)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(init_noise_std)))

    @classmethod
    def from_stage1(cls, source: Stage1Policy, *, priv_dim: int | None = None) -> "Stage2Policy":
        """Construct Stage 2 and transfer all compatible Stage 1 parameters."""
        intent_dim = source.ifm.q_proj.in_features
        target = cls(priv_dim=source.critic.backbone.net[0].in_features - intent_dim
                     if priv_dim is None else priv_dim,
                     init_noise_std=float(source.log_std.exp().mean().detach()))
        target.load_stage1_state_dict(source.state_dict())
        return target

    def load_stage1_state_dict(self, state_dict, *, strict: bool = True):
        """Load Stage 1 weights, inserting a freshly initialized terrain head."""
        state = dict(state_dict)
        # The shared policy modules and actor/log_std have identical keys.
        own = self.state_dict()
        required_prefixes = ("history_encoder.", "ifm.", "actor.",
                             "critic.backbone.", "log_std")
        required = [key for key in own
                    if any(key.startswith(prefix) for prefix in required_prefixes)]
        missing = [key for key in required if key not in state]
        mismatched = [key for key in required
                      if key in state and own[key].shape != state[key].shape]
        if missing or mismatched:
            raise ValueError(f"incomplete Stage 1 checkpoint; missing={missing}, shape_mismatch={mismatched}")
        compatible = {key: state[key] for key in required}
        result = self.load_state_dict(compatible, strict=False)
        old_w, old_b = state.get("critic.head.weight"), state.get("critic.head.bias")
        if old_w is None or old_b is None:
            if strict:
                raise KeyError("Stage 1 checkpoint lacks critic.head.weight/bias")
        else:
            w2, b2 = stage1_to_stage2_head(old_w.to(self.critic.head.weight),
                                           old_b.to(self.critic.head.bias))
            self.critic.head.weight.data.copy_(w2)
            self.critic.head.bias.data.copy_(b2)
        if strict and (old_w is not None and tuple(old_w.shape) != tuple((3, self.critic.head.in_features))):
            raise ValueError("Stage 1 critic head must have shape (3,H)")
        return result

    def _intent(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        history = self.history_encoder(observations["obs"], observations["history"])
        elevation = observations.get("elevation", observations.get("terrain"))
        if elevation is None:
            raise KeyError("Stage2Policy requires observations['elevation']")
        if elevation.ndim == 4 and elevation.shape[1] == 1:
            elevation = elevation[:, 0]
        expected = self.glimpse_encoder.map_size
        if elevation.shape != (observations["obs"].shape[0], expected, expected):
            raise ValueError(f"elevation must have shape (B,{expected},{expected})")
        z, _, _ = self.glimpse_encoder(elevation, history, observations["future"])
        return self.ifm(history, observations["future"], z_terrain=z)

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

    def act(self, observations: Dict[str, torch.Tensor], deterministic: bool = False) -> PolicyOutput:
        distribution, values = self._distribution_and_values(observations)
        actions = distribution.mean if deterministic else distribution.sample()
        return PolicyOutput(actions, distribution.log_prob(actions).sum(-1), values,
                            distribution.entropy().sum(-1))

    def evaluate_actions(self, observations: Dict[str, torch.Tensor],
                         actions: torch.Tensor) -> PolicyOutput:
        distribution, values = self._distribution_and_values(observations)
        return PolicyOutput(actions, distribution.log_prob(actions).sum(-1), values,
                            distribution.entropy().sum(-1))
