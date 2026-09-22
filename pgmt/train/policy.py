"""Bounded joint-target policies; PPO stores pre-tanh samples explicitly."""

from __future__ import annotations

import math
from typing import Dict, NamedTuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from data.retarget_lafan1 import G1_JOINT_NAMES, G1_JOINT_LIMITS

from pgmt.contracts import ACT_DIM
from pgmt.cfg.assumptions import get
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
    latent_actions: torch.Tensor | None = None


class BoundedJointPolicy(nn.Module):
    action_contract = "joint_targets_tanh_v2"

    def _init_bounds(self):
        limits = torch.tensor([G1_JOINT_LIMITS[n] for n in G1_JOINT_NAMES])
        self.register_buffer("action_mid", limits.mean(-1))
        self.register_buffer("action_half_range", (limits[:, 1] - limits[:, 0]) / 2)

    def _target(self, latent):
        return self.action_mid + self.action_half_range * latent.tanh()

    def _init_actor_targets(self, initial_joint_targets=None):
        # The standard G1 reset has all joints at zero. Preserve absolute
        # target semantics while centering a fresh policy on that posture.
        targets = (torch.zeros_like(self.action_mid) if initial_joint_targets is None
                   else torch.as_tensor(initial_joint_targets, dtype=self.action_mid.dtype))
        normalized = (targets - self.action_mid) / self.action_half_range
        if targets.shape != (ACT_DIM,) or not torch.isfinite(normalized).all() or (normalized.abs() >= 1).any():
            raise ValueError("initial joint targets must be strictly inside joint limits")
        with torch.no_grad():
            self.actor.mlp.net[-1].weight.mul_(get("A4").value.actor_output_weight_scale)
            self.actor.mlp.net[-1].bias.copy_(torch.atanh(normalized))

    def latent_distribution(self, observations):
        """Untransformed Normal; its analytic KL equals the bounded policy KL."""
        return self._distribution_and_values(observations)[0]

    @torch.no_grad()
    def detached_critic_inputs(self, observations):
        """Freeze shared features only during optional extra value fitting."""
        return observations["privileged"].detach(), self._intent(observations).detach()

    def _log_prob(self, distribution, latent):
        # Stable even when tanh(latent) rounds to +/-1 in float32.
        log_jacobian = self.action_half_range.log() + 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(-1)

    def _output(self, distribution, values, latent, actions=None):
        actions = self._target(latent) if actions is None else actions
        # Entropy of the transformed distribution, estimated with a fresh
        # reparameterized sample (never the old rollout actions).
        entropy = -self._log_prob(distribution, distribution.rsample())
        return PolicyOutput(actions, self._log_prob(distribution, latent), values, entropy, latent)

    def act(self, observations, deterministic=False):
        distribution, values = self._distribution_and_values(observations)
        latent = distribution.mean if deterministic else distribution.sample()
        return self._output(distribution, values, latent)

    def evaluate_actions(self, observations, actions, *, latent_actions=None):
        distribution, values = self._distribution_and_values(observations)
        if latent_actions is None:
            normalized = (actions - self.action_mid) / self.action_half_range
            if (normalized.abs() >= 1).any():
                raise ValueError("saturated joint targets require saved latent_actions")
            latent_actions = torch.atanh(normalized)
        elif not torch.allclose(actions, self._target(latent_actions), atol=2e-6, rtol=1e-6):
            raise ValueError("saved latent_actions do not reproduce the executed targets")
        return self._output(distribution, values, latent_actions, actions)


class Stage1Policy(BoundedJointPolicy):
    """接受 ``obs/history/future/privileged`` 字典，价值列序为 upper/lower/aux。

    ``log_std`` 是每个关节独立、与状态无关的可训练探索尺度。
    ``forward`` 返回 latent 均值映射后的确定性目标；``act`` 采样；``evaluate_actions`` 用于 PPO。
    """

    head_names = HEAD_ORDER_STAGE1

    def __init__(self, priv_dim: int = PRIV_DIM, init_noise_std: float | None = None,
                 initial_joint_targets=None, neutral_init: bool = True):
        super().__init__()
        self._init_bounds()
        if init_noise_std is None:
            init_noise_std = get("A4").value.actor_init_noise_std
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")
        self.history_encoder = HistoryEncoder()
        self.ifm = IFM()
        self.actor = Actor()
        self.critic = MultiHeadCritic(priv_dim=priv_dim, head_names=self.head_names)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(init_noise_std)))
        if neutral_init:
            self._init_actor_targets(initial_joint_targets)

    def _intent(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        history = self.history_encoder(observations["obs"], observations["history"])
        return self.ifm(history, observations["future"])

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._target(self.actor(observations["obs"], self._intent(observations)))

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(observations["privileged"], self._intent(observations))

    def _distribution_and_values(self, observations):
        intent = self._intent(observations)
        mean = self.actor(observations["obs"], intent)
        distribution = Normal(mean, self.log_std.clamp(-5, 1).exp().expand_as(mean))
        values = self.critic(observations["privileged"], intent)
        return distribution, values



class Stage2Policy(BoundedJointPolicy):
    """Perception-injected policy from Sec. IV-B of the paper.

    The history encoder, IFM, actor and critic backbone are initialized from a
    Stage 1 policy.  ``observations`` must additionally contain an elevation
    map under ``elevation`` (``terrain`` is accepted as an explicit alias).
    The critic has the paper's four split-return heads:
    ``upper/lower/terrain/aux``.
    """

    head_names = HEAD_ORDER_STAGE2

    def __init__(self, priv_dim: int = PRIV_DIM, init_noise_std: float | None = None,
                 initial_joint_targets=None, neutral_init: bool = True):
        super().__init__()
        self._init_bounds()
        if init_noise_std is None:
            init_noise_std = get("A4").value.actor_init_noise_std
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")
        self.history_encoder = HistoryEncoder()
        self.glimpse_encoder = TerrainGlimpseEncoder()
        self.ifm = IFM()
        self.actor = Actor()
        self.critic = MultiHeadCritic(priv_dim=priv_dim, head_names=self.head_names)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(init_noise_std)))
        if neutral_init:
            self._init_actor_targets(initial_joint_targets)

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
                             "critic.backbone.", "log_std", "action_mid", "action_half_range")
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
        return self._target(self.actor(observations["obs"], self._intent(observations)))

    def value(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(observations["privileged"], self._intent(observations))

    def _distribution_and_values(self, observations):
        intent = self._intent(observations)
        mean = self.actor(observations["obs"], intent)
        distribution = Normal(mean, self.log_std.clamp(-5, 1).exp().expand_as(mean))
        values = self.critic(observations["privileged"], intent)
        return distribution, values
