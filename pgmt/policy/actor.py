"""Actor（论文 Eq.4）。

a_t = MLP_A([o_t, s_int_t])

拼接当前 proprioception 与 motion-intent token：s_int 提供状态条件的
运动意图，o_t 保留瞬时反馈供精细控制。输出 29 维关节位置目标，
经环境层 PD 控制器转力矩。模块只输出 MLP logits；tanh 缩放与
动作 scale 由环境/训练层负责（M2）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pgmt.cfg.assumptions import get
from pgmt.policy.mlp import MLP

_OBS_DIM = 96
_ACT_DIM = 29  # G1 29-DoF 关节位置目标


class Actor(nn.Module):
    """MLP([o_t(96), s_int(256)]) → a_t(29)。"""

    def __init__(self, obs_dim: int = _OBS_DIM, intent_dim: int | None = None,
                 mlp_dims: tuple | None = None, act_dim: int = _ACT_DIM,
                 activation: str | None = None):
        super().__init__()
        scale = get("A3").value
        dims = get("A4").value
        intent_dim = scale.token_dim if intent_dim is None else intent_dim
        mlp_dims = dims.actor if mlp_dims is None else mlp_dims
        self.mlp = MLP([obs_dim + intent_dim, *mlp_dims, act_dim], activation)

    def forward(self, o: torch.Tensor, s_int: torch.Tensor) -> torch.Tensor:
        """Args:
            o:     当前 proprioception (B, obs_dim)
            s_int: motion-intent token (B, intent_dim)
        Returns:
            a_t: 关节位置目标 logits (B, act_dim)
        """
        return self.mlp(torch.cat([o, s_int], dim=-1))
