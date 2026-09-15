"""Multi-Head Critic（论文 Eq.6/9 + Fig.4）。

V_t = MLP_V([o_priv, s_int]) = (V_upper, V_lower, V_aux)                    Stage 1
V_inj = (V_upper, V_lower, V_terrain, V_aux)                                Stage 2（Eq.9）

各 reward 组的回报由共享 backbone + 独立末层头估计，避免量级差异
互扰。Stage 2 只改末层：3 头 → 4 头，且 terrain 头插在 aux 之前
（Eq.9 顺序）——加载 Stage 1 权重时需按此重排（见 stage1_to_stage2_head）。
消融 Aggregate-critic 对应 head_names=("total",) 的单头版本（Fig.4）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pgmt.cfg.assumptions import get
from pgmt.policy.mlp import MLP

# Eq.6 / Eq.9 的头顺序（唯一出处，train_stage2 加载权重时使用）
HEAD_ORDER_STAGE1 = ("upper", "lower", "aux")
HEAD_ORDER_STAGE2 = ("upper", "lower", "terrain", "aux")
HEAD_ORDER_AGGREGATE = ("total",)  # Fig.4 消融：单头聚合回报


class MultiHeadCritic(nn.Module):
    """共享 backbone 的多头价值网络。

    Args:
        priv_dim:   特权观测维度（由环境层给定，A9 集合）
        intent_dim: s_int 维度 256（A3）
        mlp_dims:   共享 backbone 隐层 (512,256,128)（A4.critic）
        head_names: 值头名称与顺序；默认 Stage 1 三头
    """

    def __init__(self, priv_dim: int, intent_dim: int | None = None,
                 mlp_dims: tuple | None = None, head_names: tuple = HEAD_ORDER_STAGE1,
                 activation: str | None = None):
        super().__init__()
        scale = get("A3").value
        dims = get("A4").value
        intent_dim = scale.token_dim if intent_dim is None else intent_dim
        mlp_dims = dims.critic if mlp_dims is None else mlp_dims
        if len(head_names) == 0 or len(set(head_names)) != len(head_names):
            raise ValueError(f"头名须非空且不重复: {head_names}")
        self.head_names = tuple(head_names)
        self.backbone = MLP([priv_dim + intent_dim, *mlp_dims], activation)
        self.head = nn.Linear(mlp_dims[-1], len(head_names))

    def forward(self, o_priv: torch.Tensor, s_int: torch.Tensor) -> torch.Tensor:
        """Args:
            o_priv: 特权观测 (B, priv_dim)，仅仿真训练可用
            s_int:  motion-intent token (B, intent_dim)
        Returns:
            V: (B, num_heads)，列序 = self.head_names
        """
        h = self.backbone(torch.cat([o_priv, s_int], dim=-1))
        return self.head(h)

    def values(self, o_priv: torch.Tensor, s_int: torch.Tensor) -> dict:
        """按头名返回 {name: (B,)} 字典，供分头回报/日志使用。"""
        v = self.forward(o_priv, s_int)
        return {name: v[:, i] for i, name in enumerate(self.head_names)}


def stage1_to_stage2_head(w: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1 三头末层权重 → Stage 2 四头末层权重（Eq.9 顺序重排）。

    Stage 1 头序 (upper, lower, aux) → Stage 2 (upper, lower, terrain, aux)：
    upper/lower 直接复制；aux 移到第 4 列；terrain 头（第 3 列）随机初始化，
    返回的权重与 torch 默认初始化分布一致，可直接作为新末层参数。
    """
    if w.dim() != 2 or w.shape[1] != 3 or b.shape != (3,):
        raise ValueError(f"期望 Stage 1 末层形状 (H,3)/(3,)，得到 {tuple(w.shape)}/{tuple(b.shape)}")
    w2 = torch.empty(w.shape[0], 4, dtype=w.dtype)
    b2 = torch.empty(4, dtype=b.dtype)
    nn.init.xavier_uniform_(w2)
    nn.init.zeros_(b2)
    with torch.no_grad():
        w2[:, 0], w2[:, 1], w2[:, 3] = w[:, 0], w[:, 1], w[:, 2]
        b2[0], b2[1], b2[3] = b[0], b[1], b[2]
    return w2, b2
