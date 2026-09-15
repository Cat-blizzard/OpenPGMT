"""History Encoder（论文 Eq.2）。

s_hist = CrossAttn_H(Q = o_t, K = V = H_t)

以当前 proprioception 为 query 关注 10 帧历史观测，产出紧凑的
历史条件状态 token（256 维），供 IFM 作 query 与 glimpse 位置选择器使用。
H_t 定义为当前帧之前的 10 帧（t−10..t−1，不含 o_t 本身），
由环境的观测环形缓冲维护（M2 环境层）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pgmt.cfg.assumptions import get
from pgmt.policy.mhca import MHCA

_OBS_DIM = 96  # 论文 §III：e_t(6)+ω_t(3)+q_t(29)+q̇_t(29)+a_{t−1}(29)


class HistoryEncoder(nn.Module):
    """单层 MHCA：Q=o_t(96)，K=V=H_t(10×96) → s_hist(256)。"""

    def __init__(self, obs_dim: int = _OBS_DIM, token_dim: int | None = None, heads: int | None = None):
        super().__init__()
        scale = get("A3").value
        token_dim = scale.token_dim if token_dim is None else token_dim
        heads = scale.mhca_heads if heads is None else heads
        self.q_proj = nn.Linear(obs_dim, token_dim)
        self.kv_proj = nn.Linear(obs_dim, token_dim)  # K=V=H_t 共享同一投影
        self.attn = MHCA(token_dim, heads)

    def forward(self, o: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Args:
            o: 当前 proprioception (B, 96)
            H: 历史 10 帧 (B, 10, 96)
        Returns:
            s_hist: (B, token_dim)
        """
        q = self.q_proj(o).unsqueeze(1)  # (B, 1, D)
        kv = self.kv_proj(H)  # (B, 10, D)
        return self.attn(q, kv, kv).squeeze(1)
