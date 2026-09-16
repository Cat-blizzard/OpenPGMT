"""Intent Fusion Module，IFM（论文 Eq.3 + §IV-B 注入）。

s_int = CrossAttn_C(Q = s_hist, K = V = [s_hist; C^K])          （Stage 1）
s_int = CrossAttn_C(Q = s_hist, K = V = [s_hist; C^K; z_1..z_4]) （Stage 2，注入地形 token）

MHCA 配备 RoPE：KV 中时间 token（s_hist 与未来参考帧）按帧偏移旋转，
使注意力感知参考帧间的时间距离；地形 token 为空间信息（位置已编码在
z_j 内部，见 glimpse_encoder），下标取 0 = 恒等旋转。

RoPE 位置下标（相对当前控制步 t，独立于运动数据源帧率）：
  [s_hist]            → 0
  [C^K_k, k=0..5]     → τ_k = 2^k − 1（0,1,3,7,15,31，A2）
  [z_1..z_4]          → 0（不旋转）
Q = s_hist 也取 0，因此注意力得分只依赖 KV token 与当前时刻的相对偏移。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pgmt.cfg.assumptions import get
from pgmt.contracts import REF_FRAME_DIM
from pgmt.policy.mhca import MHCA
from pgmt.policy.rope import RotaryPositionEmbedding

_REF_DIM = REF_FRAME_DIM  # Eq.1 的维度契约


class IFM(nn.Module):
    """MHCA+RoPE：融合历史状态 token 与未来参考帧（可注入地形 token）。

    Args:
        token_dim: 256（A3）
        ref_dim: 未来参考帧维度 61
        heads: 4（A3）
        rope_dim: RoPE 维度 64（A5，= head_dim）
        num_glimpses: Stage 2 地形 token 数 4（A15），仅用于预计算位置下标
    """

    def __init__(self, token_dim: int | None = None, ref_dim: int = _REF_DIM,
                 heads: int | None = None, rope_dim: int | None = None,
                 num_glimpses: int | None = None):
        super().__init__()
        scale = get("A3").value
        rope_cfg = get("A5").value
        glimpse_cfg = get("A15").value
        token_dim = scale.token_dim if token_dim is None else token_dim
        heads = scale.mhca_heads if heads is None else heads
        rope_dim = rope_cfg.dim if rope_dim is None else rope_dim
        num_glimpses = glimpse_cfg.num_glimpses if num_glimpses is None else num_glimpses
        if rope_dim != token_dim // heads:
            raise ValueError(f"RoPE 维度应等于 head_dim: {rope_dim} != {token_dim // heads}")

        self.q_proj = nn.Linear(token_dim, token_dim)
        self.kv_s_proj = nn.Linear(token_dim, token_dim)  # s_hist 进 KV
        self.kv_ref_proj = nn.Linear(ref_dim, token_dim)  # C^K 进 KV
        self.attn = MHCA(token_dim, heads,
                         rope=RotaryPositionEmbedding(rope_dim, rope_cfg.base))

        # RoPE 位置下标（控制步）：Q=[0]；KV Stage1=[0]+τ_k，Stage2 追加 num_glimpses 个 0
        offsets = get("A2").value.offsets
        self.register_buffer("q_idx", torch.tensor([0.0]), persistent=False)
        self.register_buffer("k_idx_s1", torch.tensor([0.0] + list(offsets)), persistent=False)
        self.register_buffer("k_idx_s2",
                             torch.tensor([0.0] + list(offsets) + [0.0] * num_glimpses),
                             persistent=False)

    def forward(self, s_hist: torch.Tensor, c_future: torch.Tensor,
                z_terrain: torch.Tensor | None = None,
                return_attn: bool = False):
        """Args:
            s_hist:   历史条件状态 token (B, token_dim)
            c_future: 未来参考帧 (B, K, 61)，K=A2.K
            z_terrain: 地形 token (B, num_glimpses, token_dim)，None=Stage 1
        Returns:
            s_int (B, token_dim)；return_attn=True 时返回 (s_int, attn (B, heads, 1, Lk))
        """
        q = self.q_proj(s_hist).unsqueeze(1)  # (B, 1, D)
        k = torch.cat([self.kv_s_proj(s_hist).unsqueeze(1), self.kv_ref_proj(c_future)], dim=1)
        if z_terrain is not None:
            k = torch.cat([k, z_terrain], dim=1)  # (B, 1+K+Ng, D)
            k_idx = self.k_idx_s2
        else:
            k_idx = self.k_idx_s1
        out = self.attn(q, k, k, q_idx=self.q_idx, k_idx=k_idx, return_attn=return_attn)
        if return_attn:
            s_int, attn = out
            return s_int.squeeze(1), attn
        return out.squeeze(1)
