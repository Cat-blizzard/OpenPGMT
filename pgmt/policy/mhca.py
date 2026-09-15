"""MHCA（Multi-Head Cross Attention）——History Encoder 与 IFM 共用。

实现为单层标准交叉注意力（Q/K/V/输出投影，无残差、无 LayerNorm），
与论文 "a multi-head cross-attention (MHCA) layer" 的字面描述一致。
可选 RoPE：对 Q/K 的每头特征按显式位置下标做旋转变换，
使注意力得分感知 token 间的相对位置（IFM 用于未来参考帧的时间偏移）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pgmt.policy.rope import RotaryPositionEmbedding


class MHCA(nn.Module):
    """多头交叉注意力。

    Args:
        d_model: token 维度（256，A3）。
        heads: 头数（4，A3），须整除 d_model。
        rope: RotaryPositionEmbedding 或 None；None 即普通注意力。

    输入/输出: q (B, Lq, d_model)；k, v (B, Lk, d_model) → (B, Lq, d_model)。
    RoPE 开启时须同时给 q_idx (Lq,) 与 k_idx (Lk,)，下标单位 = 帧（τ=2^k−1）。
    """

    def __init__(self, d_model: int, heads: int, rope: RotaryPositionEmbedding | None = None):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model 必须整除头数: {d_model} % {heads} != 0")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = rope

    def forward(self, q, k, v, q_idx=None, k_idx=None, return_attn=False):
        B, Lq, _ = q.shape
        q = self.q_proj(q).view(B, Lq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            if q_idx is None or k_idx is None:
                raise ValueError("启用 RoPE 的 MHCA 必须提供 q_idx 与 k_idx")
            q = self.rope(q, q_idx)
            k = self.rope(k, k_idx)

        scores = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)  # (B, heads, Lq, Lk)
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (B, heads, Lq, head_dim)
        out = out.transpose(1, 2).reshape(B, Lq, self.heads * self.head_dim)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        return out
