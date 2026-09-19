"""RoPE（Rotary Position Embedding）。

用于 IFM 中未来参考帧 C^K 的位置编码：控制步偏移 τ_k = 2^k − 1 直接作为
旋转角度下标，使 MHCA 注意力天然感知参考帧间的相对时间距离。
纯 torch 实现，可在服务器回归环境中直接测试（tests/test_rope.py）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """对最后一维特征做旋转变换，位置由显式下标给出。

    Args:
        dim: 特征维度（必须为偶数；通常 = head_dim）。
        base: 频率基数（RoPE 惯例 10000）。

    输入:  x       (..., L, dim)，L 个 token 的特征
           seq_idx (L,) int/float，每帧的位置下标（如 τ_k）
    输出:  (..., L, dim)，第 l 帧的 (2i, 2i+1) 特征对旋转
           angle = seq_idx[l] * base^{-2i/dim}
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE 维度必须为偶数，得到 {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.inv_freq.shape[0] * 2:
            raise ValueError(f"特征维度不匹配: x 为 {x.shape[-1]}，RoPE 为 {self.inv_freq.shape[0] * 2}")
        freqs = torch.outer(seq_idx.to(self.inv_freq.dtype), self.inv_freq)  # (L, dim/2)
        cos, sin = freqs.cos(), freqs.sin()

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out
