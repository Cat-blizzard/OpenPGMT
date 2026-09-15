"""Terrain-Glimpse Encoder（论文 Eq.7/8，Stage 2 组件）。

r_j = [MLP_φ(vec(M_t), s_hist, C^K)]_j,  j = 1..Ng          （位置选择器）
z_j = MLP_ψ(Crop_5×5(M_t, r_j), ρ_j)                        （局部地形 token）

位置选择器以整图高程 + 历史状态 + 未来参考为条件预测 Ng=4 个
glimpse 位置；双线性采样使裁剪可微，无需落足点标注即可学习。
ρ_j 为 glimpse 中心在机器人系下的坐标（= r_j，米），与 patch
高度值拼接后过 MLP_ψ 得到 256 维 token，注入 IFM 的 KV。

坐标约定：高程图 M ∈ R^{21×21} 覆盖机器人系 2m×2m，单元格
(i, j) 中心位于 x = −1 + i·0.1, y = −1 + j·0.1（米），
与 grid_sample(align_corners=True) 的角点对齐一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgmt.cfg.assumptions import get
from pgmt.policy.mlp import MLP


def _grid_from_centers(centers: torch.Tensor, patch_size: int, map_size: int) -> torch.Tensor:
    """由 glimpse 中心（米，机器人系）构造 grid_sample 采样网格。

    Args:
        centers: (B, Ng, 2)，(x, y) 米
    Returns:
        grid: (B, Ng*patch_size, patch_size, 2)，归一化坐标 (x, y) ∈ 约定范围
    """
    B, Ng, _ = centers.shape
    # 米 → 像素坐标：align_corners 下 −1m ↔ 0, +1m ↔ map_size−1
    p = (centers + 1.0) / 2.0 * (map_size - 1)  # (B, Ng, 2)
    off = torch.arange(patch_size, dtype=centers.dtype, device=centers.device) \
        - (patch_size - 1) / 2  # (P,)
    # grid[b,j,i,k] = (p_x(j)+off[k], p_y(j)+off[i])
    gx = (p[..., 0:1].unsqueeze(-1) + off.view(1, 1, 1, patch_size)).expand(-1, -1, patch_size, -1)
    gy = (p[..., 1:2].unsqueeze(-1) + off.view(1, 1, patch_size, 1)).expand(-1, -1, -1, patch_size)
    # 像素 → 归一化 [-1, 1]；展平为 (B, Ng*P, P, 2)：行 = glimpse j 的第 i 行
    nx = 2.0 * gx / (map_size - 1) - 1.0
    ny = 2.0 * gy / (map_size - 1) - 1.0
    return torch.stack([nx, ny], dim=-1).view(B, Ng * patch_size, patch_size, 2)


def sample_patches(M: torch.Tensor, centers: torch.Tensor, patch_size: int) -> torch.Tensor:
    """双线性采样高程图 patch（可微，越界补 0）。

    Args:
        M:       高程图 (B, map_size, map_size)，单位米
        centers: glimpse 中心 (B, Ng, 2)，米，机器人系
    Returns:
        patches: (B, Ng, patch_size, patch_size)
    """
    map_size = M.shape[-1]
    grid = _grid_from_centers(centers, patch_size, map_size)  # (B, Ng*P, P, 2)
    out = F.grid_sample(M.unsqueeze(1), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)  # (B, 1, Ng*P, P)
    return out.squeeze(1).view(M.shape[0], centers.shape[1], patch_size, patch_size)


class TerrainGlimpseEncoder(nn.Module):
    """位置选择器 + 双线性 patch 采样 + token MLP（Eq.7/8）。"""

    def __init__(self, map_size: int | None = None, num_glimpses: int | None = None,
                 patch_size: int | None = None, token_dim: int | None = None,
                 history_dim: int | None = None, ref_dim: int = 61,
                 selector_hidden: tuple | None = None, token_hidden: tuple | None = None,
                 loc_extent: float | None = None, activation: str | None = None):
        super().__init__()
        cfg = get("A15").value
        scale = get("A3").value
        map_size = get("A8").value.map_size if map_size is None else map_size
        num_glimpses = cfg.num_glimpses if num_glimpses is None else num_glimpses
        patch_size = cfg.patch_size if patch_size is None else patch_size
        token_dim = scale.token_dim if token_dim is None else token_dim
        history_dim = scale.token_dim if history_dim is None else history_dim
        K = get("A2").value.K
        selector_hidden = cfg.selector_hidden if selector_hidden is None else selector_hidden
        token_hidden = cfg.token_hidden if token_hidden is None else token_hidden
        loc_extent = cfg.loc_extent if loc_extent is None else loc_extent

        sel_in = map_size * map_size + history_dim + K * ref_dim  # 441+256+366
        self.selector = MLP([sel_in, *selector_hidden, 2 * num_glimpses], activation)
        self.token_mlp = MLP([patch_size * patch_size + 2, *token_hidden, token_dim], activation)

        self.map_size = map_size
        self.num_glimpses = num_glimpses
        self.patch_size = patch_size
        self.loc_extent = loc_extent

    def forward(self, M: torch.Tensor, s_hist: torch.Tensor, c_future: torch.Tensor):
        """Args:
            M:        高程图 (B, map_size, map_size)，米
            s_hist:   历史条件状态 token (B, history_dim)
            c_future: 未来参考帧 (B, K, ref_dim)
        Returns:
            z:       地形 token (B, Ng, token_dim)
            r:       glimpse 位置 (B, Ng, 2)，米（机器人系，∈ [−extent, extent]）
            patches: 采样 patch (B, Ng, P, P)，供调试/可视化
        """
        B = M.shape[0]
        x = torch.cat([M.flatten(1), s_hist, c_future.flatten(1)], dim=-1)
        r = torch.tanh(self.selector(x)).view(B, self.num_glimpses, 2) * self.loc_extent
        patches = sample_patches(M, r, self.patch_size)  # (B, Ng, P, P)
        psi_in = torch.cat([patches.flatten(2), r], dim=-1)  # (B, Ng, P²+2)
        z = self.token_mlp(psi_in)
        return z, r, patches
