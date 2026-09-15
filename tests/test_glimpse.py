"""glimpse_encoder.py：双线性采样数值正确性、位置界、梯度、形状。"""

import math

import pytest
import torch

from pgmt.policy.glimpse_encoder import TerrainGlimpseEncoder, sample_patches


def _pixel(c_m, map_size=21):
    """米（机器人系）→ 像素坐标（align_corners: −1m↔0, +1m↔S−1）。"""
    return (c_m + 1.0) / 2.0 * (map_size - 1)


def _manual_bilinear(M, sx, sy):
    """grid_sample(align_corners=True, zeros padding) 的手工参考实现。"""
    H, W = M.shape
    if sx < 0.0 or sx > W - 1 or sy < 0.0 or sy > H - 1:
        return 0.0  # 越界补 0
    x0, y0 = int(math.floor(sx)), int(math.floor(sy))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    wx, wy = sx - x0, sy - y0
    top = (1 - wx) * M[y0, x0] + wx * M[y0, x1]
    bot = (1 - wx) * M[y1, x0] + wx * M[y1, x1]
    return (1 - wy) * top + wy * bot


def test_sample_patches_matches_manual_bilinear():
    torch.manual_seed(0)
    B, Ng, P, S = 2, 3, 5, 21
    M = torch.rand(B, S, S) * 0.2 - 0.1
    centers = (torch.rand(B, Ng, 2) * 1.2 - 0.6)  # |c|<=0.6 → patch 全部落图内
    patches = sample_patches(M, centers, P)
    assert patches.shape == (B, Ng, P, P)
    for b in range(B):
        for j in range(Ng):
            cx, cy = _pixel(centers[b, j, 0].item()), _pixel(centers[b, j, 1].item())
            for i in range(P):
                for k in range(P):
                    sx, sy = cx + (k - 2), cy + (i - 2)
                    expect = _manual_bilinear(M[b], sx, sy)
                    assert patches[b, j, i, k].item() == pytest.approx(expect, abs=1e-5)


def test_sample_patches_out_of_bounds_is_zero():
    M = torch.rand(1, 21, 21)
    centers = torch.tensor([[[1.5, 0.0]]])  # 中心在 1.5m（图外）
    patches = sample_patches(M, centers, 5)
    assert patches.shape == (1, 1, 5, 5)
    assert torch.allclose(patches[0, 0], torch.zeros(5, 5), atol=1e-6)


def test_patch_centered_at_spike():
    """地图中心放一个 1.0 的尖峰，采样中心对准它时 patch 中心应为 1.0。"""
    M = torch.zeros(1, 21, 21)
    M[0, 10, 10] = 1.0
    centers = torch.zeros(1, 1, 2)  # 像素 (10,10) ↔ 0m
    patches = sample_patches(M, centers, 5)
    assert patches[0, 0, 2, 2].item() == pytest.approx(1.0, abs=1e-6)


def test_selector_locations_within_map():
    enc = TerrainGlimpseEncoder()
    B = 2
    M = torch.rand(B, 21, 21) * 0.1
    s_hist = torch.randn(B, 256)
    c_future = torch.randn(B, 6, 61)
    z, r, patches = enc(M, s_hist, c_future)
    assert z.shape == (B, 4, 256)
    assert r.shape == (B, 4, 2)
    assert patches.shape == (B, 4, 5, 5)
    assert (r.abs() <= enc.loc_extent + 1e-6).all(), "tanh 缩放后位置应在地图半宽内"


def test_gradient_flows_to_elevation_map():
    """双线性采样使整条通路对 M 可微。"""
    enc = TerrainGlimpseEncoder()
    M = torch.rand(1, 21, 21, requires_grad=True)  # 保持叶节点以取 .grad
    s_hist = torch.randn(1, 256)
    c_future = torch.randn(1, 6, 61)
    z, _, _ = enc(M, s_hist, c_future)
    z.sum().backward()
    assert M.grad is not None and M.grad.abs().sum().item() > 0


def test_glimpses_respond_to_map_change():
    enc = TerrainGlimpseEncoder()
    s_hist = torch.randn(1, 256)
    c_future = torch.randn(1, 6, 61)
    M1, M2 = torch.rand(1, 21, 21), torch.rand(1, 21, 21)
    z1, _, _ = enc(M1, s_hist, c_future)
    z2, _, _ = enc(M2, s_hist, c_future)
    assert not torch.allclose(z1, z2), "不同高程图应产生不同地形 token"
