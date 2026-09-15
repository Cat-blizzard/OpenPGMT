"""RoPE 数值性质校验（纯 torch，CPU 可跑）。

验证三条核心性质：
  1. 旋转保范数（旋转是正交变换）
  2. 位置 0 为恒等映射
  3. 相对位移不变性：<R(q,m), R(k,n)> = <R(q,0), R(k,n−m)>
     —— MHCA 注意力因此只依赖帧间相对距离
外加 2D 解析解与输入校验。
"""

import math

import pytest
import torch

from pgmt.policy.rope import RotaryPositionEmbedding


@pytest.fixture
def rng():
    g = torch.Generator().manual_seed(0)
    return g


@pytest.mark.parametrize("dim", [16, 64])
def test_norm_preserved(rng, dim):
    rope = RotaryPositionEmbedding(dim)
    x = torch.randn(3, 8, dim, generator=rng)
    idx = torch.tensor([0, 1, 3, 7, 15, 31, 100, 1000])
    out = rope(x, idx)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_identity_at_position_zero(rng):
    rope = RotaryPositionEmbedding(32)
    x = torch.randn(5, 32, generator=rng)
    out = rope(x, torch.zeros(5))
    assert torch.allclose(out, x, atol=1e-6)


@pytest.mark.parametrize("dim", [16, 64])
def test_relative_shift_invariance(rng, dim):
    rope = RotaryPositionEmbedding(dim)
    q = torch.randn(4, dim, generator=rng)
    k = torch.randn(7, dim, generator=rng)
    m = torch.tensor([2, 5, 9, 40])
    n = torch.tensor([0, 3, 8, 20, 33, 60, 100])

    lhs = rope(q, m) @ rope(k, n).T  # (4, 7)
    rhs = torch.empty_like(lhs)
    for i in range(4):
        # rhs[i,j] = <R(q_i,0), R(k_j, n_j − m_i)>
        rhs[i] = rope(q[i:i + 1], torch.zeros(1)) @ rope(k, n - m[i]).T
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_2d_analytic(rng):
    # dim=2, base=10000 → inv_freq=[1]，角度 = 位置值
    rope = RotaryPositionEmbedding(2, base=10000.0)
    x = torch.tensor([[1.0, 2.0], [-0.5, 3.0]])
    m = torch.tensor([1.0, 2.0])
    out = rope(x, m)
    for i in range(2):
        a, b = x[i].tolist()
        c, s = math.cos(m[i].item()), math.sin(m[i].item())
        assert out[i, 0] == pytest.approx(a * c - b * s, abs=1e-5)
        assert out[i, 1] == pytest.approx(a * s + b * c, abs=1e-5)


def test_odd_dim_raises():
    with pytest.raises(ValueError):
        RotaryPositionEmbedding(7)


def test_feature_dim_mismatch_raises():
    rope = RotaryPositionEmbedding(8)
    with pytest.raises(ValueError):
        rope(torch.randn(3, 10), torch.zeros(3))
