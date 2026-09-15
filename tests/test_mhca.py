"""mhca.py：形状、RoPE 位置敏感性、注意力权重返回。"""

import torch

from pgmt.policy.mhca import MHCA
from pgmt.policy.rope import RotaryPositionEmbedding


def _identity_proj(m: MHCA):
    """把 Q/K/V/输出投影置为单位阵，便于做精确的注意力数值断言。"""
    with torch.no_grad():
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.out_proj):
            proj.weight.copy_(torch.eye(proj.weight.shape[0]))
            proj.bias.zero_()


def test_shapes_and_attn_weights():
    m = MHCA(d_model=8, heads=2)
    q = torch.randn(3, 1, 8)
    k = torch.randn(3, 5, 8)
    v = torch.randn(3, 5, 8)
    out, attn = m(q, k, v, return_attn=True)
    assert out.shape == (3, 1, 8)
    assert attn.shape == (3, 2, 1, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 2, 1), atol=1e-6)


def test_rope_makes_nearer_position_score_higher():
    """q 与两个内容相同的 key 对齐时，RoPE 使更近位置（τ=0）得分更高。

    内容取 one-hot e0：q·R(τ)e0 = cos(τ·1.0)（首对旋转角 = τ × base^0），
    τ=1 时得分为 cos(1) < 1。
    """
    rope = RotaryPositionEmbedding(dim=4, base=10000.0)  # 与 head_dim 一致
    m = MHCA(d_model=4, heads=1, rope=rope)
    _identity_proj(m)

    q = torch.zeros(1, 1, 4)
    q[..., 0] = 1.0
    k = torch.zeros(1, 2, 4)
    k[..., 0] = 1.0  # 两个 key 内容完全相同
    v = torch.randn(1, 2, 4)

    _, attn = m(q, k, v, q_idx=torch.tensor([0.0]),
                k_idx=torch.tensor([0.0, 1.0]), return_attn=True)
    w0, w1 = attn[0, 0, 0, 0].item(), attn[0, 0, 0, 1].item()
    assert w0 > w1, f"τ=0 应比 τ=1 得分高: {w0} vs {w1}"


def test_no_rope_identical_keys_get_equal_weights():
    m = MHCA(d_model=4, heads=1)
    _identity_proj(m)
    q = torch.zeros(1, 1, 4)
    q[..., 0] = 1.0
    k = torch.zeros(1, 2, 4)
    k[..., 0] = 1.0
    v = torch.randn(1, 2, 4)
    _, attn = m(q, k, v, return_attn=True)
    w0, w1 = attn[0, 0, 0, 0].item(), attn[0, 0, 0, 1].item()
    assert abs(w0 - w1) < 1e-6, "无 RoPE 时相同 key 权重应相等"


def test_rope_requires_position_indices():
    m = MHCA(d_model=4, heads=1, rope=RotaryPositionEmbedding(4))
    q = torch.randn(1, 1, 4)
    k = torch.randn(1, 2, 4)
    try:
        m(q, k, k)
    except ValueError:
        return
    raise AssertionError("启用 RoPE 未给位置下标时应报错")


def test_deterministic():
    torch.manual_seed(0)
    m = MHCA(d_model=8, heads=2)
    q, k = torch.randn(2, 1, 8), torch.randn(2, 3, 8)
    torch.manual_seed(0)  # 重置种子 → 第二个模型权重与第一个一致
    m2 = MHCA(d_model=8, heads=2)
    out1, out2 = m(q, k, k), m2(q, k, k)
    assert torch.allclose(out1, out2), "同种子初始化 + 同输入应完全确定"
