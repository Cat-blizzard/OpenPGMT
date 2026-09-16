"""actor.py / multi_head_critic.py：形状、头顺序、Stage1→2 权重重排。"""

import pytest
import torch

from pgmt.policy.actor import Actor
from pgmt.policy.multi_head_critic import (
    HEAD_ORDER_AGGREGATE,
    HEAD_ORDER_STAGE1,
    HEAD_ORDER_STAGE2,
    MultiHeadCritic,
    stage1_to_stage2_head,
)


def test_actor_shape():
    actor = Actor()
    a = actor(torch.randn(4, 96), torch.randn(4, 256))
    assert a.shape == (4, 29)
    assert torch.isfinite(a).all()


def test_actor_input_sensitivity():
    actor = Actor()
    s_int = torch.randn(1, 256)
    a1 = actor(torch.randn(1, 96), s_int)
    a2 = actor(torch.randn(1, 96), s_int)
    assert not torch.allclose(a1, a2)


def test_critic_stage1_three_heads():
    critic = MultiHeadCritic(priv_dim=20)
    assert critic.head_names == HEAD_ORDER_STAGE1 == ("upper", "lower", "aux")
    v = critic(torch.randn(4, 20), torch.randn(4, 256))
    assert v.shape == (4, 3)
    d = critic.values(torch.randn(4, 20), torch.randn(4, 256))
    assert set(d) == {"upper", "lower", "aux"}
    assert all(d[k].shape == (4,) for k in d)


def test_critic_stage2_four_heads_eq9_order():
    critic = MultiHeadCritic(priv_dim=20, head_names=HEAD_ORDER_STAGE2)
    assert critic.head_names == ("upper", "lower", "terrain", "aux")  # Eq.9：terrain 在 aux 前
    v = critic(torch.randn(4, 20), torch.randn(4, 256))
    assert v.shape == (4, 4)


def test_critic_aggregate_single_head():
    critic = MultiHeadCritic(priv_dim=20, head_names=HEAD_ORDER_AGGREGATE)
    assert critic.head_names == ("total",)
    assert critic(torch.randn(4, 20), torch.randn(4, 256)).shape == (4, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_stage1_to_stage2_head_remap_real_state_dict(dtype, device):
    """迁移真实 checkpoint 后，原来三个值头的输出必须保持一致。"""
    torch.manual_seed(0)
    old = MultiHeadCritic(priv_dim=48).to(device=device, dtype=dtype)
    new = MultiHeadCritic(priv_dim=48, head_names=HEAD_ORDER_STAGE2).to(device=device, dtype=dtype)
    state = old.state_dict()
    w, b = state["head.weight"], state["head.bias"]
    w_before, b_before = w.clone(), b.clone()
    w2, b2 = stage1_to_stage2_head(w, b)
    assert w2.shape == new.head.weight.shape == (4, 128)
    assert b2.shape == (4,)
    assert w2.dtype == dtype and b2.dtype == dtype
    assert w2.device == w.device and b2.device == b.device
    assert torch.equal(w2[[0, 1, 3]], w) and torch.equal(b2[[0, 1, 3]], b)
    assert torch.equal(w, w_before) and torch.equal(b, b_before)
    # 新 terrain 行遵循 Linear 默认初始化范围。
    bound = 128 ** -0.5
    assert torch.isfinite(w2[2]).all() and (w2[2].abs() <= bound).all()
    assert torch.isfinite(b2[2]) and b2[2].abs() <= bound
    state["head.weight"], state["head.bias"] = w2, b2
    new.load_state_dict(state, strict=True)
    priv = torch.randn(5, 48, dtype=dtype, device=device)
    intent = torch.randn(5, 256, dtype=dtype, device=device)
    assert torch.allclose(new(priv, intent)[:, [0, 1, 3]], old(priv, intent), atol=1e-7)


@pytest.mark.parametrize("shape", [(128, 3), (4, 128), (3, 0)])
def test_stage1_to_stage2_head_bad_shape_raises(shape):
    with pytest.raises(ValueError):
        stage1_to_stage2_head(torch.randn(*shape), torch.randn(3))
