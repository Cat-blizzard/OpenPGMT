"""actor.py / multi_head_critic.py：形状、头顺序、Stage1→2 权重重排。"""

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


def test_stage1_to_stage2_head_remap():
    torch.manual_seed(0)
    w = torch.randn(128, 3)
    b = torch.randn(3)
    w2, b2 = stage1_to_stage2_head(w, b)
    assert w2.shape == (128, 4) and b2.shape == (4,)
    # upper/lower 直接复制；aux 移到第 4 列；terrain（第 3 列）随机初始化
    assert torch.equal(w2[:, 0], w[:, 0]) and torch.equal(w2[:, 1], w[:, 1])
    assert torch.equal(w2[:, 3], w[:, 2])
    assert not torch.equal(w2[:, 2], w[:, 0])  # 新 terrain 头非复制
    assert torch.equal(b2[0], b[0]) and torch.equal(b2[1], b[1]) and torch.equal(b2[3], b[2])
    assert b2[2].item() == 0.0  # 新头零初始化


def test_stage1_to_stage2_head_bad_shape_raises():
    try:
        stage1_to_stage2_head(torch.randn(128, 4), torch.randn(3))
    except ValueError:
        return
    raise AssertionError("形状不符应报错")
