"""端到端集成：Stage 1 / Stage 2 完整策略组装（论文 Fig.2 架构）。

把 History Encoder、IFM、Glimpse Encoder、Actor、Multi-Head Critic
按论文接线拼成完整前向，验证维度与梯度通路。环境层（M2）接入时
观测张量由此处的 mock 形状定义。
"""

import torch

from pgmt.policy.actor import Actor
from pgmt.policy.glimpse_encoder import TerrainGlimpseEncoder
from pgmt.policy.history_encoder import HistoryEncoder
from pgmt.policy.ifm import IFM
from pgmt.policy.multi_head_critic import (
    HEAD_ORDER_STAGE1,
    HEAD_ORDER_STAGE2,
    MultiHeadCritic,
)

PRIV_DIM = 20  # A9 特权观测集展开维度（环境层最终确认）


def _stage1_inputs(B=8):
    return {
        "o": torch.randn(B, 96),          # o_t
        "H": torch.randn(B, 10, 96),      # 10 帧历史
        "C": torch.randn(B, 6, 61),       # C^K 未来参考帧
        "priv": torch.randn(B, PRIV_DIM),
    }


def test_stage1_full_forward_and_backward():
    torch.manual_seed(0)
    hist = HistoryEncoder()
    ifm = IFM()
    actor = Actor()
    critic = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE1)
    x = _stage1_inputs()

    s_hist = hist(x["o"], x["H"])            # (B, 256)
    s_int = ifm(s_hist, x["C"])              # (B, 256)
    a = actor(x["o"], s_int)                 # (B, 29)
    v = critic(x["priv"], s_int)             # (B, 3)

    assert a.shape == (8, 29) and v.shape == (8, 3)
    (a.mean() + v.sum()).backward()
    params = list(hist.parameters()) + list(ifm.parameters()) \
        + list(actor.parameters()) + list(critic.parameters())
    assert all(p.grad is not None for p in params)
    assert all(p.grad.abs().sum().item() > 0 for p in params), "每个模块都应收到梯度"


def test_stage2_full_forward_and_backward():
    torch.manual_seed(0)
    hist = HistoryEncoder()
    ifm = IFM()
    glimpse = TerrainGlimpseEncoder()
    actor = Actor()
    critic = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE2)
    x = _stage1_inputs()
    M = torch.randn(8, 21, 21) * 0.05        # 高程图 M_t（Stage 2 新增）

    s_hist = hist(x["o"], x["H"])
    z, r, patches = glimpse(M, s_hist, x["C"])   # (B,4,256) / (B,4,2) / (B,4,5,5)
    s_int = ifm(s_hist, x["C"], z)               # 注入地形 token
    a = actor(x["o"], s_int)
    v = critic(x["priv"], s_int)                 # (B, 4)：upper/lower/terrain/aux

    assert z.shape == (8, 4, 256) and r.shape == (8, 4, 2)
    assert a.shape == (8, 29) and v.shape == (8, 4)
    (a.mean() + v.sum() + z.sum()).backward()
    for name, module in [("hist", hist), ("ifm", ifm), ("glimpse", glimpse),
                         ("actor", actor), ("critic", critic)]:
        for p in module.parameters():
            assert p.grad is not None, f"{name} 参数应收到梯度"


def test_param_count_sane():
    """参数量应远小于 10M（论文 VRAM 29GB 主要来自 15k 并行环境）。"""
    hist = HistoryEncoder()
    ifm = IFM()
    glimpse = TerrainGlimpseEncoder()
    actor = Actor()
    critic = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE2)
    n = sum(p.numel() for m in (hist, ifm, glimpse, actor, critic) for p in m.parameters())
    assert n < 10_000_000, f"参数量异常: {n}"
