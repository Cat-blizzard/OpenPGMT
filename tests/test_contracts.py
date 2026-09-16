"""维度契约闭环：策略模块的默认输入/输出维度必须来自 `pgmt.contracts`。

`pgmt/policy/history_encoder.py` 与 `pgmt/policy/actor.py` 原先各自硬编码
`_OBS_DIM = 96` / `_ACT_DIM = 29`，现已改为引用 `pgmt.contracts`。本文件用
**默认构造 + 按契约形状喂数据**的方式锁住这条闭环：若哪个模块又退回硬编码
或改错维度，这里会立刻失败（而不是等到 M2 训练时才报形状不匹配）。
"""

import numpy as np
import pytest
import torch

from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.policy.actor import Actor
from pgmt.policy.history_encoder import HistoryEncoder
from pgmt.policy.ifm import IFM
from pgmt.policy.multi_head_critic import (
    HEAD_ORDER_AGGREGATE,
    HEAD_ORDER_STAGE1,
    HEAD_ORDER_STAGE2,
    MultiHeadCritic,
)


def test_contract_values():
    assert OBS_DIM == 96
    assert ACT_DIM == 29
    assert HISTORY_LEN == 10
    assert REF_FRAME_DIM == 61


def test_history_encoder_defaults_to_shared_obs_dim():
    """默认构造的 HistoryEncoder 必须接受 (B, OBS_DIM) 并接受 (B, 10, OBS_DIM)。"""
    enc = HistoryEncoder()
    B = 3
    o = torch.zeros(B, OBS_DIM)
    H = torch.zeros(B, HISTORY_LEN, OBS_DIM)
    out = enc(o, H)
    assert out.shape[0] == B
    assert out.shape[1] == 256  # A3.token_dim


def test_history_encoder_rejects_wrong_obs_dim():
    """契约是硬的：维度不符必须报错，而不是静默广播。"""
    enc = HistoryEncoder()
    with pytest.raises(Exception):
        enc(torch.zeros(2, OBS_DIM + 1), torch.zeros(2, HISTORY_LEN, OBS_DIM))


def test_actor_defaults_to_shared_obs_and_act_dim():
    actor = Actor()
    B = 4
    a = actor(torch.zeros(B, OBS_DIM), torch.zeros(B, 256))
    assert a.shape == (B, ACT_DIM), f"动作维度应为 {ACT_DIM}，得到 {tuple(a.shape)}"


def test_ifm_defaults_to_shared_ref_dim_and_history_len():
    ifm = IFM()
    B = 2
    s_hist = torch.zeros(B, 256)
    C = torch.zeros(B, 6, REF_FRAME_DIM)  # A2.K=6, Eq.1 的 61 维
    s_int = ifm(s_hist, C)
    assert s_int.shape == (B, 256)


def test_ifm_stage2_accepts_glimpse_tokens():
    """Stage 2 注入 4 个地形 token（A15）后形状不变。"""
    from pgmt.cfg.assumptions import get
    ng = get("A15").value.num_glimpses
    ifm = IFM()
    B = 2
    s_int = ifm(torch.zeros(B, 256), torch.zeros(B, 6, REF_FRAME_DIM),
                torch.zeros(B, ng, 256))
    assert s_int.shape == (B, 256)


def test_critic_head_orders_match_eq_6_and_eq_9():
    assert HEAD_ORDER_STAGE1 == ("upper", "lower", "aux")            # 论文 Eq.6
    assert HEAD_ORDER_STAGE2 == ("upper", "lower", "terrain", "aux")  # 论文 Eq.9
    assert HEAD_ORDER_AGGREGATE == ("total",)                        # Fig.4 消融


def test_critic_heads_have_declared_widths():
    from pgmt.envs.observations import PRIV_DIM

    crit1 = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE1)
    v = crit1(torch.zeros(2, PRIV_DIM), torch.zeros(2, 256))
    assert v.shape == (2, 3)

    crit2 = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE2)
    v2 = crit2(torch.zeros(2, PRIV_DIM), torch.zeros(2, 256))
    assert v2.shape == (2, 4)


def test_priv_dim_from_layout_is_usable_by_critic():
    """A9 布局给出的维度必须能直接喂给 critic（闭环：契约→网络）。"""
    from pgmt.envs.observations import PRIV_DIM
    crit = MultiHeadCritic(priv_dim=PRIV_DIM)
    out = crit(torch.zeros(1, PRIV_DIM), torch.zeros(1, 256))
    assert out.shape == (1, len(HEAD_ORDER_STAGE1))


def test_stage1_to_stage2_head_remap_preserves_values():
    """Stage1→2 末层重排：upper/lower 的权重与偏置必须原样搬过去。"""
    from pgmt.policy.multi_head_critic import stage1_to_stage2_head

    torch.manual_seed(0)
    critic = MultiHeadCritic(priv_dim=48, mlp_dims=(8,))
    w = critic.head.weight
    b = critic.head.bias
    w2, b2 = stage1_to_stage2_head(w, b)
    assert w2.shape == (4, 8) and b2.shape == (4,)
    assert torch.equal(w2[0], w[0])   # upper
    assert torch.equal(w2[1], w[1])   # lower
    assert torch.equal(w2[3], w[2])   # aux → 第 4 行（Eq.9 顺序）
    assert torch.equal(b2[0], b[0]) and torch.equal(b2[3], b[2])


def test_obs_layout_total_matches_policy_expectation():
    """布局的 96 维必须就是策略接受的 96 维（两处常量的最终一致性检查）。"""
    from pgmt.envs.observations import OBS_LAYOUT
    assert OBS_LAYOUT.total_dim == OBS_DIM

    # 用布局真实拼一帧，喂给默认 HistoryEncoder —— 端到端形状闭环
    from pgmt.envs.observations import build_obs
    o = build_obs(
        e_t=np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        omega_t=np.zeros(3, dtype=np.float32),
        q_t=np.zeros(ACT_DIM, dtype=np.float32),
        qd_t=np.zeros(ACT_DIM, dtype=np.float32),
        a_prev=np.zeros(ACT_DIM, dtype=np.float32),
    )
    assert o.shape == (OBS_DIM,)
    enc = HistoryEncoder()
    H = torch.from_numpy(np.tile(o, (HISTORY_LEN, 1)))
    out = enc(torch.from_numpy(o).unsqueeze(0), H.unsqueeze(0))
    assert out.shape == (1, 256)
