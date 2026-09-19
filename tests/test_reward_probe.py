"""奖励尺度探针的数学与 σ 表的可用性（服务器可跑，无需数据也可）。

探针 `data/probe_reward_scales.py` 是 A18 里 σ 取值的**实测依据**。它的
`responsive_band` 若算错，我就会用一条错误的曲线去论证 σ 合理 —— 所以这里
把它的数学与 `exp_tracking_reward` 交叉验证，而不是各算各的。

另外校验 σ 表本身的内部一致性：上下半身的相对严格程度必须符合论文 §IV
的设计意图（上半身保姿态保真 → 容差更小）。
"""

import math

import pytest

from pgmt.rewards.spec import (
    SIGMAS,
    exp_tracking_reward,
    sigma_for,
)


def test_responsive_band_is_consistent_with_the_kernel():
    """`responsive_band` 给出的边界必须真的对应 reward = 0.9 / 0.1。

    这是两套独立实现（解析解 vs 核函数）的交叉验证：若探针的公式写错，
    我会用错误的曲线去论证 σ，而这里会失败。
    """
    from data.probe_reward_scales import responsive_band

    for sigma in (0.06, 0.5, 1.0, 20.0):
        e_lo, e_hi = responsive_band(sigma)
        assert e_lo < e_hi, "0.9 边界应比 0.1 边界更靠近 0"
        assert exp_tracking_reward(e_lo, sigma) == pytest.approx(0.9, rel=1e-12)
        assert exp_tracking_reward(e_hi, sigma) == pytest.approx(0.1, rel=1e-12)
        # 区间内奖励应落在 (0.1, 0.9)
        mid = 0.5 * (e_lo + e_hi)
        assert 0.1 < exp_tracking_reward(mid, sigma) < 0.9


def test_responsive_band_scales_with_sqrt_sigma():
    """响应区宽度 ∝ √σ（σ 是平方量纲的直接推论）。"""
    from data.probe_reward_scales import responsive_band

    w1 = math.log(responsive_band(1.0)[1] / responsive_band(1.0)[0])
    w4 = math.log(responsive_band(4.0)[1] / responsive_band(4.0)[0])
    assert w4 == pytest.approx(w1, rel=1e-12), "响应区宽度应与 σ 无关（只决定位置）"
    # 但位置按 √σ 平移
    assert responsive_band(4.0)[1] == pytest.approx(2.0 * responsive_band(1.0)[1], rel=1e-12)


def test_responsive_band_rejects_bad_sigma():
    from data.probe_reward_scales import responsive_band

    with pytest.raises(ValueError):
        responsive_band(0.0)
    with pytest.raises(ValueError):
        responsive_band(-1.0)


def test_report_sigma_half_decay_matches_kernel():
    from data.probe_reward_scales import report_sigma

    r = report_sigma("link_pos", "m", 0.06)
    assert r["half_decay_error"] == pytest.approx(math.sqrt(0.06))
    assert exp_tracking_reward(r["half_decay_error"], 0.06) == pytest.approx(1.0 / math.e)


# ---------------------------------------------------------------------------
# σ 表的设计意图一致性（论文 §IV）
# ---------------------------------------------------------------------------

def test_upper_body_stricter_than_lower_body():
    """论文 §IV：上半身保姿态保真，下半身让位给平衡与接触 —— 容差应更小。"""
    assert math.sqrt(sigma_for("link_pos")) < math.sqrt(sigma_for("ta_link_pos"))


def test_all_sigmas_positive_and_finite():
    for term, s in SIGMAS.items():
        assert s > 0.0, f"{term} 的 σ 应为正"
        assert math.isfinite(s), f"{term} 的 σ 应有限"


def test_sigma_for_is_strict():
    """未登记的项必须报错，不能静默返回默认 σ 把项调坏。"""
    with pytest.raises(KeyError, match="未登记"):
        sigma_for("definitely_not_a_term")


def test_position_sigmas_imply_sane_half_decay():
    """位置类 σ 的半衰误差必须落在"物理上合理"的区间。

    参考：G1 踝高约 0.79m、脚长 ~0.2m。若半衰误差到米级，等于允许
    整条腿错位还拿高分；若到毫米级，等于要求完美跟踪（项会全程为 0）。
    """
    for term in ("link_pos", "ta_link_pos"):
        hd = math.sqrt(sigma_for(term))
        assert 0.02 <= hd <= 2.0, f"{term} 的半衰误差 {hd:.3f}m 超出合理区间"


def test_joint_sigma_half_decay_is_sub_radian_scale():
    """关节位置项：半衰误差在弧度上应是"明显但可见"的量级。"""
    for term in ("joint_pos", "ta_joint_pos"):
        hd = math.sqrt(sigma_for(term))
        assert 0.05 <= hd <= 1.5, f"{term} 的半衰误差 {hd:.3f}rad 超出合理区间"


def test_velocity_sigmas_larger_than_position_ones():
    """速度量的数值范围远大于位置量，σ 也应相应更大（单位一致性）。"""
    assert sigma_for("joint_vel") > sigma_for("joint_pos")
    assert sigma_for("link_lin_vel") > sigma_for("link_pos")
