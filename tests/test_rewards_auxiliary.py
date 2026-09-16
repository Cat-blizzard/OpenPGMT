"""辅助项（Table I 的 Auxiliary 组，10 项）。

## 本文件的重点：两类项的**符号语义**

Table I 的权重符号已经指明分组：
  - 4 个正向项（权重为正）→ 高斯核，**与参考更一致 ⇒ 值更大**（∈ (0,1]）
  - 6 个惩罚项（权重为负）→ **非负代价量**，**越差 ⇒ 值越大**；无违规恰为 0

⚠️ **值是"代价量"而非"惩罚量"**：惩罚项返回 `‖·‖² ≥ 0`，由 Table I 的负权重
把它变成负贡献。本模块第一版让惩罚项返回 `−‖·‖²`，负 × 负 = **正贡献**，
于是 6 项集体从惩罚反转成奖励（动作变化越大奖励越高）。
`test_penalty_terms_reduce_group_reward_as_violation_grows` 就是锁这个的。

因此有两组对照测试：
  - 正向项：与参考一致 ⇒ **恰为 1**；误差增大 ⇒ 单调下降
  - 惩罚项：无违规 ⇒ **恰为 0**；违规增大 ⇒ 单调上升（代价变大）

后者的"无违规恰为 0"尤其重要 —— 例如 `joint_limit` 在界内必须**完全没有**
代价，否则等于把关节往中间推，而论文该项是软约束。
"""

import math

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.rewards.auxiliary import (
    AuxCfg,
    AuxState,
    action_rate_cost,
    aux_cfg,
    check_aux_values_complete,
    compute_aux_values,
    corrected_root_vel_residual,
    ee_accel_mismatch_cost,
    floating_anchor_pos_residual,
    head_torso_impact_cost,
    joint_limit_cost,
    pelvis_vert_accel_cost,
    recovery_upward_vel_residual,
    root_ori_residual,
    undesired_contact_cost,
)


def _quat_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


PENALTY_TERMS = ("pelvis_vert_accel", "ee_accel_mismatch", "action_rate",
                 "joint_limit", "undesired_contact", "head_torso_impact")
POSITIVE_TERMS = ("root_ori", "corrected_root_vel", "floating_anchor_pos",
                  "recovery_upward_vel")


# ---------------------------------------------------------------------------
# 正向项：与参考一致 ⇒ 恰为 1
# ---------------------------------------------------------------------------

def test_root_ori_residual_equals_known_angle():
    a, b = _quat_z(0.2), _quat_z(0.2 + math.radians(20.0))
    assert root_ori_residual(a, b) == pytest.approx(math.radians(20.0), abs=1e-9)


def test_corrected_root_vel_residual_equals_known_difference():
    v = np.array([1.0, 0.0, 0.0])
    assert corrected_root_vel_residual(v, v) == pytest.approx(0.0)
    assert corrected_root_vel_residual(v + np.array([0.0, 0.6, 0.0]), v) \
        == pytest.approx(0.6)


def test_floating_anchor_pos_residual_equals_known_distance():
    p = np.array([0.1, 0.2, 0.8])
    assert floating_anchor_pos_residual(p, p) == pytest.approx(0.0)
    assert floating_anchor_pos_residual(p + np.array([0.3, 0.4, 0.0]), p) \
        == pytest.approx(0.5)


def test_recovery_upward_vel_is_asymmetric():
    """**非对称**：只惩罚不足，不惩罚"起得更快"。

    目标 0.5 m/s：
      - 上行 0.3 → 残差 0.2（不足）
      - 上行 0.5 → 残差 0
      - 上行 1.2 → 残差 0（超出不罚）
    """
    assert recovery_upward_vel_residual(0.3, 0.5) == pytest.approx(0.2)
    assert recovery_upward_vel_residual(0.5, 0.5) == pytest.approx(0.0)
    assert recovery_upward_vel_residual(1.2, 0.5) == pytest.approx(0.0)


def test_recovery_rejects_negative_target():
    with pytest.raises(ValueError, match="非负"):
        recovery_upward_vel_residual(0.1, -0.1)


def test_positive_terms_reach_one_when_matched():
    """4 个正向项在与参考一致时，核取值**恰为 1**。"""
    q = _quat_z(0.4)
    v = np.array([0.5, -0.2, 0.0])
    p = np.array([0.1, 0.3, 0.8])
    vals = compute_aux_values(_state(base_quat=q, ref_quat=q,
                                     base_lin_vel=v, ref_anchor_lin_vel=v,
                                     anchor_pos=p, ref_anchor_pos=p,
                                     upward_vel=aux_cfg()
                                     .recovery_target_upward_vel))
    for term in POSITIVE_TERMS:
        assert vals[term] == pytest.approx(1.0), f"{term} 应为 1，得到 {vals[term]}"


# ---------------------------------------------------------------------------
# 惩罚项：无违规 ⇒ 恰为 0，且值**非负**
# ---------------------------------------------------------------------------

def test_pelvis_vert_accel_cost_is_zero_at_rest():
    assert pelvis_vert_accel_cost(0.0) == pytest.approx(0.0)
    assert pelvis_vert_accel_cost(2.0) == pytest.approx(4.0)
    assert pelvis_vert_accel_cost(-2.0) == pytest.approx(4.0), "平方应对称"


def test_ee_accel_mismatch_cost_zero_when_matching():
    a = np.array([1.0, 2.0, 3.0])
    assert ee_accel_mismatch_cost(a, a) == pytest.approx(0.0)
    # 偏移 (1,0,0) → 1
    assert ee_accel_mismatch_cost(a + np.array([1.0, 0.0, 0.0]), a) \
        == pytest.approx(1.0)


def test_action_rate_cost_zero_when_unchanged():
    a = np.array([0.1, -0.2, 0.3])
    assert action_rate_cost(a, a) == pytest.approx(0.0)
    assert action_rate_cost(a + 0.1, a) == pytest.approx(3 * 0.01)


def test_joint_limit_cost_is_zero_strictly_inside():
    """**界内必须完全没有代价** —— 否则等于把关节往中位推（论文是软约束）。"""
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    for q in (np.array([0.0, 0.0]), np.array([-1.0, 1.0]),   # 恰好贴限位
              np.array([0.5, -0.5])):
        assert joint_limit_cost(q, lo, hi) == pytest.approx(0.0), f"q={q} 不应罚"


def test_joint_limit_cost_counts_only_the_violation():
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    # 关节 0 越界 0.2、关节 1 界内 → 只有 0 计入：0.2² = 0.04
    assert joint_limit_cost(np.array([1.2, 0.0]), lo, hi) == pytest.approx(0.04)
    assert joint_limit_cost(np.array([-1.3, 0.0]), lo, hi) == pytest.approx(0.09)


def test_joint_limit_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower > upper"):
        joint_limit_cost(np.zeros(1), np.array([1.0]), np.array([0.0]))


def test_undesired_contact_cost_zero_when_only_allowed_bodies_touch():
    forces = {"left_ankle_roll_link": 100.0, "right_ankle_roll_link": 80.0}
    assert undesired_contact_cost(forces, ("left_ankle_roll_link",
                                           "right_ankle_roll_link")) \
        == pytest.approx(0.0)


def test_undesired_contact_cost_counts_other_bodies_over_threshold():
    allowed = ("left_ankle_roll_link", "right_ankle_roll_link")
    # torso 接触力 3.0，阈值 1.0 → 超出 2.0 → 4.0
    got = undesired_contact_cost({"torso_link": 3.0}, allowed, threshold=1.0)
    assert got == pytest.approx(4.0)
    # 未超阈值 → 0
    assert undesired_contact_cost({"torso_link": 0.5}, allowed,
                                  threshold=1.0) == pytest.approx(0.0)


def test_head_torso_impact_cost_shape():
    assert head_torso_impact_cost(10.0, 50.0) == pytest.approx(0.0)
    assert head_torso_impact_cost(50.0, 50.0) == pytest.approx(0.0)
    assert head_torso_impact_cost(53.0, 50.0) == pytest.approx(9.0)


def test_head_torso_impact_is_documented_as_unresolved():
    """该项**依据不足**，必须在语义表里标为 UNRESOLVED —— 防止被当成已复现。

    这是本模块唯一一个"论文完全没说怎么度量"的项。实现存在只是为了连通组合，
    不能据此声称复现了论文的该项。
    """
    from pgmt.rewards.semantics import Corr, semantics_for

    s = semantics_for("aux", "head_torso_impact")
    assert s.corr is Corr.UNRESOLVED
    assert "待" in s.note


def test_all_terms_are_non_negative():
    """**不变量**：10 项的值全部 ≥ 0，符号只在 Table I 的权重里。

    这是本模块的核心符号约定。若惩罚项返回负值，它与负权重相乘会变成正贡献。
    """
    st = _state(pelvis_vert_accel=1.0, upward_vel=0.0,
                action=np.array([0.2, 0.0, 0.0]),
                joint_pos=np.array([1.5, 0.0, 0.0]),
                contact_forces={"torso_link": 9.0},
                head_torso_impact_force=80.0)
    vals = compute_aux_values(st)
    for term, v in vals.items():
        assert v >= 0.0, f"{term} 的值应为 ≥ 0，得到 {v}"


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def _state(**kw) -> AuxState:
    """构造一个"全部无违规"的 AuxState，可用关键字覆盖任意字段。"""
    c = aux_cfg()
    nq = 3
    base = dict(
        base_quat=_quat_z(0.0), base_lin_vel=np.zeros(3),
        anchor_pos=np.array([0.0, 0.0, 0.8]),
        ref_quat=_quat_z(0.0), ref_anchor_lin_vel=np.zeros(3),
        ref_anchor_pos=np.array([0.0, 0.0, 0.8]),
        upward_vel=c.recovery_target_upward_vel,
        pelvis_vert_accel=0.0,
        ee_accel=np.zeros(3), ref_ee_accel=np.zeros(3),
        action=np.zeros(nq), prev_action=np.zeros(nq),
        joint_pos=np.zeros(nq),
        joint_lower=-np.ones(nq), joint_upper=np.ones(nq),
        contact_forces={"left_ankle_roll_link": 50.0},
        head_torso_impact_force=0.0,
    )
    base.update(kw)
    return AuxState(**base)


def test_compute_aux_values_keys_match_table_i():
    """返回键必须与 `spec.AUX_TERMS` 完全一致（不漏不多）。"""
    from pgmt.rewards.spec import AUX_TERMS

    vals = compute_aux_values(_state())
    assert set(vals) == {n for n, _ in AUX_TERMS}
    check_aux_values_complete(vals)     # 自检函数本身不应抛错


def test_check_aux_values_complete_detects_mismatch():
    with pytest.raises(KeyError, match="项名不匹配"):
        check_aux_values_complete({"root_ori": 1.0})


def test_check_aux_values_complete_rejects_negative_value():
    """自检函数也要拦负值 —— 与 `RewardGroup.sum` 的守卫互为冗余。"""
    vals = compute_aux_values(_state())
    vals = dict(vals, action_rate=-0.03)
    with pytest.raises(ValueError, match="负值"):
        check_aux_values_complete(vals)


def test_perfect_state_gives_max_positive_and_zero_penalties():
    """"完美状态"下：4 个正向项 = 1、6 个惩罚项 = 0。"""
    vals = compute_aux_values(_state())
    for term in POSITIVE_TERMS:
        assert vals[term] == pytest.approx(1.0), f"{term}={vals[term]}"
    for term in PENALTY_TERMS:
        assert vals[term] == pytest.approx(0.0), f"{term}={vals[term]}"


def test_aux_group_sum_is_maximal_at_perfect_state():
    """交给 `RewardGroup.sum` 后，完美状态应取到该组的理论最大值。

    Table I 的正权重之和 = 0.5+2.0+1.0+12.5 = 16.0；惩罚项为 0。
    """
    from pgmt.rewards.spec import AUX

    vals = compute_aux_values(_state())
    assert AUX.sum(vals) == pytest.approx(0.5 + 2.0 + 1.0 + 12.5)


def test_penalty_terms_reduce_group_reward_as_violation_grows():
    """**回归测试**：违规越大 ⇒ 组奖励越小。

    第一版惩罚项返回 `−代价`，与 Table I 的负权重相乘得到**正**贡献 ——
    违规越大组奖励越大，6 项集体反转。这里逐项验证"代价升、组奖励降"。

    每项单独构造，且只改影响该项的字段，从而把该项隔离出来。
    """
    from pgmt.rewards.spec import AUX

    c = aux_cfg()
    base = compute_aux_values(_state())
    base_sum = AUX.sum(base)

    cases = {
        "pelvis_vert_accel": dict(pelvis_vert_accel=3.0),
        "ee_accel_mismatch": dict(ee_accel=np.array([2.0, 0.0, 0.0])),
        "action_rate": dict(action=np.array([0.4, 0.0, 0.0])),
        "joint_limit": dict(joint_pos=np.array([2.0, 0.0, 0.0])),
        "undesired_contact": dict(contact_forces={
            "torso_link": c.contact_force_threshold + 4.0}),
        "head_torso_impact": dict(head_torso_impact_force=(
            c.head_torso_impact_threshold + 100.0)),
    }
    assert set(cases) == set(PENALTY_TERMS), "每个惩罚项都要有对照用例"

    for term, kw in cases.items():
        worse = compute_aux_values(_state(**kw))
        assert worse[term] > base[term], \
            f"{term} 是代价量，违规增大时应变大（{base[term]} → {worse[term]}）"
        assert AUX.sum(worse) < base_sum, \
            f"{term} 违规后组奖励应下降（第一版在此反向：惩罚变奖励）"


def test_action_rate_cost_lowers_group_sum_through_that_term_alone():
    """动作变化率代价必须**只**通过 `action_rate` 项降低组奖励。

    第一版我写的是"与完美状态相比组奖励更低"，失败后才想清楚：断言"总和一定
    更低"本身没问题，**问题在值与权重的符号都反了**。现在值非负、权重为负，
    贡献恒 ≤ 0，这项断言才有意义。同时仍要验证其它项不受 `action` 影响。
    """
    from pgmt.rewards.spec import AUX

    v0 = compute_aux_values(_state())
    v1 = compute_aux_values(_state(action=np.array([0.1, 0.1, 0.1]),
                                   prev_action=np.zeros(3)))

    # 其它项不受 action 影响
    for term in ("root_ori", "corrected_root_vel", "floating_anchor_pos",
                 "recovery_upward_vel", "pelvis_vert_accel",
                 "ee_accel_mismatch", "joint_limit", "undesired_contact",
                 "head_torso_impact"):
        assert v1[term] == pytest.approx(v0[term]), f"{term} 不应受 action 影响"

    assert v0["action_rate"] == pytest.approx(0.0)
    assert v1["action_rate"] == pytest.approx(0.03)   # 3 × 0.1²
    # 组和差值恰为 权重(−0.05) × 代价(+0.03) = −0.0015
    assert AUX.sum(v1) - AUX.sum(v0) == pytest.approx(-0.05 * 0.03)
    assert AUX.sum(v1) < AUX.sum(v0), "该项对组和的贡献应为负"


def test_aux_state_validates_shapes():
    with pytest.raises(ValueError, match="base_quat"):
        _state(base_quat=np.zeros(3))
    with pytest.raises(ValueError, match="base_lin_vel"):
        _state(base_lin_vel=np.zeros(2))
    with pytest.raises(ValueError, match="joint_lower"):
        _state(joint_lower=np.zeros(2))
    # 错误信息用"须一致"（不是"形状不一致"）—— 断言须与实现的实际措辞一致
    with pytest.raises(ValueError, match="须一致"):
        _state(ref_ee_accel=np.zeros(6))


# ---------------------------------------------------------------------------
# A21
# ---------------------------------------------------------------------------

def test_a21_is_registered_with_required_fields():
    cfg = get("A21").value
    assert isinstance(cfg, AuxCfg)
    for name in ("sigma_root_ori", "sigma_corrected_root_vel",
                 "sigma_floating_anchor", "sigma_recovery_upward",
                 "recovery_target_upward_vel", "contact_force_threshold",
                 "head_torso_impact_threshold"):
        assert getattr(cfg, name) > 0.0, f"{name} 应为正"
    assert cfg.allowed_contact_bodies, "必须至少允许某个部位接触"


def test_sigmas_are_in_squared_error_units():
    """σ 的量纲是误差**平方** —— 用半衰误差 √σ 反推其物理含义。"""
    c = aux_cfg()
    # 根朝向：√σ 应是可观的角度（几度到几十度），不是 0.01 rad 或 3 rad
    half_ang = math.degrees(math.sqrt(c.sigma_root_ori))
    assert 5.0 <= half_ang <= 60.0, f"根朝向半衰角 {half_ang:.1f}° 不合理"
    # 锚点位置：√σ 应在厘米~分米级
    half_pos = math.sqrt(c.sigma_floating_anchor)
    assert 0.05 <= half_pos <= 1.0, f"锚点半衰距离 {half_pos:.3f}m 不合理"
    # 速度：√σ 在合理速度量级
    half_vel = math.sqrt(c.sigma_corrected_root_vel)
    assert 0.1 <= half_vel <= 5.0, f"速度半衰 {half_vel:.3f}m/s 不合理"


def test_default_allowed_contact_bodies_are_feet():
    """默认允许接触的应是双足（人形机器人站立/行走的正常接触点）。"""
    for b in aux_cfg().allowed_contact_bodies:
        assert "ankle" in b or "foot" in b, f"{b} 看起来不是足部"
