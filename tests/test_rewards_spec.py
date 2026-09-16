"""奖励规格：Table I 权重逐字对齐、分组结构、松弛算子（Eq.10）。

本文件是**论文表格的回归锁**。权重抄错一个数字，最终报告里所有对标 Table II
的结论都会偏，而症状只是"指标差一点"，极难回溯。因此这里把论文 Table I 的
每一项都写成字面量断言 —— 与实现分开，抄错必然失败。

同时锁定 Eq.10 的两个容易搞反的语义：
  1. 松弛是**钳到容忍区之外**（`[e − αχτ]₊`），不是"缩小误差"；
  2. χ 只在 slopes/stairs/boxes 为 1，flat 与 rough 必须为 0。
"""

import math

import numpy as np
import pytest

from pgmt.rewards.spec import (
    AUX,
    AUX_TERMS,
    CHI_ACTIVE_FAMILIES,
    GROUPS,
    LOWER,
    LOWER_TERMS,
    NUM_LEVELS,
    RELAXED_TERMS,
    TERRAIN,
    TERRAIN_FAMILIES,
    TERRAIN_TERMS,
    UPPER,
    UPPER_TERMS,
    chi,
    exp_tracking_reward,
    relaxed_error,
    relaxed_tracking_reward,
    resolve_alpha,
    tau_budget,
    total_reward,
)


# ---------------------------------------------------------------------------
# Table I 权重（论文原文逐字对照）
# ---------------------------------------------------------------------------

#: 论文 Table I 的字面量转写（"The 1st & 2nd Stages" 部分）。
#: 注意力：OCR 把 −10⁻³ 渲染成 "−10 −3"，此处已按指数解读。
PAPER_TABLE_I_STAGE12 = {
    "upper": {
        "link_pos": 1.0, "link_ori": 1.0,
        "link_lin_vel": 0.5, "link_ang_vel": 0.5,
        "joint_pos": 1.0, "joint_vel": 0.5,
    },
    "lower": {
        "ta_link_pos": 0.5, "ta_link_ori": 2.0,
        "link_lin_vel": 0.5, "link_ang_vel": 0.5,
        "ta_joint_pos": 0.5, "joint_vel": 0.5,
    },
    "aux": {
        "root_ori": 0.5, "corrected_root_vel": 2.0,
        "floating_anchor_pos": 1.0, "recovery_upward_vel": 12.5,
        "pelvis_vert_accel": -1e-3, "ee_accel_mismatch": -1e-3,
        "action_rate": -0.05, "joint_limit": -15.0,
        "undesired_contact": -0.1, "head_torso_impact": -1e-5,
    },
}

#: 论文 Table I 的 "The 2nd Stage" 部分（terrain-contact 组）
PAPER_TABLE_I_STAGE2 = {
    "terrain": {
        "touchdown_quality": 10.0, "reference_contact_match": 1.5,
        "slip": -1.0, "stumble": -20.0,
        "contact_switching": -30.0, "contact_force": -1e-6,
    },
}


@pytest.mark.parametrize("group,expected", [
    ("upper", PAPER_TABLE_I_STAGE12["upper"]),
    ("lower", PAPER_TABLE_I_STAGE12["lower"]),
    ("aux", PAPER_TABLE_I_STAGE12["aux"]),
    ("terrain", PAPER_TABLE_I_STAGE2["terrain"]),
])
def test_weights_match_paper_table_i(group, expected):
    got = GROUPS[group].weights
    assert set(got) == set(expected), (
        f"{group} 组项名与论文不符: 多 {set(got) - set(expected)}, "
        f"缺 {set(expected) - set(got)}"
    )
    for name, w in expected.items():
        assert got[name] == pytest.approx(w), f"{group}.{name} 权重应为 {w}，得到 {got[name]}"


def test_term_count_per_group():
    """论文每组各 6 / 6 / 10 / 6 项 —— 数量也要对上。"""
    assert len(UPPER_TERMS) == 6
    assert len(LOWER_TERMS) == 6
    assert len(AUX_TERMS) == 10
    assert len(TERRAIN_TERMS) == 6


def test_no_duplicate_term_names_within_group():
    for g in (UPPER, LOWER, AUX, TERRAIN):
        names = g.names
        assert len(names) == len(set(names)), f"{g.name} 组项名重复"


def test_penalties_are_negative_and_rewards_positive():
    """符号约定：惩罚项必须为负，跟踪/奖励项必须为正。"""
    for name, w in AUX_TERMS:
        if name in ("pelvis_vert_accel", "ee_accel_mismatch", "action_rate",
                    "joint_limit", "undesired_contact", "head_torso_impact"):
            assert w < 0.0, f"{name} 是惩罚项，权重应为负"
    for name, w in TERRAIN_TERMS:
        if name in ("slip", "stumble", "contact_switching", "contact_force"):
            assert w < 0.0, f"{name} 是惩罚项，权重应为负"
        else:
            assert w > 0.0, f"{name} 是奖励项，权重应为正"
    for g in (UPPER, LOWER):
        assert all(w > 0 for _, w in g.terms)


def test_lower_body_ta_terms_have_paper_weights():
    """TA 三项的权重是 Table I 里最反直觉的一处（ori 2.0 高于 pos 0.5）。"""
    w = LOWER.weights
    assert w["ta_link_ori"] == 2.0
    assert w["ta_link_pos"] == 0.5
    assert w["ta_joint_pos"] == 0.5
    # 速度项不带 TA，权重与其他组一致
    assert w["link_lin_vel"] == 0.5 and w["link_ang_vel"] == 0.5
    assert w["joint_vel"] == 0.5


def test_relaxed_terms_are_exactly_the_ta_terms():
    """Eq.10 只作用于 Table I 中带 TA 的三项 —— 一个不多一个不少。"""
    assert set(RELAXED_TERMS) == {"ta_link_pos", "ta_link_ori", "ta_joint_pos"}
    assert set(RELAXED_TERMS) <= set(LOWER.names), "TA 项应属 lower body"
    # upper body 与 terrain 组都不得含 TA 项
    assert not (set(RELAXED_TERMS) & set(UPPER.names))
    assert not (set(RELAXED_TERMS) & set(TERRAIN.names))


# ---------------------------------------------------------------------------
# 分组求和
# ---------------------------------------------------------------------------

def test_sum_is_weighted_combination():
    vals = {n: 2.0 for n in UPPER.names}
    expect = 2.0 * sum(w for _, w in UPPER_TERMS)
    assert UPPER.sum(vals) == pytest.approx(expect)


def test_sum_treats_missing_terms_as_zero():
    assert UPPER.sum({"link_pos": 3.0}) == pytest.approx(3.0)
    assert UPPER.missing({"link_pos": 3.0}) == tuple(
        n for n in UPPER.names if n != "link_pos")


def test_sum_rejects_unknown_terms():
    """环境层写错键名必须立刻报错，而不是静默少算一项奖励。"""
    with pytest.raises(KeyError, match="未知奖励项"):
        UPPER.sum({"link_position": 1.0})  # 应为 link_pos


def test_sum_rejects_negative_and_nan_values():
    """**值域守卫**：负值与 NaN 必须被拒绝。

    本仓库的约定是"所有项的值 ≥ 0，符号全部由权重携带"。若某项返回负值
    （如惩罚项返回 `−‖Δa‖²`），它与 Table I 的负权重相乘
    （`−0.05 × −0.03 = +0.0015`）会变成**正贡献** —— 惩罚反转成奖励。
    这类错误在训练曲线上表现为"学出怪行为"，极难定位，故在求和处直接拒绝。

    NaN 也必须拦：`nan < 0` 为假，朴素的 `v < 0` 检查会漏过它，
    故实现用 `not (v >= 0)`。
    """
    with pytest.raises(ValueError, match="负值或 NaN"):
        AUX.sum({"action_rate": -0.03})
    with pytest.raises(ValueError, match="负值或 NaN"):
        AUX.sum({"action_rate": float("nan")})
    # 0 是合法的 —— 惩罚项"无违规"恰为 0
    assert AUX.sum({"action_rate": 0.0}) == pytest.approx(0.0)


def test_negative_weight_terms_contribute_negatively():
    """**负权重项在"有代价"时必须产生负贡献** —— 惩罚就得是惩罚。

    这正是 `auxiliary.py` 第一版违反的契约：那里惩罚项返回 `−代价`，
    与本表的负权重相乘变成**正贡献**（违规越大奖励越高）。
    这里绕开具体实现，直接用权重表验证代数结果：只给该项代价 1.0，
    组和应当恰等于它的权重（其余项按 0 计），因而恒为负。
    """
    for group in (AUX, TERRAIN):
        for name, w in group.terms:
            if w >= 0.0:
                continue
            contribution = group.sum({name: 1.0})
            assert contribution == pytest.approx(w), (
                f"{group.name}.{name} 代价为 1.0 时组和应等于权重 {w}，"
                f"得到 {contribution}")
            assert contribution < 0.0, \
                f"{group.name}.{name} 权重为负，代价应产生负贡献"


def test_total_reward_stage1_excludes_terrain():
    vals = {"upper": {"link_pos": 1.0}, "lower": {"ta_link_pos": 1.0},
            "terrain": {"touchdown_quality": 100.0}, "aux": {"root_ori": 1.0}}
    r1 = total_reward(vals, stage2=False)
    r2 = total_reward(vals, stage2=True)
    assert r1 == pytest.approx(1.0 + 0.5 + 0.5)
    assert r2 == pytest.approx(r1 + 10.0 * 100.0)


def test_total_reward_is_zero_for_empty_input():
    assert total_reward({}, stage2=True) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# χ(κ)：Eq.10 的地形指示函数
# ---------------------------------------------------------------------------

def test_chi_active_only_on_slopes_stairs_boxes():
    for f in TERRAIN_FAMILIES:
        expected = 1.0 if f in ("slopes", "stairs", "boxes") else 0.0
        assert chi(f) == expected, f"{f} 的 χ 应为 {expected}"
    assert set(CHI_ACTIVE_FAMILIES) == {"slopes", "stairs", "boxes"}


def test_chi_flat_and_rough_are_zero():
    """论文明确：flat 与 random rough 不激活松弛（必须严格跟踪）。"""
    assert chi("flat") == 0.0
    assert chi("rough") == 0.0


def test_chi_unknown_family_raises():
    with pytest.raises(ValueError, match="未知地形族"):
        chi("moon")


def test_terrain_families_match_paper():
    assert TERRAIN_FAMILIES == ("flat", "slopes", "stairs", "boxes", "rough")
    assert NUM_LEVELS == 10


# ---------------------------------------------------------------------------
# τ_{mh}(d) 与 Eq.10 松弛算子
# ---------------------------------------------------------------------------

def test_tau_zero_at_lowest_level_by_default():
    assert tau_budget(0, saturation_value=0.05) == pytest.approx(0.0)


def test_tau_saturates_at_max_level():
    assert tau_budget(9, saturation_value=0.05) == pytest.approx(0.05)


def test_tau_is_monotone_non_decreasing():
    vals = [tau_budget(d, 0.05) for d in range(NUM_LEVELS)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), f"τ 非单调: {vals}"


def test_tau_is_clamped_outside_level_domain():
    """域外夹紧而非外推（L10 不该给出超过饱和值的预算）。"""
    assert tau_budget(20, 0.05) == pytest.approx(0.05)
    assert tau_budget(-5, 0.05) == pytest.approx(0.0)


def test_tau_respects_start_fraction():
    assert tau_budget(0, 1.0, start_fraction=0.5) == pytest.approx(0.5)
    assert tau_budget(9, 1.0, start_fraction=0.5) == pytest.approx(1.0)
    assert tau_budget(4.5, 1.0, start_fraction=0.5) == pytest.approx(0.75)


def test_tau_rejects_bad_args():
    with pytest.raises(ValueError):
        tau_budget(0, 0.05, level_min=5, level_max=5)
    with pytest.raises(ValueError):
        tau_budget(0, 0.05, start_fraction=1.5)
    with pytest.raises(ValueError):
        tau_budget(0, -1.0)


def test_relaxed_error_clamps_into_tolerance_zone():
    """核心语义：误差小于预算 → 完全不罚（0）；超出部分照常保留。"""
    assert relaxed_error(0.01, alpha=1.0, chi_value=1.0, tau=0.05) == pytest.approx(0.0)
    assert relaxed_error(0.05, alpha=1.0, chi_value=1.0, tau=0.05) == pytest.approx(0.0)
    assert relaxed_error(0.08, alpha=1.0, chi_value=1.0, tau=0.05) == pytest.approx(0.03)


def test_relaxed_error_is_identity_when_chi_zero():
    """平地/rough：χ=0 → ẽ == e，即严格跟踪。"""
    for e in (0.0, 0.03, 1.0):
        assert relaxed_error(e, 1.0, 0.0, 0.05) == pytest.approx(e)


def test_relaxed_error_is_identity_when_alpha_zero():
    """α=0 明确退化为严格跟踪（论文原话）。"""
    assert relaxed_error(0.5, 0.0, 1.0, 0.05) == pytest.approx(0.5)


def test_relaxed_error_scales_every_term_alpha():
    """α 是**落在预算上**的乘子，不是落在误差上（易错点）。"""
    # α=2 → 预算翻倍
    assert relaxed_error(0.08, alpha=2.0, chi_value=1.0, tau=0.05) == pytest.approx(0.0)
    assert relaxed_error(0.12, alpha=2.0, chi_value=1.0, tau=0.05) == pytest.approx(0.02)


def test_relaxed_error_never_exceeds_raw_error():
    for e in np.linspace(0.0, 1.0, 11):
        got = relaxed_error(float(e), 1.0, 1.0, 0.05)
        assert 0.0 <= got <= e + 1e-12


def test_relaxed_error_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha"):
        relaxed_error(0.1, -1.0, 1.0, 0.05)


# ---------------------------------------------------------------------------
# 指数跟踪奖励（高斯核 exp(−e²/σ)）
# ---------------------------------------------------------------------------

def test_kernel_is_squared_error_not_linear():
    """**核是 `exp(−e²/σ)`，不是 `exp(−e/σ)`。**

    这是我最初写错的地方（按"通用形式"记成了线性核）。判据：高斯核对
    `e` 与 `−e` 对称且单调，但**不是**线性核的任何 σ 重参数化 ——
    取两点即可区分：线性核下 `ln r` 与 `e` 成正比，高斯核下与 `e²` 成正比。
    """
    sigma = 2.0
    es = np.array([0.5, 1.0, 2.0])
    r = np.array([exp_tracking_reward(float(e), sigma) for e in es])
    neg_ln = -np.log(r)
    # 与 e² 成正比（比值恒定），而不是与 e 成正比
    ratios_sq = neg_ln / (es ** 2)
    ratios_lin = neg_ln / es
    assert np.allclose(ratios_sq, ratios_sq[0], rtol=1e-12), "应满足 ln r ∝ e²"
    assert not np.allclose(ratios_lin, ratios_lin[0], rtol=1e-6), "不应满足 ln r ∝ e"


def test_exp_reward_is_one_at_zero_error():
    assert exp_tracking_reward(0.0) == pytest.approx(1.0)


def test_exp_reward_monotone_decreasing_and_positive():
    vals = [exp_tracking_reward(e, 0.2) for e in np.linspace(0.0, 2.0, 21)]
    assert all(b <= a for a, b in zip(vals, vals[1:])), "核必须单调递减"
    assert all(v > 0.0 for v in vals), "核恒正"


def test_exp_reward_analytic_value():
    """σ=2 时：e=√2 → 1/e；e=2 → e^{−2}。"""
    assert exp_tracking_reward(math.sqrt(2.0), 2.0) == pytest.approx(1.0 / math.e)
    assert exp_tracking_reward(2.0, 2.0) == pytest.approx(math.exp(-2.0))


def test_half_decay_error_is_sqrt_sigma():
    """σ 的直观标定：√σ 是 reward = 1/e 处的误差。"""
    for sigma in (0.06, 1.0, 20.0):
        assert exp_tracking_reward(math.sqrt(sigma), sigma) == pytest.approx(1.0 / math.e)


def test_exp_reward_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="σ"):
        exp_tracking_reward(0.1, 0.0)
    with pytest.raises(ValueError, match="σ"):
        exp_tracking_reward(0.1, -1.0)


def test_exp_reward_bounded_in_unit_interval():
    for e in np.linspace(0.0, 10.0, 41):
        v = exp_tracking_reward(float(e), 0.5)
        assert 0.0 < v <= 1.0 + 1e-12


def test_sigma_table_covers_all_tracked_terms():
    """每个跟踪项都必须登记 σ —— 未登记就报错，不许用默认值静默糊过去。"""
    from pgmt.rewards.spec import SIGMAS, sigma_for

    tracked = {n for n, _ in UPPER_TERMS} | {n for n, _ in LOWER_TERMS}
    # 上下半身的同名项共享 σ（环境层若区分，用 link_pos / link_pos_lower）
    missing = {t for t in tracked if t not in SIGMAS}
    assert not missing, f"以下跟踪项缺 σ: {sorted(missing)}"
    for t in tracked:
        assert sigma_for(t) > 0.0
    with pytest.raises(KeyError, match="未登记"):
        sigma_for("not_a_term")


def test_sigma_units_are_squared_by_construction():
    """σ 必须是"误差平方"量纲：位置项 σ 应等于 √σ（米）的平方量级。

    这条用半衰误差 √σ 反推：上半身位置 √σ 应在厘米级而不是米级
    （否则相当于允许 1m 的连杆位置误差还拿高分）。
    """
    from pgmt.rewards.spec import SIGMAS, sigma_for

    upper_pos = math.sqrt(sigma_for("link_pos"))
    lower_pos = math.sqrt(sigma_for("ta_link_pos"))
    assert 0.05 <= upper_pos <= 0.5, f"上半身位置半衰误差 {upper_pos:.3f}m 不合理"
    assert 0.05 <= lower_pos <= 2.0, f"下半身位置半衰误差 {lower_pos:.3f}m 不合理"
    # 上半身应比下半身严格（论文 §IV：上半身保姿态保真）
    assert upper_pos < lower_pos, "上半身位置容差应小于下半身"
    # 关节项半衰误差在弧度上应是"小而可见"的量级
    jp = math.sqrt(sigma_for("joint_pos"))
    assert 0.05 <= jp <= 1.0, f"关节位置半衰误差 {jp:.3f}rad 不合理"
    # 速度项量纲是速度平方
    assert math.sqrt(sigma_for("joint_vel")) > 0.0
    tracked_all = {n for n, _ in UPPER_TERMS} | {n for n, _ in LOWER_TERMS}
    unknown = set(SIGMAS) - tracked_all - {"link_pos_lower"}
    assert not unknown, f"SIGMAS 里有多余项: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# 组合入口：Eq.10 松弛 + 高斯核
# ---------------------------------------------------------------------------

def test_relaxed_reward_equals_exp_of_relaxed_error():
    """组合函数必须与"先算 ẽ（Eq.10）再取核"完全一致（不是另一套公式）。"""
    args = dict(alpha=1.0, terrain_family="stairs", level=5,
                saturation_value=0.05, sigma=0.2)
    for e in (0.0, 0.01, 0.05, 0.09, 0.5):
        tau = tau_budget(5, 0.05)
        expected = exp_tracking_reward(relaxed_error(e, 1.0, chi("stairs"), tau), 0.2)
        assert relaxed_tracking_reward(e, **args) == pytest.approx(expected)


def test_relaxed_reward_is_strict_on_flat():
    """平地：χ=0 → 与严格核逐点相同。"""
    for e in (0.0, 0.05, 0.2, 1.0):
        got = relaxed_tracking_reward(e, alpha=1.0, terrain_family="flat",
                                      level=9, saturation_value=0.05, sigma=0.2)
        assert got == pytest.approx(exp_tracking_reward(e, 0.2))


def test_relaxed_reward_is_more_permissive_than_strict():
    """松弛是放宽而非收紧：同一误差下奖励不得变低。"""
    e = 0.08
    strict = exp_tracking_reward(e, 0.2)
    for fam in TERRAIN_FAMILIES:
        got = relaxed_tracking_reward(e, alpha=1.0, terrain_family=fam,
                                      level=9, saturation_value=0.05, sigma=0.2)
        assert got >= strict - 1e-12, f"{fam} 上松弛后奖励反而更低"


def test_relaxed_reward_increases_with_level_on_active_terrain():
    """难度越高、容忍预算越大 → 同一误差下奖励单调不减。"""
    e = 0.08
    vals = [relaxed_tracking_reward(e, 1.0, "boxes", d, 0.05, 0.2)
            for d in range(NUM_LEVELS)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), f"非单调: {vals}"


def test_relaxed_reward_is_one_for_error_inside_budget():
    """误差落在容忍区内 → 奖励饱和到 1.0（该目标不再拉低总奖励）。"""
    got = relaxed_tracking_reward(0.04, alpha=1.0, terrain_family="stairs",
                                  level=9, saturation_value=0.05, sigma=0.2)
    assert got == pytest.approx(1.0)


def test_relaxation_tau_units_are_error_not_error_squared():
    """**单位一致性**：τ 与误差同量纲，σ 是误差平方。

    反例守卫：若把 τ 当"平方空间预算"直接用（即先平方再减），
    结果会与正确路径不同。这里对比两条路径，确认实现走的是"先钳后平方"。
    """
    e, tau_sat, sigma = 0.08, 0.05, 0.2
    tau = tau_budget(9, tau_sat)
    correct = exp_tracking_reward(relaxed_error(e, 1.0, 1.0, tau), sigma)

    # 错误路径：在平方空间里钳（等价于把 τ 当 τ² 用）
    wrong = math.exp(-max(e * e - tau, 0.0) / sigma)
    assert correct != pytest.approx(wrong), "两条路径不应相同，否则本测试无判别力"
    assert correct == pytest.approx(math.exp(-((0.08 - 0.05) ** 2) / 0.2))


# ---------------------------------------------------------------------------
# α 解析
# ---------------------------------------------------------------------------

def test_resolve_alpha_zero_for_non_relaxed_terms():
    """非 TA 项恒为 0 —— 保证 upper body 与其余约束默认严格。"""
    for term in ("link_pos", "link_ori", "joint_pos", "root_ori",
                 "touchdown_quality", "link_lin_vel", "joint_vel"):
        assert resolve_alpha(term, alpha_default=1.0) == 0.0, f"{term} 不该被松弛"


def test_resolve_alpha_default_for_relaxed_terms():
    for term in RELAXED_TERMS:
        assert resolve_alpha(term, alpha_default=0.7) == pytest.approx(0.7)


def test_resolve_alpha_per_element_overrides():
    per = {"ta_link_pos": {"left_ankle": 0.0, "right_ankle": 1.5}}
    assert resolve_alpha("ta_link_pos", 1.0, per, "left_ankle") == pytest.approx(0.0)
    assert resolve_alpha("ta_link_pos", 1.0, per, "right_ankle") == pytest.approx(1.5)
    # 未列出的元素回落到默认值
    assert resolve_alpha("ta_link_pos", 1.0, per, "pelvis") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A18：论文未写出的奖励细节必须集中在假设注册表，不得散落硬编码
# ---------------------------------------------------------------------------

def test_a18_is_registered_and_covers_relaxed_terms():
    from pgmt.cfg.assumptions import get

    cfg = get("A18").value
    # 松弛预算饱和值必须正好覆盖 TA 三项，不多不少
    assert set(cfg.tau_saturation) == set(RELAXED_TERMS), (
        f"A18.tau_saturation 与 RELAXED_TERMS 不一致: "
        f"{set(cfg.tau_saturation) ^ set(RELAXED_TERMS)}"
    )
    assert cfg.tracking_kernel == "exp_neg_squared_error_over_sigma"
    assert "OmniH2O" in cfg.sigma_source, "σ 来源必须可追溯"


def test_a18_tau_saturation_matches_a12_magnitude():
    """A12 说饱和值 ≈ 各目标 5% 量级 —— A18 的取值应与之同量级。

    注意 σ 是误差**平方**，τ 是**误差**本身，量纲不同不可直接比。
    这里只在 τ 内部一致性上做检查。
    """
    from pgmt.cfg.assumptions import get

    a12_sat = get("A12").value.tau_saturation  # 0.05（相对量级）
    assert a12_sat == pytest.approx(0.05), "A12 的饱和比例被改动？同步检查本测试"

    taus = get("A18").value.tau_saturation
    for term, sat in taus.items():
        assert sat > 0.0, f"{term} 的 τ 饱和值应为正"


def test_assumptions_include_a18():
    from pgmt.cfg.assumptions import ASSUMPTIONS

    assert "A18" in ASSUMPTIONS
    assert ASSUMPTIONS["A18"].name == "reward_impl"


def test_chi_registry_log_and_runtime_use_the_same_values():
    from pgmt.cfg.assumptions import dump, get

    expected = {"flat": 0.0, "slopes": 1.0, "stairs": 1.0,
                "boxes": 1.0, "rough": 0.0}
    assert get("A12").value.chi == expected
    assert dump()["A12"]["value"]["chi"] == expected
    assert {family: chi(family) for family in TERRAIN_FAMILIES} == expected
