"""地形接触项（Table I 的 terrain-contact 组，6 项，仅 Stage 2）。

## 本文件的重点

1. **符号约定**（与 auxiliary 组相同）：所有项的值 ≥ 0，符号由权重携带。
   2 个正向项 + 4 个代价项。
2. **落足质量是事件门控的**：只在"本步新落足"的足上评价。若持续评价，
   站立不动会每步拿满分 —— 那是奖励"站着别动"。见
   `test_touchdown_quality_only_evaluates_new_touchdown_feet`。
3. **`contact_switching` 不能惩罚"接触状态变化"**：正常步态每步都在切换，
   那样会把走路本身罚掉。故按"未持续满 dwell 就再翻转"衡量，见
   `test_contact_switching_is_zero_for_a_normal_gait`。
4. `contact_force` 的权重只有 1e−6 —— 用**平方**代价才有可见量级，
   见 `test_contact_force_squared_form_is_visible_under_its_tiny_weight`。
"""

import math

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.rewards.spec import TERRAIN, TERRAIN_TERMS
from pgmt.rewards.terrain_contact import (
    TerrainContactCfg,
    TerrainContactState,
    check_terrain_values_complete,
    compute_terrain_values,
    contact_force_cost,
    contact_switching_cost,
    local_height_variation,
    reference_contact_match,
    slip_cost,
    stumble_cost,
    terrain_contact_cfg,
    touchdown_events,
    touchdown_quality_reward,
    update_contact_age,
)

COST_TERMS = ("slip", "stumble", "contact_switching", "contact_force")
POSITIVE_TERMS = ("touchdown_quality", "reference_contact_match")


# ---------------------------------------------------------------------------
# 局部高度变化与落足质量
# ---------------------------------------------------------------------------

def test_local_height_variation_is_zero_on_flat_ground():
    assert local_height_variation(np.zeros(9)) == pytest.approx(0.0)


def test_local_height_variation_equals_known_std():
    # 两点 ±0.1：均值 0、总体标准差 0.1
    assert local_height_variation(np.array([-0.1, 0.1])) == pytest.approx(0.1)
    # 两点 0 与 0.2：均值 0.1，偏差 ±0.1 → 标准差 0.1
    assert local_height_variation(np.array([0.0, 0.2])) == pytest.approx(0.1)
    # 四点 ±0.1 各两个：均值 0、偏差 ±0.1 → 标准差 0.1
    assert local_height_variation(np.array([-0.1, -0.1, 0.1, 0.1])) \
        == pytest.approx(0.1)


def test_local_height_variation_is_translation_invariant():
    """高程图存的是相对高度（h − robot_z），故变化量必须与常数平移无关。"""
    h = np.array([0.1, -0.2, 0.35, 0.0])
    assert local_height_variation(h + 7.5) == pytest.approx(
        local_height_variation(h))


def test_local_height_variation_rejects_too_few_samples():
    """少于 2 点**报错**而不是返回 0 —— 返回 0 会被当成"完全平坦"给满分。"""
    with pytest.raises(ValueError, match="至少需 2 个采样点"):
        local_height_variation(np.array([0.3]))


def test_touchdown_quality_reward_is_one_on_flat_ground():
    assert touchdown_quality_reward(np.zeros(9), sigma=0.01) \
        == pytest.approx(1.0)


def test_touchdown_quality_reward_uses_squared_error_units():
    """σ 是**误差平方**量纲：std = √σ 处恰为 1/e（与 A18 的核一致）。"""
    sigma = 0.01
    h = np.array([-math.sqrt(sigma), math.sqrt(sigma)])   # std = √σ
    assert local_height_variation(h) == pytest.approx(math.sqrt(sigma))
    assert touchdown_quality_reward(h, sigma=sigma) \
        == pytest.approx(math.exp(-1.0))


def test_touchdown_quality_reward_decreases_with_roughness():
    sigma = 0.01
    flat = touchdown_quality_reward(np.zeros(4), sigma)
    rough = touchdown_quality_reward(np.array([-0.2, 0.2]), sigma)
    rougher = touchdown_quality_reward(np.array([-0.4, 0.4]), sigma)
    assert flat > rough > rougher > 0.0


def test_touchdown_events_only_marks_new_contacts():
    sim = np.array([True, False, True])
    prev = np.array([False, False, True])
    assert list(touchdown_events(sim, prev)) == [True, False, False]


def test_touchdown_events_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="须一致"):
        touchdown_events(np.array([True, False]), np.array([True]))


# ---------------------------------------------------------------------------
# 参考接触一致性
# ---------------------------------------------------------------------------

def test_reference_contact_match_is_one_when_identical():
    c = np.array([True, False])
    assert reference_contact_match(c, c) == pytest.approx(1.0)


def test_reference_contact_match_counts_both_error_directions():
    """两个方向都要算错：参考无接触而仿真踩上（绊到台阶）、以及反之。"""
    # 参考无接触、仿真有 → 不一致
    assert reference_contact_match(np.array([True, False]),
                                   np.array([True, True])) == pytest.approx(0.5)
    # 参考有接触、仿真没有 → 也扣
    assert reference_contact_match(np.array([True, False]),
                                   np.array([False, False])) == pytest.approx(0.5)
    # 全不一致
    assert reference_contact_match(np.array([True, True]),
                                   np.array([False, False])) == pytest.approx(0.0)


def test_reference_contact_match_rejects_bad_input():
    with pytest.raises(ValueError, match="须一致"):
        reference_contact_match(np.array([True, False]), np.array([True]))
    with pytest.raises(ValueError, match="至少需要一只足"):
        reference_contact_match(np.array([], dtype=bool), np.array([], dtype=bool))


# ---------------------------------------------------------------------------
# 4 个代价项：无违规恰为 0、值非负
# ---------------------------------------------------------------------------

def test_slip_cost_is_zero_without_contact_or_motion():
    assert slip_cost(np.zeros((2, 2)), np.array([True, True])) == pytest.approx(0.0)
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert slip_cost(v, np.array([False, False])) == pytest.approx(0.0)


def test_slip_cost_counts_only_contacting_feet():
    v = np.array([[1.0, 0.0], [3.0, 4.0]])   # 范数² 分别为 1 与 25
    assert slip_cost(v, np.array([True, False])) == pytest.approx(1.0)
    assert slip_cost(v, np.array([False, True])) == pytest.approx(25.0)


def test_slip_cost_requires_horizontal_only_input():
    """`foot_vel_xy` 必须是 `(n, 2)` —— 传三维速度（含竖直分量）应报错。

    打滑只与**切向**速度有关：抬脚/落脚时的竖直速度不是打滑。若接口接受
    三维速度，调用方很容易把竖直分量一起算进去而毫无提示，故这里只收水平分量。
    （这也是为什么本测试不是"传竖直速度看它被忽略" —— 那种输入根本进不来。）
    """
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        slip_cost(np.zeros((2, 3)), np.array([True, True]))
    with pytest.raises(ValueError, match="须一致"):
        slip_cost(np.zeros((2, 2)), np.array([True]))


def test_stumble_cost_is_zero_for_pure_vertical_force():
    """正常支撑：水平力≈0 → 不罚。"""
    f = np.array([[0.0, 0.0, 300.0], [1.0, 0.0, 400.0]])
    assert stumble_cost(f, ratio_threshold=5.0) == pytest.approx(0.0)


def test_stumble_cost_counts_side_impact():
    """侧向撞上棱边：f_xy = 10 而 f_z = 1，10 > 5×1 → 计入 10²。"""
    f = np.array([[0.0, 0.0, 300.0], [10.0, 0.0, 1.0]])
    assert stumble_cost(f, ratio_threshold=5.0) == pytest.approx(100.0)


def test_stumble_cost_boundary_is_strict():
    """恰好等于阈值不算撞击（判据是 `>` 而非 `>=`）。"""
    f = np.array([[5.0, 0.0, 1.0]])          # f_xy = 5 = 5×1 → 不罚
    assert stumble_cost(f, ratio_threshold=5.0) == pytest.approx(0.0)
    f2 = np.array([[5.0 + 1e-9, 0.0, 1.0]])
    assert stumble_cost(f2, ratio_threshold=5.0) > 0.0


def test_stumble_cost_rejects_bad_ratio():
    with pytest.raises(ValueError, match="应为正"):
        stumble_cost(np.zeros((1, 3)), ratio_threshold=0.0)


def test_contact_switching_is_zero_for_a_normal_gait():
    """**关键**：走路的正常切换不该被罚 —— 否则等于把行走本身罚掉。

    一次翻转，翻转前该状态已稳定 5 步（= dwell）→ 代价 0。
    """
    flipped = np.array([True, False])
    age = np.array([5, 9])
    assert contact_switching_cost(flipped, age, min_dwell=5) == pytest.approx(0.0)
    # 超过 dwell 也一样（不该因为"稳定得久"反而多罚）
    assert contact_switching_cost(np.array([True]), np.array([500]), 5) \
        == pytest.approx(0.0)


def test_contact_switching_grows_as_dwell_shortens():
    """代价随"已持续步数"线性衰减：翻转越快，代价越大。"""
    for age, expect in ((5, 0.0), (4, 0.2), (1, 0.8), (0, 1.0)):
        got = contact_switching_cost(np.array([True]), np.array([age]),
                                     min_dwell=5)
        assert got == pytest.approx(expect), f"age={age} 应为 {expect}，得到 {got}"


def test_contact_switching_ignores_feet_that_did_not_flip():
    flipped = np.array([False, False])
    age = np.array([1, 1])
    assert contact_switching_cost(flipped, age, min_dwell=5) == pytest.approx(0.0)


def test_contact_switching_bounds():
    """值域 [0, n]：最坏情况是每只足都在持续步数为 0 时翻转。"""
    n = 4
    worst = contact_switching_cost(np.ones(n, dtype=bool), np.zeros(n, dtype=int),
                                   min_dwell=5)
    assert worst == pytest.approx(float(n))
    assert worst <= float(n)


def test_contact_switching_rejects_bad_dwell_and_negative_age():
    with pytest.raises(ValueError, match="min_dwell"):
        contact_switching_cost(np.array([True]), np.array([1]), min_dwell=0)
    with pytest.raises(ValueError, match="不能为负"):
        contact_switching_cost(np.array([True]), np.array([-1]), min_dwell=5)


def test_contact_force_cost_is_zero_below_threshold():
    f = np.array([[0.0, 0.0, 500.0], [0.0, 0.0, 100.0]])
    assert contact_force_cost(f, np.array([True, True]), f_max=500.0) \
        == pytest.approx(0.0)


def test_contact_force_cost_counts_squared_excess_on_contacting_feet():
    f = np.array([[0.0, 0.0, 600.0], [0.0, 0.0, 900.0]])   # 超出 100 与 400
    assert contact_force_cost(f, np.array([True, False]), f_max=500.0) \
        == pytest.approx(100.0 ** 2)
    assert contact_force_cost(f, np.array([True, True]), f_max=500.0) \
        == pytest.approx(100.0 ** 2 + 400.0 ** 2)


def test_contact_force_cost_uses_full_magnitude_not_just_vertical():
    """合力大小参与判据（与 legged_gym 的 feet_contact_forces 一致）。"""
    f = np.array([[300.0, 400.0, 0.0]])      # 合力 500，恰好等于阈值
    assert contact_force_cost(f, np.array([True]), f_max=500.0) \
        == pytest.approx(0.0)
    f2 = np.array([[300.0, 400.0, 100.0]])   # 合力 ≈ 509.9
    assert contact_force_cost(f2, np.array([True]), f_max=500.0) > 0.0


def test_contact_force_cost_squared_form_is_visible_under_its_tiny_weight():
    """权重只有 1e−6 —— 用**平方**代价才能让超载产生可见惩罚。

    这一条反过来校验 A22 的度量选择：若代价改回一次方，超载 100 N 只贡献
    −1e−4，在总奖励里等于不存在。
    """
    vals = compute_terrain_values(_state(
        contact_forces=np.array([[0.0, 0.0, 600.0], [0.0, 0.0, 500.0]])))
    assert vals["contact_force"] == pytest.approx(100.0 ** 2)
    assert TERRAIN.sum({"contact_force": vals["contact_force"]}) \
        == pytest.approx(-1e-6 * 100.0 ** 2)
    assert abs(TERRAIN.sum({"contact_force": vals["contact_force"]})) >= 1e-3


# ---------------------------------------------------------------------------
# 接触年龄
# ---------------------------------------------------------------------------

def test_update_contact_age_resets_on_flip_and_increments_otherwise():
    age = np.array([7, 7])
    sim = np.array([True, False])
    prev = np.array([True, True])       # 足 1 翻转（接触→摆动）
    assert list(update_contact_age(age, sim, prev)) == [8, 1]


def test_update_contact_age_follows_the_documented_table():
    """复现 docstring 里的示例表（age[t] = 截至 t 该状态已持续的步数）。

        t:      0     1     2     3
        接触:   0     1     1     1
        age:    1     1     2     3
    """
    contact = [False, True, True, True]
    prev_contact = [True, False, True, True]   # t=0 视为一次翻转
    age = np.array([0])
    got = []
    for t in range(4):
        age = update_contact_age(age, np.array([contact[t]]),
                                 np.array([prev_contact[t]]))
        got.append(int(age[0]))
    assert got == [1, 1, 2, 3]


def test_update_contact_age_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="须一致"):
        update_contact_age(np.array([1, 2]), np.array([True, True]),
                           np.array([True]))


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def _state(**kw) -> TerrainContactState:
    """构造"完美落足"的状态：平地新落足、参考与仿真一致、无力无滑动。

    默认 `prev_sim_contact` 为全 False 而 `sim_contact` 为全 True，
    即两只足都**刚刚落足**（触发 touchdown_quality）。`contact_age` 取满
    dwell，使 `contact_switching` 为 0（翻转但已稳定够久）。
    """
    n, P = 2, 4
    base = dict(
        ref_contact=np.array([True, True]),
        sim_contact=np.array([True, True]),
        prev_sim_contact=np.array([False, False]),
        contact_age=np.array([5, 5]),
        foot_vel_xy=np.zeros((n, 2)),
        contact_forces=np.zeros((n, 3)),
        local_heights=np.zeros((n, P)),
    )
    base.update(kw)
    return TerrainContactState(**base)


def test_compute_terrain_values_keys_match_table_i():
    vals = compute_terrain_values(_state())
    assert set(vals) == {n for n, _ in TERRAIN_TERMS}
    check_terrain_values_complete(vals)


def test_check_terrain_values_complete_detects_mismatch_and_negatives():
    with pytest.raises(KeyError, match="项名不匹配"):
        check_terrain_values_complete({"slip": 0.0})
    vals = dict(compute_terrain_values(_state()), slip=-1.0)
    with pytest.raises(ValueError, match="负值"):
        check_terrain_values_complete(vals)


def test_perfect_touchdown_gives_full_positive_and_zero_costs():
    vals = compute_terrain_values(_state())
    for term in POSITIVE_TERMS:
        assert vals[term] == pytest.approx(1.0), f"{term}={vals[term]}"
    for term in COST_TERMS:
        assert vals[term] == pytest.approx(0.0), f"{term}={vals[term]}"


def test_terrain_group_sum_is_maximal_at_perfect_state():
    """Table I 的正权重之和 = 10.0 + 1.5 = 11.5；代价项为 0。"""
    vals = compute_terrain_values(_state())
    assert TERRAIN.sum(vals) == pytest.approx(10.0 + 1.5)


def test_no_touchdown_event_gives_zero_touchdown_quality():
    """已在接触中的足不重复触发落足质量（否则站立不动就白拿 +10）。"""
    vals = compute_terrain_values(_state(prev_sim_contact=np.array([True, True])))
    assert vals["touchdown_quality"] == pytest.approx(0.0)
    assert vals["reference_contact_match"] == pytest.approx(1.0), "该项仍应评价"


def test_touchdown_quality_only_evaluates_new_touchdown_feet():
    """只有**新落足**的足参与落足质量 —— 已支撑的足不该拉高或拉低它。

    左足落在粗糙处（高度 ±0.1 → std 0.1 = √σ → 核取 1/e），右足已在支撑
    且其下方是平地（std 0 → 核取 1）。若把右足也算进去，结果会被拉高到
    (1/e + 1)/2 ≈ 0.684；只算新落足的左足才是 1/e ≈ 0.368。
    """
    sigma = terrain_contact_cfg().sigma_touchdown_quality
    vals = compute_terrain_values(_state(
        prev_sim_contact=np.array([False, True]),
        local_heights=np.array([[-0.1, 0.1], [0.0, 0.0]]),
    ))
    assert sigma == pytest.approx(0.01), "本测试按 √σ = 0.1 构造，σ 变了要同步"
    assert vals["touchdown_quality"] == pytest.approx(math.exp(-1.0))
    assert vals["touchdown_quality"] != pytest.approx((math.exp(-1.0) + 1.0) / 2)


def test_all_terms_are_non_negative():
    """**不变量**：6 项的值全部 ≥ 0，符号只在 Table I 的权重里。"""
    st = _state(
        ref_contact=np.array([True, False]),
        sim_contact=np.array([True, True]),
        prev_sim_contact=np.array([False, False]),
        contact_age=np.array([0, 0]),
        foot_vel_xy=np.array([[2.0, -1.0], [0.5, 0.5]]),
        contact_forces=np.array([[50.0, 0.0, 1.0], [0.0, 0.0, 800.0]]),
        local_heights=np.array([[-0.3, 0.3], [0.0, 0.1]]),
    )
    vals = compute_terrain_values(st)
    for term, v in vals.items():
        assert v >= 0.0, f"{term} 的值应为 ≥ 0，得到 {v}"


def test_cost_terms_reduce_group_reward_as_violation_grows():
    """回归：违规越大 ⇒ 组奖励越小（负权重 × 非负代价）。

    每项单独构造，只改影响该项的字段。
    """
    base = compute_terrain_values(_state())
    base_sum = TERRAIN.sum(base)

    cases = {
        "slip": dict(foot_vel_xy=np.array([[1.0, 0.0], [0.0, 0.0]])),
        "stumble": dict(contact_forces=np.array([[100.0, 0.0, 1.0],
                                                 [0.0, 0.0, 0.0]])),
        "contact_switching": dict(contact_age=np.array([1, 1])),
        "contact_force": dict(contact_forces=np.array([[0.0, 0.0, 900.0],
                                                       [0.0, 0.0, 0.0]])),
    }
    assert set(cases) == set(COST_TERMS), "每个代价项都要有对照用例"

    for term, kw in cases.items():
        worse = compute_terrain_values(_state(**kw))
        assert worse[term] > base[term], \
            f"{term} 是代价量，违规增大时应变大（{base[term]} → {worse[term]}）"
        assert TERRAIN.sum(worse) < base_sum, \
            f"{term} 违规后组奖励应下降"


def test_terrain_state_validates_shapes():
    with pytest.raises(ValueError, match="ref_contact"):
        _state(ref_contact=np.array([True]))
    with pytest.raises(ValueError, match="foot_vel_xy"):
        _state(foot_vel_xy=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="contact_forces"):
        _state(contact_forces=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="local_heights"):
        _state(local_heights=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="至少需要一只足"):
        _state(ref_contact=np.array([], dtype=bool),
               sim_contact=np.array([], dtype=bool),
               prev_sim_contact=np.array([], dtype=bool),
               contact_age=np.array([], dtype=int),
               foot_vel_xy=np.zeros((0, 2)),
               contact_forces=np.zeros((0, 3)),
               local_heights=np.zeros((0, 4)))


# ---------------------------------------------------------------------------
# A22
# ---------------------------------------------------------------------------

def test_a22_is_registered_with_required_fields():
    cfg = get("A22").value
    assert isinstance(cfg, TerrainContactCfg)
    assert cfg.sigma_touchdown_quality > 0.0
    assert cfg.touchdown_patch_radius > 0.0
    assert cfg.stumble_force_ratio > 0.0
    assert isinstance(cfg.contact_switching_min_dwell, int)
    assert cfg.contact_switching_min_dwell >= 1
    assert cfg.contact_force_max > 0.0


def test_a22_sigma_matches_a_realistic_foot_scale():
    """√σ 应是"台阶/箱面边缘"与"平地"之间的分辨尺度。

    A7 的地形几何到 L9 约 0.4 m（箱高/台阶高），足底半长约 0.1 m 量级；
    若 √σ 只有几毫米，则任何地形都拿 0 分；若到米级，则任何地形都拿满分。
    """
    half = math.sqrt(terrain_contact_cfg().sigma_touchdown_quality)
    assert 0.02 <= half <= 0.3, f"落足质量半衰尺度 {half:.3f} m 不合理"


def test_a22_dwell_is_a_short_fraction_of_a_second():
    """dwell 用**步**表示（50 Hz，A1）：应是零点几秒量级的"抖动"尺度。"""
    dt = get("A1").value.dt
    dwell_s = terrain_contact_cfg().contact_switching_min_dwell * dt
    assert 0.02 <= dwell_s <= 0.3, f"dwell {dwell_s:.3f} s 不合理"
