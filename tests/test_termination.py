"""终止条件：容忍区（与 A12 松弛同源）、延迟、完成/失败判定。

## 本文件要锁住的三件事（都来自论文的具体表述）

1. **completion 的定义**（Table II 脚注）："reaching the 30-s horizon **without
   early termination**" → 跑满上限 = 成功，提前终止 = 失败。因此
   `episode_outcome` 里 `success = timeout & ~terminated`，两者不是互斥的补集。
2. **终止是地形感知的**（Fig.2 图例 "terrain aware rewards **& terminations**"
   + "Tolerance zone"）：容忍区在 slopes/stairs/boxes 上更大 —— 与 A12 松弛
   预算同源，有测试断言两者一致。
3. **延迟确实由难度控制**（§V-C "selected termination delays"）：同一触发在
   高难度上需要持续更久才终止。

我特意把第 2 条写成"与 A12 一致"的断言而不是两个独立的数 —— 若哪天有人改了
松弛的 χ 却没改终止，那条测试会失败。这是论文把 rewards 与 terminations 并列
画在 Fig.2 里的直接含义。
"""

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.envs.termination import (
    FAILURE_REASONS,
    TerminationReason,
    TerminationState,
    compute_termination,
    episode_outcome,
    ref_deviation_exceeded,
    root_too_low,
    termination_cfg,
    termination_delay,
    tilt_deg_from_projected_gravity,
    tilted,
    tolerance_budget,
)
from pgmt.envs.terrain.generators import FAMILIES


# ---------------------------------------------------------------------------
# completion 定义（Table II 脚注）
# ---------------------------------------------------------------------------

def test_timeout_without_termination_is_success():
    terminated = np.array([False, False, True, True])
    timeout = np.array([True, False, True, False])
    success, failed = episode_outcome(terminated, timeout)
    assert list(success) == [True, False, False, False]
    assert list(failed) == [False, False, True, True]


def test_timeout_is_not_a_failure():
    """跑满上限即成功 —— 时间到而**未**提前终止不能被算作失败。"""
    success, failed = episode_outcome(np.array([False]), np.array([True]))
    assert bool(success[0]) and not bool(failed[0])


def test_terminated_before_timeout_is_failure():
    success, failed = episode_outcome(np.array([True]), np.array([False]))
    assert not bool(success[0]) and bool(failed[0])


def test_outcome_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="形状不一致"):
        episode_outcome(np.array([True, False]), np.array([True]))


def test_failure_reasons_exclude_timeout():
    """`FAILURE_REASONS` 不得包含 TIMEOUT（它不是失败）。"""
    assert TerminationReason.TIMEOUT not in FAILURE_REASONS
    assert set(FAILURE_REASONS) == {
        TerminationReason.REF_DEVIATION,
        TerminationReason.ROOT_LOW,
        TerminationReason.TILTED,
    }


# ---------------------------------------------------------------------------
# 容忍区：与 A12 松弛同源（Fig.2 把两者并列的含义）
# ---------------------------------------------------------------------------

def test_flat_and_rough_use_the_base_budget():
    """χ=0 的族（flat / rough）不放大容忍区（与松弛的 χ 一致）。"""
    from pgmt.rewards.spec import chi

    cfg = termination_cfg()
    for fam in ("flat", "rough"):
        assert chi(fam) == 0.0
        for lv in range(10):
            assert tolerance_budget(fam, lv) == pytest.approx(cfg.ref_deviation_base)


def test_active_terrain_families_enlarge_the_budget():
    """χ=1 的族上容忍区**不小于**基准值，且随难度不减。

    注意边界：`τ(L0) = 0`（A12 把 L0 的预算定为 0），所以 **L0 处容忍区恰好
    等于基准值**，L1+ 才严格更大。我第一版断言所有级别都严格大于，忽略了 L0
    这个边界 —— 实现是对的。
    """
    from pgmt.rewards.spec import chi

    cfg = termination_cfg()
    for fam in ("slopes", "stairs", "boxes"):
        assert chi(fam) == 1.0
        vals = [tolerance_budget(fam, lv) for lv in range(10)]
        assert vals[0] == pytest.approx(cfg.ref_deviation_base), "L0 应恰为基准值"
        assert all(v >= cfg.ref_deviation_base for v in vals), f"{fam} 不应小于基准"
        assert all(b >= a for a, b in zip(vals, vals[1:])), f"{fam} 非单调"
        assert vals[9] > cfg.ref_deviation_base, f"{fam} L9 应严格大于基准"


def test_tolerance_budget_uses_the_same_chi_as_relaxation():
    """**关键一致性**：终止容忍区为基准值的族集合，必须与松弛 χ=0 的族集合相同。

    论文 Fig.2 把 "terrain aware rewards & terminations" 并列画在一起，
    含义就是两者用同一套地形指示 χ。若有人只改了一边，这条会失败。
    """
    from pgmt.rewards.spec import chi

    cfg = termination_cfg()
    for fam in FAMILIES:
        enlarged = tolerance_budget(fam, 9) > cfg.ref_deviation_base
        assert enlarged == (chi(fam) == 1.0), f"{fam}: 容忍区与 χ 不一致"


def test_ref_deviation_boundary_is_inside_outside():
    cfg = termination_cfg()
    b = tolerance_budget("flat", 0)
    assert not ref_deviation_exceeded(b, "flat", 0), "恰好等于容忍区不应终止"
    assert ref_deviation_exceeded(b + 1e-9, "flat", 0)
    assert not ref_deviation_exceeded(b - 1e-9, "flat", 0)
    assert cfg.ref_deviation_base == pytest.approx(b)


def test_high_difficulty_tolerates_more_deviation():
    """同一偏差：平地判负，高难度 stairs 不判负（drift-tolerant）。

    **期望值从配置导出，不硬编码**：我第一版写了 `err = 0.4`，但
    `ref_deviation_base` 默认是 0.5 —— 0.4 在平地上根本没超限，测试自己错了。
    正确做法是取一个"必然超过平地容忍区、但仍落在 stairs L9 容忍区内"的误差。
    """
    cfg = termination_cfg()
    err = cfg.ref_deviation_base + 0.01          # 刚超过基准
    assert ref_deviation_exceeded(err, "flat", 9), "平地应判负"
    # stairs L9 的容忍区 = 基准 + τ(L9)，更大
    assert tolerance_budget("stairs", 9) > err, "前置条件：L9 容忍区应更大"
    assert not ref_deviation_exceeded(err, "stairs", 9), "高难度 stairs 应容忍"

    # 远超任何容忍区的偏差在所有族上都应判负
    for fam in FAMILIES:
        assert ref_deviation_exceeded(tolerance_budget(fam, 9) + 1.0, fam, 9)


def test_ref_deviation_rejects_negative_error():
    with pytest.raises(ValueError, match="非负"):
        ref_deviation_exceeded(-0.1, "flat", 0)


# ---------------------------------------------------------------------------
# 姿态与高度判据
# ---------------------------------------------------------------------------

def test_tilt_from_projected_gravity_endpoints():
    assert tilt_deg_from_projected_gravity(-1.0) == pytest.approx(0.0)
    assert tilt_deg_from_projected_gravity(1.0) == pytest.approx(0.0), "符号无关"
    assert tilt_deg_from_projected_gravity(0.0) == pytest.approx(90.0)


def test_tilt_matches_analytic_angle():
    """倾角 60° 时 |g_z| = cos60° = 0.5，反演应得 60°。"""
    import math
    gz = math.cos(math.radians(60.0))
    assert tilt_deg_from_projected_gravity(gz) == pytest.approx(60.0)


def test_tilted_threshold():
    cfg = termination_cfg()
    import math
    just_under = math.cos(math.radians(cfg.tilt_max_deg - 1.0))
    just_over = math.cos(math.radians(cfg.tilt_max_deg + 1.0))
    assert not tilted(just_under)
    assert tilted(just_over)


def test_root_too_low_threshold():
    cfg = termination_cfg()
    assert not root_too_low(cfg.root_height_min)
    assert root_too_low(cfg.root_height_min - 1e-6)
    assert not root_too_low(cfg.root_height_min + 1e-6)


def test_tilt_clips_out_of_range_gravity():
    """|g_z| > 1（数值噪声）不得产生 NaN。"""
    assert np.isfinite(tilt_deg_from_projected_gravity(1.0000001))
    assert np.isfinite(tilt_deg_from_projected_gravity(-1.0000001))


# ---------------------------------------------------------------------------
# 延迟（§V-C: "selected termination delays"）
# ---------------------------------------------------------------------------

def test_ref_deviation_delay_grows_with_difficulty_on_active_terrain():
    """高难度上终止延迟更长（论文的 selected termination delays）。"""
    vals = [termination_delay(TerminationReason.REF_DEVIATION, "stairs", lv)
            for lv in range(10)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), f"延迟非单调: {vals}"
    assert vals[9] > vals[0], "L9 的延迟应大于 L0"


def test_delay_on_flat_is_the_base_value():
    cfg = termination_cfg()
    base = cfg.delay_s[TerminationReason.REF_DEVIATION.value]
    for lv in range(10):
        assert termination_delay(TerminationReason.REF_DEVIATION, "flat", lv) \
            == pytest.approx(base)


def test_non_ref_reasons_have_constant_delay():
    """基座过低 / 翻倒不随地形延长延迟 —— 地形不会让"摔倒"变得可容忍。"""
    for reason in (TerminationReason.ROOT_LOW, TerminationReason.TILTED):
        vals = {termination_delay(reason, f, lv) for f in FAMILIES
                for lv in (0, 9)}
        assert len(vals) == 1, f"{reason} 的延迟不应随族/难度变化"


def test_delay_is_never_negative():
    for reason in FAILURE_REASONS:
        for f in FAMILIES:
            for lv in range(10):
                assert termination_delay(reason, f, lv) >= 0.0


# ---------------------------------------------------------------------------
# 状态机：延迟计时
# ---------------------------------------------------------------------------

def _const(n, v):
    return np.full(n, v)


def test_immediate_reason_fires_on_first_true_step():
    """延迟为 0 的原因（root_low）应在条件成立的**当步**触发。"""
    st = TerminationState(num_envs=1, dt=0.02)
    active = {
        TerminationReason.REF_DEVIATION: np.array([False]),
        TerminationReason.ROOT_LOW: np.array([True]),
        TerminationReason.TILTED: np.array([False]),
    }
    delay = {r: np.array([0.0]) for r in FAILURE_REASONS}
    term, elapsed, primary = st.step(active, delay)
    assert bool(term[0])
    assert bool(primary[TerminationReason.ROOT_LOW][0])
    assert not bool(primary[TerminationReason.REF_DEVIATION][0])


def test_delayed_reason_requires_sustained_condition():
    """延迟 0.5s、dt 0.02 → 需连续 25 步才触发（第 25 步 elapsed 恰为 0.5）。"""
    st = TerminationState(num_envs=1, dt=0.02)
    delay_s = 0.5
    n_steps = int(round(delay_s / 0.02))
    active = {
        TerminationReason.REF_DEVIATION: np.array([True]),
        TerminationReason.ROOT_LOW: np.array([False]),
        TerminationReason.TILTED: np.array([False]),
    }
    delay = {r: np.array([delay_s if r is TerminationReason.REF_DEVIATION else 0.0])
             for r in FAILURE_REASONS}
    for i in range(n_steps - 1):
        term, _, _ = st.step(active, delay)
        assert not bool(term[0]), f"第 {i + 1} 步不应触发（延迟未到）"
    term, elapsed, _ = st.step(active, delay)
    assert bool(term[0]), "延迟到点应触发"
    assert elapsed[TerminationReason.REF_DEVIATION][0] == pytest.approx(delay_s)


def test_condition_clearing_resets_the_timer():
    """条件中途解除必须**清零**计时 —— 否则"持续满足"退化成"累计满足"。"""
    st = TerminationState(num_envs=1, dt=0.02)
    delay = {r: np.array([0.2]) for r in FAILURE_REASONS}
    on = {TerminationReason.REF_DEVIATION: np.array([True]),
          TerminationReason.ROOT_LOW: np.array([False]),
          TerminationReason.TILTED: np.array([False])}
    off = dict(on, **{TerminationReason.REF_DEVIATION: np.array([False])})

    for _ in range(5):
        st.step(on, delay)
    assert st.elapsed_of(TerminationReason.REF_DEVIATION, 0) == pytest.approx(0.1)
    st.step(off, delay)
    assert st.elapsed_of(TerminationReason.REF_DEVIATION, 0) == pytest.approx(0.0)
    # 再满足仍需重新累计满 0.2s。步数由配置导出（0.2/0.02 = 10 步），
    # 而不是手数 —— 手数容易差一步。
    n_steps = int(round(0.2 / 0.02))
    for i in range(n_steps - 1):
        term, _, _ = st.step(on, delay)
        assert not bool(term[0]), f"第 {i + 1} 步不应触发"
    term, _, _ = st.step(on, delay)
    assert bool(term[0]), f"第 {n_steps} 步（累计满 0.2s）应触发"


def test_envs_are_independent():
    """每环境的延迟计时互不影响 —— 15k 并行环境的关键性质。"""
    st = TerminationState(num_envs=3, dt=0.02)
    delay = {r: np.array([0.1, 0.1, 0.1]) for r in FAILURE_REASONS}
    active = {TerminationReason.REF_DEVIATION: np.array([True, False, True]),
              TerminationReason.ROOT_LOW: np.array([False, False, True]),
              TerminationReason.TILTED: np.array([False, False, False])}
    for _ in range(5):
        term, _, _ = st.step(active, delay)
    assert list(term) == [True, False, True], "env1 条件为假不应触发"


def test_reset_clears_timers():
    st = TerminationState(num_envs=2, dt=0.02)
    delay = {r: np.array([0.2, 0.2]) for r in FAILURE_REASONS}
    active = {TerminationReason.REF_DEVIATION: np.array([True, True]),
              TerminationReason.ROOT_LOW: np.array([False, False]),
              TerminationReason.TILTED: np.array([False, False])}
    for _ in range(5):
        st.step(active, delay)
    assert st.elapsed_of(TerminationReason.REF_DEVIATION, 0) > 0.0
    st.reset([0])
    assert st.elapsed_of(TerminationReason.REF_DEVIATION, 0) == pytest.approx(0.0)
    assert st.elapsed_of(TerminationReason.REF_DEVIATION, 1) > 0.0, "不应影响 env1"


def test_state_validates_construction_and_inputs():
    with pytest.raises(ValueError, match="num_envs"):
        TerminationState(0, 0.02)
    with pytest.raises(ValueError, match="dt"):
        TerminationState(1, 0.0)

    st = TerminationState(num_envs=2, dt=0.02)
    delay = {r: np.array([0.0, 0.0]) for r in FAILURE_REASONS}
    bad = {TerminationReason.REF_DEVIATION: np.array([True]),      # 长度 1
           TerminationReason.ROOT_LOW: np.array([False, False]),
           TerminationReason.TILTED: np.array([False, False])}
    with pytest.raises(ValueError, match="形状"):
        st.step(bad, delay)

    missing = {TerminationReason.REF_DEVIATION: np.array([True, True])}
    with pytest.raises(KeyError, match="缺少原因"):
        st.step(missing, delay)

    neg = {r: np.array([-0.1, 0.0]) for r in FAILURE_REASONS}
    ok = {r: np.array([False, False]) for r in FAILURE_REASONS}
    with pytest.raises(ValueError, match="不应为负"):
        st.step(ok, neg)


def test_reset_rejects_out_of_range():
    st = TerminationState(num_envs=2, dt=0.02)
    with pytest.raises(IndexError):
        st.reset([2])


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------

def test_compute_termination_mixes_reasons_across_envs():
    """三个环境分别因不同原因终止（或都不终止），验证归因正确。"""
    st = TerminationState(num_envs=3, dt=0.5)
    cfg = termination_cfg()
    fams = ["flat", "flat", "flat"]
    level = np.array([0, 0, 0])
    # env0: 参考偏差超限；env1: 基座过低；env2: 正常
    err = np.array([cfg.ref_deviation_base + 0.1, 0.0, 0.0])
    h = np.array([0.8, cfg.root_height_min - 0.1, 0.8])
    gz = np.array([-1.0, -1.0, -1.0])
    term, primary, elapsed = compute_termination(st, err, h, gz, fams, level)
    assert list(term) == [True, True, False]
    assert bool(primary["ref_deviation"][0])
    assert bool(primary["root_low"][1])
    assert not bool(primary["ref_deviation"][1])
    assert set(elapsed) == {r.value for r in FAILURE_REASONS}


def test_compute_termination_respects_per_env_delay():
    """同一批里高难度环境延迟更长 → 同一步内不触发，低难度已触发。"""
    st = TerminationState(num_envs=2, dt=0.5)
    fams = ["flat", "stairs"]
    level = np.array([0, 9])
    cfg = termination_cfg()
    # 误差超平地的容忍区，但仍在 stairs L9 的容忍区内
    err = np.array([cfg.ref_deviation_base + 0.05, cfg.ref_deviation_base + 0.05])
    h = np.full(2, 0.8)
    gz = np.full(2, -1.0)
    term, primary, _ = compute_termination(st, err, h, gz, fams, level)
    assert list(term) == [True, False], "高难度环境不应在同一步触发"
    assert bool(primary["ref_deviation"][0])


def test_compute_termination_validates_shapes_and_families():
    st = TerminationState(num_envs=2, dt=0.02)
    ok = np.zeros(2)
    with pytest.raises(ValueError, match="形状"):
        compute_termination(st, np.zeros(3), ok, ok, ["flat", "flat"], ok.astype(int))
    with pytest.raises(ValueError, match="长度"):
        compute_termination(st, ok, ok, ok, ["flat"], ok.astype(int))
    with pytest.raises(ValueError, match="未知地形族"):
        compute_termination(st, ok, ok, ok, ["moon", "flat"], ok.astype(int))


def test_a20_is_registered():
    """A20 的值必须是 `assumptions` 里那个类型 —— 不是别处的同名类。

    这条原来是 `isinstance(cfg, TerminationCfg)` 而 `TerminationCfg` 从
    `pgmt.envs.termination` 导入 —— 但那里曾有**第二份定义**，导致 isinstance
    失败。测试当场抓住了这个"同一概念两处定义"的问题（现已删除重复定义）。
    """
    from pgmt.cfg.assumptions import TerminationCfg as AssumptionTerminationCfg

    cfg = get("A20").value
    assert isinstance(cfg, AssumptionTerminationCfg)
    assert cfg.root_height_min > 0.0
    assert 0.0 < cfg.tilt_max_deg < 90.0
    assert cfg.ref_deviation_base > 0.0
    assert set(cfg.delay_s) == {
        TerminationReason.REF_DEVIATION.value,
        TerminationReason.ROOT_LOW.value,
        TerminationReason.TILTED.value,
    }, "delay_s 的键必须覆盖全部失败原因"
    assert cfg.terrain_delay_scale >= 0.0
