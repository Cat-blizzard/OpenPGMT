"""地形课程与 A11 兼容规则。

本文件的重点是**把论文的两句自然语言变成可测的判据**：

  - "environments advance to harder terrain after successful completion and
    regress to easier levels after tracking failures"
  - "a coarse behavior–terrain compatibility rule to exclude clearly invalid
    combinations when sampling references"

因此测试分两块：课程状态机的**转移表**（穷举几种情形，而不是只跑一遍），
以及兼容规则的**保守性**（未知类别不得被误排除）。
"""

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.curriculum import (
    MOTION_CATEGORIES,
    MOTION_PREFIX_TO_CATEGORY,
    CurriculumState,
    TerrainCurriculum,
    excluded_families,
    filter_sequences,
    filter_sequences_or_all,
    incompatible_categories,
    is_compatible,
    min_excluded_level,
    motion_category,
)
from pgmt.envs.terrain.generators import FAMILIES, NUM_LEVELS


# ---------------------------------------------------------------------------
# 运动类别解析
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("walk1_subject1", "upright_locomotion"),
    ("run2_subject4", "upright_locomotion"),
    ("sprint1_subject2", "upright_locomotion"),
    ("obstacles6_subject4", "upright_locomotion"),
    ("dance2_subject3", "dynamic_whole_body"),
    ("jumps1_subject5", "dynamic_whole_body"),
    ("fightAndSports1_subject1", "dynamic_whole_body"),
    ("aiming2_subject2", "dynamic_whole_body"),
    ("multipleActions1_subject3", "dynamic_whole_body"),
    ("ground1_subject1", "lying_prone"),
    ("ground2_subject2", "lying_prone"),
    ("fallAndGetUp1_subject4", "push_perturbation"),
    ("pushAndFall1_subject1", "push_perturbation"),
    ("pushAndStumble1_subject3", "push_perturbation"),
])
def test_motion_category_resolves_known_prefixes(name, expected):
    assert motion_category(name) == expected


def test_longest_prefix_rule_directly():
    """**最长匹配规则**用人工表验证 —— 真实表测不出它。

    真实表里重叠的前缀（`push` / `pushAndFall` / `pushAndStumble`）恰好都映到
    同一类别，所以即便实现"取首个匹配"也能通过。构造一张重叠且**类别不同**的表，
    才能把规则本身钉住。
    """
    from pgmt.envs.terrain.curriculum import longest_prefix_match

    table = {"a": "short", "abc": "long", "ab": "mid"}
    assert longest_prefix_match("abcdef", table) == "abc", "应取最长匹配"
    assert longest_prefix_match("abzzz", table) == "ab"
    assert longest_prefix_match("azz", table) == "a"
    assert longest_prefix_match("zzz", table) is None, "无匹配应返回 None"


def test_longest_prefix_is_order_independent():
    """匹配结果不得依赖 dict 的插入顺序（否则是"碰巧对"）。"""
    from pgmt.envs.terrain.curriculum import longest_prefix_match

    forward = {"a": "x", "abc": "y"}
    backward = {"abc": "y", "a": "x"}
    assert longest_prefix_match("abcd", forward) == longest_prefix_match("abcd", backward)


def test_motion_category_unknown_falls_back_to_other():
    """未知前缀 → `other`（保守：不被任何规则排除）。"""
    assert motion_category("totallyNewMotion_subject9") == "other"
    assert motion_category("") == "other"


def test_every_known_prefix_maps_to_a_declared_category():
    """前缀表的值必须都在声明的类别集合里（防止拼写漂移）。"""
    for prefix, cat in MOTION_PREFIX_TO_CATEGORY.items():
        assert cat in MOTION_CATEGORIES, f"{prefix} → 未声明的类别 {cat}"


# ---------------------------------------------------------------------------
# A11 兼容规则
# ---------------------------------------------------------------------------

def test_rule_keys_are_all_families_and_declared_categories():
    """A11 的 excluded 表必须只引用已知族与已知类别。"""
    for cat, families in get("A11").value.excluded:
        assert cat in MOTION_CATEGORIES, f"规则引用了未声明的类别 {cat}"
        for f in families:
            assert f in FAMILIES, f"规则引用了未知地形族 {f}"


def test_excluded_families_matches_a11():
    assert set(excluded_families("lying_prone")) == {"stairs", "boxes"}
    assert set(excluded_families("inverted")) == {"stairs", "boxes"}
    assert set(excluded_families("sitting")) == {"slopes"}
    assert excluded_families("upright_locomotion") == ()


def test_rules_only_apply_at_high_levels():
    """A11：仅 L >= min_level_excluded 才排除（低难度地形温和，各类动作可行）。"""
    lv = min_excluded_level()
    assert lv == get("A11").value.min_level_excluded
    for low in range(lv):
        assert is_compatible("lying_prone", "stairs", low), f"L{low} 不应排除"
    assert not is_compatible("lying_prone", "stairs", lv), f"L{lv} 应排除"


def test_lying_prone_excluded_on_stairs_and_boxes_at_high_level():
    lv = min_excluded_level()
    for fam in ("stairs", "boxes"):
        assert not is_compatible("lying_prone", fam, lv)
        assert not is_compatible("lying_prone", fam, NUM_LEVELS - 1)
    # 但斜坡与粗糙地形不排除躺地类
    for fam in ("flat", "slopes", "rough"):
        assert is_compatible("lying_prone", fam, NUM_LEVELS - 1)


def test_sitting_excluded_on_slopes_at_high_level():
    lv = min_excluded_level()
    assert not is_compatible("sitting", "slopes", lv)
    assert is_compatible("sitting", "stairs", lv)
    assert is_compatible("sitting", "boxes", lv)


def test_unruled_categories_are_always_allowed():
    """未被任何规则提及的类别在所有组合上都应允许（规则是**排除**表）。"""
    for cat in ("upright_locomotion", "dynamic_whole_body",
                "push_perturbation", "other"):
        for fam in FAMILIES:
            for lv in (0, 5, 9):
                assert is_compatible(cat, fam, lv), f"{cat} × {fam} L{lv} 被误排除"


def test_other_category_is_never_excluded():
    """**保守性原则**：未分类运动不得被排除。

    误排除会悄悄丢掉可用数据（且难以察觉）；误允许只是让某个组合更难。
    因此方向必须是"宁可允许"。
    """
    for fam in FAMILIES:
        for lv in range(NUM_LEVELS):
            assert is_compatible("other", fam, lv)


def test_is_compatible_rejects_bad_input():
    with pytest.raises(ValueError, match="未知地形族"):
        is_compatible("other", "moon", 0)
    with pytest.raises(ValueError, match="难度级"):
        is_compatible("other", "flat", NUM_LEVELS)
    with pytest.raises(ValueError, match="难度级"):
        is_compatible("other", "flat", -1)


def test_incompatible_categories_is_consistent_with_is_compatible():
    for fam in FAMILIES:
        for lv in (0, min_excluded_level(), NUM_LEVELS - 1):
            bad = set(incompatible_categories(fam, lv))
            for cat in MOTION_CATEGORIES:
                assert (cat in bad) == (not is_compatible(cat, fam, lv))


# ---------------------------------------------------------------------------
# 序列过滤
# ---------------------------------------------------------------------------

SEQ_NAMES = ["walk1_subject1", "ground1_subject1", "dance1_subject1",
             "pushAndFall1_subject1", "sprint1_subject2"]


def test_filter_keeps_all_on_flat():
    """平地不排除任何类别 → 过滤后应保持全集。"""
    for lv in range(NUM_LEVELS):
        assert filter_sequences(SEQ_NAMES, "flat", lv) == SEQ_NAMES


def test_filter_removes_lying_prone_on_high_stairs():
    lv = min_excluded_level()
    kept = filter_sequences(SEQ_NAMES, "stairs", lv)
    assert "ground1_subject1" not in kept, "躺地类应被排除"
    assert "walk1_subject1" in kept
    assert "dance1_subject1" in kept
    # 低难度应保留
    assert filter_sequences(SEQ_NAMES, "stairs", 0) == SEQ_NAMES


def test_filter_preserves_input_order():
    kept = filter_sequences(SEQ_NAMES, "stairs", NUM_LEVELS - 1)
    idx = [SEQ_NAMES.index(s) for s in kept]
    assert idx == sorted(idx), "过滤不应改变顺序"


def test_filter_or_all_reports_when_it_falls_back():
    """全部被排除时必须**报告回退**，不能静默吞掉。"""
    only_ground = ["ground1_subject1"]
    lv = min_excluded_level()
    kept, fell_back = filter_sequences_or_all(only_ground, "stairs", lv)
    assert kept == only_ground
    assert fell_back is True, "回退必须被报告（说明规则与数据不匹配）"

    kept2, fell_back2 = filter_sequences_or_all(SEQ_NAMES, "flat", 0)
    assert fell_back2 is False
    assert kept2 == SEQ_NAMES


def test_filter_or_all_never_returns_empty():
    for fam in FAMILIES:
        for lv in range(NUM_LEVELS):
            kept, _ = filter_sequences_or_all(["ground1_subject1"], fam, lv)
            assert kept, f"{fam} L{lv} 返回了空集"


# ---------------------------------------------------------------------------
# 课程状态机
# ---------------------------------------------------------------------------

def test_curriculum_starts_at_level_zero():
    c = TerrainCurriculum(["flat", "stairs"])
    assert c.num_envs == 2
    assert list(c.levels()) == [0, 0]
    assert c.families() == ("flat", "stairs")


def test_promote_after_n_consecutive_successes():
    c = TerrainCurriculum(["slopes"], promote_after=3)
    assert c.update([0], success=True) == [], "第 1 次成功不应升级"
    assert c.level_of(0) == 0
    assert c.update([0], success=True) == [], "第 2 次成功不应升级"
    assert c.level_of(0) == 0
    ev = c.update([0], success=True)
    assert c.level_of(0) == 1, "第 3 次连续成功应升级"
    assert len(ev) == 1 and "升级" in ev[0]


def test_failure_resets_success_streak():
    """成功中途失败必须清零成功计数 —— 否则"连续"就退化成"累计"。"""
    c = TerrainCurriculum(["slopes"], promote_after=3, demote_after=99)
    c.update([0], success=True)
    c.update([0], success=True)
    c.update([0], success=False)      # 打断
    c.update([0], success=True)
    c.update([0], success=True)
    assert c.level_of(0) == 0, "被打断后不应升级（计数未清零）"
    c.update([0], success=True)
    assert c.level_of(0) == 1


def test_demote_after_n_consecutive_failures():
    c = TerrainCurriculum(["boxes"], promote_after=1, demote_after=2)
    c.update([0], success=True)
    assert c.level_of(0) == 1
    assert c.update([0], success=False) == []
    assert c.level_of(0) == 1
    c.update([0], success=False)
    assert c.level_of(0) == 0, "连续 2 次失败应降级"


def test_level_never_goes_below_zero():
    c = TerrainCurriculum(["rough"], demote_after=1)
    for _ in range(10):
        ev = c.update([0], success=False)
        assert ev == [], "L0 无处可降，不应产生事件"
    assert c.level_of(0) == 0


def test_level_never_exceeds_max_level():
    c = TerrainCurriculum(["flat"], promote_after=1, max_level=2)
    for _ in range(10):
        c.update([0], success=True)
    assert c.level_of(0) == 2, "应停在 max_level"


def test_promotions_and_demotions_are_counted():
    c = TerrainCurriculum(["slopes"], promote_after=1, demote_after=1)
    c.update([0], success=True)
    c.update([0], success=True)
    c.update([0], success=False)
    assert c.states[0].promotions == 2
    assert c.states[0].demotions == 1


def test_update_handles_multiple_envs_independently():
    """每环境的课程必须**互不影响** —— 否则 15k 并行环境会互相干扰。"""
    c = TerrainCurriculum(["slopes"] * 3, promote_after=1, demote_after=1)
    c.update([1], success=True)
    assert list(c.levels()) == [0, 1, 0]
    c.update([0, 2], success=True)
    assert list(c.levels()) == [1, 1, 1]
    c.update([1], success=False)
    assert list(c.levels()) == [1, 0, 1]


def test_update_processes_duplicate_env_ids_as_separate_updates():
    """重复的 env_id 应被当作**多次**更新，不做去重。

    我最初把断言写成 `level_of(0) == 1`（隐含"批内会去重"），与我自己在
    `TerrainCurriculum.update` docstring 里写的"不做去重（去重会掩盖调用方的
    bug）"**自相矛盾**。实现按序列逐个处理，`promote_after=1` 下 `[0, 0]` 即两次
    成功 → 升两级。

    为什么不该去重：环境层若因掩码写法重复传入同一 id，那是**调用方的 bug**，
    静默去重会把它藏起来；而多数一次只会多升一级，很快就能从训练曲线看出来。
    """
    c = TerrainCurriculum(["slopes"], promote_after=1)
    c.update([0, 0], success=True)
    assert c.level_of(0) == 2, "两个重复 id 应各算一次更新 → 升两级"
    assert c.states[0].promotions == 2

    # 混合成功/失败按顺序生效
    c2 = TerrainCurriculum(["slopes"], promote_after=1, demote_after=1)
    c2.update([0, 0], success=True)
    assert c2.level_of(0) == 2
    c2.update([0, 0], success=False)
    assert c2.level_of(0) == 0, "两次失败应各降一级"


def test_update_rejects_out_of_range_env_id():
    c = TerrainCurriculum(["flat"])
    with pytest.raises(IndexError):
        c.update([5], success=True)


def test_curriculum_rejects_bad_construction():
    with pytest.raises(ValueError, match="不能为空"):
        TerrainCurriculum([])
    with pytest.raises(ValueError, match="promote_after"):
        TerrainCurriculum(["flat"], promote_after=0)
    with pytest.raises(ValueError, match="max_level"):
        TerrainCurriculum(["flat"], max_level=0)
    with pytest.raises(ValueError, match="max_level"):
        TerrainCurriculum(["flat"], max_level=NUM_LEVELS)


def test_curriculum_state_validates_family_and_level():
    with pytest.raises(ValueError, match="未知地形族"):
        CurriculumState(family="moon")
    with pytest.raises(ValueError, match="难度级"):
        CurriculumState(family="flat", level=NUM_LEVELS)


def test_summary_reports_histogram_and_mean():
    c = TerrainCurriculum(["flat"] * 4, promote_after=1)
    c.update([0, 1], success=True)
    s = c.summary()
    assert s["num_envs"] == 4
    assert s["level_histogram"][0] == 2
    assert s["level_histogram"][1] == 2
    assert s["mean_level"] == pytest.approx(0.5)
    assert s["max_level"] == 1
    assert s["promotions"] == 2


def test_summary_histogram_covers_all_levels():
    c = TerrainCurriculum(["flat"])
    h = c.summary()["level_histogram"]
    assert sorted(h) == list(range(NUM_LEVELS)), "直方图应覆盖全部难度级"
    assert sum(h.values()) == 1


def test_curriculum_only_changes_level_not_family():
    """族是评估协议里**固定分配**的一部分，课程只调 level（论文的 matched 协议）。"""
    c = TerrainCurriculum(["stairs", "boxes"], promote_after=1, demote_after=1)
    for _ in range(5):
        c.update([0, 1], success=True)
    for _ in range(5):
        c.update([0, 1], success=False)
    assert c.families() == ("stairs", "boxes")


def test_levels_array_reflects_state():
    c = TerrainCurriculum(["flat"] * 3, promote_after=1)
    c.update([2], success=True)
    assert np.array_equal(c.levels(), [0, 0, 1])
