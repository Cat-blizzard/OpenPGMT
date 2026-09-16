"""Table I 十六项语义对照表的同步性与完整性（A18 的语义依据）。

这张表的价值在于"逐项都有出处、无出处就明确标待办"。因此测试要保证：

  1. 覆盖面：`spec.py` 的每一项都在表里有且仅有一条语义记录（不漏、不多）
  2. 松弛标记：只有论文 Table I 中带 TA 的三项 `relaxable=True`，
     其余必须为 False —— 标错了会静默放宽不该放宽的目标
  3. 对应关系：标 `IDENTICAL` 的项必须有 counterpart 名字；
     标 `UNRESOLVED` 的项必须有 note 说明待办（不许留空过）
  4. 统计数：对应关系分布与期望一致（变了就要有人来解释）
"""

import pytest

from pgmt.rewards.semantics import (
    ALL_SEMANTICS,
    BY_GROUP_TERM,
    Corr,
    SPEC_TERMS,
    correspondence_counts,
    semantics_for,
    term_names_by_group,
)
from pgmt.rewards.spec import RELAXED_TERMS, TERRAIN_TERMS


# ---------------------------------------------------------------------------
# 覆盖面：本表与 spec.py 必须一一对应
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group", ["upper", "lower", "aux", "terrain"])
def test_every_spec_term_has_semantics(group):
    """spec.py 的每一项都必须有语义记录 —— 不许有'裸项'。"""
    for term in SPEC_TERMS[group]:
        assert (group, term) in BY_GROUP_TERM, f"缺语义: {group}.{term}"


@pytest.mark.parametrize("group", ["upper", "lower", "aux", "terrain"])
def test_no_extra_terms_in_semantics(group):
    """本表不得登记 spec.py 里不存在的项（防止改名后两边漂移）。"""
    registered = set(term_names_by_group().get(group, ()))
    assert registered == set(SPEC_TERMS[group]), (
        f"{group} 组项名不一致：仅在本表 {registered - set(SPEC_TERMS[group])}，"
        f"仅在 spec {set(SPEC_TERMS[group]) - registered}"
    )


def test_total_term_count_matches_table_i():
    """Table I 共 6 + 6 + 10 + 6 = 28 条（含上下半身同名项）。"""
    assert len(ALL_SEMANTICS) == 28
    assert len(SPEC_TERMS["upper"]) == 6
    assert len(SPEC_TERMS["lower"]) == 6
    assert len(SPEC_TERMS["aux"]) == 10
    assert len(SPEC_TERMS["terrain"]) == 6


def test_by_group_term_index_is_complete_and_unique():
    assert len(BY_GROUP_TERM) == len(ALL_SEMANTICS), "存在 (组,项名) 重复"


# ---------------------------------------------------------------------------
# 松弛标记：只有 TA 三项
# ---------------------------------------------------------------------------

def test_only_ta_terms_are_marked_relaxable():
    """论文 Eq.10 只作用于 lower-body 的 link position / link orientation /
    joint position。标错了会静默放宽不该放宽的目标（尤其是上半身）。"""
    marked = {t.term for t in ALL_SEMANTICS if t.relaxable}
    assert marked == set(RELAXED_TERMS), (
        f"可松弛项与 spec.RELAXED_TERMS 不一致: {marked ^ set(RELAXED_TERMS)}"
    )
    # 且全部属于 lower 组
    assert all(t.group == "lower" for t in ALL_SEMANTICS if t.relaxable)


def test_upper_body_terms_are_never_relaxable():
    """论文原话：upper-body tracking objectives remain strict。"""
    for t in ALL_SEMANTICS:
        if t.group == "upper":
            assert not t.relaxable, f"{t.term} 属上半身，不应可松弛"


def test_lower_body_velocity_terms_are_not_relaxable():
    """Table I 的下身速度项**不带** TA —— 速度跟踪仍严格。"""
    for term in ("link_lin_vel", "link_ang_vel", "joint_vel"):
        s = semantics_for("lower", term)
        assert not s.relaxable, f"lower.{term} 不应可松弛（Table I 无 TA 标记）"


# ---------------------------------------------------------------------------
# 对应关系：标注纪律
# ---------------------------------------------------------------------------

def test_identical_terms_have_a_named_counterpart():
    """标 IDENTICAL 就必须有对应项名（'同项'不能只有结论没有出处）。"""
    for t in ALL_SEMANTICS:
        if t.corr is Corr.IDENTICAL:
            assert t.counterpart, f"{t.group}.{t.term} 标为 identical 但无 counterpart"


def test_pgmt_specific_terms_have_no_counterpart():
    for t in ALL_SEMANTICS:
        if t.corr is Corr.PGMT_SPECIFIC:
            assert t.counterpart is None, (
                f"{t.group}.{t.term} 标为 PGMT_SPECIFIC 却给了 counterpart"
            )


def test_unresolved_terms_must_explain_the_todo():
    """标 UNRESOLVED 的项必须写清待办 —— 不许用空 note 蒙过去。"""
    for t in ALL_SEMANTICS:
        if t.corr is Corr.UNRESOLVED:
            assert t.note.strip(), f"{t.group}.{t.term} 标为 unresolved 但 note 为空"
            assert "待" in t.note or "TODO" in t.note, (
                f"{t.group}.{t.term} 的 note 未说明待办内容"
            )


def test_approx_terms_explain_their_difference():
    """标 APPROX 的项必须说明差异在哪 —— 否则'近似'是无信息的。"""
    for t in ALL_SEMANTICS:
        if t.corr is Corr.APPROX:
            assert t.note.strip(), f"{t.group}.{t.term} 标为 approx 但未说明差异"


def test_counterpart_terms_have_short_names():
    """counterpart 应是参照实现里的项名，不是一句话。"""
    for t in ALL_SEMANTICS:
        if t.counterpart:
            assert len(t.counterpart) < 120, f"{t.group}.{t.term} 的 counterpart 过长"


def test_hedged_notes_must_be_unresolved():
    """**纪律**：note 里承认"未确认/未见/依据不足"的项，不得标为已确认的对应关系。

    这条是本表第一次跑测试时被暴露出来的真实缺陷：`ta_link_ori` 标成
    `PGMT_SPECIFIC`（断言"参照实现没有"）而 note 写的是"未见"；
    `floating_anchor_pos` 标成 `APPROX` 而 note 写的是"未确认到同名项"。
    两者都是**把'没见过'当成'不存在'** —— 存在性断言必须有正面依据。

    `PGMT_SPECIFIC` 与 `APPROX` 都蕴含"我知道参照实现那边是什么样"，
    因此措辞带保留时只能标 `UNRESOLVED`。
    """
    hedges = ("未确认", "未见", "未验证", "依据强度", "依据不足", "未在")
    for t in ALL_SEMANTICS:
        if any(h in t.note for h in hedges):
            assert t.corr is Corr.UNRESOLVED, (
                f"{t.group}.{t.term} 的 note 含保留措辞却标为 {t.corr.value}；"
                f"存在性断言需要有正面依据，否则应标 unresolved"
            )


def test_unresolved_terms_are_listed_for_followup():
    """待办项必须在报告的待办清单里可枚举（数量变化时有人要解释）。"""
    unresolved = sorted(f"{t.group}.{t.term}"
                        for t in ALL_SEMANTICS if t.corr is Corr.UNRESOLVED)
    assert unresolved == [
        "aux.ee_accel_mismatch",
        "aux.floating_anchor_pos",
        "aux.head_torso_impact",
        "lower.ta_link_ori",
    ], f"待办清单变化，需同步报告: {unresolved}"


# ---------------------------------------------------------------------------
# 统计：分布变化有人要解释
# ---------------------------------------------------------------------------

def test_correspondence_distribution_is_summed_and_consistent():
    """对应关系分布：断言**内部一致**，不锁死具体数字。

    最初我把字面量写成 11/8/8/1，实测是 11/9/7/1 —— 我在测试里数错了。
    更重要的是：锁死字面量会鼓励"改测试让它变绿"。改为断言分布的自洽性
    （四类之和 == Table I 项数、四类都非空），把"具体数字是多少"交给
    `test_unresolved_terms_are_listed_for_followup` 与各项的标注纪律去保证。

    当前实测分布（供报告引用，改动本表时需同步）：
        identical 11 / approx 8 / pmgt_specific 5 / unresolved 4（和 = 28）
    """
    counts = correspondence_counts()
    assert sum(counts.values()) == 28, f"各类之和应等于 Table I 项数，得到 {counts}"
    for key, n in counts.items():
        assert n > 0, f"{key} 类为空 —— 某类对应关系被清空，需说明理由: {counts}"
    assert set(counts) == {c.value for c in Corr}


def test_correspondence_counts_are_reported_in_docs():
    """分布一旦变化，README/方案里的数字要跟着改（此处给出权威口径）。

    当前实测：identical 11 / approx 8 / pmgt_specific 5 / unresolved 4（和 = 28）。

    原先这里只断言 `pmgt + approx == 13`，把两类的**和**锁住了、却没锁各自的
    数量 —— 结果 README 写"approx 7 / PGMT-specific 6"、本文件另一处 docstring
    写"approx 7 / pmgt_specific 7 / unresolved 3"，三处互相矛盾且无人发现
    （"和是 13/28"两边都满足）。现在逐类断言，并给出可复现的核对命令：

        python -c "from pgmt.rewards.semantics import correspondence_counts as c; print(c())"
    """
    counts = correspondence_counts()
    assert counts[Corr.IDENTICAL.value] == 11
    assert counts[Corr.APPROX.value] == 8
    assert counts[Corr.PGMT_SPECIFIC.value] == 5
    assert counts[Corr.UNRESOLVED.value] == 4
    assert sum(counts.values()) == 28, "四类之和须等于 Table I 的项数"


def test_terrain_group_is_entirely_second_stage():
    """terrain-contact 组只在 Stage 2 存在（论文 Eq.9 的第四个值头）。"""
    terrain_terms = {t.term for t in ALL_SEMANTICS if t.group == "terrain"}
    assert terrain_terms == {n for n, _ in TERRAIN_TERMS}


def test_semantics_for_unknown_raises():
    with pytest.raises(KeyError, match="未登记语义"):
        semantics_for("upper", "not_a_term")


# ---------------------------------------------------------------------------
# 具体结论的回归：几个关键解读被锁定
# ---------------------------------------------------------------------------

def test_ta_interpretation_is_recorded():
    """`TA` 的解读：即 Eq.10 松弛作用的 lower-body 三项。

    这是本项目对论文缩写的推断，表里必须留痕（否则以后没人知道它从哪来）。
    """
    for term in RELAXED_TERMS:
        s = semantics_for("lower", term)
        assert s.relaxable
        assert "松弛" in s.objective, f"{term} 的 objective 未说明松弛"


def test_corrected_root_vel_is_linked_to_a13():
    """`corrected_root_vel` 必须与 A13（全局位置修正）挂钩，而不是普通速度跟踪。"""
    s = semantics_for("aux", "corrected_root_vel")
    assert "A13" in s.note or "λ_pos" in s.note, "未标明与 A13 的关系"
    assert s.corr is Corr.PGMT_SPECIFIC


def test_touchdown_and_contact_match_cite_the_paper_wording():
    """这两项的语义在论文 §IV-B 有明确表述，note 应引用到。"""
    assert "height variation" in semantics_for("terrain", "touchdown_quality").note.lower() \
        or "局部高度" in semantics_for("terrain", "touchdown_quality").note
    assert "mesh" in semantics_for("terrain", "reference_contact_match").note.lower()
