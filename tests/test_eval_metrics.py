"""质量指标的独立性（A17 复核的前提）。

`data/eval_retarget.py` 的改造目的是让"参考是否可用"有一个**不依赖 IK 优化
目标**的判据。本文件在无数据时也能运行 —— 它校验的是**关键点分组本身**：
拟合组必须与 `ik_refine.FULL_KEYPOINTS` 完全一致，留出组必须与之**零交集**。

若哪天有人往 `KEYPOINTS` 里加了一个恰好在 IK 目标中的关键点却标成 holdout，
这里会失败 —— 那正是 A17 旧裁定失效的根因（13/18 重叠）。
"""

import numpy as np
import pytest

from data.eval_retarget import (
    FITTED_KPS,
    HOLDOUT_KPS,
    KEYPOINTS,
    _mean_err_cm,
    _motion_type,
)


def _fitted_body_pairs() -> set:
    from data.ik_refine import FULL_KEYPOINTS
    return {(src, body) for src, body in FULL_KEYPOINTS}


def test_fitted_group_exactly_matches_ik_optimized_keypoints():
    """拟合组 == FULL_KEYPOINTS（IK 的优化目标），一个不多一个不少。"""
    mine = {(s, g) for s, g, kind in FITTED_KPS}
    assert mine == _fitted_body_pairs(), (
        f"拟合组与 IK 目标不一致:\n  仅在我方: {mine - _fitted_body_pairs()}\n"
        f"  仅在 IK: {_fitted_body_pairs() - mine}"
    )


def test_holdout_group_is_disjoint_from_ik_objective():
    """**核心不变量**：留出组与 IK 优化目标零交集，否则独立性不成立。"""
    holdout = {(s, g) for s, g, kind in HOLDOUT_KPS}
    assert holdout & _fitted_body_pairs() == set(), (
        f"留出组混入了被 IK 优化的关键点: {holdout & _fitted_body_pairs()}"
    )


def test_holdout_group_is_not_empty():
    """留出组不能为空，否则无法提供独立证据。"""
    assert len(HOLDOUT_KPS) >= 2


def test_every_keypoint_has_a_known_kind():
    kinds = {kind for _, _, kind in KEYPOINTS}
    assert kinds <= {"fused", "holdout", "ref"}, f"未知类别: {kinds}"


def test_exactly_one_reference_keypoint():
    """骨盆是唯一的对齐基准，不参与任何误差均值。"""
    assert [k for k in KEYPOINTS if k[2] == "ref"] == [("Hips", "pelvis", "ref")]


def test_no_duplicate_g1_bodies_in_keypoints():
    bodies = [g for _, g, _ in KEYPOINTS]
    assert len(bodies) == len(set(bodies)), "同一 G1 body 不应被匹配两次"


def test_mean_err_cm_returns_nan_for_empty_subset():
    assert np.isnan(_mean_err_cm({}, []))


def test_mean_err_cm_averages_selected_subset_only():
    errs = {
        "A->a": np.array([0.01, 0.01]),   # 1cm
        "B->b": np.array([0.03, 0.03]),   # 3cm
        "C->c": np.array([0.10, 0.10]),   # 10cm（不应被选中）
    }
    got = _mean_err_cm(errs, [("A", "a", "fused"), ("B", "b", "holdout")])
    assert np.isclose(got, 2.0), f"期望 2cm，得到 {got}"


def test_mean_err_cm_skips_missing_entries():
    errs = {"A->a": np.array([0.02, 0.02])}
    got = _mean_err_cm(errs, [("A", "a", "fused"), ("Z", "z", "holdout")])
    assert np.isclose(got, 2.0), "缺失关键点应被跳过，而不是算成 0"


@pytest.mark.parametrize("name,expected", [
    ("ground1_subject1", "ground"),
    ("walk1_subject1", "walk"),
    ("fallAndGetUp2_subject3", "fall"),
    ("pushAndStumble1_subject5", "push"),
    ("obstacles6_subject4", "obstacle"),
    ("multipleActions1_subject1", "multiple"),
    ("fightAndSports1_subject4", "fight"),
    ("unknownType_1", "other"),
])
def test_motion_type_classification(name, expected):
    assert _motion_type(name) == expected
