"""质量指标的独立性（A17 复核的前提）。

`data/eval_retarget.py` 的改造目的是让"参考是否可用"有一个**不依赖 IK 优化
目标**的判据。本文件在无数据时也能运行 —— 它校验的是**关键点分组本身**：
拟合组必须与 `ik_refine.FULL_KEYPOINTS` 完全一致，留出组必须与之**零交集**。

若哪天有人往 `KEYPOINTS` 里加了一个恰好在 IK 目标中的关键点却标成 holdout，
这里会失败 —— 那正是 A17 旧裁定失效的根因（13/18 重叠）。
"""

import os

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


# ---------------------------------------------------------------------------
# 输出路径：`--only` 绝不覆盖完整报告（数据安全）
# ---------------------------------------------------------------------------

def test_resolve_out_path_never_overwrites_full_report():
    """`--only` 必须写子集专用文件，**绝不覆盖**完整报告。

    这是一条数据安全测试。README 推荐用 `--only ground` 做 A17 复核，而改造前
    这条命令会把仓库里 77 行的 `quality_report.csv` 静默改写成 5 行 —— 没有
    任何提示，且文件已在版本控制中，一次误跑就丢掉了全量基线。
    """
    from data.eval_retarget import FULL_REPORT_PATH, resolve_out_path

    # 不带 --only ⇒ 完整报告
    assert resolve_out_path(None, None) == FULL_REPORT_PATH

    # 带 --only 且省略 --out ⇒ 子集专用文件，路径与完整报告不同
    got = resolve_out_path("ground", None)
    assert got == "data/processed/quality_report_ground.csv"
    assert os.path.abspath(got) != os.path.abspath(FULL_REPORT_PATH)

    # 显式 --out 到别处 ⇒ 尊重调用方
    assert resolve_out_path("ground", "tmp/x.csv") == "tmp/x.csv"

    # 显式 --out 指向完整报告 + 有 --only ⇒ 拒绝
    with pytest.raises(ValueError, match="拒绝写入"):
        resolve_out_path("ground", FULL_REPORT_PATH)

    # 不带 --only 时显式写完整报告是合法的（那本来就是全量结果）
    assert resolve_out_path(None, FULL_REPORT_PATH) == FULL_REPORT_PATH


def test_legacy_fk_err_cm_divides_by_18_not_17():
    """旧口径的分母是 18（含骨盆），不是 17 条误差条目。

    骨盆是对齐基准、不参与误差统计，故 `kp_errs` 只有 17 条；但旧口径把它作为
    0 计入了分母。若误用 17，均值会抬高 18/17 ≈ 5.9%，**新旧 CSV 不可比** ——
    而保留这一列的唯一用途正是对比改造前后的数值变化。
    """
    from data.eval_retarget import legacy_fk_err_cm

    assert len(KEYPOINTS) == 18, "本测试按 18 写死；关键点数变化时须同步"

    # 17 条误差、每条 1cm ⇒ 旧口径 17/18 cm（而非 1cm）
    errs = {f"k{i}->b{i}": np.array([0.01]) for i in range(17)}
    assert legacy_fk_err_cm(errs) == pytest.approx(17.0 / 18.0)
    assert legacy_fk_err_cm(errs) != pytest.approx(1.0)

    assert np.isnan(legacy_fk_err_cm({}))
