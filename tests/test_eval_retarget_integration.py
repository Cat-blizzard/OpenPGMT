"""A17 复核口径的集成前提（需真实数据）。

本文件只验证一件事：**评估口径所依赖的"独立性"陈述在真实数据上成立**。

A17 的原始裁定依据是旧 `fk_err_cm` —— 一个"在训练集上测训练损失"的自证指标
（18 个关键点里 13 个是 `ik_refine.FULL_KEYPOINTS` 的优化目标）。重裁的整个
结论都压在替代指标真的独立这一点上，因此这里把三条前提锁死：

  1. `refine_full` **从不改动** `root_pos` / `root_rot`（独立性的前提）
  2. `evaluate` 输出审计后的列集（不再有被拟合点主导的 `abs_pos_err_cm`）
  3. `upright_holdout_err_cm` 的分母是**留出组**，不是全部 17 点
  4. `root_rot_err_deg` 按构造**恒为 ~0** —— 显式锁住，防止它被当成质量证据
     （它一度被标为"完全独立"；独立是真的，**有信息量是假的**）
"""

import os

import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lafan1")
HAS_DATA = os.path.isdir(DATA_DIR) and any(
    f.endswith(".bvh") for f in os.listdir(DATA_DIR))

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="需要 data/raw/lafan1 真实数据")

SEQ = "walk1_subject1"


@pytest.fixture(scope="module")
def refined():
    """返回 `(bvh, refine_full 之前的根姿态快照, refine_full 输出)`。

    ⚠️ 快照必须在 `refine_full` **之前**拷贝：`_finalize` 用浅拷贝把
    `root_pos/root_rot` 原样带过去，所以 `out["root_pos"] is raw["root_pos"]`
    是同一个对象 —— 直接互相比较会恒真，既测不出替换也测不出原地修改。
    """
    from data.bvh import load_bvh
    from data.ik_refine import refine_full
    from data.retarget_lafan1 import retarget

    bvh = load_bvh(os.path.join(DATA_DIR, SEQ + ".bvh"))
    raw = retarget(bvh)
    before = {k: raw[k].copy() for k in ("root_pos", "root_rot")}
    return bvh, before, refine_full(raw, bvh)


@pytest.fixture(scope="module")
def row():
    from data.eval_retarget import evaluate
    return evaluate(SEQ, regenerate=True)


def test_ik_never_modifies_root_pose(refined):
    """**A17 独立性的基石**：`refine_full` 不改动 root_pos / root_rot。

    `root_rot_err_deg` 之所以是本口径里**唯一完全独立**的指标，全靠这一条：
    IK 只优化关节角，根姿态由 `retarget` 冻结。若哪天 IK 开始精修根姿态，
    该项立刻退化为训练残差，A17 的裁定依据随之失效 —— 故与**调用前的快照**
    逐位比较（这样替换与原地修改都能抓到）。

    （另两个"留出"指标只做到"未被拟合"：4 个留出点虽不在目标里，但 IK 优化
    全部 29 个关节、会移动同一条运动链，属泛化检验而非独立测量。）
    """
    _, before, out = refined
    assert np.array_equal(out["root_pos"], before["root_pos"]), "IK 改动了 root_pos"
    assert np.array_equal(out["root_rot"], before["root_rot"]), "IK 改动了 root_rot"


def test_evaluate_reports_the_audited_column_set(row):
    """输出的列必须是审计后的那一套。

    `abs_pos_err_cm` 已被移除：它对全部 17 个非基准点取均值，其中 13 个是 IK
    拟合点，因而会被训练残差主导 —— 却曾被 README 标为"独立"。若它出现在
    结果里，说明审计修正被回退了。
    """
    assert "upright_holdout_err_cm" in row
    assert "abs_pos_err_cm" not in row, "旧列名回来了 —— 它会被 IK 拟合点主导"
    for k in ("fitted_err_cm", "holdout_err_cm", "upright_holdout_err_cm",
              "root_rot_err_deg", "min_foot_z", "spike_pct", "contact_agree",
              "limit_over_pct"):
        assert np.isfinite(row[k]), f"{k} 应为有限值，得到 {row[k]}"


def test_root_rot_metric_is_a_tautology_by_construction(row):
    """`root_rot_err_deg` 按构造恒为 ~0 —— 它是**恒等式，不是质量指标**。

    `retarget()` 里 `root_rot = (qw ⊗ grot_src[Hips]) ⊗ Q_MRIG_INV`，与评估里
    调用的 `source_root_quat_to_g1_base()` 是**同一个式子**，故两者永远相等；
    剩下的 0.01° 只是 `root_rot` 存 float32 的舍入。全量 77 个序列实测均为
    0.01°，**无法区分任何序列**。

    本测试把它明确锁成"应当恒为 0"，使它不可能再被误读为"朝向误差极小、
    所以参考很好"（这正是一度把 README 里的它标成"完全独立/✅"所导致的误读）。
    同时它仍保留一项价值：**跨实现的交叉检查** —— 若有人只改了 `retarget()`
    的内联公式而没同步 helper（或反之），这里会立刻失败。

    真正的质量证据见 `test_upright_metric_uses_holdout_subset_only` 与
    `min_foot_z` 等列。
    """
    assert row["root_rot_err_deg"] < 0.5, (
        f"按构造应恒为 ~0，得到 {row['root_rot_err_deg']}° —— retarget() 的内联"
        f"根朝向公式与 source_root_quat_to_g1_base() 已不一致")


def test_upright_metric_uses_holdout_subset_only(refined, row):
    """手工复算对照：`upright_holdout_err_cm` 必须等于**只用 4 个留出点**的值。

    这里刻意重算一遍"定义"（而不是调用被测代码的内部函数），与 `spec.py`
    的字面量断言同理：口径被改动时必须失败。

    注意：全点均值与留出均值在数值上**可能**接近（若两类点误差恰好相当），
    因此本测试锁的是"用的是哪个子集"这一**定义**，而非两者数值必然不同。
    """
    from data.eval_retarget import (
        HOLDOUT_KPS,
        KEYPOINTS,
        W,
        _mean_err_cm,
        g1_forward_kinematics,
    )

    bvh, _, out = refined
    T = bvh.num_frames
    gpos = bvh.fk(unit_scale=1.0)[0] @ W.T * float(out["scale"])
    upright = g1_forward_kinematics(
        out["qpos"],
        np.tile(gpos[:, bvh.joint_index("Hips")], (1, 1)),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1)),
    )
    errs = {}
    for src, body, kind in KEYPOINTS:
        if kind == "ref":
            continue
        errs[f"{src}->{body}"] = np.linalg.norm(
            gpos[:, bvh.joint_index(src)] - upright[body], axis=-1)

    holdout_only = _mean_err_cm(errs, HOLDOUT_KPS)
    non_ref = [k for k in KEYPOINTS if k[2] != "ref"]
    assert len(HOLDOUT_KPS) < len(non_ref), "留出组与全体非基准点应真的不同"

    assert row["upright_holdout_err_cm"] == pytest.approx(holdout_only, abs=1e-9)
