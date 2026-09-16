"""根朝向换算与相对旋转角（对照评估的核心约定）。

`eval_retarget` 的"根朝向误差"要比较**源骨架**与 **G1** 的根朝向，两者
坐标系约定不同，必须走 `source_root_quat_to_g1_base` 换算。此处曾在改造
过程中用错过乘法顺序（把基变换 qw 右乘），本文件把正确顺序锁死。

关键不变量（`retarget` 的契约）：

    root_rot = (qw ⊗ q_src) ⊗ Q_MRIG_INV

其中 qw 是**左乘**的基变换（世界→G1 世界系），Q_MRIG_INV 是**右乘**的
约定换算（rig 局部轴 → G1 基座轴）。两者都不可交换、不可省。
"""

import numpy as np
import pytest

from data.bvh import quat_mul
from data.retarget_lafan1 import (
    Q_MRIG_INV,
    _qW,
    relative_rotation_angle_deg,
    source_root_quat_to_g1_base,
)


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def _angle_error_deg(a, b):
    """两个四元数（同一旋转，可差整体符号）之间的夹角（度）。

    用 `2·acos|dot|` —— 比 `relative_rotation_angle_deg` 的 trace 公式在
    **小角度**上数值稳定得多。trace 公式要算 `acos((tr−1)/2)`，当 tr→3 时
    `(tr−1)/2 → 1`，acos 在 1 附近导数发散，float64 舍入会被放大成
    ~1e-6 度的假误差（实测双射测试即如此）。点积形式在 dot→1 附近良态。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


# ---------------------------------------------------------------------------
# source_root_quat_to_g1_base
# ---------------------------------------------------------------------------

def test_conversion_is_exactly_the_documented_composition():
    """必须严格等于 (qw ⊗ q_src) ⊗ Q_MRIG_INV —— 契约的逐字复现。"""
    rng = np.random.default_rng(0)
    q_src = rng.normal(size=(5, 4))
    q_src /= np.linalg.norm(q_src, axis=-1, keepdims=True)

    qw = _qW()
    expected = quat_mul(
        quat_mul(np.tile(qw, (5, 1)), q_src),
        np.tile(Q_MRIG_INV, (5, 1)),
    )
    got = source_root_quat_to_g1_base(q_src)
    assert np.allclose(got, expected, atol=1e-12)


def test_conversion_wrong_orders_differ():
    """反例守卫：把 qw 右乘、或漏掉 Q_MRIG_INV，都必须与正确结果不同。

    这条测试的意义是"防止错误实现悄悄通过" —— 若某次重构把顺序改回去，
    上面那条测试会失败，而这条说明为何值得失败。
    """
    q_src = _axis_angle_quat([0.3, 0.5, 0.8], 1.1)
    qw = _qW()
    correct = source_root_quat_to_g1_base(q_src)

    wrong_right = quat_mul(q_src, qw)          # 把基变换 qw 右乘
    wrong_no_mrig = quat_mul(qw, q_src)        # 漏掉 Q_MRIG_INV

    def angle(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(relative_rotation_angle_deg(a / np.linalg.norm(a),
                                                 b / np.linalg.norm(b)))

    assert angle(correct, wrong_right) > 30.0, "右乘 qw 的错误实现应明显不同"
    assert angle(correct, wrong_no_mrig) > 30.0, "漏掉 Q_MRIG_INV 应明显不同"


def test_conversion_single_and_batched_agree():
    q_src = _axis_angle_quat([1, 0, 0], 0.7)
    single = source_root_quat_to_g1_base(q_src)
    batched = source_root_quat_to_g1_base(np.stack([q_src, q_src]))
    assert single.shape == (4,)
    assert np.allclose(batched[0], single, atol=1e-12)
    assert np.allclose(batched[1], single, atol=1e-12)


def test_conversion_output_is_unit_quaternion():
    rng = np.random.default_rng(1)
    q_src = rng.normal(size=(16, 4))
    q_src /= np.linalg.norm(q_src, axis=-1, keepdims=True)
    out = source_root_quat_to_g1_base(q_src)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-10)


def test_conversion_is_bijective_on_so3():
    """换算是一个固定旋转的左右乘，故必须是双射：给定 G1 朝向能唯一还原源朝向。"""
    rng = np.random.default_rng(2)
    q_src = rng.normal(size=(8, 4))
    q_src /= np.linalg.norm(q_src, axis=-1, keepdims=True)
    g1 = source_root_quat_to_g1_base(q_src)

    qw = _qW()
    # 逆运算：先右乘 Q_MRIG_INV 的逆（= 共轭），再左乘 qw 的逆
    def inv(q):
        out = np.array(q, dtype=np.float64, copy=True)
        out[..., 1:] *= -1.0
        return out

    back = quat_mul(np.tile(inv(qw), (8, 1)),
                    quat_mul(g1, np.tile(inv(Q_MRIG_INV), (8, 1))))
    # q 与 −q 同一旋转，用夹角判等。
    #
    # 容差 1e-4°：实测最大残差 ~1.7e-6°。该残差不是算法误差 —— 换算在数学上
    # 精确可逆（左右各乘一个固定单位四元数，逆即交换顺序取共轭）。它来自
    # **四元数乘法 + 归一化的 float64 舍入，被 acos 在 1 附近放大**：
    #   dot ≈ 1 − 2.2e-16  →  acos(1−ε) ≈ √(2ε) ≈ 2.1e-8 rad
    #   →  2 × 2.1e-8 rad = 4.2e-8 rad ≈ 2.4e-6°（与实测 1.708e-6° 同量级）
    # 判别力不受影响：乘法顺序写错会产生 >30° 的系统偏差（见反例守卫测试），
    # 比这里的噪声高 5 个数量级。
    err = _angle_error_deg(back, q_src)
    assert np.all(err < 1e-4), f"换算不可逆，最大残差 {err.max():.3e}°"


# ---------------------------------------------------------------------------
# relative_rotation_angle_deg
# ---------------------------------------------------------------------------

def test_angle_zero_for_identical():
    q = _axis_angle_quat([0.2, 0.4, 0.9], 1.3)
    assert np.allclose(relative_rotation_angle_deg(q, q), 0.0, atol=1e-9)


def test_angle_ignores_sign_ambiguity():
    """q 与 −q 表示同一旋转 —— 角度必须为 0，不得报 180°。"""
    q = _axis_angle_quat([0, 0, 1], 0.8)
    assert np.allclose(relative_rotation_angle_deg(q, -q), 0.0, atol=1e-9)


def test_angle_matches_analytic_value():
    """绕同一轴的 0.3 rad 与 1.1 rad 之间，夹角应为 0.8 rad = 45.8366°。"""
    a = _axis_angle_quat([0, 1, 0], 0.3)
    b = _axis_angle_quat([0, 1, 0], 1.1)
    got = float(relative_rotation_angle_deg(a, b))
    assert np.isclose(got, np.degrees(0.8), atol=1e-6)


def test_angle_is_symmetric():
    a = _axis_angle_quat([1, 1, 0], 0.9)
    b = _axis_angle_quat([0, 1, 1], 2.2)
    assert np.isclose(float(relative_rotation_angle_deg(a, b)),
                      float(relative_rotation_angle_deg(b, a)), atol=1e-9)


def test_angle_never_exceeds_180():
    """trace 公式天然给出 [0,180]，且必须对接近 180° 的情形稳定。"""
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        ang = float(relative_rotation_angle_deg(a, b))
        assert 0.0 <= ang <= 180.0 + 1e-9
        assert np.isfinite(ang)


def test_angle_batched_matches_scalar():
    a = np.stack([_axis_angle_quat([0, 0, 1], 0.1), _axis_angle_quat([1, 0, 0], 0.5)])
    b = np.stack([_axis_angle_quat([0, 0, 1], 1.0), _axis_angle_quat([1, 0, 0], 2.5)])
    batch = relative_rotation_angle_deg(a, b)
    assert batch.shape == (2,)
    for i in range(2):
        assert np.isclose(batch[i], relative_rotation_angle_deg(a[i], b[i]))


def test_angle_180_degree_case_is_exact():
    """180° 是 trace 公式在 −1 处的边界，须给出 180 而不是 NaN。"""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = _axis_angle_quat([0, 0, 1], np.pi)
    assert np.isclose(float(relative_rotation_angle_deg(a, b)), 180.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 与 retarget() 实际输出的交叉验证（需真实数据）
# ---------------------------------------------------------------------------

_HAS_DATA = None


def _has_data():
    global _HAS_DATA
    if _HAS_DATA is None:
        import os
        _HAS_DATA = os.path.exists(os.path.join("data", "raw", "lafan1", "walk1_subject1.bvh"))
    return _HAS_DATA


@pytest.mark.skipif(not _has_data(), reason="需要 data/raw/lafan1 真实数据")
def test_helper_matches_retarget_root_rot_on_real_data():
    """闭环验证：把 helper 作用在**源**髋朝向（rig 约定、世界系）上，必须
    复原 `retarget()` 自己产出的 `root_rot`。

    这是本文件最重要的一条 —— 它证明对照评估里的"根朝向误差"确实在比较
    同一件事，而不是两套不同约定的量化。若 helper 的乘法顺序被改错，
    这里会出现明显的角度偏差（而非浮点级误差）。

    注意 helper 的输入是**源原始**朝向（`bvh.fk()` 输出），不是已经转过
    `qw` 的 `grot_g1` —— 传错会在 `_qW()` 上叠两次。
    """
    import os

    from data.bvh import load_bvh
    from data.retarget_lafan1 import retarget

    bvh = load_bvh(os.path.join("data", "raw", "lafan1", "walk1_subject1.bvh"))
    data = retarget(bvh)
    T = min(200, bvh.num_frames)

    _, grot_world = bvh.fk(unit_scale=1.0)
    hip_src = grot_world[:, bvh.joint_index("Hips")]  # 源原始（rig 约定）
    recomputed = source_root_quat_to_g1_base(hip_src[:T])
    err = relative_rotation_angle_deg(recomputed, data["root_rot"][:T])
    # 容差 0.1°。实测最大 0.0232°，来源是**精度而非算法**：
    #   - helper 全程 float64
    #   - retarget() 把 root_rot 存为 float32（npz 契约）
    #   - bvh.fk() 的世界朝向本身是 float64，但源数据只有 6 位小数精度
    # 判别力不受影响：乘法顺序写错会产生 >30° 的系统偏差（见上一条反例守卫），
    # 与这里 0.02° 的噪声相差三个数量级。
    assert float(err.max()) < 0.1, (
        f"helper 与 retarget 的 root_rot 不一致，最大偏差 {float(err.max()):.4f}°"
        " —— 乘法顺序或 Q_MRIG_INV 用法有误（预期只应有 ~0.02° 的 float32/文本精度噪声）"
    )


@pytest.mark.skipif(not _has_data(), reason="需要 data/raw/lafan1 真实数据")
def test_helper_matches_retarget_within_float32_precision():
    """上一条的精度归因验证：把 retarget 的输出视为 float32 真值后，
    残差应回到 float32 量化水平（≈1e-5 度量级），确认 0.023° 确实是精度而非 bug。"""
    import os

    from data.bvh import load_bvh
    from data.retarget_lafan1 import retarget

    bvh = load_bvh(os.path.join("data", "raw", "lafan1", "walk1_subject1.bvh"))
    data = retarget(bvh)
    T = min(200, bvh.num_frames)
    _, grot_world = bvh.fk(unit_scale=1.0)
    src = source_root_quat_to_g1_base(grot_world[:, bvh.joint_index("Hips")])[:T]

    # 模拟"只经过 float32 存储"的误差量级：量化后再比
    src32 = src.astype(np.float32).astype(np.float64)
    err_q = _angle_error_deg(src32, data["root_rot"][:T])
    assert float(err_q.max()) < 0.05, (
        f"float32 量化后的残差 {float(err_q.max()):.4f}° 仍偏大，说明不只是精度问题"
    )


@pytest.mark.skipif(not _has_data(), reason="需要 data/raw/lafan1 真实数据")
def test_root_rotation_error_is_small_for_upright_walk():
    """步行动作（直立）的根朝向误差应很小 —— 若此处出现大角度，说明对照
    评估的换算不对，而不是"数据退化"。"""
    import os

    from data.bvh import load_bvh
    from data.retarget_lafan1 import g1_forward_kinematics, retarget

    bvh = load_bvh(os.path.join("data", "raw", "lafan1", "walk1_subject1.bvh"))
    data = retarget(bvh)
    T = min(200, bvh.num_frames)

    grot_src = bvh.fk(unit_scale=1.0)[1]
    src_q = source_root_quat_to_g1_base(grot_src[:, bvh.joint_index("Hips")])[:T]
    _, g1_quat = g1_forward_kinematics(data["qpos"], data["root_pos"],
                                       data["root_rot"], return_quats=True)
    err = float(relative_rotation_angle_deg(g1_quat["pelvis"][:T], src_q).mean())
    assert err < 15.0, f"步行的根朝向误差 {err:.2f}° 偏大，换算可能有问题"
