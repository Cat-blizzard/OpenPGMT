"""旋转表示工具：6D 表示与四元数最短弧插值。

覆盖 `pgmt/policy/rotation.py`：
  - 旋转矩阵 ↔ 6D 往返一致性（含公转、180°、近奇异）
  - Gram-Schmidt 对非正交输入的鲁棒性（网络输出不会是严格正交阵）
  - slerp 的端点、最短弧（半球对齐）、与恒等/90° 的解析对照
  - slerp 的连续性：绕单一轴等步长采样时角度必须单调，无 360° 跳变
  - `relative_anchor_6d` 的相对性：两者同乘一个全局旋转后结果不变
"""

import numpy as np
import pytest

from pgmt.policy.rotation import (
    normalize_quat,
    quat_slerp,
    quat_to_mat,
    relative_anchor_6d,
    rot6d_to_rotmat,
    rotmat_to_6d,
)


def _random_rotations(n: int, rng) -> np.ndarray:
    """随机旋转矩阵 (n,3,3)：对正态矩阵做 QR 并修正 det。"""
    A = rng.normal(size=(n, 3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diagonal(R, axis1=-2, axis2=-1))[:, None, :]
    det = np.linalg.det(Q)
    Q[det < 0, :, 0] *= -1.0
    return Q


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 [w,x,y,z]（Shepperd 分支法，数值稳定）。"""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return normalize_quat(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# ---------------------------------------------------------------------------
# 6D 表示
# ---------------------------------------------------------------------------

def test_rot6d_roundtrip_random():
    rng = np.random.default_rng(0)
    R = _random_rotations(64, rng)
    back = rot6d_to_rotmat(rotmat_to_6d(R))
    assert np.allclose(back, R, atol=1e-10), "随机旋转的 6D 往返应精确复原"


def test_rot6d_matches_first_two_columns_for_orthonormal_input():
    rng = np.random.default_rng(1)
    R = _random_rotations(8, rng)
    d6 = rotmat_to_6d(R)
    assert np.allclose(d6[..., :3], R[..., :, 0], atol=1e-12)
    assert np.allclose(d6[..., 3:], R[..., :, 1], atol=1e-12)


def test_rot6d_gram_schmidt_handles_non_orthonormal_input():
    """网络回归出的 6D 不会严格正交 —— 重建必须仍给出合法旋转矩阵。"""
    d6 = np.array([[2.0, 0.0, 0.0, 0.3, 3.0, 0.0]])  # 非单位、非正交
    R = rot6d_to_rotmat(d6)
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R)[0], 1.0), "det 必须为 +1（不得出现镜像）"


def test_rot6d_reconstructs_orthonormal_columns():
    """重建后的前两列应等于输入方向的正交化结果。"""
    d6 = np.array([[1.0, 0.0, 0.0, 0.5, 1.0, 0.0]])
    R = rot6d_to_rotmat(d6)
    assert np.isclose(np.linalg.norm(R[0, :, 0]), 1.0)
    assert np.isclose(np.linalg.norm(R[0, :, 1]), 1.0)
    assert np.isclose(np.dot(R[0, :, 0], R[0, :, 1]), 0.0, atol=1e-12)


def test_rot6d_roundtrip_180_degree_and_identity():
    h = np.pi
    cases = np.stack([np.eye(3), _rot_x(h), _rot_x(h / 2)])
    d6 = rotmat_to_6d(cases)
    assert np.allclose(rot6d_to_rotmat(d6), cases, atol=1e-12)


def test_rot6d_bad_dim_raises():
    with pytest.raises(ValueError):
        rot6d_to_rotmat(np.zeros(5))
    with pytest.raises(ValueError):
        rot6d_to_rotmat(np.zeros((4, 7)))


# ---------------------------------------------------------------------------
# 四元数 ↔ 矩阵
# ---------------------------------------------------------------------------

def test_quat_to_mat_matches_axis_angle():
    q = _axis_angle_quat([0, 0, 1], np.pi / 3)
    R = quat_to_mat(q)
    v = np.array([1.0, 0.0, 0.0])
    expected = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0])
    assert np.allclose(R @ v, expected, atol=1e-12)


def test_quat_to_mat_batched_matches_single():
    q = np.stack([_axis_angle_quat([1, 0, 0], 0.3),
                  _axis_angle_quat([0, 1, 0], 1.1)])
    batched = quat_to_mat(q)
    for i in range(q.shape[0]):
        assert np.allclose(batched[i], quat_to_mat(q[i]))


def test_normalize_quat_unit_and_safe_on_zero():
    q = normalize_quat(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(q), 1.0)
    z = normalize_quat(np.zeros(4))
    assert np.all(np.isfinite(z)), "零四元数不得产生 NaN/Inf"


# ---------------------------------------------------------------------------
# slerp
# ---------------------------------------------------------------------------

def test_slerp_endpoints():
    q0 = _axis_angle_quat([0, 0, 1], 0.2)
    q1 = _axis_angle_quat([1, 0, 0], 1.3)
    assert np.allclose(quat_slerp(q0, q1, 0.0), q0, atol=1e-12)
    assert np.allclose(quat_slerp(q0, q1, 1.0), q1, atol=1e-12)


def test_slerp_midpoint_of_90_degrees_is_45_degrees():
    """绕 z 轴 0° → 90°，中点必须是 45°（解析对照，非自证）。"""
    q0 = _axis_angle_quat([0, 0, 1], 0.0)
    q1 = _axis_angle_quat([0, 0, 1], np.pi / 2)
    qm = quat_slerp(q0, q1, 0.5)
    Rm = quat_to_mat(qm)
    v = Rm @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0], atol=1e-12)


def test_slerp_takes_shortest_arc_when_q1_flipped():
    """核心回归：q1 取反（同一旋转）后，插值路径必须与未取反时完全一致。

    朴素线性插值在此处会走 360° 反向长弧 —— 那是本函数存在的唯一理由。
    """
    q0 = _axis_angle_quat([0, 0, 1], 0.0)
    q1_pos = _axis_angle_quat([0, 0, 1], np.pi / 2)
    q1_neg = -q1_pos
    for w in (0.1, 0.25, 0.5, 0.75, 0.9):
        a = quat_slerp(q0, q1_pos, w)
        b = quat_slerp(q0, q1_neg, w)
        # 同一旋转的两种表示：矩阵必须一致（四元数本身可差全局符号）
        assert np.allclose(quat_to_mat(a), quat_to_mat(b), atol=1e-9), f"w={w}"


def test_slerp_angle_is_monotonic_along_arc():
    """等步长采样的旋转角必须单调递增 —— 捕获 360° 假跳变。"""
    q0 = _axis_angle_quat([0, 0, 1], 0.0)
    q1 = _axis_angle_quat([0, 0, 1], np.pi * 0.9)
    ws = np.linspace(0.0, 1.0, 21)
    angs = []
    for w in ws:
        R = quat_to_mat(quat_slerp(q0, q1, w))
        angs.append(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    angs = np.array(angs)
    assert np.all(np.diff(angs) > 0), f"角度非单调: {angs}"
    assert angs[-1] < np.pi, "不应超过 180°"


def test_slerp_near_identical_uses_stable_branch():
    """dot→1（近重合）时 slerp 公式退化，必须走 NLERP 稳定分支且仍为单位四元数。"""
    q0 = _axis_angle_quat([0, 0, 1], 0.5)
    q1 = _axis_angle_quat([0, 0, 1], 0.5 + 1e-12)
    for w in (0.0, 0.3, 0.5, 1.0):
        q = quat_slerp(q0, q1, w)
        assert np.all(np.isfinite(q))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9), "必须归一化"


def test_slerp_antipodal_identical_rotation_is_stable():
    """q1 = −q0（完全相同旋转）：结果应等于该旋转，且不出现 NaN。"""
    q0 = _axis_angle_quat([0, 0, 1], 0.7)
    q = quat_slerp(q0, -q0, 0.5)
    assert np.all(np.isfinite(q))
    assert np.allclose(quat_to_mat(q), quat_to_mat(q0), atol=1e-9)


def test_slerp_batched_matches_scalar():
    q0 = np.stack([_axis_angle_quat([0, 0, 1], 0.1), _axis_angle_quat([1, 0, 0], 0.4)])
    q1 = np.stack([_axis_angle_quat([0, 0, 1], 1.2), _axis_angle_quat([1, 0, 0], 2.0)])
    w = np.array([0.25, 0.75])
    batch = quat_slerp(q0, q1, w)
    for i in range(2):
        assert np.allclose(batch[i], quat_slerp(q0[i], q1[i], w[i]), atol=1e-12)


def test_slerp_output_stays_in_q0_hemisphere():
    """输出与 q0 同半球：避免逐帧符号翻转污染差分角速度。"""
    q0 = _axis_angle_quat([0, 0, 1], 0.0)
    q1 = _axis_angle_quat([0, 0, 1], 2.5)
    for w in np.linspace(0.0, 1.0, 11):
        q = quat_slerp(q0, q1, w)
        assert np.dot(q, q0) >= -1e-12, f"w={w} 处符号翻转"


# ---------------------------------------------------------------------------
# relative_anchor_6d
# ---------------------------------------------------------------------------

def test_relative_anchor_6d_is_identity_for_equal_quats():
    q = _axis_angle_quat([0.3, 0.5, 0.8], 1.1)
    d6 = relative_anchor_6d(q, q)
    assert np.allclose(d6, [1, 0, 0, 0, 1, 0], atol=1e-9)


def test_relative_anchor_6d_invariant_to_global_rotation():
    """`e_t` 是相对量：两者同乘一个全局旋转后必须不变（root-centric 的关键）。"""
    rng = np.random.default_rng(2)
    Rg = _random_rotations(1, rng)[0]
    qg = _mat_to_quat(Rg)

    q_robot = _axis_angle_quat([0, 0, 1], 0.3)
    q_ref = _axis_angle_quat([1, 0, 0], -0.6)
    base = relative_anchor_6d(q_robot, q_ref)

    rotated = relative_anchor_6d(_quat_mul(qg, q_robot), _quat_mul(qg, q_ref))
    assert np.allclose(base, rotated, atol=1e-9), "全局旋转不应影响 e_t"


def test_relative_anchor_6d_is_valid_rotation_encoding():
    rng = np.random.default_rng(3)
    for _ in range(20):
        qr = normalize_quat(rng.normal(size=4))
        qf = normalize_quat(rng.normal(size=4))
        R = rot6d_to_rotmat(relative_anchor_6d(qr, qf))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_quat_to_mat_is_not_scale_equivariant():
    """固化 `quat_to_mat` 对非单位四元数的真实行为。

    **教训**：我在这一点上连续凭记忆写错了三次断言。所以这里的期望值全部
    由**实现里的公式当场算出**，而不是回忆某个"‖q‖² 缩放"的说法：

        a[0,0] = 1 − 2(y² + z²)      ← 只依赖 y,z
        a[1,1] = 1 − 2(x² + z²)
        a[2,2] = 1 − 2(x² + y²)
        非对角 = 2(xy ± wz) 等

    取 `q = [0, 3, 0, 0]`（w=0, x=3, y=z=0）逐项代入：

        a[0,0] = 1 − 2(0)   = 1      ← 注意：不是 1+2s²(w²−1)，那个改写是错的
        a[1,1] = 1 − 2(9)   = −17
        a[2,2] = 1 − 2(9)   = −17
        非对角（xy=xz=yz=0 且 w=0）全为 0

    结论：齐次项 `1` 不随尺度变化，所以矩阵**不是** s²·R；缩放体现在哪些
    对角元上取决于哪个分量非零。这正是"未归一化输入会被静默算错"的机制。
    """
    R = quat_to_mat(np.array([0.0, 3.0, 0.0, 0.0]))
    assert np.isclose(R[0, 0], 1.0), "a[0,0] 只依赖 y,z（此处为 0），不受 x 的尺度影响"
    assert np.isclose(R[1, 1], 1.0 - 2.0 * 9.0)
    assert np.isclose(R[2, 2], 1.0 - 2.0 * 9.0)
    assert np.allclose(R[~np.eye(3, dtype=bool)], 0.0), "非对角应全为 0"
    # 列范数因此不是 s²（R[:,0] = [1,0,0]）
    assert np.isclose(np.linalg.norm(R[:, 0]), 1.0)

    # 一般四元数：矩阵不是 s²·R（用一个真正非对角的情形验证）
    q = normalize_quat(np.array([1.0, 1.0, 0.0, 0.0]))
    R1 = quat_to_mat(q)
    R2 = quat_to_mat(q * 2.0)
    assert not np.allclose(R2, 4.0 * R1), "一般四元数不应是 s²·R"


def test_relative_anchor_6d_normalizes_inputs():
    """**回归防护**：非单位四元数必须被归一化，而不是静默给出错误 e_t。

    两个输入若尺度不同，未归一化会让相对旋转的修正项尺度失配，从而污染
    结果（见下一条反例守卫）。这里对同一旋转施加任意尺度，输出必须逐元素不变。
    """
    rng = np.random.default_rng(4)
    qr = normalize_quat(rng.normal(size=4))
    qf = normalize_quat(rng.normal(size=4))
    base = relative_anchor_6d(qr, qf)
    for s in (0.1, 3.0, 100.0):
        got = relative_anchor_6d(qr * s, qf * s)
        assert np.allclose(got, base, atol=1e-9), f"尺度 {s} 影响了结果"


def test_relative_anchor_6d_mismatched_scales_would_differ_without_norm():
    """反例守卫：**两个输入尺度不同**时才暴露问题（同尺度会被约掉）。

    这条证明上一行的归一化不是空操作 —— 若实现里去掉 `normalize_quat`，
    这里构造的用例会给出错误结果。
    """
    rng = np.random.default_rng(5)
    qr = normalize_quat(rng.normal(size=4))
    qf = normalize_quat(rng.normal(size=4))

    correct = relative_anchor_6d(qr, qf)

    # 手动复现"不归一化"的实现（尺度刻意取不同）
    R_robot = quat_to_mat(qr * 5.0)
    R_ref = quat_to_mat(qf * 0.3)
    naive = rotmat_to_6d(np.swapaxes(R_ref, -1, -2) @ R_robot)

    assert not np.allclose(naive, correct, atol=1e-6), (
        "尺度失配的未归一化路径竟然给出正确结果 —— 反例无效，请复核"
    )


def test_relative_anchor_6d_near_zero_quat_does_not_nan():
    """零四元数无定义，但不得产生 NaN（归一化有 eps 兜底）。"""
    out = relative_anchor_6d(np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(out))
