"""旋转表示工具（观测/参考帧契约层，M2 环境与策略共用）。

提供两件论文观测格式必需、但此前缺失的转换：

1. **旋转矩阵 ↔ 6D 表示**（`rotmat_to_6d` / `rot6d_to_rotmat`）
   论文 §III 的 `o_t` 第一项是 `e_t ∈ R^6`（参考相对 anchor 朝向，6D 旋转），
   `H_t` 的每一帧同理。6D 表示取自 Zhou et al. 2019 "On the Continuity of
   Rotation Representations in Neural Networks"：取旋转矩阵前两列后做
   Gram-Schmidt 正交化。相比四元数/欧拉角，它在 SO(3) 上连续、无万向节，
   对回归友好（也是 tracking 类工作的常见选择）。

2. **四元数最短弧插值**（`quat_slerp`）
   LAFAN1 是 30 fps 而控制频率 50 Hz，参考锚点朝向需要小数帧插值。
   四元数**不能**直接线性插值（q 与 −q 表示同一旋转，符号翻转处会跳出
   360° 假旋转）；必须对齐半球后按最短弧插值。

约定：四元数一律 `[w, x, y, z]`，且与 `data/bvh.py` 的 `quat_mul` /
`quat_rot_vec` / `quat_to_mat` 保持同一约定。
"""

from __future__ import annotations

import numpy as np

from data.bvh import quat_mul

_EPS = 1e-8


# ---------------------------------------------------------------------------
# 四元数 ↔ 旋转矩阵
# ---------------------------------------------------------------------------

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] 四元数 → 旋转矩阵。支持 (...,4) → (...,3,3)。"""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray | float) -> np.ndarray:
    """四元数最短弧球面插值（含半球对齐与线性退化分支）。

    Args:
        q0, q1: (...,4) 单位四元数（可批量）
        w:      插值权重，0 → q0，1 → q1；标量或 (...,) 广播
    Returns:
        (...,4) 单位四元数

    实现要点：
      - **半球对齐**：若 dot(q0,q1) < 0 则把 q1 取反，保证走最短弧
        （否则会沿 360° 反向路径旋转，产生假的大幅转动）。
      - **近距离分支**：dot → ±1 时 sin(θ) → 0，slerp 公式数值失稳；
        改用归一化线性插值（NLERP），在 θ 极小时与 slerp 的差异为 O(θ³)，
        远低于观测噪声水平。
      - **符号选择**：插值结果与 q0 同半球（dot ≥ 0），避免逐帧符号翻转
        污染差分得到的角速度。
    """
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    w = np.asarray(w, dtype=np.float64)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # 半球对齐：整体翻转到 q0 所在的半球
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    # 近距离：NLERP
    near = dot > (1.0 - 1e-6)
    lerp = q0 * (1.0 - w[..., None]) + q1 * w[..., None]
    lerp = normalize_quat(lerp)

    # 一般情形：SLERP
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - w[..., None]) * theta) / np.maximum(sin_theta, _EPS)
    b = np.sin(w[..., None] * theta) / np.maximum(sin_theta, _EPS)
    slerp = normalize_quat(a * q0 + b * q1)

    out = np.where(near, lerp, slerp)
    # 与 q0 同半球
    flip = np.sum(out * q0, axis=-1, keepdims=True) < 0.0
    return np.where(flip, -out, out)


# ---------------------------------------------------------------------------
# 6D 旋转表示（Zhou et al. 2019）
# ---------------------------------------------------------------------------

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 6D 表示（前两列 + Gram-Schmidt 正交化）。

    Args:
        R: (...,3,3)
    Returns:
        (...,6) = [a1(3), a2(3)]
    """
    R = np.asarray(R, dtype=np.float64)
    a1 = R[..., :, 0]
    a2 = R[..., :, 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), _EPS)
    return np.concatenate([b1, b2], axis=-1)


def rot6d_to_rotmat(d6: np.ndarray) -> np.ndarray:
    """6D 表示 → 旋转矩阵（Gram-Schmidt 重建，第三列取叉积保证 det=+1）。

    已知退化情形：前两列线性相关（重建结果无法确定一个旋转）时，第二基
    向量归一化分母→0，输出方向无意义。这是 6D 表示本身的奇异性，而非实现
    缺陷；用 `_EPS` 兜底避免 NaN。实践中网络输出与参考朝向都不会落入该点。
    """
    d6 = np.asarray(d6, dtype=np.float64)
    if d6.shape[-1] != 6:
        raise ValueError(f"6D 表示末维必须为 6，得到 {d6.shape}")
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), _EPS)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def relative_anchor_6d(robot_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    """参考相对 anchor 朝向 `e_t ∈ R^6`（论文 §III 的 `o_t` 第一项）。

    `e_t` = 参考锚点朝向相对机器人锚点朝向的偏差，表达在机器人锚点系：

        R_rel = R_ref^T · R_robot      （列向量约定下的"机器人系下的参考朝向"）

    即 `R_robot = R_ref · R_rel`。取 6D 表示后作为观测，使策略在
    root-centric 表述下只关心相对朝向（与论文"减少对绝对全局朝向的依赖"
    一致），而不是世界系绝对方位。

    Args:
        robot_quat: 机器人锚点（基座）朝向 (...,4)，[w,x,y,z]
        ref_quat:   参考锚点朝向     (...,4)
    Returns:
        (...,6)

    **会先归一化输入**。非单位四元数不会报错 —— `quat_to_mat` 对未归一化
    输入给出的是带缩放的矩阵（‖q‖² 倍），而 `rotmat_to_6d` 的 Gram-Schmidt
    会把缩放归一化掉，于是结果"看起来正常"但实际是错的。真实来源包括
    slerp 的浮点残差、真机状态估计、以及手写的常量四元数。此处显式归一化，
    代价可忽略（只在拼观测时调用一次）。
    """
    R_robot = quat_to_mat(normalize_quat(robot_quat))
    R_ref = quat_to_mat(normalize_quat(ref_quat))
    R_rel = np.swapaxes(R_ref, -1, -2) @ R_robot
    return rotmat_to_6d(R_rel)
