"""LAFAN1（BVH）→ Unitree G1 29-DoF 重定向（假设 A14）。

约定：源为 cm、30 fps、y-up；世界变换 W 把源 x/y/z 映射到 G1 −x/z/y。
根朝向另用右乘 Q_MRIG_INV，把源 rig 局部轴换成 G1 基座轴。

管线：
  1. BVH 解析与 FK，按 P90 腿骨链长估算 m/cm 缩放。
  2. 以固定 M_RIG 和 G1 静止链基框架对齐源关节相对旋转。
  3. 除腰部外用旋转均值去绑定偏移，再按链做欧拉分解、奇异区保护。
     中间帧静止倾斜采用近似；IK 精修补偿位置误差。
  4. 根位置做序列 xy 居中与高度重锚。高度是启发式，躺倒/跨障需要复核。
  5. 29 个有限关节投影到机械限位，速度直接差分导出坐标，不用 wrap/unwrap。
  6. 接触标签用足速、序列高度阈值与中值滤波；批量导出默认再做全链 IK。

未映射：Neck/Head、Toe 独立自由度、手指、部分脊柱自由度。
输出四元数为 [w,x,y,z]；root_pos 是 G1 世界坐标，不是随动锚点坐标。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from pgmt.cfg.assumptions import get
from data.bvh import BVH, load_bvh, quat_inv, quat_mul, quat_rot_vec, quat_to_mat

# ---------------------------------------------------------------------------
# G1 目标定义（源自 unitree_rl_gym/resources/robots/g1_description/
# g1_29dof_rev_1_0.xml）
# ---------------------------------------------------------------------------

# 关节顺序 = XML 定义顺序（q^r ∈ R^29 的规范顺序，环境层须对齐）
G1_JOINT_NAMES: List[str] = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

G1_PELVIS_HEIGHT = 0.793  # 米（XML pelvis pos）

# G1 关节限位（rad，源自 g1_29dof_rev_1_0.xml；右侧取镜像范围）
G1_JOINT_LIMITS = {
    "left_hip_pitch": (-2.5307, 2.8798), "left_hip_roll": (-0.5236, 2.9671),
    "left_hip_yaw": (-2.7576, 2.7576), "left_knee": (-0.087267, 2.8798),
    "left_ankle_pitch": (-0.87267, 0.5236), "left_ankle_roll": (-0.2618, 0.2618),
    "right_hip_pitch": (-2.5307, 2.8798), "right_hip_roll": (-2.9671, 0.5236),
    "right_hip_yaw": (-2.7576, 2.7576), "right_knee": (-0.087267, 2.8798),
    "right_ankle_pitch": (-0.87267, 0.5236), "right_ankle_roll": (-0.2618, 0.2618),
    "waist_yaw": (-2.618, 2.618), "waist_roll": (-0.52, 0.52), "waist_pitch": (-0.52, 0.52),
    "left_shoulder_pitch": (-3.0892, 2.6704), "left_shoulder_roll": (-1.5882, 2.2515),
    "left_shoulder_yaw": (-2.618, 2.618), "left_elbow": (-1.0472, 2.0944),
    "left_wrist_roll": (-1.97222, 1.97222), "left_wrist_pitch": (-1.61443, 1.61443),
    "left_wrist_yaw": (-1.61443, 1.61443),
    "right_shoulder_pitch": (-3.0892, 2.6704), "right_shoulder_roll": (-2.2515, 1.5882),
    "right_shoulder_yaw": (-2.618, 2.618), "right_elbow": (-1.0472, 2.0944),
    "right_wrist_roll": (-1.97222, 1.97222), "right_wrist_pitch": (-1.61443, 1.61443),
    "right_wrist_yaw": (-1.61443, 1.61443),
}


# G1 静止骨架（左半 + 骨盆/腰/躯干；右半镜像生成）。
# body: (parent, pos, quat[w,x,y,z])，取自 XML body pos/quat。
_G1_REST_LEFT: Dict[str, Tuple[Optional[str], np.ndarray, np.ndarray]] = {
    "pelvis": (None, np.array([0.0, 0.0, 0.793]), np.array([1.0, 0.0, 0.0, 0.0])),
    "left_hip_pitch_link": ("pelvis", np.array([0.0, 0.064452, -0.1027]), np.array([1.0, 0.0, 0.0, 0.0])),
    "left_hip_roll_link": ("left_hip_pitch_link", np.array([0.0, 0.052, -0.030465]),
                           np.array([0.996179, 0.0, -0.0873386, 0.0])),
    "left_hip_yaw_link": ("left_hip_roll_link", np.array([0.025001, 0.0, -0.12412]),
                          np.array([1.0, 0.0, 0.0, 0.0])),
    "left_knee_link": ("left_hip_yaw_link", np.array([-0.078273, 0.0021489, -0.17734]),
                       np.array([0.996179, 0.0, 0.0873386, 0.0])),
    "left_ankle_pitch_link": ("left_knee_link", np.array([0.0, -9.4445e-05, -0.30001]),
                              np.array([1.0, 0.0, 0.0, 0.0])),
    "left_ankle_roll_link": ("left_ankle_pitch_link", np.array([0.0, 0.0, -0.017558]),
                             np.array([1.0, 0.0, 0.0, 0.0])),
    "waist_yaw_link": ("pelvis", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    "waist_roll_link": ("waist_yaw_link", np.array([-0.0039635, 0.0, 0.044]),
                        np.array([1.0, 0.0, 0.0, 0.0])),
    "torso_link": ("waist_roll_link", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    "left_shoulder_pitch_link": ("torso_link", np.array([0.0039563, 0.10022, 0.24778]),
                                 np.array([0.990264, 0.139201, 1.38722e-05, -9.86868e-05])),
    "left_shoulder_roll_link": ("left_shoulder_pitch_link", np.array([0.0, 0.038, -0.013831]),
                                np.array([0.990268, -0.139172, 0.0, 0.0])),
    "left_shoulder_yaw_link": ("left_shoulder_roll_link", np.array([0.0, 0.00624, -0.1032]),
                               np.array([1.0, 0.0, 0.0, 0.0])),
    "left_elbow_link": ("left_shoulder_yaw_link", np.array([0.015783, 0.0, -0.080518]),
                        np.array([1.0, 0.0, 0.0, 0.0])),
    "left_wrist_roll_link": ("left_elbow_link", np.array([0.1, 0.00188791, -0.01]),
                             np.array([1.0, 0.0, 0.0, 0.0])),
    "left_wrist_pitch_link": ("left_wrist_roll_link", np.array([0.038, 0.0, 0.0]),
                              np.array([1.0, 0.0, 0.0, 0.0])),
    "left_wrist_yaw_link": ("left_wrist_pitch_link", np.array([0.046, 0.0, 0.0]),
                            np.array([1.0, 0.0, 0.0, 0.0])),
}

# 每个 G1 关节：(父 body, 铰链轴[父帧])，取自 XML <joint axis=...>
_G1_JOINT_SPEC: Dict[str, Tuple[str, np.ndarray]] = {
    "left_hip_pitch": ("pelvis", np.array([0.0, 1.0, 0.0])),
    "left_hip_roll": ("left_hip_pitch_link", np.array([1.0, 0.0, 0.0])),
    "left_hip_yaw": ("left_hip_roll_link", np.array([0.0, 0.0, 1.0])),
    "left_knee": ("left_hip_yaw_link", np.array([0.0, 1.0, 0.0])),
    "left_ankle_pitch": ("left_knee_link", np.array([0.0, 1.0, 0.0])),
    "left_ankle_roll": ("left_ankle_pitch_link", np.array([1.0, 0.0, 0.0])),
    "right_hip_pitch": ("pelvis", np.array([0.0, 1.0, 0.0])),
    "right_hip_roll": ("right_hip_pitch_link", np.array([1.0, 0.0, 0.0])),
    "right_hip_yaw": ("right_hip_roll_link", np.array([0.0, 0.0, 1.0])),
    "right_knee": ("right_hip_yaw_link", np.array([0.0, 1.0, 0.0])),
    "right_ankle_pitch": ("right_knee_link", np.array([0.0, 1.0, 0.0])),
    "right_ankle_roll": ("right_ankle_pitch_link", np.array([1.0, 0.0, 0.0])),
    "waist_yaw": ("pelvis", np.array([0.0, 0.0, 1.0])),
    "waist_roll": ("waist_yaw_link", np.array([1.0, 0.0, 0.0])),
    "waist_pitch": ("waist_roll_link", np.array([0.0, 1.0, 0.0])),
    "left_shoulder_pitch": ("torso_link", np.array([0.0, 1.0, 0.0])),
    "left_shoulder_roll": ("left_shoulder_pitch_link", np.array([1.0, 0.0, 0.0])),
    "left_shoulder_yaw": ("left_shoulder_roll_link", np.array([0.0, 0.0, 1.0])),
    "left_elbow": ("left_shoulder_yaw_link", np.array([0.0, 1.0, 0.0])),
    "left_wrist_roll": ("left_elbow_link", np.array([1.0, 0.0, 0.0])),
    "left_wrist_pitch": ("left_wrist_roll_link", np.array([0.0, 1.0, 0.0])),
    "left_wrist_yaw": ("left_wrist_pitch_link", np.array([0.0, 0.0, 1.0])),
    "right_shoulder_pitch": ("torso_link", np.array([0.0, 1.0, 0.0])),
    "right_shoulder_roll": ("right_shoulder_pitch_link", np.array([1.0, 0.0, 0.0])),
    "right_shoulder_yaw": ("right_shoulder_roll_link", np.array([0.0, 0.0, 1.0])),
    "right_elbow": ("right_shoulder_yaw_link", np.array([0.0, 1.0, 0.0])),
    "right_wrist_roll": ("right_elbow_link", np.array([1.0, 0.0, 0.0])),
    "right_wrist_pitch": ("right_wrist_roll_link", np.array([0.0, 1.0, 0.0])),
    "right_wrist_yaw": ("right_wrist_pitch_link", np.array([0.0, 0.0, 1.0])),
}

# 源关节 → G1 链映射：(源关节, 相对旋转的父关节覆盖[None=源关节的父])
_MAPPING: Dict[str, Tuple[List[str], Optional[str]]] = {
    "left_hip": (["LeftUpLeg"], None),
    "left_knee": (["LeftLeg"], None),
    "left_ankle": (["LeftFoot"], None),
    "right_hip": (["RightUpLeg"], None),
    "right_knee": (["RightLeg"], None),
    "right_ankle": (["RightFoot"], None),
    "waist": (["Spine"], None),  # 特殊：全部 Spine* 复合，相对 Hips
    "left_shoulder": (["LeftShoulder"], None),
    "left_elbow": (["LeftArm"], None),
    # 腕：源 = 手相对上臂（= 前臂旋前 ⊗ 手屈曲，肘屈曲已随上臂除去）
    "left_wrist": (["LeftHand"], "LeftArm"),
    "right_shoulder": (["RightShoulder"], None),
    "right_elbow": (["RightArm"], None),
    "right_wrist": (["RightHand"], "RightArm"),
}

# LAFAN1 世界系 → G1 系：x→−x（前向 −x → G1 +x），y→z（上），z→y（左）
W = np.array([[-1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0]])
# W 的四元数形式 [w,x,y,z]（W = Rz(90°)·Rx(180°)？直接由矩阵数值验证生成）
_QW = None  # 惰性计算见 _qW()

# LAFAN1 rig 局部轴约定 → G1：(rig x=上, y=前, z=左) → (G1 z, x, y)。
# 循环置换，常数旋转（det=+1）。对齐完全基于该约定 + G1 静止姿态，
# 不依赖运动数据（舞蹈/翻滚等动态片段稳健）。
M_RIG = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0]])
# M_RIG 的逆（= 转置）：G1 轴 → rig 轴。用于把根朝向从 rig 约定
# 转到 G1 基座约定（前+x/左+y/上+z），环境层直接可用。
# M_RIG 是逆循环（x→z→y→x，q=(0.5,−0.5,−0.5,−0.5)），其逆 = 正循环。
Q_MRIG_INV = np.array([0.5, 0.5, 0.5, 0.5])  # [w,x,y,z]


def _qW() -> np.ndarray:
    """W 的四元数形式 [w,x,y,z]。

    注意 W 是 180° 旋转（对称矩阵，反对称部分为零，轴角公式退化），
    须从 (W+I)/2 的非零列提取轴：W = 绕 (0,1,1)/√2 的 180° 旋转，
    q = (0, 0, √2/2, √2/2)。
    """
    global _QW
    if _QW is None:
        tr = np.trace(W)
        if tr <= -1.0 + 1e-6:
            M = (W + np.eye(3)) / 2.0
            col = M[:, 0]
            n = np.linalg.norm(col)
            if n < 1e-9:
                col = M[:, 1]
                n = np.linalg.norm(col)
            if n < 1e-9:
                col = M[:, 2]
                n = np.linalg.norm(col)
            axis = col / n
            _QW = np.array([0.0, *axis])
        else:
            ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
            skew = (W - W.T) / 2
            axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            axis = axis / np.linalg.norm(axis)
            _QW = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
    return _QW


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------

def _mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def _mat_mean(R: np.ndarray) -> np.ndarray:
    """旋转矩阵的 SVD 均值（正交 Procrustes）。R: (T,3,3) → (3,3)。"""
    U, _, Vt = np.linalg.svd(R.mean(0))
    return U @ Vt


def _project_angle(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """旋转矩阵 R (...,3,3) 在轴 axis 上的旋转角分量。

    axis 可为 (3,) 或 (...,3)（逐帧变化的轴，用于父链实际朝向）。
    """
    a = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    u = np.cross(a, np.broadcast_to(np.array([1.0, 0.0, 0.0]), a.shape))
    degenerate = np.linalg.norm(u, axis=-1) < 1e-6
    u = np.where(degenerate[..., None],
                 np.cross(a, np.broadcast_to(np.array([0.0, 1.0, 0.0]), a.shape)), u)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    Ru = np.einsum("...ij,...j->...i", R, u)
    sin = np.einsum("...i,...i->...", a, np.cross(u, Ru))
    cos = np.einsum("...i,...i->...", Ru, u)
    return np.arctan2(sin, cos)


def _circular_mean(a: np.ndarray) -> float:
    return float(np.angle(np.exp(1j * a).mean()))


# ---------------------------------------------------------------------------
# G1 静止位姿（左 + 镜像右）
# ---------------------------------------------------------------------------

def _build_g1_rest() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    bodies = dict(_G1_REST_LEFT)
    for name, (parent, pos, quat) in list(_G1_REST_LEFT.items()):
        if name.startswith("left_"):
            rname = "right_" + name[len("left_"):]
            rparent = ("right_" + parent[len("left_"):]) if parent and parent.startswith("left_") else parent
            bodies[rname] = (rparent, pos * np.array([1.0, -1.0, 1.0]),
                             quat * np.array([1.0, -1.0, 1.0, -1.0]))
    rest: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    def resolve(name: str):
        if name in rest:
            return rest[name]
        parent, pos, quat = bodies[name]
        if parent is None:
            rest[name] = (pos.copy(), quat.copy())
        else:
            ppos, pquat = resolve(parent)
            rest[name] = (ppos + quat_rot_vec(pquat, pos), quat_mul(pquat, quat))
        return rest[name]
    for name in bodies:
        resolve(name)
    return rest

_G1_REST = _build_g1_rest()


def _chain_dir_g1(chain: str) -> Tuple[np.ndarray, str]:
    """G1 链：(d_g 世界方向, 链基 body 名)。"""
    if chain in ("left_hip", "right_hip"):
        side = chain.split("_")[0]
        d = _G1_REST[f"{side}_knee_link"][0] - _G1_REST["pelvis"][0]
        return d, "pelvis"
    if chain in ("left_knee", "right_knee"):
        side = chain.split("_")[0]
        base = f"{side}_hip_yaw_link"
        ppos, pquat = _G1_REST[base]
        d = _G1_REST[f"{side}_ankle_pitch_link"][0] - ppos
        return quat_rot_vec(quat_inv(pquat), d), base
    if chain in ("left_ankle", "right_ankle"):
        side = chain.split("_")[0]
        base = f"{side}_knee_link"
        ppos, pquat = _G1_REST[base]
        d = _G1_REST[f"{side}_ankle_roll_link"][0] - ppos
        return quat_rot_vec(quat_inv(pquat), d), base
    if chain == "waist":
        d = _G1_REST["left_shoulder_pitch_link"][0] - _G1_REST["pelvis"][0]
        return d, "pelvis"
    if chain in ("left_shoulder", "right_shoulder"):
        side = chain.split("_")[0]
        base = f"{side}_shoulder_pitch_link"
        ppos, pquat = _G1_REST[base]
        d = _G1_REST[f"{side}_elbow_link"][0] - ppos
        return quat_rot_vec(quat_inv(pquat), d), base
    if chain in ("left_elbow", "right_elbow"):
        side = chain.split("_")[0]
        base = f"{side}_shoulder_yaw_link"
        ppos, pquat = _G1_REST[base]
        d = _G1_REST[f"{side}_wrist_roll_link"][0] - ppos
        return quat_rot_vec(quat_inv(pquat), d), base
    if chain in ("left_wrist", "right_wrist"):
        side = chain.split("_")[0]
        base = f"{side}_elbow_link"
        ppos, pquat = _G1_REST[base]
        d = _G1_REST[f"{side}_wrist_yaw_link"][0] - ppos
        return quat_rot_vec(quat_inv(pquat), d), base
    raise ValueError(f"未知链: {chain}")


def _chain_of(joint: str) -> Optional[str]:
    for prefix in ("left_hip", "right_hip", "left_ankle", "right_ankle",
                   "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
                   "left_knee", "right_knee", "left_elbow", "right_elbow"):
        if joint.startswith(prefix):
            return prefix
    if joint.startswith("waist"):
        return "waist"
    return None


def contact_labels(foot_pos: np.ndarray, frame_time: float,
                   vel_thresh: float = 0.15, clearance: float = 0.05) -> np.ndarray:
    """接触标签统一协议（M1.5c 升级）：足速阈值 + 高度条件 + 中值滤波。

    foot_pos: (T, 2, 3) 双足踝位置（米）。规则：
      1. 足速 < vel_thresh（原 A14 协议）
      2. 踝 z < 地面 + clearance——地面 = 序列双踝 z 的 2% 分位
         （对轻微穿地鲁棒；排除空中缓速帧，如摆动腿最高点附近）
      3. 窗 5 中值滤波消除单帧闪烁
    源侧（retarget）与 G1 侧（ik_refine._finalize、eval）共用，接触
    一致率在同等协议下比较。
    """
    foot_vel = np.zeros_like(foot_pos)
    foot_vel[1:] = (foot_pos[1:] - foot_pos[:-1]) / frame_time
    slow = np.linalg.norm(foot_vel, axis=-1) < vel_thresh
    ground_z = float(np.percentile(foot_pos[..., 2], 2))
    low = foot_pos[..., 2] < ground_z + clearance
    labels = slow & low
    out = labels.copy()
    for t in range(2, labels.shape[0] - 2):
        out[t] = labels[t - 2:t + 3].mean(axis=0) > 0.5
    return out


def _estimate_scale(gpos_cm: np.ndarray, bvh: BVH) -> float:
    """骨骼缩放：G1 骨盆高 / P90 源腿骨链长（大腿 + 小腿，cm）。

    骨链长是刚体量，与姿态无关；旧版用"髋高 − 足底高度"（世界竖直
    差），躺地时髋足同高导致尺度爆炸（ground1 0.0318 vs walk 0.0094）。
    注意 LAFAN1 命名：LeftUpLeg 关节在髋部（Hips→LeftUpLeg 只是髋宽
    偏移 ~11cm），大腿 = LeftUpLeg→LeftLeg，小腿 = LeftLeg→LeftFoot。
    """
    chains = []
    for side in ("Left", "Right"):
        hip_j = bvh.joint_index(side + "UpLeg")  # 髋关节（大腿起点）
        knee = bvh.joint_index(side + "Leg")     # 膝关节
        ankle = bvh.joint_index(side + "Foot")   # 踝关节
        chains.append(np.linalg.norm(gpos_cm[:, hip_j] - gpos_cm[:, knee], axis=-1)
                      + np.linalg.norm(gpos_cm[:, knee] - gpos_cm[:, ankle], axis=-1))
    leg_cm = np.percentile(np.minimum(chains[0], chains[1]), 90)
    if leg_cm < 1e-6:
        raise ValueError("腿长估计异常")
    return G1_PELVIS_HEIGHT / leg_cm  # m/cm


# ---------------------------------------------------------------------------
# 重定向主体
# ---------------------------------------------------------------------------

def bounded_joint_trajectory(qpos: np.ndarray, frame_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """Project G1 hinge coordinates into their limits and differentiate them.

    All 29 actuated joints in g1_29dof_rev_1_0.xml have finite ranges inside
    (-pi, pi). They are not continuous rotation joints: an apparent short path
    across +/-pi crosses a mechanical stop. Neither wrapping positions nor
    unwrapping differences is valid for these joint coordinates.
    """
    lo = np.array([G1_JOINT_LIMITS[n][0] for n in G1_JOINT_NAMES])
    hi = np.array([G1_JOINT_LIMITS[n][1] for n in G1_JOINT_NAMES])
    positions = np.clip(np.asarray(qpos, dtype=np.float64), lo, hi).astype(np.float32)
    # Differentiate the coordinates that are actually exported, in float64 to
    # avoid adding arithmetic roundoff before the final float32 conversion.
    velocities = np.zeros_like(positions)
    velocities[1:] = np.diff(positions.astype(np.float64), axis=0) / frame_time
    return positions, velocities


def _repair_principal_angle_jumps(qpos: np.ndarray, branch_margin: float = 0.5) -> np.ndarray:
    """Remove Euler principal-value jumps before finite-hinge projection.

    ``_decompose_chain`` returns ``atan2`` principal values.  A wrist can pass
    the ``-pi/pi`` representation seam while its mechanical coordinate is
    already outside the G1 wrist-roll range; clipping the two principal values
    independently then turns one smooth stop contact into a ``+limit ->
    -limit`` jump.  Keep :func:`bounded_joint_trajectory` deliberately literal
    (finite hinges must use exported-coordinate differences), and repair only
    this identifiable representation seam in the retarget path.

    A transition is shifted by one full turn when all of the following hold:

    * the principal values are inside ``[-pi, pi]`` (so synthetic/unbounded
      inputs are left untouched);
    * the frame-to-frame change exceeds ``pi`` and one endpoint is close to the
      principal seam; and
    * at least one endpoint is outside that joint's finite mechanical range.

    The cumulative shift follows the local branch, just like angle unwrapping,
    but is gated by the finite-limit and seam checks above.  Thus a legitimate
    traversal from the lower to upper mechanical limit remains a direct
    coordinate difference, while a wrist-roll ``-3.08 -> +2.94`` seam crossing
    is represented on one branch and clips to the same stop.
    """
    values = np.asarray(qpos, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError("qpos must have shape (frames, 29)")
    if values.shape[0] < 2:
        return values
    seam = np.pi - float(branch_margin)
    for j, name in enumerate(G1_JOINT_NAMES):
        lo, hi = G1_JOINT_LIMITS[name]
        raw = values[:, j].copy()
        offset = 0.0
        for t in range(1, raw.shape[0]):
            prev, cur = raw[t - 1], raw[t]
            # Decomposition is principal-valued.  Leave values outside that
            # domain alone: bounded_joint_trajectory's clipping contract also
            # covers arbitrary caller input and test sentinels.
            principal = max(abs(prev), abs(cur)) <= np.pi + 1e-6
            seam_hit = max(abs(prev), abs(cur)) >= seam
            outside = (prev < lo - 1e-6 or prev > hi + 1e-6 or
                       cur < lo - 1e-6 or cur > hi + 1e-6)
            jump = cur - prev
            if principal and seam_hit and outside and abs(jump) > np.pi:
                offset += -2.0 * np.pi if jump > 0 else 2.0 * np.pi
            values[t, j] = cur + offset
    return values


def retarget(bvh: BVH) -> Dict[str, np.ndarray]:
    """LAFAN1 BVH → G1 29-DoF（固定 rig 对齐与数据均值去偏移）。"""
    for name in ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
                 "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
                 "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                 "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                 "Neck"):
        if name not in bvh.names:
            raise ValueError(f"BVH 缺少关节: {name}")

    T = bvh.num_frames

    # ---- 尺度：P90 源腿骨链长（髋→膝 + 膝→踝，姿势无关）→ 米 ----
    # 不能用"髋高 − 足底高度"（世界竖直差）：躺地时髋足同高，竖直差
    # →0 → 尺度爆炸（实测 ground1 scale=0.0318 vs walk 0.0094，源骨架
    # 被放大 3.4 倍）。骨长是刚体量，任何姿态下不变。
    gpos_cm, grot = bvh.fk(unit_scale=1.0)  # 世界位置（cm）/ 朝向
    scale = _estimate_scale(gpos_cm, bvh)

    # ---- 世界 → G1 系 ----
    gpos = gpos_cm @ W.T * scale  # (T,J,3) G1 系，米
    grot_q = grot  # 世界四元数 (T,J,4)
    qw = _qW()
    grot_g1 = quat_mul(np.tile(qw, (T, 1, 1)), grot_q)  # G1 系朝向

    # ---- 根 ----
    # 高度锚定：均值髋高 → G1 骨盆高 0.793（腿长比缩放后的残余系统偏差
    # 消除）；源足底最低点（相对髋）→ G1 零姿态的相对踝高
    # （0.036 − 0.793 ≈ −0.757），避免足底穿地/悬空。
    G1_ANKLE_REL = float(_G1_REST["left_ankle_roll_link"][0][2] - G1_PELVIS_HEIGHT)
    root_pos = gpos[:, bvh.joint_index("Hips")].copy()
    root_pos[:, 0] -= root_pos[:, 0].mean()
    root_pos[:, 1] -= root_pos[:, 1].mean()
    hip_z = root_pos[:, 2].copy()
    root_pos[:, 2] = G1_PELVIS_HEIGHT + (hip_z - hip_z.mean())
    foot_rel = np.minimum(gpos[:, bvh.joint_index("LeftFoot"), 2],
                          gpos[:, bvh.joint_index("RightFoot"), 2]) - hip_z
    root_pos[:, 2] += G1_ANKLE_REL - foot_rel.min()
    # 根朝向：rig 约定 → G1 基座约定（G1 局部前 +x / 左 +y / 上 +z）
    root_rot = source_root_quat_to_g1_base(grot[:, bvh.joint_index("Hips")])

    # ---- 源关节相对旋转（G1 系） ----
    def rel_rot(joint: str, parent: str) -> np.ndarray:
        Rc = _mat(grot_g1[:, bvh.joint_index(joint)])
        Rp = _mat(grot_g1[:, bvh.joint_index(parent)])
        return Rp.transpose(0, 2, 1) @ Rc  # (T,3,3)

    # ---- 逐链：常数 R_align（rig 轴约定 × G1 静止链基框架）+ 欧拉分解 ----
    # R_src 表达在源父关节的 rig 框架内（R_src = R_parent⁻¹·R_child）；
    # R_align = R_base_rest · M_RIG 把 rig 框架映射到 G1 链基静止框架。
    # 对齐完全由约定决定（rig 上+x/前+y/左+z → G1 z/x/y 循环置换 +
    # G1 静止姿态的链基倾斜 ≤16%），与运动数据无关 → 舞蹈/翻滚等
    # 动态片段同样稳健。旋转均值居中去除绑定偏移后，按链做标准欧拉
    # 分解（逐轴最佳拟合投影在大旋转下有数学歧义，欧拉分解只在万向节
    # 奇异点失效）。链内中间帧的静止倾斜（≤16°）按链基框架近似（v1）。
    qpos = np.zeros((T, 29))
    for chain, (sources, rel_parent) in _MAPPING.items():
        _, base_body = _chain_dir_g1(chain)
        q_base = _G1_REST[base_body][1]
        R_align = _mat(q_base) @ M_RIG
        if chain == "waist":
            # 复合脊柱：R_src = R_hips⁻¹·R_chest（最后一个 Spine 关节）
            spines = [n for n in bvh.names if n.startswith("Spine")]
            R_src = rel_rot(spines[-1], "Hips")
        else:
            parent = rel_parent if rel_parent is not None else _parent_of(bvh, sources[0])
            R_src = rel_rot(sources[0], parent)
        R_aligned = R_align @ R_src @ R_align.T  # (T,3,3)，链基框架内
        if chain == "waist":
            # 胸廓例外：绑定脊柱链直立（绑定偏移≈0），均值旋转=风格
            # 前倾（实测 17.4°，走路特征），居中会抹掉风格——保留
            R_c = R_aligned
        else:
            # 旋转均值居中（去绑定偏移：腿链的折叠绑定 ≈90°）
            R_c = R_aligned @ _mat_mean(R_aligned).T
        angles = _decompose_chain(R_c, chain)
        for jname, ang in zip(_chain_joints(chain), angles):
            qpos[:, G1_JOINT_NAMES.index(jname)] = ang

    # ---- 万向节保护：中间角接近 ±90° 时外角仅剩耦合自由度，提取值被
    # 0/0 噪声放大（实测一帧跳 58°）。用邻域好帧在 (sin,cos) 空间线性
    # 插值替换奇异区的外角，消除速度尖峰。 ----
    qpos = _gimbal_protect(qpos)

    # ``atan2`` uses principal values.  Repair only finite-hinge seam jumps
    # before clipping; bounded_joint_trajectory itself remains a literal
    # projection plus coordinate difference (no generic unwrap).
    qpos = _repair_principal_angle_jumps(qpos)

    # Finite G1 hinges: project into limits, then directly differentiate.
    qpos, qvel = bounded_joint_trajectory(qpos, float(np.float32(bvh.frame_time)))

    # ---- 接触标签：统一协议（足速 + 高度 + 中值滤波，见 contact_labels） ----
    thresh = get("A14").value.foot_vel_thresh
    foot_pos = np.stack([gpos[:, bvh.joint_index("LeftFoot")],
                         gpos[:, bvh.joint_index("RightFoot")]], axis=1)
    contacts = contact_labels(foot_pos, bvh.frame_time, thresh)

    return {
        "joint_names": np.array(G1_JOINT_NAMES),
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "contacts": contacts,
        "frame_time": np.float32(bvh.frame_time),
        "scale": np.float32(scale),
    }


def _parent_of(bvh: BVH, joint: str) -> str:
    return bvh.names[bvh.parents[bvh.joint_index(joint)]]


# ---------------------------------------------------------------------------
# 源根朝向 ↔ G1 基座朝向（唯一换算出处）
# ---------------------------------------------------------------------------

def source_root_quat_to_g1_base(q_src_rig: np.ndarray) -> np.ndarray:
    """源（rig 约定）根朝向 → G1 基座约定朝向。

    `retarget` 里的两步是::

        grot_g1  = qw ⊗ q_src            # 世界 → G1 世界系（qw = _qW()，作为左乘基变换）
        root_rot = grot_g1 ⊗ Q_MRIG_INV  # rig 局部轴 → G1 基座轴（前+x/左+y/上+z）

    于是 ``root_rot = (qw ⊗ q_src) ⊗ Q_MRIG_INV``。

    **乘法顺序陷阱**：qw 必须左乘（它是基变换，作用在源局部旋转之外），
    而 Q_MRIG_INV 必须右乘（它把 rig 的局部轴约定换到 G1 的基座轴约定）。
    写成 ``qw ⊗ q_src ⊗ Q_MRIG_INV`` 之外的任何组合（例如把 qw 右乘、
    或漏掉 Q_MRIG_INV）都会得到错误朝向 —— 这正是对照评估里最易错的一处，
    故单独抽成本函数并配单测。

    Args:
        q_src_rig: (...,4) 源骨架根关节世界朝向，[w,x,y,z]（如 ``bvh.fk()`` 的输出）
    Returns:
        (...,4) G1 基座朝向
    """
    q_src_rig = np.asarray(q_src_rig, dtype=np.float64)
    qw = _qW()
    shape = q_src_rig.shape[:-1]
    return quat_mul(
        quat_mul(np.tile(qw, (*shape, 1)), q_src_rig),
        np.tile(Q_MRIG_INV, (*shape, 1)),
    )


def relative_rotation_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """两个四元数所代表旋转之间的最小夹角（度）。

    数值上取 ``trace`` 公式：``R_rel = Raᵀ·Rb``，夹角 ``= acos((tr(R_rel)−1)/2)``。
    相比 ``2·acos(|w(rel)|)``，它在接近 0°/180° 两端更稳定。

    符号歧义由公式本身消解 —— q 与 −q 对应同一个 R，故无需额外取绝对值
    （`quat_to_mat` 是二次型，天然对整体符号不变）。
    """
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    R_rel = np.swapaxes(quat_to_mat(q_a), -1, -2) @ quat_to_mat(q_b)
    tr = np.trace(R_rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _chain_joints(chain: str) -> List[str]:
    """链内 G1 关节名（XML 顺序）。"""
    return [j for j in G1_JOINT_NAMES if _chain_of(j) == chain]


def _gimbal_protect(qpos: np.ndarray, threshold_deg: float = 80.0) -> np.ndarray:
    """3 铰链链的万向节保护（原地修改并返回 qpos）。

    中间角（hips/shoulders/waist 的 roll、wrist 的 pitch）接近 ±90° 时，
    外角（首/末角）的欧拉提取退化为 0/0，噪声被放大产生速度尖峰。
    奇异区外角用邻域好帧在 (sin,cos) 空间的线性插值替换（物理上外角
    仅剩耦合自由度，插值路径是连续且近似正确的）。

    阈值经验：80° 硬阈值。更低（65°/锥形过渡）会误伤舞蹈片段的
    中速手势（源数据本身帧间旋转可达 50-100°，非伪影），且过渡
    边界引入新的跳变。
    """
    thr = np.radians(threshold_deg)
    T = qpos.shape[0]
    for chain in ("left_hip", "right_hip", "left_shoulder", "right_shoulder",
                  "left_wrist", "right_wrist", "waist"):
        js = _chain_joints(chain)
        mid = qpos[:, G1_JOINT_NAMES.index(js[1])]
        mask = np.abs(mid) > thr
        n_bad = int(mask.sum())
        if n_bad == 0 or n_bad == T:
            continue
        good = ~mask
        idx = np.arange(T)
        for jname in (js[0], js[2]):
            col = G1_JOINT_NAMES.index(jname)
            a = qpos[:, col]
            s, c = np.sin(a), np.cos(a)
            s[mask] = np.interp(idx[mask], idx[good], s[good])
            c[mask] = np.interp(idx[mask], idx[good], c[good])
            qpos[mask, col] = np.arctan2(s[mask], c[mask])
    return qpos


def _decompose_chain(R: np.ndarray, chain: str) -> List[np.ndarray]:
    """链基框架内的标准欧拉分解（忽略中间帧 ≤16° 的静止倾斜）。

    链组成（理想无倾斜模型）：
      hips/shoulders: Ry(pitch)·Rx(roll)·Rz(yaw)
      waist:          Rz(yaw)·Rx(roll)·Ry(pitch)
      wrist:          Rx(roll)·Ry(pitch)·Rz(yaw)
      ankle:          Ry(pitch)·Rx(roll)
      knee/elbow:     绕 Y 铰链

    注意（2026-09-16 实验结论）：曾尝试牛顿细化求倾斜链的精确逆
    （与 FK 组合互逆），FK 位置误差反而变差（20.9→25.8cm）：精确
    末端朝向逆把中间连杆（大腿方向）推离源姿态，而跟踪奖励恰恰是
    连杆位置。朴素欧拉的中间连杆保真更好，保留此版本。
    """
    if chain in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        return [np.arctan2(R[:, 0, 2], R[:, 2, 2]),
                np.arcsin(np.clip(-R[:, 1, 2], -1.0, 1.0)),
                np.arctan2(R[:, 1, 0], R[:, 1, 1])]
    if chain == "waist":
        return [np.arctan2(-R[:, 0, 1], R[:, 1, 1]),
                np.arcsin(np.clip(R[:, 2, 1], -1.0, 1.0)),
                np.arctan2(-R[:, 2, 0], R[:, 2, 2])]
    if chain in ("left_wrist", "right_wrist"):
        return [np.arctan2(-R[:, 1, 2], R[:, 2, 2]),
                np.arcsin(np.clip(R[:, 0, 2], -1.0, 1.0)),
                np.arctan2(-R[:, 0, 1], R[:, 0, 0])]
    if chain in ("left_ankle", "right_ankle"):
        return [np.arctan2(R[:, 0, 2], R[:, 2, 2]),
                np.arctan2(-R[:, 1, 2], R[:, 1, 1])]
    if chain in ("left_knee", "right_knee", "left_elbow", "right_elbow"):
        return [np.arctan2(-R[:, 2, 0], R[:, 0, 0])]
    raise ValueError(f"未知链: {chain}")


def g1_forward_kinematics(qpos: np.ndarray, root_pos: Optional[np.ndarray] = None,
                          root_quat: Optional[np.ndarray] = None,
                          return_quats: bool = False) -> Dict[str, np.ndarray]:
    """G1 前向运动学：qpos (T,29) → {body: (T,3) 世界位置}。

    return_quats=True 时返回 (pos, quat) 两字典（IK 雅可比需要世界朝向）。
    """
    T = qpos.shape[0]
    if root_pos is None:
        root_pos = np.zeros((T, 3))
    if root_quat is None:
        root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    pos = {"pelvis": root_pos.copy()}
    quat = {"pelvis": root_quat.copy()}
    for body in _G1_REST:
        if body == "pelvis":
            continue
        # 相对偏移与父 body（左表原始值；右 = 左镜像；_G1_REST 是绝对坐标，
        # 不能用于 FK 叠加）
        if body.startswith("left_"):
            parent, rest_pos, rest_quat = _G1_REST_LEFT[body]
        elif body.startswith("right_"):
            lb = "left_" + body[len("right_"):]
            lp = _G1_REST_LEFT[lb][0]
            parent = ("right_" + lp[len("left_"):]) if lp and lp.startswith("left_") else lp
            rest_pos = _G1_REST_LEFT[lb][1] * np.array([1.0, -1.0, 1.0])
            rest_quat = _G1_REST_LEFT[lb][2] * np.array([1.0, -1.0, 1.0, -1.0])
        else:
            parent, rest_pos, rest_quat = _G1_REST_LEFT[body]
        # body → 关节映射：一般 body 名 = 关节名 + "_link"；
        # 特例 torso_link 的关节 = waist_pitch（XML 中 waist_pitch_joint
        # 定义在 torso_link 元素内）
        jname = "waist_pitch" if body == "torso_link" else body[:-len("_link")]
        if jname in G1_JOINT_NAMES:
            axis = _G1_JOINT_SPEC[jname][1]
            theta = qpos[:, G1_JOINT_NAMES.index(jname)]
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            jquat = np.stack([c, s * axis[0], s * axis[1], s * axis[2]], axis=-1)
            r_rot = quat_mul(quat[parent], quat_mul(rest_quat, jquat))
        else:
            r_rot = quat_mul(quat[parent], rest_quat)
        quat[body] = r_rot
        pos[body] = pos[parent] + quat_rot_vec(quat[parent], rest_pos)
    if return_quats:
        return pos, quat
    return pos


# ---------------------------------------------------------------------------
# 批量导出
# ---------------------------------------------------------------------------

def retarget_all(bvh_dir: str, out_dir: str, verbose: bool = True,
                 refine: bool = True) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for fname in sorted(os.listdir(bvh_dir)):
        if not fname.endswith(".bvh"):
            continue
        path = os.path.join(bvh_dir, fname)
        try:
            data = retarget(load_bvh(path))
            if refine:
                from data.ik_refine import refine_full
                data = refine_full(data, load_bvh(path))
        except (ValueError, AssertionError) as e:
            print(f"[跳过] {fname}: {e}")
            continue
        out = os.path.join(out_dir, fname[:-4] + ".npz")
        np.savez_compressed(out, **data)
        outs.append(out)
        if verbose:
            print(f"[导出] {fname} -> {out}  ({data['qpos'].shape[0]} 帧)")
    return outs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LAFAN1 BVH → G1 29-DoF 重定向")
    ap.add_argument("--bvh-dir", default="data/raw/lafan1")
    ap.add_argument("--out-dir", default="data/processed/lafan1_g1")
    ap.add_argument("--no-ik", action="store_true", help="跳过 IK 精修（对比用）")
    args = ap.parse_args()
    retarget_all(args.bvh_dir, args.out_dir, refine=not args.no_ik)
