"""IK 精修（M1.5 路线 C）——位置驱动的参考质量提升。

以现有旋转映射结果为初值，逐帧用几何雅可比高斯-牛顿（LM）优化
关节角，直接最小化 G1 骨架与源骨架的关键点位置误差（评估同款口径）。

设计要点：
  - 腿链先行（M1.5a）：12 个腿关节，目标 = 膝/踝位置（趾因 G1 无脚趾
    关节存在几何下限，不参与优化，仅评估报告）
  - 批量向量化：全部帧同时迭代（FK/雅可比/求解均按 (T,·) 批处理）；
    时间平滑 = 向**原始轨迹**（旋转映射输出，本身平滑）的正则，
    替代逐帧热启动（热启动版 378s/文件，批量版约一个量级快）
  - LM 自适应阻尼按帧独立（接受/拒绝的向量化掩码）+ 关节限位硬投影
  - 接触标签与 qvel 在精修后重算（脚位置贴合源 → 接触一致率上升）

论文未指定参考生成方法（只有"LAFAN1 retargeted"一句），本模块处于
复现自由空间；数据源与评估协议不变，所有策略共用同一参考，内部
对标结论不受影响。

雅可比正确性三坑（已由 tests/test_ik_refine.py 差分锁定）：
  1. 上游判定须为**严格祖先**（关节的子 body = 关键点自身时不影响
     其位置——旋转发生在位置之后）
  2. 世界轴 = 子 body **当前**朝向 × 局部轴（含子 body 自身关节角）
  3. 旋转支点 = 子 body 原点（非父原点）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from data.bvh import BVH, quat_rot_vec
from data.retarget_lafan1 import (
    G1_JOINT_LIMITS,
    G1_JOINT_NAMES,
    W,
    _G1_JOINT_SPEC,
    _G1_REST,
    _G1_REST_LEFT,
    g1_forward_kinematics,
)

# 腿链关节（G1_JOINT_NAMES 顺序）
LEG_JOINTS = G1_JOINT_NAMES[:12]

# 腿链关键点：源关节 → G1 body（评估同款匹配；不含趾）
LEG_KEYPOINTS: List[Tuple[str, str]] = [
    ("LeftLeg", "left_knee_link"),
    ("LeftFoot", "left_ankle_pitch_link"),
    ("RightLeg", "right_knee_link"),
    ("RightFoot", "right_ankle_pitch_link"),
]

# 权重：踝（足端，接触相关）高于膝
KEYPOINT_WEIGHTS = np.array([1.0, 2.0, 1.0, 2.0])


def _body_ancestors(body: str) -> List[str]:
    path = [body]
    while body != "pelvis":
        if body.startswith("left_"):
            body = _G1_REST_LEFT[body][0]
        elif body.startswith("right_"):
            lb = "left_" + body[len("right_"):]
            lp = _G1_REST_LEFT[lb][0]
            body = ("right_" + lp[len("left_"):]) if lp and lp.startswith("left_") else lp
        else:
            body = _G1_REST_LEFT[body][0]
        path.append(body)
    return path


def _joint_child(jname: str) -> str:
    return "torso_link" if jname == "waist_pitch" else jname + "_link"


def _source_keypoints(bvh: BVH, scale: float) -> np.ndarray:
    """源关键点位置（G1 系、米）：(T, 4, 3)，膝/踝 × 左右。"""
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale
    return np.stack([gpos[:, bvh.joint_index(src)] for src, _ in LEG_KEYPOINTS], axis=1)


def _batch_fk(qpos: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray):
    """批量 FK（T,29）→ pos/quat 字典。"""
    pos, quat = g1_forward_kinematics(qpos, root_pos, root_rot, return_quats=True)
    return pos, quat


def _batch_jacobian(pos: Dict[str, np.ndarray], quat: Dict[str, np.ndarray],
                    keypoint_bodies: List[str], joint_names: List[str]) -> np.ndarray:
    """批量几何雅可比 (T, 3K, N)。

    正确性三坑（tests/test_ik_refine.py 差分锁定）：严格祖先、子 body
    当前朝向轴、子 body 原点支点。
    """
    T = next(iter(pos.values())).shape[0]
    J = np.zeros((T, len(keypoint_bodies) * 3, len(joint_names)))
    for k, g1_body in enumerate(keypoint_bodies):
        p_k = pos[g1_body]  # (T,3)
        ancestors = _body_ancestors(g1_body)
        for j, jname in enumerate(joint_names):
            child = _joint_child(jname)
            if child not in ancestors or child == g1_body:
                continue
            axis_local = _G1_JOINT_SPEC[jname][1]
            a = quat_rot_vec(quat[child], axis_local)  # (T,3) 当前朝向轴
            p_j = pos[child]
            col = np.cross(a, p_k - p_j)  # (T,3)
            J[:, 3 * k:3 * k + 3, j] = col
    return J


def _refine(data: Dict[str, np.ndarray], targets: np.ndarray,
            joint_idx: List[int], keypoint_bodies: List[str],
            weights: np.ndarray, max_iter: int = 8,
            damp_init: float = 1e-2, lambda_smooth: float = 0.05) -> np.ndarray:
    """通用批量 LM 精修：joint_idx 关节 + keypoint_bodies 目标。

    targets: (T, K, 3)（已对齐）；返回精修后的完整 qpos。
    """
    qpos = data["qpos"].copy()
    T = qpos.shape[0]
    root_pos, root_rot = data["root_pos"], data["root_rot"]
    n_j = len(joint_idx)
    lo = np.array([G1_JOINT_LIMITS[G1_JOINT_NAMES[i]][0] for i in joint_idx])
    hi = np.array([G1_JOINT_LIMITS[G1_JOINT_NAMES[i]][1] for i in joint_idx])
    theta0 = qpos[:, joint_idx].copy()
    theta = theta0.copy()
    lam = np.full(T, damp_init)

    joint_names = [G1_JOINT_NAMES[i] for i in joint_idx]

    def residuals(theta_: np.ndarray) -> np.ndarray:
        q_full = qpos.copy()
        q_full[:, joint_idx] = theta_
        pos_, _ = _batch_fk(q_full, root_pos, root_rot)
        r = np.stack([weights[k] * (pos_[b] - targets[:, k])
                      for k, b in enumerate(keypoint_bodies)], axis=1)
        return r.reshape(T, -1)

    for _ in range(max_iter):
        q_full = qpos.copy()
        q_full[:, joint_idx] = theta
        pos, quat = _batch_fk(q_full, root_pos, root_rot)
        r = residuals(theta)
        J = _batch_jacobian(pos, quat, keypoint_bodies, joint_names)
        Jw = J * np.repeat(weights, 3)[None, :, None]
        A = np.einsum("tij,tik->tjk", Jw, Jw) \
            + (lam + lambda_smooth)[:, None, None] * np.eye(n_j)
        b = np.einsum("tij,ti->tj", Jw, r) + lambda_smooth * (theta0 - theta)
        dtheta = np.linalg.solve(A, b)
        trial = np.clip(theta - dtheta, lo, hi)
        trial_r = residuals(trial)
        cur_cost = np.einsum("ti,ti->t", r, r)
        trial_cost = np.einsum("ti,ti->t", trial_r, trial_r)
        accept = trial_cost < cur_cost * 0.99
        theta[accept] = trial[accept]
        lam = np.where(accept, np.maximum(lam * 0.5, 1e-8), np.minimum(lam * 10.0, 1e2))
        if not accept.any():
            break
    qpos[:, joint_idx] = theta
    return qpos


# 全链关键点（评估同款匹配；不含趾与骨盆自身）
FULL_KEYPOINTS: List[Tuple[str, str]] = [
    ("Spine2", "torso_link"),
    ("LeftShoulder", "left_shoulder_pitch_link"),
    ("LeftArm", "left_elbow_link"),
    ("LeftForeArm", "left_wrist_roll_link"),
    ("LeftHand", "left_wrist_yaw_link"),
    ("RightShoulder", "right_shoulder_pitch_link"),
    ("RightArm", "right_elbow_link"),
    ("RightForeArm", "right_wrist_roll_link"),
    ("RightHand", "right_wrist_yaw_link"),
    ("LeftLeg", "left_knee_link"),
    ("LeftFoot", "left_ankle_pitch_link"),
    ("RightLeg", "right_knee_link"),
    ("RightFoot", "right_ankle_pitch_link"),
]

# 权重：踝(接触) > 手 > 肘/膝 > 肩 > 躯干（躯干/肩位置受比例限制，低权重）
FULL_WEIGHTS = np.array([
    0.3,   # Spine2
    0.5, 1.0, 1.0, 2.0,   # 左肩/肘/前臂/手
    0.5, 1.0, 1.0, 2.0,   # 右肩/肘/前臂/手
    1.0, 2.0, 1.0, 2.0,   # 左膝/踝/右膝/踝
])


def refine_legs(data: Dict[str, np.ndarray], bvh: BVH,
                max_iter: int = 8, damp_init: float = 1e-2,
                lambda_smooth: float = 0.05) -> Dict[str, np.ndarray]:
    """腿链 IK 精修（批量 LM，M1.5a 接口保留）。data = retarget() 输出。"""
    scale = float(data["scale"])
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale
    delta = gpos[:, bvh.joint_index("Hips")] - data["root_pos"]
    targets = _source_keypoints(bvh, scale) - delta[:, None, :]
    qpos = _refine(data, targets, list(range(12)),
                   [b for _, b in LEG_KEYPOINTS],
                   KEYPOINT_WEIGHTS, max_iter, damp_init, lambda_smooth)
    return _finalize(data, qpos)


def refine_full(data: Dict[str, np.ndarray], bvh: BVH,
                max_iter: int = 10, damp_init: float = 1e-2,
                lambda_smooth: float = 0.05) -> Dict[str, np.ndarray]:
    """全链 IK 精修（29 关节，M1.5b）。data = retarget() 输出。

    纯位置残差（FULL_KEYPOINTS）。链接方向残差（上臂/前臂）已实验
    否决：两阶段版（先位置后方向）与联合单阶段版均使肘/手/膝变差
    （walk1 均值 14.68→15.70cm）——源与 G1 肢体比例不同，方向目标
    与位置目标几何不一致，LM 只能折中。中间连杆保真交给位置权重。
    """
    scale = float(data["scale"])
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale
    delta = gpos[:, bvh.joint_index("Hips")] - data["root_pos"]
    pos_targets = np.stack([gpos[:, bvh.joint_index(src)] for src, _ in FULL_KEYPOINTS],
                           axis=1) - delta[:, None, :]
    pos_bodies = [b for _, b in FULL_KEYPOINTS]
    qpos = _refine(data, pos_targets, list(range(29)), pos_bodies,
                   FULL_WEIGHTS, max_iter, damp_init, lambda_smooth)
    return _finalize(data, qpos)


def _finalize(data: Dict[str, np.ndarray], qpos: np.ndarray) -> Dict[str, np.ndarray]:
    """重算 qvel 与接触标签。

    注意导出约定：qpos 保持 wrap 后的规范值（(−π,π]）——环境的 PD
    参考/关节位置奖励按规范角度比较，unwrap 的 2π 偏移会导致错误的
    绝对参考值；qvel 才由 unwrap 后的连续序列差分。
    """
    root_pos, root_rot = data["root_pos"], data["root_rot"]
    qpos_w = (qpos + np.pi) % (2 * np.pi) - np.pi
    qpos_u = np.unwrap(qpos_w, axis=0)
    qvel = np.zeros_like(qpos_u)
    qvel[1:] = (qpos_u[1:] - qpos_u[:-1]) / float(data["frame_time"])
    out = {k: v for k, v in data.items()}
    out["qpos"] = qpos_w.astype(np.float32)
    out["qvel"] = qvel.astype(np.float32)
    pos = g1_forward_kinematics(qpos_u, root_pos, root_rot)
    feet = np.stack([pos["left_ankle_roll_link"], pos["right_ankle_roll_link"]], axis=1)
    from data.retarget_lafan1 import contact_labels
    from pgmt.cfg.assumptions import get
    out["contacts"] = contact_labels(
        feet, float(data["frame_time"]), get("A14").value.foot_vel_thresh)
    return out
