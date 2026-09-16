"""跟踪残差与奖励（M2 待办 #1，论文 Table I 的上半部 + Eq.4/5 分组）。

## 论文说了什么

  - Table I 的 upper / lower body 两组各有 6 项：link position / orientation /
    linear velocity / angular velocity / joint position / joint velocity
  - lower body 的 link position / orientation / joint position 三项标 **TA**
    （受 Eq.10 地形松弛），速度项**不带** TA
  - `r_t = r_upper + r_lower + r_aux`（Eq.5）
  - 每项都是 "the exponential tracking reward" 作用在残差上（核函数与 σ 见
    `spec.py`，依据 OmniH2O 的 `exp(−0.5‖p−p̂‖²)`）

## 论文没说的（本项目拍定，登记在 A21）

  - 残差的**具体形式**：位置取欧氏距离、朝向取**旋转角**（非 6D 差的范数）、
    速度取差向量的范数、关节取绝对差 —— 这些是各种选择中语义最直接的一组
  - **upper / lower 如何切分 body 集合**（论文只说"upper-body objectives
    preserve motion expression"），取躯干+双臂 = upper、双腿 = lower
  - 残差在 body / joint 维上如何归约到标量 —— 取**均值**（与"误差尺度"的
    语义一致；求和会让项的量级随 body 数漂移）

## 设计要点

  - 残差函数是**纯函数**，输入显式状态，便于用构造状态做解析验证
    （"与参考完全一致 ⇒ 残差为 0；偏移已知量 ⇒ 残差等于该量"）
  - `link_*` 与 `joint_*` 残差**按 body/joint 子集分别给 upper 与 lower**，
    但速度残差用的是同一批 body（因为 Table I 里上下半身的速度项同名同权重）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgmt.rewards.spec import SIGMAS, exp_tracking_reward, relaxed_error

# ---------------------------------------------------------------------------
# 状态与参考（显式数据结构，便于构造测试）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotState:
    """仿真侧的单步状态（跟踪残差所需的量）。

    所有数组都是**单环境**（无 batch 维）；批量在环境层自行堆叠后在
    `torch`/`numpy` 上批处理 —— 本模块保持逐环境纯函数，便于解析验证。

    Args:
        base_pos:      基座世界位置 (3,)
        base_quat:     基座世界朝向 (4,) [w,x,y,z]
        base_lin_vel:  基座线速度 (**世界系**) (3,)
        base_ang_vel:  基座角速度 (**基座系**) (3,)
        gravity_z:     投影重力的 z 分量（用于终止判据；跟踪残差不用）
        joint_pos:     关节位置 (nq,)
        joint_vel:     关节速度 (nq,)
        joint_acc:     关节加速度 (nq,) —— 用于 aux 的平滑惩罚
        link_pos:      {body: (3,)} 世界位置
        link_quat:     {body: (4,)} 世界朝向
        link_lin_vel:  {body: (3,)} 世界线速度
        link_ang_vel:  {body: (3,)} 世界角速度
    """

    base_pos: np.ndarray
    base_quat: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    gravity_z: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_acc: np.ndarray
    link_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    link_quat: Dict[str, np.ndarray] = field(default_factory=dict)
    link_lin_vel: Dict[str, np.ndarray] = field(default_factory=dict)
    link_ang_vel: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        checks = [
            ("base_pos", self.base_pos, 3),
            ("base_quat", self.base_quat, 4),
            ("base_lin_vel", self.base_lin_vel, 3),
            ("base_ang_vel", self.base_ang_vel, 3),
        ]
        for name, arr, dim in checks:
            if np.shape(arr) != (dim,):
                raise ValueError(f"{name} 形状应为 ({dim},)，得到 {np.shape(arr)}")
        nq = np.shape(self.joint_pos)[0]
        if nq == 0:
            raise ValueError("joint_pos 不能为空")
        for name, arr in (("joint_vel", self.joint_vel),
                          ("joint_acc", self.joint_acc)):
            if np.shape(arr) != (nq,):
                raise ValueError(
                    f"{name} 形状应为 ({nq},)，得到 {np.shape(arr)}")
        for dname in ("link_pos", "link_quat", "link_lin_vel", "link_ang_vel"):
            d = getattr(self, dname)
            want = 4 if dname == "link_quat" else 3
            for body, v in d.items():
                if np.shape(v) != (want,):
                    raise ValueError(
                        f"{dname}[{body}] 形状应为 ({want},)，得到 {np.shape(v)}")

    @property
    def num_joints(self) -> int:
        return int(np.shape(self.joint_pos)[0])


@dataclass(frozen=True)
class ReferenceFrame:
    """参考运动在某一时刻的状态（由 `reference_sampler` 提供）。

    与 `RobotState` 同构，但只有跟踪残差需要的量。调用单项残差时，
    `bodies` 可显式选择非空子集；所选 body 在该项的两侧字典中都必须存在。
    `compute_residuals` 计算全部六项，因此需要所选 body 的全部四类连杆量。
    """

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    #: 参考**锚点**（根）的世界位置与朝向 —— root-centric 表述下用于对齐
    anchor_pos: np.ndarray
    anchor_quat: np.ndarray
    link_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    link_quat: Dict[str, np.ndarray] = field(default_factory=dict)
    link_lin_vel: Dict[str, np.ndarray] = field(default_factory=dict)
    link_ang_vel: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        for name, arr in (("anchor_pos", self.anchor_pos),
                          ("anchor_quat", self.anchor_quat)):
            want = 3 if name == "anchor_pos" else 4
            if np.shape(arr) != (want,):
                raise ValueError(f"{name} 形状应为 ({want},)，得到 {np.shape(arr)}")
        nq = np.shape(self.joint_pos)[0]
        if np.shape(self.joint_vel) != (nq,):
            raise ValueError("joint_vel 与 joint_pos 长度不一致")


# ---------------------------------------------------------------------------
# 可分组的 body / joint 集合
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Partitions:
    """把 body 与 joint 划分到 upper / lower（论文未给划分，登记为 A21）。

    划分依据论文的意图："Upper-body objectives primarily preserve motion
    expression and pose fidelity, whereas lower-body objectives are closely
    coupled with balance and contact transitions."
    → **躯干 + 双臂 = upper**（保姿态表达），**双腿 = lower**（管平衡与接触）。
    """

    upper_bodies: Tuple[str, ...]
    lower_bodies: Tuple[str, ...]
    upper_joints: Tuple[str, ...] = ()
    lower_joints: Tuple[str, ...] = ()

    def __post_init__(self):
        if set(self.upper_bodies) & set(self.lower_bodies):
            raise ValueError("upper / lower 的 body 集合不得重叠")
        if set(self.upper_joints) & set(self.lower_joints):
            raise ValueError("upper / lower 的 joint 集合不得重叠")
        if not self.upper_bodies and not self.lower_bodies:
            raise ValueError("至少要有若干 body")


def default_partitions() -> Partitions:
    """G1 的默认划分（29-DoF）。

    upper = torso + 双臂（含肩/肘/腕），lower = 双腿（髋/膝/踝）。
    关节名与 `data/retarget_lafan1.G1_JOINT_NAMES` 的前缀约定一致。
    """
    upper_bodies = (
        "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    )
    lower_bodies = (
        "pelvis",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    )
    upper_joints = (
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    )
    lower_joints = (
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
        "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
        "right_ankle_pitch", "right_ankle_roll",
    )
    return Partitions(upper_bodies=upper_bodies, lower_bodies=lower_bodies,
                      upper_joints=upper_joints, lower_joints=lower_joints)


# ---------------------------------------------------------------------------
# 基本度量
# ---------------------------------------------------------------------------


def ref_anchor_from_links(ref: ReferenceFrame) -> Tuple[np.ndarray, np.ndarray]:
    """从参考的 body 字典里取**锚点**（pelvis）的位置与朝向。

    论文的 "reference root anchor" 在重定向数据里就是 `pelvis`。抽成函数是为了
    把"锚点是哪个 body"这件事显式化 —— 若数据管线换了锚点定义，这里会立刻报错，
    而不是静默用错参考系算 root-centric 误差。
    """
    if "pelvis" not in ref.link_pos or "pelvis" not in ref.link_quat:
        raise KeyError(
            "参考缺少 pelvis（root-centric 对齐全依赖锚点）。"
            f"已有 body: {sorted(ref.link_pos)}")
    return (np.asarray(ref.link_pos["pelvis"], dtype=np.float64),
            np.asarray(ref.link_quat["pelvis"], dtype=np.float64))


def relative_rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """两个四元数朝向之间的最短旋转角（rad）∈ [0, π]。

    对相对四元数用 atan2，避免 trace/acos 在零误差附近将舍入噪声放大。
    q 与 -q 等价；有限非零四元数会先归一化。
    """
    qa, qb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    for name, q in (("a", qa), ("b", qb)):
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{name} 必须为有限的 (4,) 四元数")
        if np.linalg.norm(q) < 1e-12:
            raise ValueError(f"{name} 不能是零四元数")
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    # conjugate(qa) * qb；实部取绝对值选择最短弧。
    vector = qa[0] * qb[1:] - qb[0] * qa[1:] - np.cross(qa[1:], qb[1:])
    scalar = abs(float(np.dot(qa, qb)))
    return float(2.0 * np.arctan2(np.linalg.norm(vector), scalar))


def _mean(values: Sequence[float]) -> float:
    if not len(values):
        raise ValueError("跟踪目标不能为空；空残差不能作为完美跟踪")
    return float(np.mean(values))


def _require_bodies(state: RobotState, ref: ReferenceFrame,
                    bodies: Sequence[str], field_name: str) -> None:
    """子集由调用者显式选定；缺测量不能静默缩小目标集并提高奖励。"""
    if not len(bodies):
        raise ValueError("bodies 不能为空；请显式选择要跟踪的 body 子集")
    if len(set(bodies)) != len(bodies):
        raise ValueError("bodies 不得重复，否则会重复加权")
    for side, values in (("state", getattr(state, field_name)),
                         ("ref", getattr(ref, field_name))):
        missing = set(bodies) - set(values)
        if missing:
            raise KeyError(f"{side}.{field_name} 缺少请求的 body: {sorted(missing)}")


def _check_same_length(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if np.shape(a) != np.shape(b):
        raise ValueError(f"{what} 形状不一致: {np.shape(a)} vs {np.shape(b)}")


# ---------------------------------------------------------------------------
# 跟踪残差（Table I 的 12 项，按 body/joint 子集算）
# ---------------------------------------------------------------------------


def link_pos_residual(state: RobotState, ref: ReferenceFrame,
                      bodies: Sequence[str]) -> float:
    """连杆位置残差（m）：各自位置差的欧氏距离，再对 body 取**均值**。

    root-centric：调用方应先把参考锚点对齐到机器人锚点（见 `align_reference`），
    因此这里直接比较世界位置即可。
    """
    _require_bodies(state, ref, bodies, "link_pos")
    errs: List[float] = []
    for b in bodies:
        errs.append(float(np.linalg.norm(
            np.asarray(state.link_pos[b]) - np.asarray(ref.link_pos[b]))))
    return _mean(errs)


def link_ori_residual(state: RobotState, ref: ReferenceFrame,
                      bodies: Sequence[str]) -> float:
    """连杆朝向残差（rad）：各自的旋转角，再对 body 取均值。"""
    _require_bodies(state, ref, bodies, "link_quat")
    errs: List[float] = []
    for b in bodies:
        errs.append(relative_rotation_angle(state.link_quat[b], ref.link_quat[b]))
    return _mean(errs)


def link_lin_vel_residual(state: RobotState, ref: ReferenceFrame,
                          bodies: Sequence[str]) -> float:
    """连杆线速度残差（m/s）：差向量范数，对 body 取均值。"""
    _require_bodies(state, ref, bodies, "link_lin_vel")
    errs: List[float] = []
    for b in bodies:
        errs.append(float(np.linalg.norm(
            np.asarray(state.link_lin_vel[b]) - np.asarray(ref.link_lin_vel[b]))))
    return _mean(errs)


def link_ang_vel_residual(state: RobotState, ref: ReferenceFrame,
                          bodies: Sequence[str]) -> float:
    """连杆角速度残差（rad/s）：差向量范数，对 body 取均值。"""
    _require_bodies(state, ref, bodies, "link_ang_vel")
    errs: List[float] = []
    for b in bodies:
        errs.append(float(np.linalg.norm(
            np.asarray(state.link_ang_vel[b]) - np.asarray(ref.link_ang_vel[b]))))
    return _mean(errs)


def joint_pos_residual(state: RobotState, ref: ReferenceFrame,
                       joint_idx: Optional[Sequence[int]] = None) -> float:
    """关节位置残差（rad）：绝对差，对关节取均值。"""
    _check_same_length(state.joint_pos, ref.joint_pos, "joint_pos")
    d = np.abs(np.asarray(state.joint_pos) - np.asarray(ref.joint_pos))
    if joint_idx is not None:
        d = d[np.asarray(list(joint_idx), dtype=np.int64)]
    return _mean(d)


def joint_vel_residual(state: RobotState, ref: ReferenceFrame,
                       joint_idx: Optional[Sequence[int]] = None) -> float:
    """关节速度残差（rad/s）：绝对差，对关节取均值。"""
    _check_same_length(state.joint_vel, ref.joint_vel, "joint_vel")
    d = np.abs(np.asarray(state.joint_vel) - np.asarray(ref.joint_vel))
    if joint_idx is not None:
        d = d[np.asarray(list(joint_idx), dtype=np.int64)]
    return _mean(d)


# ---------------------------------------------------------------------------
# 参考对齐（root-centric 的关键一步）
# ---------------------------------------------------------------------------


def align_reference(state: RobotState, ref: ReferenceFrame,
                    yaw_only: bool = True) -> ReferenceFrame:
    """把参考整体平移（可选：只按 yaw 旋转）到机器人锚点，实现 root-centric 比较。

    论文：tracking errors are computed relative to the reference root anchor,
    使策略从不同初始位姿/朝向都能复现同一运动意图。

    Args:
        yaw_only: True 时只用绕 z 的偏航把参考"转正"到机器人朝向 ——
            **不平移**机器人的俯仰/横滚。这样机器人倾斜时不会被误判为
            "参考也倾斜"。论文强调减少对绝对朝向的依赖，取 yaw-only 更贴合。
    Returns:
        平移/旋转后的新 `ReferenceFrame`（不修改入参）
    """
    dpos = np.asarray(state.base_pos) - np.asarray(ref.anchor_pos)
    if not yaw_only:
        out = ReferenceFrame(
            joint_pos=ref.joint_pos, joint_vel=ref.joint_vel,
            anchor_pos=np.asarray(state.base_pos, dtype=np.float64),
            anchor_quat=ref.anchor_quat,
            link_pos={k: np.asarray(v) + dpos for k, v in ref.link_pos.items()},
            link_quat=dict(ref.link_quat),
            link_lin_vel=dict(ref.link_lin_vel),
            link_ang_vel=dict(ref.link_ang_vel),
        )
        return out

    # 将参考朝向主动旋转到机器人朝向，所有输出仍在世界坐标系。
    # 先减参考锚点、再转 R(yaw_robot-yaw_ref)、最后加机器人基座。
    yaw_r = _yaw_of(state.base_quat)
    yaw_f = _yaw_of(ref.anchor_quat)
    dyaw = yaw_r - yaw_f
    c, s = float(np.cos(dyaw)), float(np.sin(dyaw))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    anchor = np.asarray(ref.anchor_pos, dtype=np.float64)
    base = np.asarray(state.base_pos, dtype=np.float64)

    def to_robot_frame(v):
        return Rz @ (np.asarray(v, dtype=np.float64) - anchor) + base

    def rot_q(q):
        # 与位置、速度使用同一个主动 yaw 旋转。
        hq = np.array([np.cos(dyaw / 2.0), 0.0, 0.0, np.sin(dyaw / 2.0)])
        w1, x1, y1, z1 = hq
        w2, x2, y2, z2 = np.asarray(q, dtype=np.float64)
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    return ReferenceFrame(
        joint_pos=ref.joint_pos, joint_vel=ref.joint_vel,
        anchor_pos=base.copy(),
        anchor_quat=rot_q(ref.anchor_quat),
        link_pos={k: to_robot_frame(v) for k, v in ref.link_pos.items()},
        link_quat={k: rot_q(v) for k, v in ref.link_quat.items()},
        link_lin_vel={k: Rz @ np.asarray(v, dtype=np.float64)
                      for k, v in ref.link_lin_vel.items()},
        link_ang_vel={k: Rz @ np.asarray(v, dtype=np.float64)
                      for k, v in ref.link_ang_vel.items()},
    )


def _yaw_of(q: np.ndarray) -> float:
    """四元数的偏航角（绕 z，rad）。

    **不重复实现旋转公式**，而是复用 `data.bvh.quat_rot_vec`（数据管线与
    重定向已依赖的同一实现，且有自己的单测）。理由：本模块第一版手写
    "从旋转矩阵元素取 atan2" 时把符号搞反了（被
    `test_align_reference_makes_matching_state_zero_residual` 抓住 ——
    残差 0.978 恰是绕 z 转 2ψ 的位移量，说明对齐用了 −ψ 而非 +ψ）。

    做法：把机器人本体的 x 轴（前进方向）用四元数旋转，取其在 xy 平面上的
    方位角。偏航定义与 `quat_to_mat`/`quat_rot_vec` 的约定自动一致。
    """
    from data.bvh import quat_rot_vec

    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("零四元数无偏航角")
    v = quat_rot_vec(q / n, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


# ---------------------------------------------------------------------------
# 分组奖励（论文 Eq.5 的 r_upper / r_lower）
# ---------------------------------------------------------------------------


def compute_residuals(state: RobotState, ref: ReferenceFrame,
                      bodies: Sequence[str],
                      joint_idx: Optional[Sequence[int]] = None,
                      ) -> Dict[str, float]:
    """算齐一组（upper 或 lower）的 6 个残差。

    Returns:
        `{"link_pos": …, "link_ori": …, "link_lin_vel": …, "link_ang_vel": …,
           "joint_pos": …, "joint_vel": …}`
    """
    return {
        "link_pos": link_pos_residual(state, ref, bodies),
        "link_ori": link_ori_residual(state, ref, bodies),
        "link_lin_vel": link_lin_vel_residual(state, ref, bodies),
        "link_ang_vel": link_ang_vel_residual(state, ref, bodies),
        "joint_pos": joint_pos_residual(state, ref, joint_idx),
        "joint_vel": joint_vel_residual(state, ref, joint_idx),
    }


def residuals_by_partition(state: RobotState, ref: ReferenceFrame,
                           parts: "Partitions",
                           joint_names: Sequence[str],
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """按 `Partitions` 一次算出 `(upper 残差, lower 残差)`。

    `joint_names` 用于把 upper/lower 的**关节名**翻译成 `joint_pos` 数组里的
    下标 —— 关节顺序由数据管线固定（`G1_JOINT_NAMES`），此处显式对齐而不是
    假设顺序，避免"顺序变了却静默算错"。
    """
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    missing = (set(parts.upper_joints) | set(parts.lower_joints)) - set(name_to_idx)
    if missing:
        raise KeyError(f"joint_names 缺少: {sorted(missing)}")
    up_idx = tuple(sorted(name_to_idx[n] for n in parts.upper_joints))
    lo_idx = tuple(sorted(name_to_idx[n] for n in parts.lower_joints))
    upper = compute_residuals(state, ref, parts.upper_bodies, up_idx)
    lower = compute_residuals(state, ref, parts.lower_bodies, lo_idx)
    return upper, lower


def group_term_values(residuals: Mapping[str, float],
                      group: str,
                      terrain_family: Optional[str] = None,
                      level: int = 0,
                      alpha: float = 1.0,
                      tau_m: float = 0.0,
                      tau_rad: float = 0.0) -> Dict[str, float]:
    """把残差映射为**按 Table I 项名索引**的值（含地形松弛）。

    单独抽出来是为了可测：`group_reward` 的输出只是一个标量，无法直接验证
    "lower 组的三个 TA 项确实被映射到了 `ta_*` 项名、且只有它们被松弛"。
    这里返回逐项的值，测试可以直接核对键名与逐项取值。

    项名映射：lower 组在 Table I 中名为 `ta_link_pos` / `ta_link_ori` /
    `ta_joint_pos`，而残差键无前缀 —— 在此加前缀，使 `spec.RewardGroup.sum`
    能直接消费。非 TA 项与残差键同名。
    """
    from pgmt.rewards.spec import LOWER, UPPER, chi

    if group not in ("upper", "lower"):
        raise ValueError(f"group 应为 'upper' 或 'lower'，得到 {group!r}")
    grp = UPPER if group == "upper" else LOWER

    #: Table I 项名 → (残差键, 松弛预算)。只有这三个 TA 项带预算
    rename = {
        "ta_link_pos": ("link_pos", tau_m),
        "ta_link_ori": ("link_ori", tau_rad),
        "ta_joint_pos": ("joint_pos", tau_rad),
    }
    out: Dict[str, float] = {}
    for term in grp.names:
        src, tau = rename.get(term, (term, 0.0))
        if src not in residuals:
            raise KeyError(
                f"{group} 组需要残差 {src!r}，未提供（已有 {sorted(residuals)}）")
        e = float(residuals[src])
        if terrain_family is not None and tau > 0.0:
            e = relaxed_error(e, alpha, chi(terrain_family), tau)
        out[term] = exp_tracking_reward(e, SIGMAS[term])
    return out


def group_reward(residuals: Mapping[str, float],
                 group: str,
                 terrain_family: Optional[str] = None,
                 level: int = 0,
                 alpha: float = 1.0,
                 tau_m: float = 0.0,
                 tau_rad: float = 0.0) -> float:
    """把一组残差映射成该组的加权标量和（对应 `spec.RewardGroup.sum`）。

    Args:
        group: `"upper"` 或 `"lower"`
        terrain_family / level / alpha: 地形松弛参数；`terrain_family=None`
            表示**不松弛**（Stage 1 平地）。
        tau_m / tau_rad: 位置类与角度类目标的松弛预算（单位分别为米、弧度）。
            由调用方按 A12/A18 从难度算出后传入（本模块不重复推导，避免两处口径）

    注：零残差时 upper 与 lower 的奖励**恰好相等**（权重和都是 4.5），
    这不是 bug —— 别用"两组不等"来验证映射（我第一版就这么错过）。
    """
    from pgmt.rewards.spec import LOWER, UPPER

    grp = UPPER if group == "upper" else LOWER
    values = group_term_values(residuals, group, terrain_family, level,
                               alpha, tau_m, tau_rad)
    return grp.sum(values)


#: 受地形松弛作用的三项（即 Table I 中带 TA 的项）
TA_TERMS: Tuple[str, ...] = ("ta_link_pos", "ta_link_ori", "ta_joint_pos")
