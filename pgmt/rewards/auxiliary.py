"""辅助项残差（M2 待办 #1，论文 Table I 的 Auxiliary 组，10 项）。

## 与跟踪组的结构差异（重要）

Table I 的 auxiliary 组**不是**统一的"正向跟踪"项：

  - **4 个正向项**（权重为正）：`root_ori` 0.5、`corrected_root_vel` 2.0、
    `floating_anchor_pos` 1.0、`recovery_upward_vel` 12.5
    → 用与跟踪组相同的高斯核 `exp(−e²/σ)`
  - **6 个惩罚项**（Table I 权重为负）：`pelvis_vert_accel` −1e−3、
    `ee_accel_mismatch` −1e−3、`action_rate` −0.05、`joint_limit` −15.0、
    `undesired_contact` −0.1、`head_torso_impact` −1e−5
    → 返回**非负代价量**（`‖·‖²`、越界量平方），由组权重把它变成惩罚

论文没有明说这个区分，但 Table I 的权重符号已经指明了 —— 若把惩罚项也套高斯核
（取值 ∈ (0,1]），负权重会变成"越差越接近 0"的古怪激励。故按符号分流。

## ⚠️ 符号约定（本模块第一版写错的地方，6 项全反）

**所有项的值恒 ≥ 0，符号全部由 Table I 的权重携带。**

第一版让惩罚项返回 `−‖·‖²`（负值），配上 Table I 本来的负权重
（如 `action_rate` −0.05）：负 × 负 = **正贡献**。后果是
"动作变化率惩罚"在**奖励**动作变化，`undesired_contact` 在**奖励**用力撞击
非期望部位 —— 惩罚项集体反转成奖励。

因此函数名统一用 `*_cost` 而非 `*_penalty`，表示"这是要被负权重惩罚的代价量"。
并且 `spec.RewardGroup.sum` 现在会在运行时拒绝负值/NaN（见那里的说明），
使这类符号错误在第一次调用就炸掉，而不是安静地把训练带偏。

注意这与**正向项**一致：正向项返回 `exp(−e²/σ) ∈ (0,1]` 也 ≥ 0。
两类项的值域因此**都是非负**，符号只在权重里 —— 这是一条可单测的强不变量。

## 论文说了什么、没说什么

**明确的**：
  - `corrected_root_vel` 对应 §IV-B 的 Global Position Correction
    （`ṽ^r = v^r + clip(g(‖v^r‖)λ_pos e^p, ±v̄)`，见 A13）
  - `recovery_upward_vel` 属 recovery curriculum（论文 follows RGMT）
  - `joint_limit` 是关节限位软约束
  - 其余项只有项名与权重

**未说的**（登记 A21）：各惩罚的**度量方式**与**尺度阈值**：
  - `head_torso_impact`：论文完全没说怎么度量 → 本实现取"头/躯干接触力超过
    阈值的部分"，并**显式标为待定**（见 `semantics.py` 的 UNRESOLVED）
  - `ee_accel_mismatch`：与**参考**比还是与**上一帧**比 → 取与参考比
  - `pelvis_vert_accel`：加速度平方（论文未说是否取绝对值/平方）
  - `undesired_contact`：哪些 body 属"非期望" → 取 config 的白名单

**因此本模块的参数集中在 `AuxCfg`（A21），且每一项都标明依据强度。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from pgmt.cfg.assumptions import AuxCfg, get  # noqa: F401  （类型在此转出）
from pgmt.rewards.spec import exp_tracking_reward

# ---------------------------------------------------------------------------
# 正向项（高斯核）
# ---------------------------------------------------------------------------


def root_ori_residual(state_quat: np.ndarray, ref_quat: np.ndarray) -> float:
    """根朝向残差（rad）：两个朝向之间的旋转角。

    root-centric 下参考已被对齐到机器人锚点系（`tracking.align_reference`），
    故这里直接比较即可。取**旋转角**而非 6D 差范数，与跟踪组保持一致。
    """
    from pgmt.rewards.tracking import relative_rotation_angle

    return relative_rotation_angle(state_quat, ref_quat)


def corrected_root_vel_residual(state_lin_vel_world: np.ndarray,
                                ref_anchor_lin_vel_world: np.ndarray) -> float:
    """修正后根速度残差（m/s）：世界系速度差的范数。

    `ref_anchor_lin_vel_world` 应是**已修正**的参考锚点速度 —— 即 A13 的
    `ṽ^r = v^r + clip(g(‖v^r‖)λ_pos e^p, ±v̄)`。
    `reference_sampler` 在参考锚点系计算该目标；环境层必须用对应参考锚点的
    完整朝向旋转回世界系，再与机器人的世界系速度一起传入，不能混用坐标系。
    本函数不重复施加修正，避免两处口径。
    """
    a = np.asarray(state_lin_vel_world, dtype=np.float64)
    b = np.asarray(ref_anchor_lin_vel_world, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError(f"速度应为 (3,)，得到 {a.shape} / {b.shape}")
    return float(np.linalg.norm(a - b))


def floating_anchor_pos_residual(state_anchor_pos: np.ndarray,
                                 ref_anchor_pos: np.ndarray) -> float:
    """浮动 anchor 位置残差（m）：两个锚点位置的欧氏距离。

    该项与"根朝向"共同约束锚点本身 —— 论文的 root-centric 表述只把**本体
    跟踪误差**相对锚点计算，锚点自身仍需要一个跟踪信号，否则锚点会自由漂移。
    """
    a = np.asarray(state_anchor_pos, dtype=np.float64)
    b = np.asarray(ref_anchor_pos, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError(f"锚点位置应为 (3,)，得到 {a.shape} / {b.shape}")
    return float(np.linalg.norm(a - b))


def recovery_upward_vel_residual(upward_vel: float, target_upward_vel: float
                                 ) -> float:
    """恢复上行速度残差（m/s）：`max(0, target − 实际上行速度)`。

    论文的 recovery curriculum 要"从摔倒状态起身"，因此该项**只惩罚不足**：
    上行速度低于目标时才有残差，超过目标不罚（不该惩罚"起得更快"）。
    这与其它跟踪项的对称残差不同，是本项刻意的非对称设计。
    """
    if target_upward_vel < 0.0:
        raise ValueError(f"target_upward_vel 应为非负，得到 {target_upward_vel}")
    return float(max(0.0, float(target_upward_vel) - float(upward_vel)))


# ---------------------------------------------------------------------------
# 惩罚项 —— 返回**非负代价量**（无违规 ⇒ 恰为 0），符号由 Table I 的负权重携带
# ---------------------------------------------------------------------------


def pelvis_vert_accel_cost(vert_accel: float) -> float:
    """骨盆竖直加速度代价：`(a_z)² ≥ 0`（权重 −1e−3 负责变负）。

    取**平方**（而非绝对值）使量纲与其它平方代价一致、且在 0 处可导。
    论文未说明形式，登记 A21。
    """
    return float(vert_accel) ** 2


def ee_accel_mismatch_cost(state_ee_accel: np.ndarray,
                           ref_ee_accel: np.ndarray) -> float:
    """末端执行器加速度不匹配代价：`‖a_state − a_ref‖² ≥ 0`。

    "mismatch" 的对象论文未说明 —— 取**与参考比**（而非与上一帧比），
    因为该项与其它跟踪项同组，语义上应与参考一致。属 A21 拍定，
    在 `semantics.py` 中标记为 UNRESOLVED（依据不足）。
    """
    a = np.asarray(state_ee_accel, dtype=np.float64)
    b = np.asarray(ref_ee_accel, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"加速度形状不一致: {a.shape} vs {b.shape}")
    return float(np.sum((a - b) ** 2))


def action_rate_cost(action: np.ndarray, prev_action: np.ndarray) -> float:
    """动作变化率代价：`‖a_t − a_{t−1}‖² ≥ 0`（控制平滑）。"""
    a = np.asarray(action, dtype=np.float64)
    p = np.asarray(prev_action, dtype=np.float64)
    if a.shape != p.shape:
        raise ValueError(f"动作形状不一致: {a.shape} vs {p.shape}")
    return float(np.sum((a - p) ** 2))


def joint_limit_cost(joint_pos: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray) -> float:
    """关节限位代价：**越界量的平方和** ≥ 0（界内恰为 0）。

    取"[越界量]²"而非"到限位的距离"：界内不应有任何代价，否则等于把关节
    往中间推（论文的该项是**软约束**，不是"偏好中位"）。
    """
    q = np.asarray(joint_pos, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if not (q.shape == lo.shape == hi.shape):
        raise ValueError(f"形状不一致: q{q.shape} lo{lo.shape} hi{hi.shape}")
    if np.any(lo > hi):
        raise ValueError("存在 lower > upper 的关节")
    over = np.maximum(lo - q, 0.0) + np.maximum(q - hi, 0.0)
    return float(np.sum(over ** 2))


def undesired_contact_cost(contact_forces: Mapping[str, float],
                           allowed_bodies: Sequence[str],
                           threshold: float = 0.0) -> float:
    """非期望部位接触代价：除允许部位外，接触力超过阈值的部位数。

    A21 工程定义；撞击强度由独立的 head_torso_impact 项度量。

    Args:
        contact_forces: {body: 法向接触力大小}
        allowed_bodies: 允许接触的部位（通常是双足）—— 其余部位一旦接触即计入
        threshold: 力的死区（N）；论文未给，登记 A21
    """
    if threshold < 0.0:
        raise ValueError(f"threshold 应为非负，得到 {threshold}")
    allowed = set(allowed_bodies)
    total = 0.0
    for body, f in contact_forces.items():
        if body in allowed:
            continue
        total += float(float(f) > threshold)
    return float(total)


def head_torso_impact_cost(impact_force: float, threshold: float) -> float:
    """头/躯干撞击代价：超过阈值的部分平方 ≥ 0。

    ⚠️ **论文完全没说"撞击"如何度量**（接触力？冲量？穿透深度？）。本实现取
    "接触力超过阈值的部分"，并在 `pgmt/rewards/semantics.py` 中把该项标为
    `UNRESOLVED`（依据不足、实现前需拍定）。此处保留实现以便组合项连通，
    但**不得据此声称已复现论文该项**。
    """
    if threshold < 0.0:
        raise ValueError(f"threshold 应为非负，得到 {threshold}")
    over = max(float(impact_force) - threshold, 0.0)
    return float(over ** 2)


# ---------------------------------------------------------------------------
# 配置（A21）与组合
# ---------------------------------------------------------------------------

# `AuxCfg` 的**唯一定义在 `pgmt/cfg/assumptions.py`**（A21 的值类型），此处复用。
#
# ⚠️ 这个坑我在 `TerminationCfg` 上已经犯过一次（两处定义 → `isinstance` 失败），
# 写本模块时**又复制了一遍**。规则：**假设表里出现的 dataclass 只在
# `assumptions.py` 定义一次，实现模块一律 import**，不得再写一份"等价的"。


def aux_cfg() -> AuxCfg:
    """取 A21 的配置（唯一出处）。"""
    return get("A21").value


def compute_aux_values(state: "AuxState", cfg: Optional[AuxCfg] = None
                       ) -> Dict[str, float]:
    """算齐 Table I 的 10 个 auxiliary 项值（**未加权**）。

    返回的键与 `spec.AUX_TERMS` 的项名一致，可直接交给 `RewardGroup.sum`。

    值域（见模块头的符号约定）：正向项 `exp(−e²/σ) ∈ (0,1]`，
    惩罚项为**非负代价量**（无违规恰为 0）。**全部 ≥ 0**，符号在权重里。
    """
    c = cfg or aux_cfg()
    return {
        "root_ori": exp_tracking_reward(
            root_ori_residual(state.base_quat, state.ref_quat),
            c.sigma_root_ori),
        "corrected_root_vel": exp_tracking_reward(
            corrected_root_vel_residual(state.base_lin_vel,
                                        state.ref_anchor_lin_vel),
            c.sigma_corrected_root_vel),
        "floating_anchor_pos": exp_tracking_reward(
            floating_anchor_pos_residual(state.anchor_pos, state.ref_anchor_pos),
            c.sigma_floating_anchor),
        "recovery_upward_vel": exp_tracking_reward(
            recovery_upward_vel_residual(state.upward_vel,
                                         c.recovery_target_upward_vel),
            c.sigma_recovery_upward),
        "pelvis_vert_accel": pelvis_vert_accel_cost(state.pelvis_vert_accel),
        "ee_accel_mismatch": ee_accel_mismatch_cost(state.ee_accel,
                                                    state.ref_ee_accel),
        "action_rate": action_rate_cost(state.action, state.prev_action),
        "joint_limit": joint_limit_cost(state.joint_pos,
                                        state.joint_lower, state.joint_upper),
        "undesired_contact": undesired_contact_cost(
            state.contact_forces, c.allowed_contact_bodies,
            c.contact_force_threshold),
        "head_torso_impact": head_torso_impact_cost(
            state.head_torso_impact_force, c.head_torso_impact_threshold),
    }


@dataclass(frozen=True)
class AuxState:
    """辅助项所需的单步状态（逐环境，无 batch 维）。"""

    #: 机器人锚点（基座）世界朝向与线速度
    base_quat: np.ndarray
    base_lin_vel: np.ndarray
    #: 机器人锚点位置（= 基座位置）
    anchor_pos: np.ndarray
    #: 参考侧：对齐后的参考朝向、**已修正**的参考锚点速度、参考锚点位置
    ref_quat: np.ndarray
    ref_anchor_lin_vel: np.ndarray
    ref_anchor_pos: np.ndarray
    #: 上行速度（m/s，世界系 z 分量）—— recovery 项用
    upward_vel: float
    #: 骨盆竖直加速度（m/s²）
    pelvis_vert_accel: float
    #: 末端执行器加速度：当前 (3k,) 与参考 (3k,)
    ee_accel: np.ndarray
    ref_ee_accel: np.ndarray
    #: 动作与上一动作 (nq,)
    action: np.ndarray
    prev_action: np.ndarray
    #: 关节位置与限位 (nq,)
    joint_pos: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    #: 接触力 {body: N}
    contact_forces: Dict[str, float] = field(default_factory=dict)
    #: 头/躯干撞击力（N）
    head_torso_impact_force: float = 0.0

    def __post_init__(self):
        for name in ("base_quat", "ref_quat"):
            if np.shape(getattr(self, name)) != (4,):
                raise ValueError(f"{name} 应为 (4,)")
        for name in ("base_lin_vel", "ref_anchor_lin_vel", "anchor_pos",
                     "ref_anchor_pos"):
            if np.shape(getattr(self, name)) != (3,):
                raise ValueError(f"{name} 应为 (3,)")
        nq = np.shape(self.joint_pos)[0]
        for name in ("action", "prev_action"):
            if np.shape(getattr(self, name)) != (nq,):
                raise ValueError(f"{name} 应为 ({nq},)")
        for name in ("joint_lower", "joint_upper"):
            if np.shape(getattr(self, name)) != (nq,):
                raise ValueError(f"{name} 应为 ({nq},)")
        if np.shape(self.ee_accel) != np.shape(self.ref_ee_accel):
            raise ValueError("ee_accel 与 ref_ee_accel 形状须一致")


def check_aux_values_complete(values: Mapping[str, float]) -> None:
    """自检：项名与 `spec.AUX_TERMS` 完全一致（不漏不多），且值域合法。

    值域检查即模块头的符号约定：**所有项 ≥ 0**。负值/NaN 会让负权重变成
    正贡献（惩罚反转成奖励），属静默错误，故在这里也拦一道 —— 与
    `RewardGroup.sum` 的运行时守卫重复是刻意的（本函数可在不求和时单独调用）。
    """
    from pgmt.rewards.spec import AUX_TERMS

    want = {n for n, _ in AUX_TERMS}
    got = set(values)
    if got != want:
        raise KeyError(f"auxiliary 项名不匹配：缺 {sorted(want - got)}，"
                       f"多 {sorted(got - want)}")
    bad = {n: v for n, v in values.items() if not (float(v) >= 0.0)}
    if bad:
        raise ValueError(f"auxiliary 项出现负值/NaN: {bad}；"
                         f"本仓库约定所有项的值 ≥ 0，符号由权重携带")
