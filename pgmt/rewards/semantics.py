"""Table I 十六项与参照实现的语义对照表（A18 的语义依据）。

## 为什么需要它

`pgmt/rewards/spec.py` 照搬了论文 Table I 的**权重**，但每一项的**语义**
（在哪个坐标系里量什么误差、对哪些 body/关节、单位是什么）论文没写。
而这套 tracking 奖励是 PGMT 明示继承的既有约定（论文用定冠词称
"**the** exponential tracking reward"），因此可以用参照实现交叉验证语义解读。

本表把每一项标注为四种对应关系之一，并记录置信度与依据：

  - `IDENTICAL`  ：参照实现里有明确同项（依据可直接核对）
  - `APPROX`     ：参照实现里有近似项，语义有差异（差异写进 note）
  - `PGMT_SPECIFIC`：论文新引入，参照实现无对应项（如 terrain-contact 组）
  - `UNRESOLVED` ：未找到足够依据判断，**明确留下待办**而不是猜

## 这张表能证实什么 / 不能证实什么

**能**：验证我对 16 项语义的解读是否有参照依据；把 `TA` 的含义从推断
变成有证据的判断；暴露"我以为同项、其实语义不同"的风险。

**不能**：验证权重数值。各实现的权重表**不可直接数值比对** ——
legged_gym 系常按 `dt` 归一化、且跟踪 body 数量不同（PGMT 用 G1 的 29-DoF
body 集，OmniH2O 用其自身 body 集），量级自然不同。数值只以论文 Table I 为准。

## 参照来源

- OmniH2O（human2humanoid）论文奖励表 + 其
  `legged_gym/legged_gym/cfg/rewards/rewards_teleop_omnih2o_teacher.yaml`
- OmniH2O 配置里 `tracking_sigma : 0.25   # tracking reward : exp(-error^2/sigma)`
  确立了核函数；`teleop_joint_pos_sigma` 的注释 `# 0.05 -> 0.1 lower body`
  确立了"下半身 σ 更大"这一结构，与论文 §IV 的表述一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from pgmt.rewards.spec import AUX_TERMS, LOWER_TERMS, TERRAIN_TERMS, UPPER_TERMS


class Corr(str, Enum):
    """与参照实现的对应关系。"""

    IDENTICAL = "identical"          # 参照实现有明确同项
    APPROX = "approx"                # 近似项，语义有差异（见 note）
    PGMT_SPECIFIC = "pmgt_specific"  # 论文新引入，参照实现无对应
    UNRESOLVED = "unresolved"        # 依据不足，**待办**


@dataclass(frozen=True)
class TermSemantics:
    """一项奖励的语义与出处。

    Args:
        term:     与 `spec.py` 中项名一致（由测试保证同步）
        group:    upper / lower / aux / terrain
        objective: 量的对象与方式（人可读，供实现与报告引用）
        relaxable: 是否受 Eq.10 地形松弛作用（论文：仅 TA 三项）
        counterpart: 参照实现里的对应项名（无则 None）
        corr:     对应关系
        note:     差异、单位、坐标系等必须写清的细节
    """

    term: str
    group: str
    objective: str
    relaxable: bool
    counterpart: str | None
    corr: Corr
    note: str = ""


#: Table I "The 1st & 2nd Stages" 部分
STAGE12_SEMANTICS: Tuple[TermSemantics, ...] = (
    # ---- Upper body（论文：保动作表达与姿态保真）----
    TermSemantics(
        "link_pos", "upper",
        "上身连杆位置误差（root-centric，参考相对 anchor 系）",
        False, "teleop_body_position_extend_upper / _vr_3keypoints",
        Corr.IDENTICAL,
        "参照实现按 keypoint 选择（含 VR 3 点），PGMT 用上身连杆集；σ 同属'严格'档",
    ),
    TermSemantics(
        "link_ori", "upper",
        "上身连杆朝向误差（旋转角或 6D 差）",
        False, "teleop_body_rotation",
        Corr.IDENTICAL,
        "参照实现 rot σ=1.0（其论文表记 exp(−0.5‖·‖²)）",
    ),
    TermSemantics(
        "link_lin_vel", "upper",
        "上身连杆线速度误差",
        False, "teleop_body_vel",
        Corr.IDENTICAL,
    ),
    TermSemantics(
        "link_ang_vel", "upper",
        "上身连杆角速度误差",
        False, "teleop_body_ang_vel",
        Corr.IDENTICAL,
    ),
    TermSemantics(
        "joint_pos", "upper",
        "上身关节位置误差",
        False, "teleop_selected_joint_position",
        Corr.IDENTICAL,
        "参照实现按关节加权（teleop_joint_pos_selection，上身 2.0）",
    ),
    TermSemantics(
        "joint_vel", "upper",
        "上身关节速度误差",
        False, "teleop_selected_joint_vel",
        Corr.IDENTICAL,
    ),
    # ---- Lower body（论文：与平衡和接触转换强耦合）----
    TermSemantics(
        "ta_link_pos", "lower",
        "下身连杆位置误差，**受地形松弛**（TA）",
        True, "teleop_body_position_extend（下半身权重 0.5）",
        Corr.APPROX,
        "参照实现用单独的下半身权重与更大的 σ（配置注释 `0.05 -> 0.1 lower body`），"
        "但**没有** PGMT 的地形松弛项；TA 是 PGMT 新增",
    ),
    TermSemantics(
        "ta_link_ori", "lower",
        "下身连杆朝向误差，受地形松弛",
        True, None,
        Corr.UNRESOLVED,
        "**待办**：参照实现有 body_rotation 项（σ=1.0），但**未验证**它是否按上下半身"
        "分离，因此无法判断 PGMT 的 TA 朝向项是'新增'还是'分离既有项'。"
        "PGMT 给该项权重 2.0（Table I 中最高之一），值得查清。"
        "注：本项原标 PGMT_SPECIFIC，但那是**存在性断言**（'参照实现没有'），"
        "而我只做到'没见过'，故降级为待办",
    ),
    TermSemantics(
        "link_lin_vel", "lower",
        "下身连杆线速度误差 —— **不带 TA**，仍严格",
        False, "teleop_body_vel",
        Corr.APPROX,
        "Table I 的下身速度项无 TA 标记，即速度跟踪不受地形松弛保护（论文原文："
        "relaxation 只作用于 lower-body 的 link position / link orientation / "
        "joint position）",
    ),
    TermSemantics(
        "link_ang_vel", "lower",
        "下身连杆角速度误差 —— 不带 TA，仍严格",
        False, "teleop_body_ang_vel",
        Corr.APPROX,
        "同 link_lin_vel",
    ),
    TermSemantics(
        "ta_joint_pos", "lower",
        "下身关节位置误差，受地形松弛",
        True, "teleop_selected_joint_position（下半身权重 0.5）",
        Corr.APPROX,
        "同 ta_link_pos：权重结构对应，地形松弛为 PGMT 新增",
    ),
    TermSemantics(
        "joint_vel", "lower",
        "下身关节速度误差 —— 不带 TA，仍严格",
        False, "teleop_selected_joint_vel",
        Corr.IDENTICAL,
    ),
    # ---- Auxiliary ----
    TermSemantics(
        "root_ori", "aux",
        "根（基座）朝向偏离期望姿态的惩罚",
        False, "orientation（legged_gym 系通用）",
        Corr.APPROX,
        "各实现的'期望'不同：legged_gym 常惩罚 projected gravity 的 xy 分量；"
        "PGMT 在 root-centric 表述下应为相对参考根朝向 —— **待实现时确认**",
    ),
    TermSemantics(
        "corrected_root_vel", "aux",
        "修正后根速度的跟踪误差（论文 §IV-B Global Position Correction 的 ṽ^r）",
        False, None,
        Corr.PGMT_SPECIFIC,
        "论文自有的全局位置修正机制；λ_pos 与门控 g(·) 见 A13。"
        "该项即'跟踪修正后的速度指令'，与 A13 的 e^p 反馈闭合",
    ),
    TermSemantics(
        "floating_anchor_pos", "aux",
        "浮动 anchor 位置跟踪（root-centric 的锚点本身）",
        False, None,
        Corr.UNRESOLVED,
        "**待办**：语义依据来自 HOVER/PHC 的 floating-base 表述，但本项目读过的"
        "配置里**未确认到同名项**，因此既不能断言'同项'也不能断言'无对应'。"
        "注：原标 APPROX，但 APPROX 要求有可点的近似项，此处只有'某个体系里应存在'，"
        "依据强度不足，降级为待办",
    ),
    TermSemantics(
        "recovery_upward_vel", "aux",
        "摔倒恢复时的向上速度激励（权重 12.5，Table I 中最大的正权重之一）",
        False, None,
        Corr.PGMT_SPECIFIC,
        "论文 recovery curriculum（follows RGMT）的一部分；用于从摔倒状态起身",
    ),
    TermSemantics(
        "pelvis_vert_accel", "aux",
        "骨盆竖直加速度惩罚（平滑性）",
        False, "dof_acc / 平滑类惩罚",
        Corr.APPROX,
        "参照实现的平滑惩罚作用在关节量（dof_acc/dof_vel）；PGMT 明确作用在骨盆竖直加速度",
    ),
    TermSemantics(
        "ee_accel_mismatch", "aux",
        "末端执行器加速度与参考不一致的惩罚",
        False, None,
        Corr.UNRESOLVED,
        "**待办**：论文权重 −1e−3，但未说 'mismatch' 是与**参考**比还是与**上一帧**比；"
        "参照实现里也**未确认到**同项（其平滑惩罚作用在关节量 dof_acc/dof_vel）。"
        "实现前需拍定误差定义并登记为假设。"
        "注：原标 PGMT_SPECIFIC，但那只做到'没见过'，不构成存在性依据",
    ),
    TermSemantics(
        "action_rate", "aux",
        "动作变化率惩罚（控制平滑）",
        False, "lower_action_rate / upper_action_rate",
        Corr.APPROX,
        "参照实现按上下半身**分设**两个 action_rate；PGMT 只有一个 —— "
        "Table I 未分上下身，按论文实现",
    ),
    TermSemantics(
        "joint_limit", "aux",
        "关节限位软约束惩罚",
        False, "dof_pos_limits",
        Corr.IDENTICAL,
        "权重数值不可比：参照实现 −100×1.25，论文 −15.0（各实现归一化口径不同）",
    ),
    TermSemantics(
        "undesired_contact", "aux",
        "非期望部位接触惩罚",
        False, "feet_contact_forces / termination_contact",
        Corr.APPROX,
        "参照实现分'足部接触力'与'终止接触'两类；PGMT 的 undesired_contact 更接近"
        "后者的连续化版本",
    ),
    TermSemantics(
        "head_torso_impact", "aux",
        "头/躯干撞击惩罚（权重 −1e−5，Table I 中最小的权重）",
        False, None,
        Corr.UNRESOLVED,
        "**待办**：论文只给项名与权重，未说撞击如何度量（接触力？冲量？）。"
        "参照实现未见等价项。实现前需拍定并登记为假设",
    ),
)

#: Table I "The 2nd Stage" 部分（terrain-contact 组）
STAGE2_SEMANTICS: Tuple[TermSemantics, ...] = (
    TermSemantics(
        "touchdown_quality", "terrain",
        "落足质量：用局部高度变化评价落点是否平坦/合适",
        False, None,
        Corr.PGMT_SPECIFIC,
        "论文 §IV-B 明写'Local height variation is used to evaluate touchdown quality'",
    ),
    TermSemantics(
        "reference_contact_match", "terrain",
        "参考接触标签与仿真接触的一致性（标签来自离线地形网格查询）",
        False, None,
        Corr.PGMT_SPECIFIC,
        "论文明写接触标签由 offline terrain-mesh queries 得到",
    ),
    TermSemantics(
        "slip", "terrain",
        "打滑惩罚",
        False, "slippage",
        Corr.IDENTICAL,
        "权重不可比：参照实现 −30×1.25",
    ),
    TermSemantics(
        "stumble", "terrain",
        "绊倒惩罚（权重 −20.0）",
        False, "stumble",
        Corr.IDENTICAL,
        "权重不可比：参照实现 −1000×1.25（含终止语义）",
    ),
    TermSemantics(
        "contact_switching", "terrain",
        "接触状态快速切换惩罚（权重 −30.0，Table I 中绝对值最大的权重）",
        False, None,
        Corr.PGMT_SPECIFIC,
        "参照实现有 feet_air_time 类项但语义不同（鼓励足够腾空时间）；"
        "PGMT 明确惩罚'rapid contact switching'",
    ),
    TermSemantics(
        "contact_force", "terrain",
        "过大接触力惩罚",
        False, "feet_contact_forces",
        Corr.IDENTICAL,
        "参照实现带 max_contact_force=500 的阈值化；PGMT 权重 −1e−6",
    ),
)

ALL_SEMANTICS: Tuple[TermSemantics, ...] = STAGE12_SEMANTICS + STAGE2_SEMANTICS

#: 项名 → 语义（便捷索引）
BY_TERM: Dict[str, TermSemantics] = {t.term: t for t in ALL_SEMANTICS}

#: 每个组的项名（含重复项 link_lin_vel 等，故按 (group, term) 索引）
BY_GROUP_TERM: Dict[Tuple[str, str], TermSemantics] = {
    (t.group, t.term): t for t in ALL_SEMANTICS
}


def semantics_for(group: str, term: str) -> TermSemantics:
    """按 (组, 项名) 取语义。**不存在则报错** —— 不加默认值。"""
    key = (group, term)
    if key not in BY_GROUP_TERM:
        raise KeyError(f"未登记语义的奖励项: {group}.{term}")
    return BY_GROUP_TERM[key]


def correspondence_counts() -> Dict[str, int]:
    """各对应关系的项数统计（供报告引用）。"""
    out: Dict[str, int] = {c.value: 0 for c in Corr}
    for t in ALL_SEMANTICS:
        out[t.corr.value] += 1
    return out


def term_names_by_group() -> Dict[str, Tuple[str, ...]]:
    """本表登记的项名，便于与 `spec.py` 做同步校验。"""
    out: Dict[str, list] = {}
    for t in ALL_SEMANTICS:
        out.setdefault(t.group, []).append(t.term)
    return {g: tuple(v) for g, v in out.items()}


#: `spec.py` 的权威项名（本表必须与之完全一致）
SPEC_TERMS: Dict[str, Tuple[str, ...]] = {
    "upper": tuple(n for n, _ in UPPER_TERMS),
    "lower": tuple(n for n, _ in LOWER_TERMS),
    "aux": tuple(n for n, _ in AUX_TERMS),
    "terrain": tuple(n for n, _ in TERRAIN_TERMS),
}
