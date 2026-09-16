"""奖励规格（论文 Table I + §IV-B 松弛），M2/M4 环境层的调用契约。

本模块只负责**奖励的代数结构**：分组、逐项权重、松弛算子、指数映射。
各项**残差的具体计算**依赖仿真状态（连杆/关节量、接触、地形），属环境层
（`pgmt/envs/g1_env.py`）职责 —— 这样分组与权重可以脱离 Isaac Gym 单测，
而环境层只负责把残差算对。

## 分组（论文 Eq.5 / Eq.9）

    Stage 1:  r_t = r_upper + r_lower + r_aux
    Stage 2:  r_t = r_upper + r_lower + r_terrain + r_aux

多头 critic 的价值头顺序必须与这里的分组一一对应
（`HEAD_ORDER_STAGE1` / `HEAD_ORDER_STAGE2`），每个头估计**该组的回报**
而不是总量 —— 这正是论文 Fig.4 的 split-return 设计。

## 权重（论文 Table I，逐字照搬）

| 组 | 项（权重） |
|---|---|
| Upper body | link pos 1.0 · link ori 1.0 · link lin vel 0.5 · link ang vel 0.5 · joint pos 1.0 · joint vel 0.5 |
| Lower body | **TA** link pos 0.5 · **TA** link ori 2.0 · link lin vel 0.5 · link ang vel 0.5 · **TA** joint pos 0.5 · joint vel 0.5 |
| Auxiliary | root ori 0.5 · corrected root vel 2.0 · floating-anchor pos 1.0 · recovery upward vel 12.5 · pelvis vert accel −1e−3 · EE accel mismatch −1e−3 · action rate −0.05 · joint limit −15.0 · undesired contact −0.1 · head/torso impact −1e−5 |
| Terrain-contact（仅 Stage 2） | touchdown quality 10.0 · reference contact match 1.5 · slip −1.0 · stumble −20.0 · contact switching −30.0 · contact force −1e−6 |

### 值的符号约定（不变量）

**所有项的值恒 ≥ 0，正负完全由上表的权重携带。**

这条看似显然，但很容易写反：若惩罚项自己也返回负数（如 `−‖Δa‖²`），
它与表里的负权重相乘（`−0.05 × −0.03 = +0.0015`）就变成**正贡献** ——
"惩罚动作变化率"实际在**奖励**动作变化。`auxiliary.py` 第一版 6 个惩罚项
全部踩了这个坑。因此：

  - 惩罚项返回**代价量**（`‖·‖²`、越界量平方），无违规恰为 0
  - 正向项返回 `exp(−e²/σ) ∈ (0,1]`
  - `RewardGroup.sum` 在运行时**拒绝负值与 NaN**（见该方法），
    使这类符号错误立刻暴露，而不是表现为"训练不动/学出怪行为"

NaN 同样被拒绝：`NaN < 0` 为假会漏过朴素检查，故用 `not (v >= 0)`。

**TA = "Terrain-Adaptive"**：论文 Table I 用 TA 标记的就是 §IV-B 中受
**地形感知松弛**（Eq.10）作用的那三项 —— lower body 的 link position、
link orientation、joint position。upper body 与其余约束保持严格
（论文原话："upper-body tracking objectives and remaining motion
constraints remain strict"）。注意 lower body 的 link/angular velocity
**不带** TA，即速度跟踪仍然严格。

## 松弛（论文 Eq.10）

    ẽ_{t,h,j} = [ e_{t,h,j} − α_{h,j} · χ(κ_t) · τ_{mh}(d_t) ]_+

其中 `e` 是原始误差，`ẽ` 替换进指数跟踪奖励。`χ(κ) ∈ {0,1}` 只在
slopes/stairs/boxes 上激活（平地与 rough 不松弛），`τ_{mh}(d)` 随难度
线性增长并饱和，`α_{h,j} ≥ 0` 逐元素缩放（α=0 即退化为严格跟踪）。

**术语提醒**：论文里 `τ` 一符两用 —— Eq.1 的 τ 是未来参考帧的**时间偏移**，
Eq.10 的 τ_{mh} 是误差空间里的**容忍预算**（单位同该目标的量纲）。本模块
一律用 `tau` 指后者，并在参数名上带 `budget` 以免混淆。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

# ---------------------------------------------------------------------------
# 逐项权重（论文 Table I）
# ---------------------------------------------------------------------------

#: 一组奖励项：(名称, 权重)。名称即环境层需要提供的残差键。
Terms = Tuple[Tuple[str, float], ...]

#: Upper body —— 保留动作表达与姿态保真
UPPER_TERMS: Terms = (
    ("link_pos", 1.0),
    ("link_ori", 1.0),
    ("link_lin_vel", 0.5),
    ("link_ang_vel", 0.5),
    ("joint_pos", 1.0),
    ("joint_vel", 0.5),
)

#: Lower body —— 与平衡和接触转换强耦合；TA 三项受地形松弛
LOWER_TERMS: Terms = (
    ("ta_link_pos", 0.5),
    ("ta_link_ori", 2.0),
    ("link_lin_vel", 0.5),
    ("link_ang_vel", 0.5),
    ("ta_joint_pos", 0.5),
    ("joint_vel", 0.5),
)

#: Auxiliary —— 根运动、恢复、安全、控制正则化
AUX_TERMS: Terms = (
    ("root_ori", 0.5),
    ("corrected_root_vel", 2.0),
    ("floating_anchor_pos", 1.0),
    ("recovery_upward_vel", 12.5),
    ("pelvis_vert_accel", -1e-3),
    ("ee_accel_mismatch", -1e-3),
    ("action_rate", -0.05),
    ("joint_limit", -15.0),
    ("undesired_contact", -0.1),
    ("head_torso_impact", -1e-5),
)

#: Terrain-contact —— 仅 Stage 2 引入（论文 Eq.9 的第四个值头）
TERRAIN_TERMS: Terms = (
    ("touchdown_quality", 10.0),
    ("reference_contact_match", 1.5),
    ("slip", -1.0),
    ("stumble", -20.0),
    ("contact_switching", -30.0),
    ("contact_force", -1e-6),
)

#: 受地形感知松弛（Eq.10）作用的目标 —— 即 Table I 中带 TA 的三项
RELAXED_TERMS: Tuple[str, ...] = ("ta_link_pos", "ta_link_ori", "ta_joint_pos")

#: χ(κ)：哪些地形族激活松弛。论文：只在 slopes / stairs / boxes 上激活，
#: flat 与 random rough 不激活。
CHI_ACTIVE_FAMILIES: Tuple[str, ...] = ("slopes", "stairs", "boxes")

#: 地形族全集（论文 §V-A：5 族 × L0–L9）
TERRAIN_FAMILIES: Tuple[str, ...] = ("flat", "slopes", "stairs", "boxes", "rough")

#: 难度级数（L0–L9）
NUM_LEVELS = 10


# ---------------------------------------------------------------------------
# 分组
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardGroup:
    """一个奖励分组：名称 + 权重表 + 求和。

    `sum()` 的三条约定（都是"让错误立刻炸"而非静默降级）：
      - **缺失项按 0 处理** —— 便于分阶段补齐；`missing()` 供自检
      - **未知项直接报错** —— 环境层若拼错项名会立刻暴露，而不是少算一项
      - **负值/NaN 直接报错** —— 见模块头"值的符号约定"，防止惩罚反转成奖励
    """

    name: str
    terms: Terms

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in self.terms)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self.terms)

    def sum(self, values: Mapping[str, float]) -> float:
        unknown = set(values) - set(self.names)
        if unknown:
            raise KeyError(f"{self.name} 组收到未知奖励项 {sorted(unknown)}；"
                           f"已知项为 {list(self.names)}")
        # 值域守卫：见模块头"值的符号约定"。负值/NaN 与负权重相乘会变成正贡献，
        # 即惩罚反转成奖励 —— 这类错误在训练曲线上极难辨认，故在此直接拒绝。
        # 用 `not (v >= 0)` 而非 `v < 0`：前者同时拦住 NaN。
        bad = {n: float(values[n]) for n in self.names
               if n in values and not (float(values[n]) >= 0.0)}
        if bad:
            raise ValueError(
                f"{self.name} 组的奖励项出现负值或 NaN: {bad}。"
                f"本仓库约定**所有项的值 ≥ 0、符号全部由权重携带**；"
                f"某项若返回负值，与负权重相乘会变成正贡献（惩罚被反转成奖励）。"
                f"惩罚项应返回非负代价量（‖·‖² 等），无违规时为 0。")
        return sum(w * float(values.get(n, 0.0)) for n, w in self.terms)

    def missing(self, values: Mapping[str, float]) -> Tuple[str, ...]:
        """返回尚未提供的项名（环境层自检用；缺项按 0 计，不报错）。"""
        return tuple(n for n in self.names if n not in values)


UPPER = RewardGroup("upper", UPPER_TERMS)
LOWER = RewardGroup("lower", LOWER_TERMS)
AUX = RewardGroup("aux", AUX_TERMS)
TERRAIN = RewardGroup("terrain", TERRAIN_TERMS)

#: 分组注册表：名称 → RewardGroup。顺序即多头 critic 的价值头顺序。
GROUPS: Dict[str, RewardGroup] = {
    "upper": UPPER,
    "lower": LOWER,
    "terrain": TERRAIN,
    "aux": AUX,
}


def total_reward(values: Mapping[str, Mapping[str, float]],
                 stage2: bool = False) -> float:
    """按论文 Eq.5（Stage 1）/ 隐含的 Stage 2 分组求和。

    Args:
        values: {组名: {项名: 值}}
        stage2: True 时把 terrain 组计入
    """
    r = UPPER.sum(values.get("upper", {})) + LOWER.sum(values.get("lower", {})) \
        + AUX.sum(values.get("aux", {}))
    if stage2:
        r += TERRAIN.sum(values.get("terrain", {}))
    return r


# ---------------------------------------------------------------------------
# 跟踪奖励映射
# ---------------------------------------------------------------------------

#: 各跟踪目标的 σ（论文未给；取自 PGMT 所继承的 tracking 实现的约定）
#:
#: **核函数是高斯式 `exp(−e²/σ)`，不是 `exp(−e/σ)`。** 依据：
#:   - OmniH2O 论文奖励表把 body position 项写作 `exp(−0.5‖p−p̂‖²₂)`
#:   - 其配置注释明写 `tracking reward : exp(-error^2/sigma)`
#:   - PGMT 论文用定冠词称 "**the** exponential tracking reward"，
#:     即引用该领域既有约定而非自行定义（论文只给权重，未给尺度）
#:
#: σ 的量纲是**误差的平方**（位置项 m²、关节项 rad²…），这点与线性核不同，
#: 不可直接照搬。取值的量级与"上/下半身分离"的结构参照 OmniH2O：
#: 上半身位置 σ 明显小于下半身（约 15 倍），这正是论文 §IV
#: "upper-body 保姿态保真 / lower-body 让位给平衡与接触"这一意图的量化表达。
#:
#: 单位换算说明：OmniH2O 的位置项还带一个 0.5 系数（`exp(−0.5e²/σ_oh)`），
#: 这里把它吸收进 σ（`σ_us = 2·σ_oh`），使核统一为 `exp(−e²/σ)`。
SIGMAS: Dict[str, float] = {
    # 连杆/身体位置：上半身严格（保姿态），下半身宽松（让位平衡与接触）
    "ta_link_pos": 1.0,        # m²  —— 下半身（OmniH2O lowerbody σ 0.5，×2 吸收系数）
    "link_pos": 0.06,          # m²  —— 上半身（OmniH2O upperbody σ 0.03，×2）
    "link_pos_lower": 1.0,     # m²  —— 下半身非 TA 位置项（若环境层区分上下半身）
    "ta_link_ori": 2.0,        # rad² —— OmniH2O body_rot σ 1.0，×2
    "link_ori": 2.0,           # rad²
    "link_lin_vel": 20.0,      # (m/s)² —— OmniH2O body_vel σ 10，×2
    "link_ang_vel": 20.0,      # (rad/s)²
    "ta_joint_pos": 1.0,       # rad² —— OmniH2O joint_pos σ 0.5，×2
    "joint_pos": 1.0,          # rad²
    "joint_vel": 20.0,         # (rad/s)² —— OmniH2O joint_vel σ 10，×2
}


def sigma_for(term: str) -> float:
    """取该跟踪目标的 σ；未登记则报错（避免静默用默认值把项调坏）。"""
    if term not in SIGMAS:
        raise KeyError(f"未登记 σ 的跟踪项: {term}（已登记 {sorted(SIGMAS)}）")
    return SIGMAS[term]


def exp_tracking_reward(error: float, sigma: float = 1.0) -> float:
    """指数跟踪奖励 `exp(−error² / σ)` ∈ (0, 1]。

    **是误差的平方**（高斯式），不是线性误差 —— 见 `SIGMAS` 的依据说明。
    `σ` 的量纲是误差平方，单位同 `SIGMAS` 各条注释。

    性质（被单测锁定）：error=0 → 1；单调递减；恒正；大误差 → 0；
    且 `error = √σ` 处恰为 `1/e`（这是 σ 的直观标定：√σ 是"1/e 误差"）。
    """
    if sigma <= 0.0:
        raise ValueError(f"σ 必须为正，得到 {sigma}")
    e = max(float(error), 0.0)
    return math.exp(-(e * e) / sigma)


# ---------------------------------------------------------------------------
# 地形感知松弛（论文 Eq.10）
# ---------------------------------------------------------------------------

def chi(terrain_family: str) -> float:
    """χ(κ) ∈ {0,1}：该地形族是否激活松弛。

    论文：只在 slopes / stairs / boxes 上激活；flat 与 rough 为 0。
    """
    if terrain_family not in TERRAIN_FAMILIES:
        raise ValueError(f"未知地形族 {terrain_family!r}（可选 {TERRAIN_FAMILIES}）")
    return 1.0 if terrain_family in CHI_ACTIVE_FAMILIES else 0.0


def tau_budget(level: int, saturation_value: float, *,
               level_min: int = 0, level_max: int = NUM_LEVELS - 1,
               start_fraction: float = 0.0) -> float:
    """τ_{mh}(d)：随难度线性增长并饱和的基础容忍预算（论文 Eq.10）。

    论文只说"grows linearly with d and saturates"，未给斜率与饱和值；这里
    以 `saturation_value` 表示该目标上的饱和量（属 A12 的实现选择，需在报告
    中说明）。`start_fraction` 给出最低难度处占饱和值的比例（默认 0 起步）。

    Args:
        level: 难度级别 d（默认域 L0..L9）
        saturation_value: 饱和预算，单位同该目标的误差量纲
        level_min/level_max: 难度域
        start_fraction: d=level_min 时的预算占饱和值比例 ∈ [0, 1]
    """
    if level_max <= level_min:
        raise ValueError("level_max 必须大于 level_min")
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError(f"start_fraction 应在 [0,1]，得到 {start_fraction}")
    if saturation_value < 0.0:
        raise ValueError(f"saturation_value 不能为负，得到 {saturation_value}")
    frac = (float(level) - level_min) / (level_max - level_min)
    frac = min(max(frac, 0.0), 1.0)          # 域外夹紧（域外不外推）
    shaped = start_fraction + (1.0 - start_fraction) * frac
    return float(saturation_value) * shaped


def relaxed_error(error: float, alpha: float, chi_value: float,
                  tau: float) -> float:
    """论文 Eq.10：`ẽ = [ e − α·χ(κ)·τ ]_+`。

    意图是把这个误差**钳到容忍区之外**：误差小于预算时完全不计罚
    （地形迫使的合理偏离不应被惩罚），超出部分照常按指数核衰减。

    Args:
        error: 原始误差 e（非负）
        alpha: α_{h,j} ≥ 0 逐元素缩放；α=0 即退化为严格跟踪
        chi_value: χ(κ) ∈ {0,1}（用 `chi()` 求）
        tau: τ_{mh}(d) 基础容忍预算（用 `tau_budget()` 求）
    """
    if alpha < 0.0:
        raise ValueError(f"alpha 必须 ≥ 0，得到 {alpha}")
    return max(float(error) - alpha * float(chi_value) * float(tau), 0.0)


def relaxed_tracking_reward(error: float, alpha: float, terrain_family: str,
                            level: int, saturation_value: float,
                            sigma: float = 1.0) -> float:
    """松弛 + 指数映射的组合（Eq.10 代入指数跟踪奖励）。

    环境层最常用的入口：给原始误差、逐元素 α、地形族与难度，直接得到该目标
    的奖励。

    单位一致性（易错点）：Eq.10 的 τ 与**误差 e 同量纲**（它是要减掉的误差量），
    而本核的 σ 是**误差平方的量纲**。所以 τ 必须先被钳到误差空间、再平方：
    `exp(−ẽ²/σ)`。若环境层想用"平方空间"的预算 τ²，应先开方再传进来。
    """
    tau = tau_budget(level, saturation_value)
    e_tilde = relaxed_error(error, alpha, chi(terrain_family), tau)
    return exp_tracking_reward(e_tilde, sigma)


def resolve_alpha(term: str, alpha_default: float = 1.0,
                  per_element: Mapping[str, Mapping[str, float]] | None = None,
                  element: str | None = None) -> float:
    """取 α_{h,j}：优先逐元素覆盖表，否则用默认值；非松弛项恒为 0。

    非松弛项（upper body 与其余约束）返回 0，保证"严格跟踪"是默认行为 ——
    这样调用方可以直接对每个目标统一走 `relaxed_tracking_reward`，而不必
    自己判断哪一项该松弛（判断错了会静默放宽 upper body，违反论文约束）。
    """
    if term not in RELAXED_TERMS:
        return 0.0
    if per_element and term in per_element and element is not None:
        return float(per_element[term].get(element, alpha_default))
    return float(alpha_default)
