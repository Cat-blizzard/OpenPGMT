"""地形接触项（M2 待办 #1 收尾，论文 Table I 的 terrain-contact 组，6 项，仅 Stage 2）。

## 论文给出的全部信息

§IV-B "Multi-Head Critic Extension" 只有一段话：

    "To encourage stable contacts, we introduce a new terrain-contact reward
     group r^terrain_t (summarized in Table I). **Local height variation** is used
     to evaluate **touchdown quality**, while **contact labels obtained from
     offline terrain-mesh queries** supervise the consistency between reference
     and simulated contacts. Additional penalties discourage **foot slippage,
     stumbling, rapid contact switching, and excessive contact forces**."

**意图 6 项都点明了，公式与阈值一个都没给。** 度量方式与尺度集中在 A22
（`TerrainContactCfg`），本模块只做"把残差算对"。

## 与 auxiliary 组相同的符号约定

**所有项的值恒 ≥ 0，符号全部由 Table I 的权重携带**（详见 `spec.py` 的
"值的符号约定"）。6 项分两类：

  - **2 个正向项**（权重 > 0）：`touchdown_quality` 10.0、`reference_contact_match` 1.5
  - **4 个代价项**（权重 < 0）：`slip` −1.0、`stumble` −20.0、
    `contact_switching` −30.0、`contact_force` −1e−6
    → 返回**非负代价量**，无违规恰为 0

`RewardGroup.sum` 的运行时守卫会拒绝负值/NaN。

`contact_switching` 的 |权重| = 30 是 Table I 中最大的，`contact_force` 的
1e−6 是最小的。后者恰好能反过来校验度量形式：若代价取"超出阈值的**一次方**"，
典型超出量（~100 N）只贡献 −1e−4，在总奖励里完全不可见；取**平方**则贡献
−0.01 量级，与其它项可比。故本模块的代价统一取平方（见 A22）。

## 参考接触标签的来源

新版物理 Stage 2 要求 `data.build_mesh_contacts` 生成的离线网格查询标签。
作者没有发布参考地形资产，本项目显式采用 z=0 平面参考网格、URDF 碰撞球
和 2cm 容差（也支持外部高度场）。这些参数属于工程假设，不能称作作者原始标签。
旧足速阈值标签保留在 `contacts_heuristic` 供比较。

## 单位与坐标系

  - 时间单位：步（50 Hz，A1），故 `contact_switching` 的 dwell 也用步
  - `local_heights` 可以是**相对高度**（如 A8 高程图存的 `h − robot_z`）：
    标准差对常数平移不变，故不需要绝对高度
  - `foot_vel_xy` 是世界系水平速度；`slip` 只关心切向（水平）分量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pgmt.cfg.assumptions import TerrainContactCfg, get  # noqa: F401
from pgmt.rewards.spec import exp_tracking_reward

# ---------------------------------------------------------------------------
# 单项：2 个正向项
# ---------------------------------------------------------------------------


def local_height_variation(heights: np.ndarray) -> float:
    """局部高度变化（m）：采样点高度的**标准差**。

    论文原话是 "local height variation is used to evaluate touchdown quality"，
    未定义 "variation"。取标准差（而非极差）的理由：对单个异常采样点不敏感，
    且量纲是长度、便于与 `sigma_touchdown_quality`（m²）配合。

    **少于 2 个采样点直接报错**：一个点谈不上"变化"，若静默返回 0 会被当成
    "完全平坦"从而给出满分落足奖励 —— 那是把配置错误伪装成好结果。
    """
    h = np.asarray(heights, dtype=np.float64).ravel()
    if h.size < 2:
        raise ValueError(f"局部高度变化至少需 2 个采样点，得到 {h.size}")
    return float(np.std(h))


def touchdown_quality_reward(heights: np.ndarray,
                             sigma: float) -> float:
    """落足质量奖励 `exp(−σ_h² / σ) ∈ (0,1]`（σ_h = 局部高度标准差）。

    `σ` 单位是 **m²**（误差平方量纲，与 A18 的核一致，见 A22）。
    """
    return exp_tracking_reward(local_height_variation(heights), sigma)


def touchdown_events(sim_contact: np.ndarray,
                     prev_sim_contact: np.ndarray) -> np.ndarray:
    """本步**新落足**的足（本步接触、上步不接触）→ `(n,)` bool。

    落足质量只在落足**事件**上评价（论文该项名为 touchdown quality），
    不是持续评价 —— 详见 `TerrainContactState` 的说明。
    """
    s = np.asarray(sim_contact, dtype=bool)
    p = np.asarray(prev_sim_contact, dtype=bool)
    if s.shape != p.shape:
        raise ValueError(f"接触标签形状须一致: {s.shape} vs {p.shape}")
    return s & ~p


def reference_contact_match(ref_contact: np.ndarray,
                            sim_contact: np.ndarray) -> float:
    """参考接触与仿真接触的**一致率** ∈ [0, 1]。

    论文明说是 "consistency between reference and simulated contacts"，
    一致率本身就是归一化奖励，**不需要 σ** —— 凭空引入一个尺度反而会
    把"完全一致"从 1 拉低。

    返回的是**逐足一致率**（含双脚都在摆动/都在支撑的情形），而不是
    "只在参考接触为真的帧上统计"：后者会漏掉"参考无接触但仿真踩上去了"
    这一类错误，而那正是地形上最该防的（绊到台阶边缘）。
    """
    r = np.asarray(ref_contact, dtype=bool)
    s = np.asarray(sim_contact, dtype=bool)
    if r.shape != s.shape:
        raise ValueError(f"接触标签形状须一致: {r.shape} vs {s.shape}")
    if r.size == 0:
        raise ValueError("接触标签为空 —— 至少需要一只足")
    return float(np.mean(r == s))


# ---------------------------------------------------------------------------
# 单项：4 个代价项（返回**非负**代价量，无违规恰为 0）
# ---------------------------------------------------------------------------


def slip_cost(foot_vel_xy: np.ndarray, sim_contact: np.ndarray) -> float:
    """打滑代价：接触足的**水平速度平方和** ≥ 0。

    取平方（与仓库其它 `‖·‖²` 代价同族）；论文只写 "foot slippage"，
    没说用速度还是一次方 —— 见 A22。
    """
    v = np.asarray(foot_vel_xy, dtype=np.float64)
    c = np.asarray(sim_contact, dtype=bool)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"foot_vel_xy 应为 (n, 2)，得到 {v.shape}")
    if v.shape[0] != c.shape[0]:
        raise ValueError(f"足数与接触标签数须一致: {v.shape[0]} vs {c.shape[0]}")
    return float(np.sum(np.sum(v ** 2, axis=1) * c))


def stumble_cost(contact_forces: np.ndarray,
                 ratio_threshold: float = 5.0) -> float:
    """绊倒代价：**水平接触力显著超过竖直分量**时的水平力平方和 ≥ 0。

    判据取自 legged_gym 系的同名项（PGMT 未给定义）：
    `‖f_xy‖ > ratio · f_z` 即认为足/小腿**侧向撞上了地形棱边**，而非正常支撑。
    正常站立时 `f_xy ≈ 0`，故该项在平地上恒为 0。

    竖直分量取 `max(f_z, 0)`：法向力不应为负，若传感器给出负值（该体未接触），
    则 `f_xy > 0` 即判为撞击。

    Args:
        contact_forces: `(n, 3)` 各足的三维接触力（N），世界系或体坐标系均可
                        （判据只用比值，但**必须在同一坐标系内**）
        ratio_threshold: 水平/竖直比阈值，A22 取 5.0
    """
    f = np.asarray(contact_forces, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"contact_forces 应为 (n, 3)，得到 {f.shape}")
    if ratio_threshold <= 0.0:
        raise ValueError(f"ratio_threshold 应为正，得到 {ratio_threshold}")
    f_xy = np.linalg.norm(f[:, :2], axis=1)
    f_z = np.maximum(f[:, 2], 0.0)
    hit = f_xy > ratio_threshold * f_z
    return float(np.sum(f_xy[hit] ** 2))


def contact_switching_cost(flipped: np.ndarray, age_steps: np.ndarray,
                           min_dwell: int) -> float:
    """接触状态快速切换代价 ∈ [0, n] ≥ 0。

    **不能**直接惩罚"接触状态发生变化" —— 正常步态每一步都在切换，那样会把
    走路本身罚掉。论文说的是 "**rapid** contact switching"，故按"切换得太快"
    衡量：代价随该状态**已持续步数**线性衰减，持续满 `min_dwell` 步后归 0。

        cost_i = [本步翻转] · (1 − min(age_i, D) / D)

    Args:
        flipped: `(n,)` bool，本步接触状态是否相对上步翻转
        age_steps: `(n,)` 翻转前该状态**已持续的步数**（含上一步本身）。
                   由 `update_contact_age` 维护
        min_dwell: 认为"稳定"所需的最少步数 D ≥ 1

    例（D = 5）：连续两次翻转（age=1）→ 代价 0.8；翻转前已稳定 5 步以上 → 0。
    """
    fl = np.asarray(flipped, dtype=bool)
    age = np.asarray(age_steps, dtype=np.float64)
    if fl.shape != age.shape:
        raise ValueError(f"flipped 与 age_steps 形状须一致: {fl.shape} vs {age.shape}")
    if min_dwell < 1:
        raise ValueError(f"min_dwell 应为 ≥ 1 的整数，得到 {min_dwell}")
    if np.any(age < 0.0):
        raise ValueError("age_steps 不能为负")
    stable = np.minimum(age, float(min_dwell)) / float(min_dwell)
    return float(np.sum((1.0 - stable) * fl))


def contact_force_cost(contact_forces: np.ndarray, sim_contact: np.ndarray,
                       f_max: float) -> float:
    """过大接触力代价：超过阈值的部分**平方和** ≥ 0。

    取**合力大小**（含竖直分量）与阈值比较，与 legged_gym 的
    `feet_contact_forces`（`max_contact_force`，传统取 500 N）一致。
    平方的理由见模块头（权重 1e−6 只在平方下才有可见量级）。

    Args:
        contact_forces: `(n, 3)` 各足接触力（N）
        sim_contact: `(n,)` bool，只对接触中的足计代价
        f_max: 阈值（N），A22 取 500
    """
    f = np.asarray(contact_forces, dtype=np.float64)
    c = np.asarray(sim_contact, dtype=bool)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"contact_forces 应为 (n, 3)，得到 {f.shape}")
    if f.shape[0] != c.shape[0]:
        raise ValueError(f"足数与接触标签数须一致: {f.shape[0]} vs {c.shape[0]}")
    if f_max < 0.0:
        raise ValueError(f"f_max 应为非负，得到 {f_max}")
    over = np.maximum(np.linalg.norm(f, axis=1) - float(f_max), 0.0)
    return float(np.sum(over ** 2 * c))


# ---------------------------------------------------------------------------
# 接触年龄的状态推进（纯函数，供环境层调用）
# ---------------------------------------------------------------------------


def update_contact_age(age_steps: np.ndarray, sim_contact: np.ndarray,
                       prev_sim_contact: np.ndarray) -> np.ndarray:
    """推进接触年龄 → `(n,)` int64：翻转则归 1，否则 +1。

    与 `contact_switching_cost` 的口径配套：**返回值是"下一步"用的 age**，
    即"截至当前步，该状态已持续了多少步"。

    约定（含一步的例子）：

        t:      0     1     2     3
        接触:   0     1     1     1
        age:    1     1     2     3      ← age[t] 表示"截至 t，状态已持续 t 步"
        翻转:   是    是    否    否
                 ↑ 第 1 步翻转时 age=1（第 0 步的状态只持续了 1 步）

    于是 `contact_switching_cost(flipped[t], age[t], D)` 表达的正是
    "翻转发生时，被结束的那个状态持续了多久"。
    """
    a = np.asarray(age_steps, dtype=np.int64)
    s = np.asarray(sim_contact, dtype=bool)
    p = np.asarray(prev_sim_contact, dtype=bool)
    if not (a.shape == s.shape == p.shape):
        raise ValueError(f"形状须一致: age{a.shape} sim{s.shape} prev{p.shape}")
    return np.where(s != p, 1, a + 1).astype(np.int64)


# ---------------------------------------------------------------------------
# 组合
# ---------------------------------------------------------------------------

# `TerrainContactCfg` 的**唯一定义在 `pgmt/cfg/assumptions.py`**（A22 的值类型）。
#
# ⚠️ 我在 `TerminationCfg` 与 `AuxCfg` 上各犯过一次"两处定义 → isinstance 失败"，
# 规则：**假设表里出现的 dataclass 只在 assumptions.py 定义一次，实现模块一律 import**。


def terrain_contact_cfg() -> TerrainContactCfg:
    """取 A22 的配置（唯一出处）。"""
    return get("A22").value


@dataclass(frozen=True)
class TerrainContactState:
    """地形接触项所需的单步状态（逐环境，无 batch 维）。

    足的顺序由调用方固定（建议 `("left_ankle_roll_link",
    "right_ankle_roll_link")`），所有 `(n, ...)` 字段按同一顺序排列。

    **为什么 `touchdown_quality` 只在落足事件上评价**：该项名为 touchdown
    quality，论文也只说用局部高度变化评价"落足"。若持续评价，站立不动时
    双脚会每步都拿满分 +10.0×2，等于奖励"站着别动"，与跟踪目标冲突。
    故取事件门控（A22）；代价是该项较稀疏 —— 这可由 +10.0 的大权重补偿。
    """

    #: 参考接触标签与仿真接触 `(n,)` bool
    ref_contact: np.ndarray
    sim_contact: np.ndarray
    #: 上一步的仿真接触 `(n,)` bool（用于检测落足事件与切换）
    prev_sim_contact: np.ndarray
    #: 翻转前各状态已持续的步数 `(n,)` int（由 `update_contact_age` 维护）
    contact_age: np.ndarray
    #: 各足世界系水平速度 `(n, 2)`（m/s）
    foot_vel_xy: np.ndarray
    #: 各足三维接触力 `(n, 3)`（N）
    contact_forces: np.ndarray
    #: 各足落点附近的局部高度采样 `(n, P)`（m；可为相对高度）
    local_heights: np.ndarray

    def __post_init__(self):
        age = np.asarray(self.contact_age)
        if age.ndim != 1:
            raise ValueError(f"contact_age 应为 (n,)，得到 {age.shape}")
        n = age.shape[0]
        if n == 0:
            raise ValueError("至少需要一只足")
        for name in ("ref_contact", "sim_contact", "prev_sim_contact"):
            if np.shape(getattr(self, name)) != (n,):
                raise ValueError(f"{name} 应为 ({n},)")
        if np.shape(self.foot_vel_xy) != (n, 2):
            raise ValueError(f"foot_vel_xy 应为 ({n}, 2)")
        if np.shape(self.contact_forces) != (n, 3):
            raise ValueError(f"contact_forces 应为 ({n}, 3)")
        lh = np.shape(self.local_heights)
        if len(lh) != 2 or lh[0] != n:
            raise ValueError(f"local_heights 应为 ({n}, P)，得到 {lh}")
        if lh[1] < 2:
            raise ValueError(f"local_heights 每足至少 2 个采样点，得到 {lh[1]}")


def compute_terrain_values(state: TerrainContactState,
                           cfg: Optional[TerrainContactCfg] = None
                           ) -> Dict[str, float]:
    """算齐 Table I 的 6 个 terrain-contact 项值（**未加权**）。

    返回键与 `spec.TERRAIN_TERMS` 一致，可直接交给 `RewardGroup.sum`。
    值域：`touchdown_quality ∈ (0,1]`（无落足事件时为 0）、
    `reference_contact_match ∈ [0,1]`、4 个代价项 ≥ 0。**全部 ≥ 0**。
    """
    c = cfg or terrain_contact_cfg()

    # 落足质量：只对"本步新落足"的足求平均；无落足事件 → 0（不加不减）
    new_td = touchdown_events(state.sim_contact, state.prev_sim_contact)
    if bool(np.any(new_td)):
        q = float(np.mean([
            touchdown_quality_reward(state.local_heights[i],
                                     c.sigma_touchdown_quality)
            for i in np.flatnonzero(new_td)
        ]))
    else:
        q = 0.0

    return {
        "touchdown_quality": q,
        "reference_contact_match": reference_contact_match(
            state.ref_contact, state.sim_contact),
        "slip": slip_cost(state.foot_vel_xy, state.sim_contact),
        "stumble": stumble_cost(state.contact_forces, c.stumble_force_ratio),
        "contact_switching": contact_switching_cost(
            state.sim_contact != state.prev_sim_contact,
            state.contact_age,
            c.contact_switching_min_dwell),
        "contact_force": contact_force_cost(
            state.contact_forces, state.sim_contact, c.contact_force_max),
    }


def check_terrain_values_complete(values: Dict[str, float]) -> None:
    """自检：项名与 `spec.TERRAIN_TERMS` 完全一致，且值域合法（全部 ≥ 0）。

    与 `RewardGroup.sum` 的值域守卫重复是刻意的 —— 本函数可在不求和时单独调用。
    """
    from pgmt.rewards.spec import TERRAIN_TERMS

    want = {n for n, _ in TERRAIN_TERMS}
    got = set(values)
    if got != want:
        raise KeyError(f"terrain-contact 项名不匹配：缺 {sorted(want - got)}，"
                       f"多 {sorted(got - want)}")
    bad = {n: v for n, v in values.items() if not (float(v) >= 0.0)}
    if bad:
        raise ValueError(f"terrain-contact 项出现负值/NaN: {bad}；"
                         f"本仓库约定所有项的值 ≥ 0，符号由权重携带")
