"""终止条件与容忍区（M2，论文 Fig.2 / §V-C 的 "terrain aware ... terminations"）。

## 论文说了什么

  - **完成定义**（Table II 脚注）："Completion denotes reaching the 30-s horizon
    **without early termination**" → 时间上限本身**不是**失败，episode 跑满 30s
    即算成功（`A1.steps_per_episode = 1500`）
  - **Fig.2 图例**："Terrain aware rewards **& terminations**"，并画出
    "**Tolerance zone**" 与 "**Drift-tolerant tracking**" → 终止条件与奖励一样
    是**地形感知**的：误差落在容忍区内不算失败
  - **§V-C**："The terrain level jointly controls geometric difficulty, tracking
    relaxation, and **selected termination delays**" → 高难度地形上，终止被**延迟**
    （给策略反应时间），而不是立刻结束
  - **§V-B**："regress to easier levels after **tracking failures**" → 终止 = 跟踪失败

## 论文没说的

  - 具体有哪些终止触发条件（论文从未列出）
  - 各阈值的取值
  - 延迟如何随难度增长

→ 登记为 A20，并在最终报告中说明。

## 设计

三类触发：

  1. `REF_DEVIATION`：跟踪误差超出容忍区。容忍区 = **A12 松弛预算 τ** ——
     直接复用 Stage 2 松弛用的同一个量，使"奖励不罚"与"不终止"口径一致
     （两者都由地形族与难度决定，这正是 Fig.2 把 rewards 与 terminations 并列的原因）
  2. `ROOT_LOW`：基座低于阈值 —— 摔倒/塌陷的粗判据（根高度是物理量，与地形无关）
  3. `TILTED`：姿态倾斜超过阈值 —— 翻倒的粗判据

**延迟语义**：某触发条件连续成立达到 `delay_s` 秒后才真正终止；期间条件若解除则
延迟计时清零。这实现论文的 "selected termination delays"，且 `delay_s` 由
**地形难度**给出（高难度 ⇒ 更长延迟）。

`ROOT_LOW` 与 `TILTED` 的阈值属"可从服务器失败模式校准"的参数（见复现方案
M2 待办清单），因此做成显式配置而非埋在代码里。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgmt.cfg.assumptions import TerminationCfg, get  # noqa: F401  （类型在此转出）
from pgmt.envs.terrain.generators import FAMILIES

#: 延迟比较的容差（秒）。`dt` 与 `delay_s` 都不是二进制精确可表示的十进制小数，
#: 逐帧累加会有 ~1e-16 量级误差；用裸 `>=` 比较会让"恰好等于延迟"的边界抖动
#: （实测 0.02 累加 10 次得 0.19999999999999998 < 0.2）。取 1e-9 远小于任何
#: 有意义的物理时间尺度，只用于吸收浮点噪声。
_DELAY_EPS = 1e-9

# ---------------------------------------------------------------------------
# 触发原因
# ---------------------------------------------------------------------------


class TerminationReason(str, Enum):
    """终止触发原因。`TIMEOUT` **不是**失败（论文的 completion 定义）。"""

    REF_DEVIATION = "ref_deviation"
    ROOT_LOW = "root_low"
    TILTED = "tilted"
    TIMEOUT = "timeout"


#: 论文定义的"失败"类原因（TIMEOUT 不计入）
FAILURE_REASONS: Tuple[TerminationReason, ...] = (
    TerminationReason.REF_DEVIATION,
    TerminationReason.ROOT_LOW,
    TerminationReason.TILTED,
)


# ---------------------------------------------------------------------------
# 配置（A20）
# ---------------------------------------------------------------------------

# `TerminationCfg` 的**唯一定义在 `pgmt/cfg/assumptions.py`**（A20 的值类型），
# 此处直接复用。若在两边各定义一份，`isinstance` 会失败、且改动易漂移 ——
# 这个坑在写 test_termination.py 时被 isinstance 断言当场抓住。


def termination_cfg() -> TerminationCfg:
    """取 A20 的配置（唯一出处）。"""
    return get("A20").value


# ---------------------------------------------------------------------------
# 容忍区（与 A12 松弛共用同一预算）
# ---------------------------------------------------------------------------


def tolerance_budget(terrain_family: str, level: int,
                     cfg: Optional[TerminationCfg] = None) -> float:
    """终止用的容忍区（米）—— 与 A12 的松弛预算同源。

    平地与 rough 的 χ=0（`spec.chi`），故容忍区退化为基准值；slopes/stairs/boxes
    上按难度线性增长并饱和，与 `spec.tau_budget` 完全一致的形状。

    **为什么不直接调用 `spec.tau_budget`**：那个函数作用在**各目标自己的误差
    量纲**上（连杆位置用米、关节角用弧度），而终止判据只有"参考偏差"这一个
    标量（单位米）。因此这里用 A12 的 χ 与 A18 的饱和值给出"米"这一路的预算，
    形状与松弛一致、量纲统一。
    """
    from pgmt.rewards.spec import chi, tau_budget

    cfg = cfg or termination_cfg()
    chi_v = chi(terrain_family)
    if chi_v == 0.0:
        return cfg.ref_deviation_base
    # 复用 A18 里 ta_link_pos 的饱和值（米），保持与松弛同量级。
    # 注意 τ(L0) = 0，因此 **L0 的容忍区恰好等于基准值**，L1+ 才严格更大。
    sat = get("A18").value.tau_saturation.get("ta_link_pos", 0.05)
    return cfg.ref_deviation_base + tau_budget(level, sat)


def termination_delay(reason: TerminationReason, terrain_family: str,
                      level: int, cfg: Optional[TerminationCfg] = None) -> float:
    """该原因在该 (族, 级) 上的延迟（秒）。

    地形**只会延长**延迟（`terrain_delay_scale · τ`），不会缩短 ——
    高难度地形上策略需要更多时间调整，这正是 §V-C 的意图。
    """
    cfg = cfg or termination_cfg()
    base = cfg.delay_s.get(reason.value, 0.0)
    if reason is not TerminationReason.REF_DEVIATION:
        return base
    from pgmt.rewards.spec import chi, tau_budget

    if chi(terrain_family) == 0.0:
        return base
    sat = get("A18").value.tau_saturation.get("ta_link_pos", 0.05)
    return base + cfg.terrain_delay_scale * tau_budget(level, sat)


# ---------------------------------------------------------------------------
# 触发判据（纯函数，便于单测）
# ---------------------------------------------------------------------------


def ref_deviation_exceeded(mean_body_pos_error: float, terrain_family: str,
                           level: int,
                           cfg: Optional[TerminationCfg] = None) -> bool:
    """跟踪误差是否**超出容忍区**（论文的 drift-tolerant：区内不算失败）。"""
    if mean_body_pos_error < 0.0:
        raise ValueError(f"误差应为非负，得到 {mean_body_pos_error}")
    return bool(mean_body_pos_error > tolerance_budget(terrain_family, level, cfg))


def root_too_low(root_height: float, cfg: Optional[TerminationCfg] = None) -> bool:
    cfg = cfg or termination_cfg()
    return bool(root_height < cfg.root_height_min)


def tilt_deg_from_projected_gravity(gravity_z: float) -> float:
    """由"投影重力"的 z 分量反推基座倾角（度）。

    基座竖直时 `projected_gravity = (0, 0, −1)`（体坐标系下重力指向下方），
    故 `g_z = -cos(倾角)`，相对直立的有向倾角范围是 [0°, 180°]。
    直立 g_z=-1、侧躺 g_z=0、倒立 g_z=+1；不能取绝对值抹去倒立方向。
    """
    if not np.isfinite(gravity_z):
        raise ValueError("gravity_z 必须为有限值")
    return float(np.degrees(np.arccos(np.clip(-gravity_z, -1.0, 1.0))))


def tilted(gravity_z: float, cfg: Optional[TerminationCfg] = None) -> bool:
    """基座是否倾斜过度（翻倒的粗判据）。"""
    cfg = cfg or termination_cfg()
    return tilt_deg_from_projected_gravity(gravity_z) > cfg.tilt_max_deg


# ---------------------------------------------------------------------------
# 状态机（延迟）
# ---------------------------------------------------------------------------


class TerminationState:
    """每环境的终止延迟计时器。

    对每个失败类原因各存一个"已持续满足"的秒数；条件不满足则清零。
    达阈值的那些原因即**现在应终止**，取其中第一个（按 `FAILURE_REASONS` 顺序）
    作为主原因（用于日志/课程归因）。
    """

    def __init__(self, num_envs: int, dt: float,
                 cfg: Optional[TerminationCfg] = None):
        if num_envs <= 0:
            raise ValueError(f"num_envs 必须为正，得到 {num_envs}")
        if dt <= 0.0:
            raise ValueError(f"dt 必须为正，得到 {dt}")
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.cfg = cfg or termination_cfg()
        # 每个原因一个 (num_envs,) 的秒数缓冲
        self.elapsed: Dict[TerminationReason, np.ndarray] = {
            r: np.zeros(self.num_envs, dtype=np.float64) for r in FAILURE_REASONS
        }

    # ---- 更新 ----
    def step(self,
             active: Mapping[TerminationReason, np.ndarray],
             delay: Mapping[TerminationReason, np.ndarray],
             ) -> Tuple[np.ndarray, Dict[TerminationReason, np.ndarray],
                        Dict[TerminationReason, np.ndarray]]:
        """推进一个控制步。

        Args:
            active: 每个原因在当前步是否满足**原始**判据，各为 `(num_envs,)` bool
            delay:  每个原因在该环境 (族, 级) 上的延迟秒数，`(num_envs,)` float
        Returns:
            `(terminated, elapsed, primary)`
              - `terminated`: `(num_envs,)` bool，是否应终止
              - `elapsed`:    原因 → `(num_envs,)` float，**已持续满足的秒数**。
                调用方在终止瞬间读它，即为该 episode 的"终止延迟"（M3 的
                "selected termination delays" 要观测的量）
              - `primary`:    原因 → `(num_envs,)` bool，哪些环境的终止由其触发
                （可能同时多个；调用方按 `FAILURE_REASONS` 顺序取首个来归因）
        """
        terminated = np.zeros(self.num_envs, dtype=bool)
        primary: Dict[TerminationReason, np.ndarray] = {}
        for reason in FAILURE_REASONS:
            if reason not in active:
                raise KeyError(f"active 缺少原因 {reason.value}")
            a = np.asarray(active[reason], dtype=bool)
            if a.shape != (self.num_envs,):
                raise ValueError(
                    f"{reason.value} 的形状应为 ({self.num_envs},)，得到 {a.shape}")
            d = np.asarray(delay[reason], dtype=np.float64)
            if d.shape != (self.num_envs,):
                raise ValueError(
                    f"{reason.value} 的 delay 形状应为 ({self.num_envs},)，得到 {d.shape}")
            if np.any(d < 0.0):
                raise ValueError(f"{reason.value} 的 delay 不应为负")

            buf = self.elapsed[reason]
            # 条件满足则累加、否则清零
            buf += np.where(a, self.dt, 0.0)
            buf[~a] = 0.0
            fired = a & (buf >= d - _DELAY_EPS)
            primary[reason] = fired
            terminated |= fired
        return terminated, dict(self.elapsed), primary

    def reset(self, env_ids) -> None:
        """episode 结束时清零对应环境的所有计时（必须调用，否则跨 episode 累积）。"""
        ids = np.asarray(list(env_ids), dtype=np.int64)
        if ids.size == 0:
            return
        if ids.min() < 0 or ids.max() >= self.num_envs:
            raise IndexError(f"env_ids 超出范围 [0, {self.num_envs})")
        for reason in FAILURE_REASONS:
            self.elapsed[reason][ids] = 0.0

    def elapsed_of(self, reason: TerminationReason, env_id: int) -> float:
        """查询某环境在某原因上已持续满足的秒数（调试/日志用）。"""
        return float(self.elapsed[reason][env_id])


# ---------------------------------------------------------------------------
# 组合入口
# ---------------------------------------------------------------------------


def compute_termination(state: TerminationState,
                        mean_body_pos_error: np.ndarray,
                        root_height: np.ndarray,
                        gravity_z: np.ndarray,
                        terrain_family: Sequence[str],
                        level: np.ndarray,
                        ) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                                   Dict[str, np.ndarray]]:
    """一次算齐三类判据 + 延迟。

    Returns:
        `(terminated, primary, elapsed)`，其中后两者均为 `原因名 → (num_envs,)`：
          - `primary`：该环境是否**因该原因**终止（bool）
          - `elapsed`：该原因已持续满足的秒数（float）—— 终止瞬间读它即得
            "终止延迟"，用于 M3 遗留的 delay 统计与日志

    参数均为 `(num_envs,)`（`terrain_family` 为长度 num_envs 的字符串序列）。
    延迟按**每环境自己的族与难度**计算 —— 同一步里各环境的延迟可以不同。
    """
    n = state.num_envs
    for name, arr in (("mean_body_pos_error", mean_body_pos_error),
                      ("root_height", root_height),
                      ("gravity_z", gravity_z), ("level", level)):
        if np.shape(arr) != (n,):
            raise ValueError(f"{name} 的形状应为 ({n},)，得到 {np.shape(arr)}")
    if len(terrain_family) != n:
        raise ValueError(f"terrain_family 长度应为 {n}，得到 {len(terrain_family)}")

    fams = list(terrain_family)
    unknown = set(fams) - set(FAMILIES)
    if unknown:
        raise ValueError(f"未知地形族: {sorted(unknown)}")

    active = {
        TerminationReason.REF_DEVIATION: np.array(
            [ref_deviation_exceeded(float(e), f, int(l), state.cfg)
             for e, f, l in zip(mean_body_pos_error, fams, level)], dtype=bool),
        TerminationReason.ROOT_LOW: np.array(
            [root_too_low(float(h), state.cfg) for h in root_height], dtype=bool),
        TerminationReason.TILTED: np.array(
            [tilted(float(g), state.cfg) for g in gravity_z], dtype=bool),
    }
    delay = {
        reason: np.array([termination_delay(reason, f, int(l), state.cfg)
                          for f, l in zip(fams, level)], dtype=np.float64)
        for reason in FAILURE_REASONS
    }
    terminated, elapsed, primary = state.step(active, delay)
    return (terminated,
            {r.value: primary[r] for r in FAILURE_REASONS},
            {r.value: elapsed[r] for r in FAILURE_REASONS})


def episode_outcome(terminated: np.ndarray, timeout: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """由 (early_termination, timeout) 给出 `(success, failed)`。

    论文的 completion 定义：跑满时间上限且**未提前终止** = success。
    """
    terminated = np.asarray(terminated, dtype=bool)
    timeout = np.asarray(timeout, dtype=bool)
    if terminated.shape != timeout.shape:
        raise ValueError(f"形状不一致: {terminated.shape} vs {timeout.shape}")
    success = timeout & ~terminated
    failed = terminated
    return success, failed
