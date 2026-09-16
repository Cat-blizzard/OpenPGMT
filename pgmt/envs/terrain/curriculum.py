"""地形课程与 behavior–terrain 兼容规则（M3 第③块）。

## 论文说了什么

  - "we train on flat terrain, slopes, stairs, boxes, and random rough terrain
    using a **level-based curriculum**, in which environments **advance to harder
    terrain after successful completion and regress to easier levels after
    tracking failures**"
  - "The terrain level jointly controls geometric difficulty, tracking
    relaxation, and selected termination delays"
  - "Since not every motion is feasible on every terrain, we use a **coarse
    behavior–terrain compatibility rule** to exclude clearly invalid
    combinations when sampling references."
  - Stage 1 全部在平地（`All pretraining is conducted on flat terrain`），
    因此**课程只在 Stage 2 生效**

**论文没说的**（本项目拍定）：
  - 升降级是在**单族内**还是跨族 → 本实现取单族内（`level` 单调升降，
    族由 tile 分配固定），因为"族"在评估协议里是 **matched** 的
    （9600 集 = 5 族 × 10 级 × 192，族与级都是固定分配的一部分）
  - 升降级的**判据** → 取"episode 是否完成"（论文的 completion 定义：
    到达 30s 时间上限且未提前终止）
  - 兼容规则里"运动类别"**从哪来** → 论文只说"粗特征分类器"。
    本实现按序列名前缀解析（见 `MOTION_PREFIX_TO_CATEGORY`），
    对 LAFAN1 的 12 个前缀是精确的；对未知/新数据回退到 `other`
    （`other` 不被任何规则排除，因此**保守**：宁可允许也不误排除）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.generators import FAMILIES, NUM_LEVELS

# ---------------------------------------------------------------------------
# 运动类别（A11 的 excluded 表使用的键）
# ---------------------------------------------------------------------------

#: A11 规则里出现的运动类别 + 兜底 `other`
#: （与 `CompatibilityRules.excluded` 的键保持一致，有测试锁定）
MOTION_CATEGORIES: Tuple[str, ...] = (
    "lying_prone",    # 躺 / 趴（含翻滚、爬行）
    "inverted",       # 倒立 / 头朝下（侧手翻等）
    "sitting",        # 坐姿
    "upright_locomotion",  # 走 / 跑 / 冲刺 / 过障碍
    "dynamic_whole_body",  # 跳 / 舞 / 格斗 / 瞄准
    "push_perturbation",   # 受推力扰动 / 摔倒起身
    "other",          # 未分类（**不被任何规则排除**）
)

#: LAFAN1 序列名前缀 → 运动类别。
#: 前缀取自实际数据集（77 个序列共 12 个前缀），逐一人工归类。
#: 这不是"猜"：`ground*` 是躺地/翻滚、`fallAndGetUp*` 是摔倒起身，
#: 属数据集自带的语义标注。
MOTION_PREFIX_TO_CATEGORY: Dict[str, str] = {
    "walk": "upright_locomotion",
    "run": "upright_locomotion",
    "sprint": "upright_locomotion",
    "obstacles": "upright_locomotion",
    "aiming": "dynamic_whole_body",
    "dance": "dynamic_whole_body",
    "fight": "dynamic_whole_body",
    "fightAndSports": "dynamic_whole_body",
    "jumps": "dynamic_whole_body",
    "multipleActions": "dynamic_whole_body",
    "ground": "lying_prone",
    "fallAndGetUp": "push_perturbation",
    "push": "push_perturbation",
    "pushAndFall": "push_perturbation",
    "pushAndStumble": "push_perturbation",
}


def longest_prefix_match(name: str, table: Dict[str, str]) -> Optional[str]:
    """在 `table` 中找 `name` 的**最长**匹配前缀；无匹配返回 None。

    单独抽出来是为了**可测**：把它绑定在模块级的真实表上测不出"最长匹配"这一规则
    —— 因为真实表里重叠的前缀（`push` / `pushAndFall` / `pushAndStumble`）
    恰好都映射到同一类别，用真实表无法构造"取短前缀就会得到不同答案"的用例。
    因此对它传入人工构造的表来验证规则本身。
    """
    best: Optional[str] = None
    for prefix in table:
        if name.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return best


def motion_category(seq_name: str) -> str:
    """由序列名解析运动类别。

    取**最长匹配前缀**（如 `pushAndFall` 优先于 `push`）。
    未知前缀 → `"other"`（保守：`other` 不被任何规则排除）。
    """
    prefix = longest_prefix_match(seq_name, MOTION_PREFIX_TO_CATEGORY)
    return MOTION_PREFIX_TO_CATEGORY[prefix] if prefix else "other"


# ---------------------------------------------------------------------------
# A11 兼容规则
# ---------------------------------------------------------------------------

def excluded_families(category: str) -> Tuple[str, ...]:
    """该运动类别被禁止采样的地形族（不看难度级）。"""
    rules = get("A11").value
    out: List[str] = []
    for cat, families in rules.excluded:
        if cat == category:
            out.extend(families)
    return tuple(out)


def min_excluded_level() -> int:
    """A11：仅在该难度级及以上才应用排除规则。"""
    return int(get("A11").value.min_level_excluded)


def is_compatible(category: str, family: str, level: int) -> bool:
    """论文的"粗兼容规则"：该 (运动类别, 地形族, 难度级) 组合是否可行。

    规则形式：`excluded` 表把某些类别在某些族上的**高难度**组合排除掉
    （A11.min_level_excluded 以下不排除 —— 低难度地形温和，各类动作都还可行）。

    未知类别（`other`）与未被排除的组合一律返回 True —— **保守方向正确**：
    误排除会悄悄丢掉可用数据，误允许只是让某个组合更难，后者可接受。
    """
    if family not in FAMILIES:
        raise ValueError(f"未知地形族 {family!r}（可选 {FAMILIES}）")
    if not 0 <= level < NUM_LEVELS:
        raise ValueError(f"难度级应在 [0, {NUM_LEVELS - 1}]，得到 {level}")
    if level < min_excluded_level():
        return True
    return family not in excluded_families(category)


def incompatible_categories(family: str, level: int) -> Tuple[str, ...]:
    """在给定 (族, 级) 上被排除的运动类别（供采样器快速过滤）。"""
    return tuple(c for c in MOTION_CATEGORIES if not is_compatible(c, family, level))


def filter_sequences(seq_names: Sequence[str], family: str, level: int
                     ) -> List[str]:
    """按兼容规则过滤序列名列表，返回可用的子集。

    环境层在 (重新) 采样参考时调用。若过滤后为空，调用方需要决定策略
    （本函数**不**擅自放宽规则 —— 那会静默违反论文约束）；参见
    `filter_sequences_or_all`。
    """
    return [s for s in seq_names if is_compatible(motion_category(s), family, level)]


def filter_sequences_or_all(seq_names: Sequence[str], family: str, level: int
                            ) -> Tuple[List[str], bool]:
    """同 `filter_sequences`，但为空时回退到全集。

    Returns:
        `(可用序列, 是否发生了回退)`。第二项用于**记录事件** ——
        回退意味着兼容规则在该组合下把全部候选都排除了，说明规则与数据不匹配，
        应当在日志里暴露而不是静默吞掉。
    """
    kept = filter_sequences(seq_names, family, level)
    if kept:
        return kept, False
    return list(seq_names), True


# ---------------------------------------------------------------------------
# 课程：level-based 升降级
# ---------------------------------------------------------------------------

@dataclass
class CurriculumState:
    """单环境的课程状态：当前族、当前难度级、在该级上的连续成败计数。"""

    family: str
    level: int = 0
    #: 当前级别上连续成功的 episode 数（达到阈值则升级）
    consecutive_success: int = 0
    #: 当前级别上连续失败的 episode 数（达到阈值则降级）
    consecutive_failure: int = 0
    #: 总升级 / 降级次数（监控用）
    promotions: int = 0
    demotions: int = 0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"未知地形族 {self.family!r}")
        if not 0 <= self.level < NUM_LEVELS:
            raise ValueError(f"难度级应在 [0, {NUM_LEVELS - 1}]，得到 {self.level}")


class TerrainCurriculum:
    """每环境独立的地形课程（论文：成功升级、失败降级）。

    Stage 1 不使用本类（论文：pretraining 全在平地）。Stage 2 的环境层持有一个
    实例，在每次 episode 结束时调用 `update(env_ids, success)`。

    Args:
        families: 每环境的初始地形族（长度 = 环境数）。评估协议里族是**固定分配**
            的一部分，因此构造后不再变化；课程只调整 `level`
        promote_after: 连续成功多少集升级
        demote_after: 连续失败多少集降级
        max_level: 难度上限（默认 L9）

    为什么用"连续"而不是"滑动窗口成功率"：论文只说
    "advance ... after successful completion and regress ... after tracking
    failures"，未给统计口径。连续计数是最简形式，且对单个 episode 的结果反应明确
    （成功率窗口需要一个额外的窗口长度，属另一层自由度）。此为拍定项。
    """

    def __init__(self, families: Sequence[str],
                 promote_after: int = 3, demote_after: int = 1,
                 max_level: int = NUM_LEVELS - 1):
        if len(families) == 0:
            raise ValueError("families 不能为空")
        if promote_after < 1 or demote_after < 1:
            raise ValueError("promote_after / demote_after 必须 >= 1")
        if not 0 < max_level < NUM_LEVELS:
            raise ValueError(f"max_level 应在 (0, {NUM_LEVELS})，得到 {max_level}")
        self.states: List[CurriculumState] = [
            CurriculumState(family=f, level=0) for f in families
        ]
        self.promote_after = int(promote_after)
        self.demote_after = int(demote_after)
        self.max_level = int(max_level)

    # ---- 查询 ----
    @property
    def num_envs(self) -> int:
        return len(self.states)

    def level_of(self, env_id: int) -> int:
        return self.states[env_id].level

    def family_of(self, env_id: int) -> str:
        return self.states[env_id].family

    def levels(self) -> np.ndarray:
        return np.array([s.level for s in self.states], dtype=np.int64)

    def families(self) -> Tuple[str, ...]:
        return tuple(s.family for s in self.states)

    # ---- 更新 ----
    def update(self, env_ids: Iterable[int], success: bool) -> List[str]:
        """按 episode 结果更新课程；返回发生变化的描述（供日志）。

        判定顺序：先累计同类计数并清零另一类；达到阈值则升/降级并把两个
        计数都清零（避免刚升级又立刻因上一个级别的残留计数降级）。
        """
        events: List[str] = []
        for e in env_ids:
            e = int(e)
            if not 0 <= e < len(self.states):
                raise IndexError(f"env_id {e} 超出范围 [0, {len(self.states)})")
            st = self.states[e]
            if success:
                st.consecutive_success += 1
                st.consecutive_failure = 0
                if st.consecutive_success >= self.promote_after and st.level < self.max_level:
                    st.level += 1
                    st.promotions += 1
                    events.append(f"env{e} {st.family} 升级 → L{st.level}")
                    self._reset_streaks(st)
                elif st.consecutive_success >= self.promote_after:
                    # 已在最高级：保持计数不增长，避免溢出且语义清晰
                    self._reset_streaks(st)
            else:
                st.consecutive_failure += 1
                st.consecutive_success = 0
                if st.consecutive_failure >= self.demote_after and st.level > 0:
                    st.level -= 1
                    st.demotions += 1
                    events.append(f"env{e} {st.family} 降级 → L{st.level}")
                    self._reset_streaks(st)
                elif st.consecutive_failure >= self.demote_after:
                    self._reset_streaks(st)   # 已在 L0，无处可降
        return events

    @staticmethod
    def _reset_streaks(st: CurriculumState) -> None:
        st.consecutive_success = 0
        st.consecutive_failure = 0

    # ---- 汇总 ----
    def summary(self) -> Dict[str, object]:
        """课程分布快照（训练日志用）：各级多少个环境、各族均值等。"""
        levels = self.levels()
        hist = {int(lv): int((levels == lv).sum()) for lv in range(NUM_LEVELS)}
        return {
            "num_envs": self.num_envs,
            "level_histogram": hist,
            "mean_level": float(levels.mean()) if self.num_envs else 0.0,
            "max_level": int(levels.max()) if self.num_envs else 0,
            "promotions": int(sum(s.promotions for s in self.states)),
            "demotions": int(sum(s.demotions for s in self.states)),
        }
