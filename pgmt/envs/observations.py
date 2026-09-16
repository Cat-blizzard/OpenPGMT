"""观测契约层（M2 环境与策略的唯一定义出处）。

把论文 §III 的观测规格从"散落在多个模块的局部常量 + docstring"固化成
可校验、可索引、可测试的代码。此前的问题：
  - `_OBS_DIM = 96` 在 `history_encoder.py` 与 `actor.py` 各写了一遍；
  - 96 维的**内部分段**没有任何显式定义，只能从注释猜测；
  - `H_t` 是否含 `o_t` 只写在 docstring 里，无测试；
  - critic 特权观测（A9）的实际维度是测试里的硬编码占位 `PRIV_DIM = 20`。

## 观测布局（论文 §III，共 96 维）

| 段 | 维度 | 含义 |
|---|---|---|
| `e_t`      | 6  | 参考相对 anchor 朝向（6D 旋转，见 `rotation.relative_anchor_6d`）|
| `omega_t`  | 3  | 基座角速度（基座系，rad/s）|
| `q_t`      | 29 | 关节位置（rad，`G1_JOINT_NAMES` 顺序）|
| `qd_t`     | 29 | 关节速度（rad/s）|
| `a_prev`   | 29 | 上一时刻动作（关节位置目标）|

合计 **96**。注意 **`o_t` 不含基座线速度**——线速度属特权观测（A9）只给
critic；参考的平面速度通过 `C^K` 的 `ṽ^r` 进入策略（论文 Eq.1）。

## 历史 `H_t`

`H_t ∈ R^{10×96}`，与 `o_t` **同布局**。本项目约定 **`H_t` 不含当前帧
`o_t`**，即 `H_t = [o_{t-10}, ..., o_{t-1}]`（论文只写 "a ten-frame
proprioception history"，未明确是否含当前帧）。理由：`o_t` 已单独作为
History Encoder 的 query 与 Actor 的输入，含入只是重复信息并增加维度。
该约定由 `HistoryBuffer` 强制，并有边界测试锁定。

## 特权观测（critic，A9）

`PrivilegedLayout` 与 `OBS_LAYOUT` 分离：前者仅训练期可用，**不进 actor
输入**。维度可配置（`n_envs_dynamics`），默认按论文级仿真随机化形状给出。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM
from pgmt.policy.rotation import relative_anchor_6d

__all__ = [
    "OBS_SEGMENTS",
    "OBS_DIM",
    "HISTORY_LEN",
    "ObsSegment",
    "ObsLayout",
    "OBS_LAYOUT",
    "build_obs",
    "build_obs_from_state",
    "HistoryBuffer",
    "E6_IDENTITY",
    "e_t_identity_fill",
    "PrivSegment",
    "PrivilegedLayout",
    "PRIV_LAYOUT",
    "PRIV_DIM",
    "privileged_segments",
]

# ---------------------------------------------------------------------------
# 96 维观测布局
# ---------------------------------------------------------------------------

#: `o_t` 的分段定义：(名称, 维度, 说明)。顺序即论文 §III 的书写顺序。
OBS_SEGMENTS: Tuple[Tuple[str, int, str], ...] = (
    ("e_t", 6, "参考相对 anchor 朝向（6D 旋转）"),
    ("omega_t", 3, "基座角速度（基座系，rad/s）"),
    ("q_t", ACT_DIM, "关节位置（rad）"),
    ("qd_t", ACT_DIM, "关节速度（rad/s）"),
    ("a_prev", ACT_DIM, "上一时刻动作（关节位置目标）"),
)

# 加载期一致性校验：分段维度之和必须等于 pgmt/contracts.py 的 OBS_DIM。
# 放在这里（而非仅写测试）是为了让不一致在 import 时立刻暴露。
if sum(d for _, d, _ in OBS_SEGMENTS) != OBS_DIM:
    raise AssertionError(
        f"OBS_SEGMENTS 维度和 {sum(d for _, d, _ in OBS_SEGMENTS)} "
        f"≠ contracts.OBS_DIM {OBS_DIM}"
    )


@dataclass(frozen=True)
class ObsSegment:
    name: str
    offset: int
    dim: int
    description: str

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.dim)


class ObsLayout:
    """`o_t` 的显式布局：按名称取切片，并做构造期一致性校验。"""

    def __init__(self, segments: Sequence[Tuple[str, int, str]] = OBS_SEGMENTS):
        self._segments: List[ObsSegment] = []
        offset = 0
        seen: Dict[str, int] = {}
        for name, dim, desc in segments:
            if name in seen:
                raise ValueError(f"观测段名重复: {name}")
            if dim <= 0:
                raise ValueError(f"观测段 {name} 维度必须为正，得到 {dim}")
            self._segments.append(ObsSegment(name, offset, dim, desc))
            seen[name] = offset
            offset += dim
        self.total_dim = offset

    def __iter__(self) -> Iterator[ObsSegment]:
        return iter(self._segments)

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self._segments)

    def segment(self, name: str) -> ObsSegment:
        for s in self._segments:
            if s.name == name:
                return s
        raise KeyError(f"未知观测段: {name}（已有: {self.names}）")

    def index(self, name: str) -> slice:
        return self.segment(name).slice

    def dim(self, name: str) -> int:
        return self.segment(name).dim


OBS_LAYOUT = ObsLayout()


def build_obs(e_t: np.ndarray, omega_t: np.ndarray, q_t: np.ndarray,
              qd_t: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
    """按 `OBS_LAYOUT` 拼装 `o_t`，并校验各段维度。

    各入参支持 (B, dim) 或 (dim,)；返回同 batch 形状的 (..., 96)。
    """
    parts = []
    for name, arr in (("e_t", e_t), ("omega_t", omega_t), ("q_t", q_t),
                      ("qd_t", qd_t), ("a_prev", a_prev)):
        arr = np.asarray(arr, dtype=np.float32)
        want = OBS_LAYOUT.dim(name)
        if arr.shape[-1] != want:
            raise ValueError(f"{name} 维度应为 {want}，得到 {arr.shape[-1]}")
        parts.append(arr)
    return np.concatenate(parts, axis=-1)


def build_obs_from_state(robot_quat: np.ndarray, ref_quat: np.ndarray,
                         omega_t: np.ndarray, q_t: np.ndarray,
                         qd_t: np.ndarray, a_prev: np.ndarray) -> np.ndarray:
    """从锚点朝向 + 关节状态直接构造 `o_t`（`e_t` 由 `pgmt.policy.rotation` 计算）。"""
    e_t = relative_anchor_6d(robot_quat, ref_quat)
    return build_obs(e_t, omega_t, q_t, qd_t, a_prev)


# ---------------------------------------------------------------------------
# 历史环形缓冲 H_t
# ---------------------------------------------------------------------------

#: 6D 恒等旋转 `[1,0,0,0,1,0]`（`rotmat_to_6d(I)`）—— 用于冷启动分段填充
E6_IDENTITY: np.ndarray = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def e_t_identity_fill() -> np.ndarray:
    """冷启动的**分段填充向量**：`e_t` 段填 6D 恒等，其余段填 0。

    零填充的问题在于 `e_t` 的 6D 零向量不是合法旋转编码（分布外输入）；
    用恒等旋转更接近"机器人朝向与参考一致"这一中性状态。是否采用属工程
    取舍（论文未规定），作为 `HistoryBuffer(fill=...)` 的可选值提供。
    """
    fill = np.zeros(OBS_DIM, dtype=np.float32)
    fill[OBS_LAYOUT.index("e_t")] = E6_IDENTITY
    return fill


class HistoryBuffer:
    """固定容量 `HISTORY_LEN` 的 `o_t` 环形缓冲，产出 `H_t`（10, 96）。

    **语义约定（本项目拍定）**：`H_t` = 当前帧之前的 10 帧，**不含当前帧
    `o_t`**。即按 `o_0, o_1, ...` 顺序 push、推进到 `t` 后，`as_tensor()`
    返回 `[o_{t-10}, ..., o_{t-1}]`（最近的一帧在**末行**）。

    冷启动（尚未积累满 10 帧）时用 `fill` 在**前端**填充，使前 10 步的观测
    维度与后续一致，而不是输出可变长度的历史。默认零填充（HOVER/OmniH2O
    系惯例）；`fill=e_t_identity_fill()` 可改用分段填充，使 `e_t` 段为合法
    的 6D 恒等旋转 —— 见 `__init__` 的说明。

    内部按**时间正序**存储（与 `as_tensor()` 的行序一致），避免方向混淆：
    早期版本用 `appendleft` 存储却按正序读取，文档与数据方向相反。
    """

    def __init__(self, obs_dim: int = OBS_DIM, length: int = HISTORY_LEN,
                 fill: float | np.ndarray = 0.0):
        """Args:
            fill: 冷启动填充值。标量（默认 0.0，全局填充）或 `(obs_dim,)`
                  向量（**分段填充**，见下）。

        ⚠️ **`fill=0.0` 的已知副作用**：`e_t` 段的 6D 零向量不是合法的旋转
        编码（`rot6d_to_rotmat([0]*6)` 得到退化矩阵），因此零填充相当于给策略
        喂了 10 帧**分布外**输入，而不是"中性历史"。

        工程上仍以零填充为默认（HOVER/OmniH2O 系惯例；且只有 episode 前
        10 步受影响，随后被真实观测冲刷掉）。若要更保守，传
        `e_t_identity_fill()` 返回的分段向量：`e_t` 段填 6D 恒等
        `[1,0,0,0,1,0]`，其余段填 0。

        两种取值论文均未规定，属工程取舍；**无论选哪个都要在最终报告里说明**，
        不要让它成为隐藏差异。
        """
        if length <= 0:
            raise ValueError(f"历史长度必须为正，得到 {length}")
        if obs_dim <= 0:
            raise ValueError(f"观测维度必须为正，得到 {obs_dim}")
        self.obs_dim = int(obs_dim)
        self.length = int(length)

        fill_arr = np.asarray(fill, dtype=np.float32)
        if fill_arr.ndim == 0:
            self._fill = np.full(self.obs_dim, float(fill_arr), dtype=np.float32)
        elif fill_arr.shape == (self.obs_dim,):
            self._fill = fill_arr.copy()
        else:
            raise ValueError(
                f"fill 应为标量或形状 ({self.obs_dim},)，得到 {fill_arr.shape}")
        # 标量视图（仅供自省/日志；分段填充时取 e_t 之外无意义）
        self.fill = float(self._fill[0]) if np.all(self._fill == self._fill[0]) else float("nan")
        self._buf: deque = deque(maxlen=self.length)

    # ---- 状态 ----
    @property
    def count(self) -> int:
        """已积累的真实帧数（不含填充）。"""
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) == self.length

    def reset(self) -> None:
        self._buf.clear()

    # ---- 写入 ----
    def push(self, o: np.ndarray) -> None:
        """推入一帧 `o_t`（(obs_dim,)）。超过容量时自动丢弃最旧一帧。

        **会复制输入**。环境层通常复用一个 `obs_buf` 并按行切片喂进来
        （如 `hb.push(obs_buf[i])`），若不复制，缓冲区里存的是同一个底层
        内存的视图 —— 下一次写入就会篡改已经记入"历史"的那一帧，而症状是
        策略输入悄悄变成当前帧的重复，极难定位。
        `np.asarray` 对已经是 float32 的输入是**零拷贝**，所以这里显式判断
        是否共享内存后再复制：既保证正确，又避免对本来就会新建的输入
        （float64 / list）做多余拷贝。
        """
        arr = np.asarray(o, dtype=np.float32)
        if arr.shape != (self.obs_dim,):
            raise ValueError(f"o 形状应为 ({self.obs_dim},)，得到 {arr.shape}")
        if isinstance(o, np.ndarray) and np.shares_memory(arr, o):
            arr = arr.copy()
        self._buf.append(arr)

    # ---- 读取 ----
    def as_tensor(self) -> np.ndarray:
        """(length, obs_dim)，时间正序（末行 = 最近一帧）。未满则前端填充。"""
        n_pad = self.length - len(self._buf)
        pad = np.tile(self._fill, (n_pad, 1)) if n_pad else np.zeros((0, self.obs_dim), np.float32)
        if n_pad == self.length:
            return pad
        return np.concatenate([pad, np.stack(list(self._buf), axis=0)], axis=0)

    def as_batch(self, batch_size: int | None = None) -> np.ndarray:
        """(B, length, obs_dim)，供策略前向使用。"""
        h = self.as_tensor()
        if batch_size is None:
            return h
        return np.repeat(h[None], batch_size, axis=0)

    def observe_then_advance(self, o: np.ndarray) -> np.ndarray:
        """一步的**推荐调用顺序**：先按当前 `H_t` 取值，再推入本帧 `o_t`。

        等价于 `h = as_tensor(); push(o); return h`，但把顺序固定下来。
        顺序搞反（先 push 再取）就会让 `H_t` 含入当前帧 —— 症状是策略输入
        比论文多一帧自身的观测，且**不会报任何错**，只是性能悄悄变差。
        环境层每步用这一个方法即可，不必自己记顺序。

        Returns:
            (length, obs_dim) 本步应喂给 History Encoder 的 `H_t`
        """
        h = self.as_tensor()
        self.push(o)
        return h

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# 特权观测布局（A9，仅 critic）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrivSegment:
    name: str
    dim: int
    description: str


def privileged_segments(n_envs_dynamics: int = 1) -> Tuple[PrivSegment, ...]:
    """A9 特权观测集的分段与维度。

    Args:
        n_envs_dynamics: 动力学随机化按环境索引存储时的组数。
            1 = 单值（所有 env 同一随机化，基础情形）；
            N = 每 env 独立随机化（Isaac Gym 常规做法），
                "每 env 一组"的量（摩擦、质量、质心、电机强度）按 N 展开。

    维度选择说明（论文未给数值，属 A9 的实现细节，在此固化）：
      - `friction_coefficients`: 2 —— 每足一个（静/动合并为单值）
      - `base_mass_perturbation`: 1 —— 相对质量缩放
      - `com_perturbation`: 3 —— 质心偏移
      - `push_perturbation`: 3 —— 当前外力扰动（基座系），全局非按 env
      - `terrain_height_at_feet`: 2 —— 每足一个
      - `motor_strength_scale`: 29 —— 每关节一个
    """
    if n_envs_dynamics < 1:
        raise ValueError(f"n_envs_dynamics 必须 >= 1，得到 {n_envs_dynamics}")
    n = n_envs_dynamics
    return (
        PrivSegment("base_lin_vel", 3, "基座线速度（基座系，m/s）—— 论文未给 actor，属特权"),
        PrivSegment("base_ang_vel", 3, "基座角速度（与 o_t 冗余，供 critic）"),
        PrivSegment("foot_contact_states", 2, "左右足接触状态"),
        PrivSegment("friction_coefficients", 2 * n, "足端摩擦系数"),
        PrivSegment("terrain_height_at_feet", 2, "足端正下方地形高度"),
        PrivSegment("base_mass_perturbation", 1 * n, "基座质量扰动（相对缩放）"),
        PrivSegment("com_perturbation", 3 * n, "质心偏移"),
        PrivSegment("push_perturbation", 3, "当前外力扰动（基座系）"),
        PrivSegment("motor_strength_scale", ACT_DIM * n, "关节电机强度缩放"),
    )


class PrivilegedLayout:
    """critic 特权观测布局：按名称取切片，维度由 A9 + `n_envs_dynamics` 决定。

    默认（n_envs_dynamics=1）总维度为 **48**。此前测试里的 `PRIV_DIM = 20`
    是占位值，并非来自任何出处 —— 现以本表为准。
    """

    def __init__(self, n_envs_dynamics: int = 1):
        self._segments: List[PrivSegment] = []
        self._slices: Dict[str, slice] = {}
        offset = 0
        for seg in privileged_segments(n_envs_dynamics):
            self._segments.append(seg)
            self._slices[seg.name] = slice(offset, offset + seg.dim)
            offset += seg.dim
        self.total_dim = offset
        self.n_envs_dynamics = n_envs_dynamics

    def __iter__(self) -> Iterator[PrivSegment]:
        return iter(self._segments)

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self._segments)

    def index(self, name: str) -> slice:
        if name not in self._slices:
            raise KeyError(f"未知特权观测段: {name}（已有: {self.names}）")
        return self._slices[name]

    def dim(self, name: str) -> int:
        s = self.index(name)
        return s.stop - s.start


PRIV_LAYOUT = PrivilegedLayout()
PRIV_DIM = PRIV_LAYOUT.total_dim
