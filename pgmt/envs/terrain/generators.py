"""五族地形生成器（M3，论文 §V-A 的 five terrain families）。

## 论文说了什么、没说什么

**论文明确的**（照搬）：
  - 五个地形族：flat / slopes / stairs / box obstacles / randomly rough
  - 每族十个难度级 L0–L9，难度递增
  - 难度级"jointly controls geometric difficulty, tracking relaxation, and
    selected termination delays" → 因此**难度级是一个单一标量**，几何、松弛、
    终止延迟都由它导出（这正是本模块只暴露 `level` 而不单独暴露几何参数的原因）
  - 训练在平地上做 tracking 预训练（Stage 1），Stage 2 才引入地形

**论文没说的**（本项目拍定，登记在 A7 / A18 体系，需在报告说明）：
  - 各族 L0→L9 的**具体数值端点** → A7（slope 5°→30°、stairs 阶高 4→24cm、
    boxes 5→40cm、rough 幅值 1→12cm）。boxes L9=40cm 对齐真机"37cm 上限"
  - **tile 几何**（边长、边界留白、楼梯/方块的排布）→ 本模块的模块级常量
  - **rough 噪声的具体形式** → 取"整数周期正弦叠加"：在 `x,y` 上以 tile 尺寸为
    周期，从而**跨 tile 无缝**（否则相邻 tile 高度不连续，机器人会撞到看不见的墙）；
    确定性由 seed 决定

## 与高程图的约定耦合（重要）

`pgmt/policy/glimpse_encoder.py` 与 A8 都假定高程图覆盖机器人系 **2m×2m**、
21×21 格（`map_res = 0.1`）。因此 **tile 边长必须 > 2m**，否则一张高程图会跨越
多个 tile，"机器人位于哪个地形族"就变得没有定义。

本模块取 `TILE_SIZE = 4.0`（4 倍余量），并有测试锁定 `TILE_SIZE > 2 * half_extent`。

## 坐标与"边界留白"

每个 tile 是边长 `TILE_SIZE` 的正方形，局部坐标 `x, y ∈ [−TILE_SIZE/2, +TILE_SIZE/2]`，
`z` 向上。tile 四周留 `BORDER` 宽的**平台区（z=0）**，几何特征只出现在内部 ——
这样相邻 tile 在边界处高度一致（都是 0），不会产生跨越 tile 的台阶状裂缝。
这是 legged_gym 系 terrain 的通行做法，也是论文未规定的实现细节。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from pgmt.cfg.assumptions import get

#: 地形族（论文 §V-A 的五个族，顺序即报告中的展示顺序）
FAMILIES: Tuple[str, ...] = ("flat", "slopes", "stairs", "boxes", "rough")

#: 难度级数（L0–L9）
NUM_LEVELS: int = 10

#: tile 几何取自 A19（**唯一出处**，勿在此硬编码）。
_A19 = get("A19").value

#: tile 边长（m）。必须 > 2m —— 见模块 docstring 的"与高程图的约定耦合"
TILE_SIZE: float = _A19.tile_size

#: tile 四周的平坦留白宽度（m）。相邻 tile 在边界处同为 z=0，避免裂缝
BORDER: float = _A19.border

#: 高程图半宽（m），= A8.map_size * map_res / 2 = 21*0.1/2 = 1.05
#: 用于断言 tile 足够大，使一张高程图落在单个 tile 内
ELEVATION_HALF_EXTENT: float = get("A8").value.map_size * get("A8").value.map_res / 2.0

#: stairs 的台阶级数（A19.stair_steps）。取固定级数使**总上升量正比于难度**：
#: L9 最大总上升 = (4−1) × 24cm ≈ 0.72m，与踏步高度同量级、几何上自洽。
#: 若改为"阶距固定"，L9 会在 1m 内爬升 ≈1.4m（约 55° 的阶梯），不合物理。
STAIR_STEPS: int = _A19.stair_steps

#: rough 噪声的谐波数（A19.rough_harmonics）
ROUGH_HARMONICS: int = _A19.rough_harmonics


@dataclass(frozen=True)
class TerrainParams:
    """一个 (族, 难度级) 的完整几何参数。

    由 `params_for()` 从 A7 的端点表线性插值得到，**不要在别处硬编码几何数值**。
    """

    family: str
    level: int
    #: slopes：内部平台相对留白处的高度差（m），由坡度与内部半宽推出
    slope_rise: float = 0.0
    #: stairs：单级台阶高度（m）
    step_height: float = 0.0
    #: stairs / boxes：沿 x 方向的格距（m）
    spacing: float = 0.5
    #: boxes：方块高度（m）
    box_height: float = 0.0
    #: rough：噪声幅值（m）
    rough_amp: float = 0.0
    #: rough：噪声 seed（由 level 导出，保证可复现）
    seed: int = 0

    @property
    def inner_half(self) -> float:
        """内部特征区的半宽（m）：tile 半宽减去留白。"""
        return TILE_SIZE / 2.0 - BORDER


def _lerp_table(table: Tuple[float, ...], level: int) -> float:
    if not 0 <= level < len(table):
        raise ValueError(f"难度级应在 [0, {len(table) - 1}]，得到 {level}")
    return float(table[level])


def params_for(family: str, level: int) -> TerrainParams:
    """取 (族, 难度级) 的几何参数。

    对 flat：无几何，难度由 `A7.flat_randomize` 表示（随机化强度），
    几何参数全为 0 —— 调用方用 `flat_randomize_strength()` 取该强度。
    """
    if family not in FAMILIES:
        raise ValueError(f"未知地形族 {family!r}（可选 {FAMILIES}）")
    if not 0 <= level < NUM_LEVELS:
        raise ValueError(f"难度级应在 [0, {NUM_LEVELS - 1}]，得到 {level}")

    t = get("A7").value
    inner = TILE_SIZE / 2.0 - BORDER

    if family == "flat":
        return TerrainParams(family=family, level=level)

    if family == "slopes":
        # 金字塔式：内部平台高出留白处 slope_rise，由坡度与内部半宽推出
        deg = _lerp_table(t.slope_deg, level)
        rise = inner * math.tan(math.radians(deg))
        return TerrainParams(family=family, level=level, slope_rise=rise)

    if family == "stairs":
        # 沿 +x 上升的固定级数阶梯；阶高取 A7 端点
        h = _lerp_table(t.stairs_h_cm, level) / 100.0
        return TerrainParams(family=family, level=level, step_height=h,
                             spacing=2.0 * inner / STAIR_STEPS)

    if family == "boxes":
        h = _lerp_table(t.boxes_h_cm, level) / 100.0
        # 方块间距随高度增大，避免高难度时方块互相重叠成一堵墙
        spacing = max(2.5 * h, 0.3)
        return TerrainParams(family=family, level=level, box_height=h, spacing=spacing)

    # rough
    amp = _lerp_table(t.rough_amp_cm, level) / 100.0
    return TerrainParams(family=family, level=level, rough_amp=amp,
                         seed=1000 + level)


def flat_randomize_strength(level: int) -> float:
    """flat 族的"难度" = 随机化强度 0→1（A7.flat_randomize）。"""
    return _lerp_table(get("A7").value.flat_randomize, level)


# ---------------------------------------------------------------------------
# 高度场
# ---------------------------------------------------------------------------

def _periodic_noise(x: np.ndarray, y: np.ndarray, amp: float,
                    n_harmonics: int = ROUGH_HARMONICS, seed: int = 0) -> np.ndarray:
    """以 `TILE_SIZE` 为周期、关于 0 对称的确定性高度噪声。

    形式：`n_harmonics` 个 `sin(2π(kx·x + ky·y)/TILE_SIZE + φ)` 的加权和，
    再除以权重和。**每个谐波的 `kx, ky` 都取整数波数**，因此在 x、y 方向上的
    周期正好是 `TILE_SIZE` —— 噪声在 tile 上相位自洽，不会因 tile 索引不同而突变。

    这是本实现的刻意选择：任意随机场（如逐 tile 独立采样的 Perlin）在边界处不连续。

    **幅值**：`|输出| ≤ amp`，但这是**上界不是保证**。除以权重和（L1 归一化）
    使得峰值随谐波相位差增大而下降，实测峰值约为 `amp` 的 50~80%。
    且由于各项正弦相位随机，实际极值**未必在采样网格上取到** ——
    因此测试只断言边界不越出 `±amp`，不断言峰值一定达到某个比例。
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(np.broadcast_arrays(x, y)[0], dtype=np.float64)
    weight_sum = 0.0
    for _ in range(n_harmonics):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        w = float(rng.uniform(0.5, 1.0))
        out += w * np.sin(2.0 * math.pi * (kx * x + ky * y) / TILE_SIZE + phase)
        weight_sum += w
    return out / max(weight_sum, 1e-9) * amp


def height_at(p: TerrainParams, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """地形高度场 `z = h(x, y)`，单位米。支持标量或数组（广播）。

    约定：
      - `x, y` 是 **tile 局部坐标**（米），范围 `[−TILE_SIZE/2, +TILE_SIZE/2]`
      - 留白区（`max(|x|, |y|) > inner_half`）恒为 **0**，保证跨 tile 连续
      - 所有族都满足 `height_at(..., 边界处) == 0`

    **值域（各族不同，按论文用词区分）**：

    | 族 | 论文用词 | 值域 |
    |---|---|---|
    | flat / slopes / stairs / boxes | flat / slopes / stairs / **obstacles** | `[0, 上限]` —— 只在平台上向上长 |
    | rough | randomly **rough** terrain | `[−amp, +amp]` —— **关于 0 对称，有凸有凹** |

    rough 之所以允许负值：论文对它的用词是"粗糙"而非"障碍"。一个只有突起的
    表面是"布满小方块"（即 boxes），不是粗糙地面。对称噪声才符合"粗糙"的语义。
    代价是地面可低于 z=0，落到表面的机器人会被抬升 `|amp|`（L9 约 12cm）——
    这是可接受的后果，且若需要可整体上移；本项目**不做**该上移，以保持与论文
    "随机粗糙地形"的一致。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inner = p.inner_half
    inside = (np.abs(x) <= inner) & (np.abs(y) <= inner)

    if p.family == "flat":
        return np.zeros_like(np.broadcast_arrays(x, y)[0], dtype=np.float64)

    if p.family == "slopes":
        # 金字塔：距内部的中心越近越高。中心处为 slope_rise，内部边缘处为 0
        d = np.maximum(np.abs(x), np.abs(y))
        t = np.clip((inner - d) / inner, 0.0, 1.0)   # 边缘 0 → 中心 1
        return np.where(inside, p.slope_rise * t, 0.0)

    if p.family == "stairs":
        # 沿 +x 逐级上升：把内部区均分为 STAIR_STEPS 级，前 spacing 宽为起步平地
        idx = np.floor((x + inner) / p.spacing).astype(np.int64)
        rise = np.clip(idx - 1, 0, STAIR_STEPS - 1) * p.step_height
        return np.where(inside, rise, 0.0)

    if p.family == "boxes":
        # 离散方块阵：格心处一个方块，格间为空。
        # 格数向下取整 → 方块不会越出内部区（有测试锁定）
        n = max(int(2 * inner / p.spacing), 1)
        i = np.floor((x + inner) / p.spacing).astype(np.int64)
        j = np.floor((y + inner) / p.spacing).astype(np.int64)
        # 方块占每格中心的 60%，形成有间隙的离散障碍
        cx = -inner + (i + 0.5) * p.spacing
        cy = -inner + (j + 0.5) * p.spacing
        w = 0.3 * p.spacing
        on_box = (np.abs(x - cx) <= w) & (np.abs(y - cy) <= w)
        on_box &= (i >= 0) & (i < n) & (j >= 0) & (j < n)
        return np.where(inside & on_box, p.box_height, 0.0)

    if p.family == "rough":
        return np.where(inside, _periodic_noise(x, y, p.rough_amp, seed=p.seed), 0.0)

    raise ValueError(f"未实现的地形族: {p.family}")   # pragma: no cover


def sample_grid(p: TerrainParams, n: int, extent: float | None = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 tile 局部坐标上以 `n×n` 网格采样高度场。

    Args:
        n: 网格边长（点数）
        extent: 采样半宽（m）；默认 `TILE_SIZE/2`（覆盖整个 tile）
    Returns:
        (xs, ys, zs)，`xs`、`ys` 为 (n,) 坐标，`zs` 为 (n, n) 高度（`zs[i, j]` 对应
        `y = ys[i]`、`x = xs[j]`，与图像行列一致）
    """
    if n < 2:
        raise ValueError(f"网格边长至少为 2，得到 {n}")
    half = TILE_SIZE / 2.0 if extent is None else float(extent)
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, height_at(p, X, Y)


# ---------------------------------------------------------------------------
# 便利入口
# ---------------------------------------------------------------------------

def all_params() -> Dict[Tuple[str, int], TerrainParams]:
    """全部 5 族 × 10 级的参数表（供可视化与测试遍历）。"""
    return {(f, lv): params_for(f, lv) for f in FAMILIES for lv in range(NUM_LEVELS)}


def is_inside_tile(p: TerrainParams, x: float, y: float) -> bool:
    """`(x, y)` 是否落在 tile 的内部特征区内（非留白）。"""
    return abs(x) <= p.inner_half and abs(y) <= p.inner_half
