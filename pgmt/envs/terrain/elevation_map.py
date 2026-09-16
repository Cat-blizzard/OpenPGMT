"""高程图 `M_t ∈ R^{21×21}`（论文 §III），yaw 对齐、覆盖机器人系 2m×2m。

## 论文明确的

  - `M_t ∈ R^{21×21}`，覆盖 yaw 对齐的 **2m×2m** 区域
  - 只在 Stage 2 引入（Stage 1 是 terrain-agnostic 的）
  - Stage 2 的松弛与课程都依赖"机器人当前所在地形族与难度"，因此高程图
    必须与 tile 的正负号、朝向、原点一致

## 坐标约定（**必须与 `pgmt/policy/glimpse_encoder.py` 逐位一致**）

`glimpse_encoder._grid_from_centers` 把米映射到 `grid_sample` 的归一化坐标：

    p = (center_m + 1.0) / 2.0 * (map_size - 1)      # 米 → 像素下标（浮点）

因此 `grid_sample(align_corners=True)` 下：

    像素下标 i=0    ↔  x = −1.0 m
    像素下标 i=n−1  ↔  x = +1.0 m
    格距            = 2.0 / (n − 1) m

本模块的 `cell_centers()` 与之一致：**格心**位于 `i·res − floor(n/2)·res`，
其中 `res = 2.0 / (n − 1)`。取 A8 的 `map_size=21`、`map_res=0.1` 时：

    2.0 / (21 − 1) = 0.1 = map_res        ← 两者自洽（有测试锁定）
    floor(21/2) × 0.1 = 1.0 m             ← 正好覆盖 ±1m

即 **A8 的 `map_res` 与 `map_size` 不是独立参数** —— 给定 map_size=21，
res 必须等于 0.1，否则与 glimpse 采样器的约定冲突。这条由测试强制。

**行列约定**：`M[i, j]` 对应 `y = centers[i]`、`x = centers[j]`（与图像一致）。
glimpse 的 `gx` 用 `p[..., 0]`（x）作列、`gy` 用 `p[..., 1]`（y）作行，与此一致。

## 相对高度（关键语义）

`M_t` 存的是**相对机器人基座高度 `z_robot` 的高度差**：`h(x, y) − z_robot`。
原因：论文对 terrain-contact 奖励说 "Local height variation is used to evaluate
touchdown quality" —— 判断落足质量看的是**局部高差**，不是海拔。若存绝对高度，
策略必须自己减掉基座高度才能得到该信息，等于把一个恒等变换塞进网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.generators import TerrainParams, height_at

#: 高程图规格取自 A8（唯一出处）
_A8 = get("A8").value

#: 格数（论文 §III：21×21）
MAP_SIZE: int = _A8.map_size


def grid_resolution(map_size: int = MAP_SIZE) -> float:
    """格距（m/格）—— 由 `glimpse_encoder` 的 `align_corners` 约定推出。

    推导：`(center + 1)/2 * (n−1)` 把 `[−1, +1] m` 映射到像素 `[0, n−1]`，
    故 1 m 对应 `(n−1)/2` 个像素 → 格距 `= 2/(n−1)` m。
    """
    if map_size < 2:
        raise ValueError(f"格数至少为 2，得到 {map_size}")
    return 2.0 / (map_size - 1)


def cell_centers(map_size: int = MAP_SIZE) -> np.ndarray:
    """格心坐标（m），`(map_size,)`，覆盖 `±floor(n/2)·res`。

    n=21 时得到 `[−1.0, −0.9, …, 0.9, 1.0]`，与 `glimpse_encoder` 完全一致。
    """
    res = grid_resolution(map_size)
    return (np.arange(map_size) - map_size // 2) * res


def elevation_extent(map_size: int = MAP_SIZE) -> float:
    """高程图覆盖的半宽（m）。n=21 时为 1.0。"""
    return float(map_size // 2) * grid_resolution(map_size)


@dataclass(frozen=True)
class ElevationNoiseModel:
    """A8 的高程图扰动模型。

    Args:
        sigma: 逐格高斯噪声标准差（m）。随难度线性增长（`sigma_min` → `sigma_max`）
        dropout_prob: 每格被置 0 的概率。随难度线性增长到 `dropout_prob_max`

    论文提到 "observation corruption" 但未给细节，故登记为 A8。
    """

    sigma: float = 0.0
    dropout_prob: float = 0.0


def noise_model_for(level: int) -> ElevationNoiseModel:
    """按难度级取噪声模型（A8：sigma 与 dropout 均随难度线性增长）。"""
    n_levels = 10
    if not 0 <= level < n_levels:
        raise ValueError(f"难度级应在 [0, {n_levels - 1}]，得到 {level}")
    frac = level / (n_levels - 1)
    a8 = get("A8").value
    return ElevationNoiseModel(
        sigma=a8.sigma_min + (a8.sigma_max - a8.sigma_min) * frac,
        dropout_prob=a8.dropout_prob_max * frac,
    )


def sample_elevation(params: TerrainParams,
                     robot_xy: Tuple[float, float],
                     robot_z: float,
                     yaw: float = 0.0,
                     map_size: int = MAP_SIZE) -> np.ndarray:
    """在机器人处采样高程图，返回 `(n, n)` 的**相对高度**（m）。

    Args:
        params:   机器人所在 tile 的地形参数
        robot_xy: 机器人基座在 **tile 局部坐标**中的 `(x, y)`（m）
        robot_z:  机器人基座高度（m，tile 局部系）。高程图存 `h − robot_z`
        yaw:      机器人朝向（rad）。高程图 yaw 对齐 —— 格网随机器人旋转，
                  因此采样点是 `robot_xy + R(yaw) · offset`，而不是轴对齐的原点
        map_size: 格数（默认 21）

    Returns:
        `(map_size, map_size)`，`M[i, j]` 对应 `y = centers[i]`、`x = centers[j]`

    注：`height_at` 对 tile 外/留白区的点返回 0，因此机器人贴近 tile 边缘时，
    高程图里 tile 之外的部分会显示为 0（平地）—— 这在课程里是可接受的近似，
    因为 tile 四周本就是 z=0 的平台（见 `generators` 的留白设计）。
    """
    c = cell_centers(map_size)
    # 局部偏移 → 世界（tile 局部系）偏移：绕 z 轴旋转 yaw
    cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
    dx = np.outer(np.ones_like(c), c)          # (n, n)：列变化 → x 偏移
    dy = np.outer(c, np.ones_like(c))          # (n, n)：行变化 → y 偏移
    gx = robot_xy[0] + cos_y * dx - sin_y * dy
    gy = robot_xy[1] + sin_y * dx + cos_y * dy

    h = height_at(params, gx, gy)
    return h - float(robot_z)


def corrupt_elevation(M: np.ndarray, model: ElevationNoiseModel,
                      rng: np.random.Generator) -> np.ndarray:
    """按 A8 施加观测扰动：逐格高斯噪声 + 逐格 dropout（置 0）。

    Args:
        M:   `(n, n)` 干净高程图（m）
        rng: 显式随机源 —— **不提供默认值**，保证可复现性由调用方掌控
             （评估协议要求"各策略使用完全相同的观测扰动"）

    注意 dropout 是**逐格独立**的，不是整图丢弃：论文未给细节，取 A8 的设定。
    """
    out = np.asarray(M, dtype=np.float64).copy()
    if model.sigma > 0.0:
        out += rng.normal(0.0, model.sigma, size=out.shape)
    if model.dropout_prob > 0.0:
        mask = rng.random(out.shape) < model.dropout_prob
        out[mask] = 0.0
    return out


def observe_elevation(params: TerrainParams,
                      robot_xy: Tuple[float, float],
                      robot_z: float,
                      level: int,
                      yaw: float = 0.0,
                      rng: np.random.Generator | None = None,
                      map_size: int = MAP_SIZE) -> np.ndarray:
    """一步到位：采样 + 按难度施加 A8 扰动。

    `rng=None` 时**不施加扰动**（训练/评估的干净路径）；给了 `rng` 才扰动 ——
    这样"是否加噪"是显式的，而不是靠某个默认值偷偷决定。
    """
    M = sample_elevation(params, robot_xy, robot_z, yaw=yaw, map_size=map_size)
    if rng is None:
        return M
    return corrupt_elevation(M, noise_model_for(level), rng)


def elevation_stats(M: np.ndarray) -> Dict[str, float]:
    """高程图的诊断统计（供日志与测试）。"""
    arr = np.asarray(M, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "range": float(arr.max() - arr.min()),
    }
