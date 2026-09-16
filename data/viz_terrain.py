"""地形检查图：五族 × L0–L9 的高度场与高程图（M3 验收的可视化）。

生成 `eval/viz/terrain_overview.png`：5 行（族）× 10 列（难度）的小图，
每格绘该 (族, 级) 的高度场俯视图（x-y 平面 + 高度颜色）。

用法::

    python -m data.viz_terrain                       # 全族全级
    python -m data.viz_terrain --families slopes stairs
    python -m data.viz_terrain --levels 0 5 9 --out /tmp/t.png

另提供 `dump_elevation_map()`：把某 (族, 级) 的 21×21 高程图打到 stdout，
用于在无 matplotlib 的环境下快速核对数值。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence

import numpy as np

from pgmt.envs.terrain.generators import (
    BORDER,
    FAMILIES,
    NUM_LEVELS,
    TILE_SIZE,
    params_for,
    sample_grid,
)

DEFAULT_OUT = "eval/viz/terrain_overview.png"


def _grid_points(n: int = 161) -> tuple:
    """tile 局部坐标的采样网格（n×n，覆盖整个 tile 含留白）。"""
    half = TILE_SIZE / 2.0
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    return np.meshgrid(xs, ys)


def plot_families(families: Sequence[str], levels: Sequence[int],
                  out_png: str) -> None:
    """绘制 (族 × 级) 的高度场总览图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = _grid_points(161)
    n_rows, n_cols = len(families), len(levels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    # 全图统一色标范围，便于横向比较难度
    zmax = 0.0
    fields = {}
    for f in families:
        for lv in levels:
            _, _, z = sample_grid(params_for(f, lv), 161)
            fields[(f, lv)] = z
            zmax = max(zmax, float(np.abs(z).max()))

    for i, f in enumerate(families):
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            z = fields[(f, lv)]
            im = ax.imshow(z, origin="lower", cmap="terrain",
                           vmin=-zmax, vmax=zmax,
                           extent=[-TILE_SIZE / 2, TILE_SIZE / 2,
                                   -TILE_SIZE / 2, TILE_SIZE / 2])
            # 内部特征区边界（留白带）—— 取自 A19 的 border，勿硬编码
            inner = TILE_SIZE / 2.0 - BORDER
            ax.plot([-inner, inner, inner, -inner, -inner],
                    [-inner, -inner, inner, inner, -inner],
                    "k--", lw=0.5, alpha=0.4)
            if i == 0:
                ax.set_title(f"L{lv}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("PGMT 地形：五族 × 难度级（高度场，虚线 = 内部特征区/留白边界）")
    fig.colorbar(im, ax=axes, shrink=0.6, label="height (m)")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {out_png}")
    print(f"[i] 高度范围: ±{zmax:.3f} m")


def plot_elevation_maps(family: str, levels: Sequence[int],
                        out_png: str) -> None:
    """特征更清晰的一张：把高程图（策略实际观测到的 21×21）画出来。

    与 `plot_families` 的区别：这里画的是 **A8 规格的 21×21 高程图**，
    也就是 `TerrainGlimpseEncoder` 真正看到的东西；上者画的是解析高度场。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pgmt.envs.terrain.elevation_map import cell_centers, sample_elevation

    c = cell_centers(21)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.0 * len(levels), 3.4),
                             squeeze=False)
    for j, lv in enumerate(levels):
        p = params_for(family, lv)
        M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
        ax = axes[0][j]
        im = ax.imshow(M, origin="lower", cmap="terrain")
        ax.set_title(f"{family} L{lv}\nmax={M.max():.3f} m", fontsize=9)
        ax.set_xticks([0, 10, 20])
        ax.set_xticklabels([f"{c[0]:.1f}", "0", f"{c[-1]:.1f}"], fontsize=7)
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels([f"{c[0]:.1f}", "0", f"{c[-1]:.1f}"], fontsize=7)
        ax.set_xlabel("x (m)", fontsize=8)
        if j == 0:
            ax.set_ylabel("y (m)", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"高程图 M_t（21×21，机器人系 2m×2m）—— {family}")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {out_png}")


def dump_elevation_map(family: str, level: int) -> None:
    """把高程图数值打到 stdout（无 matplotlib 时也能核对）。"""
    from pgmt.envs.terrain.elevation_map import elevation_stats, sample_elevation

    p = params_for(family, level)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    s = elevation_stats(M)
    print(f"=== {family} L{level} 的高程图 M_t（21×21，单位 m）===")
    print(f"    min={s['min']:.4f} max={s['max']:.4f} "
          f"mean={s['mean']:.4f} std={s['std']:.4f}")
    with np.printoptions(precision=3, suppress=True, linewidth=200):
        print(M)


def main():
    ap = argparse.ArgumentParser(description="地形检查图")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--levels", nargs="+", type=int,
                    default=list(range(NUM_LEVELS)))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--maps-family", default=None, choices=list(FAMILIES),
                    help="额外输出该族的高程图（策略实际观测量）")
    ap.add_argument("--dump", nargs=2, metavar=("FAMILY", "LEVEL"),
                    help="把高程图数值打到 stdout，例如 --dump stairs 9")
    args = ap.parse_args()

    if args.dump:
        dump_elevation_map(args.dump[0], int(args.dump[1]))
        return

    plot_families(args.families, args.levels, args.out)
    if args.maps_family:
        maps_out = args.out.replace(".png", "_maps.png")
        plot_elevation_maps(args.maps_family, args.levels, maps_out)


if __name__ == "__main__":
    main()
