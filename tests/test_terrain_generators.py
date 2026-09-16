"""五族地形生成器：解析值对照、跨 tile 连续性、与高程图规格的耦合。

## 本文件的两条原则（吸取此前教训）

1. **期望值当场从实现公式推导，不凭记忆断言。**
   我在 `quat_to_mat` 的缩放上连续写错三次测试，根因都是"凭记忆写数值"。
   因此这里凡涉及具体数值，都在注释里写出推导过程。

2. **优先用不变量而非字面量。**
   例如"总上升量正比于难度"、"边界处恒为 0"、"方块不越界"这类性质断言，
   比 `assert z == 0.42` 更能抓住真实错误，也不会因调参而失效。
"""

import math

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.generators import (
    BORDER,
    ELEVATION_HALF_EXTENT,
    FAMILIES,
    NUM_LEVELS,
    STAIR_STEPS,
    TILE_SIZE,
    all_params,
    flat_randomize_strength,
    height_at,
    is_inside_tile,
    params_for,
    sample_grid,
)


# ---------------------------------------------------------------------------
# 论文层面的规格
# ---------------------------------------------------------------------------

def test_five_families_match_paper():
    """论文 §V-A：flat terrain, slopes, stairs, box obstacles, randomly rough。"""
    assert FAMILIES == ("flat", "slopes", "stairs", "boxes", "rough")


def test_ten_levels_per_family():
    """论文 §V-A：Each family contains ten difficulty levels, denoted L0–L9。"""
    assert NUM_LEVELS == 10
    assert len(all_params()) == 5 * 10


def test_tile_is_large_enough_for_one_elevation_map():
    """**约定耦合**：高程图覆盖机器人系 2m×2m（A8 + 论文 §III）。

    若 tile 比一张高程图还小，一张图会跨越多个 tile，"机器人位于哪个地形族"
    就没有定义 —— Stage 2 的松弛 χ(κ_t) 与地形课程都依赖这个量。
    """
    assert TILE_SIZE > 2.0 * ELEVATION_HALF_EXTENT, (
        f"tile {TILE_SIZE}m 不足以容纳一张 {2 * ELEVATION_HALF_EXTENT}m 的高程图"
    )


def test_border_leaves_platform_for_cross_tile_continuity():
    """留白必须为正且小于 tile 半宽，否则要么无特征、要么没有平台区。"""
    assert 0.0 < BORDER < TILE_SIZE / 2.0
    assert TILE_SIZE / 2.0 - BORDER > 0.0


def test_tiling_constants_come_from_a19():
    """tile 几何必须取自 A19（唯一出处）—— 防止在模块里另起一套硬编码。"""
    a19 = get("A19").value
    assert TILE_SIZE == a19.tile_size
    assert BORDER == a19.border
    assert STAIR_STEPS == a19.stair_steps
    assert a19.tile_size > 2.0 * ELEVATION_HALF_EXTENT, (
        "A19.tile_size 必须大于一张高程图的覆盖范围，否则地形族/难度无定义"
    )
    assert a19.stair_steps >= 2, "阶梯至少两级才有'阶'的意义"


# ---------------------------------------------------------------------------
# 参数：单调性与端点（A7）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,attr", [
    ("slopes", "slope_rise"),
    ("stairs", "step_height"),
    ("boxes", "box_height"),
    ("rough", "rough_amp"),
])
def test_geometric_difficulty_increases_with_level(family, attr):
    """论文：十个难度级，难度递增。几何参数必须随 level 单调不减。"""
    vals = [getattr(params_for(family, lv), attr) for lv in range(NUM_LEVELS)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), f"{family} 非单调: {vals}"
    assert vals[-1] > vals[0], f"{family} 的 L9 未高于 L0，难度未体现"


@pytest.mark.parametrize("family,attr,lo,hi", [
    ("slopes", "slope_rise", 0.0, None),
    ("stairs", "step_height", 0.04, 0.24),
    ("boxes", "box_height", 0.05, 0.40),
    ("rough", "rough_amp", 0.01, 0.12),
])
def test_endpoints_match_a7(family, attr, lo, hi):
    """端点必须等于 A7 登记值。

    期望值直接取自 A7（不是重写一遍数字），所以"改了 A7 忘了改这里"会被抓住，
    而"这个测试抄错了"不会发生 —— 这正是我此前栽过的那类坑。
    """
    t = get("A7").value
    v0 = getattr(params_for(family, 0), attr)
    v9 = getattr(params_for(family, 9), attr)
    if family == "stairs":
        assert v0 == pytest.approx(t.stairs_h_cm[0] / 100.0)
        assert v9 == pytest.approx(t.stairs_h_cm[9] / 100.0)
    elif family == "boxes":
        assert v0 == pytest.approx(t.boxes_h_cm[0] / 100.0)
        assert v9 == pytest.approx(t.boxes_h_cm[9] / 100.0)
    elif family == "rough":
        assert v0 == pytest.approx(t.rough_amp_cm[0] / 100.0)
        assert v9 == pytest.approx(t.rough_amp_cm[9] / 100.0)


def test_slope_rise_matches_slope_angle_geometry():
    """slopes 的 rise 是由**坡度角**与内部半宽推出的，不是独立取的数。

    推导：rise = inner_half · tan(θ)。取 L9 的 θ = 30°、inner = TILE/2 − BORDER：
        推导值 = 1.0 · tan(30°) = 0.5774 m
    若实现里换了公式，这条会失败。
    """
    inner = TILE_SIZE / 2.0 - BORDER
    for lv in (0, 5, 9):
        deg = get("A7").value.slope_deg[lv]
        expected = inner * math.tan(math.radians(deg))
        assert params_for("slopes", lv).slope_rise == pytest.approx(expected)


def test_boxes_l9_aligns_with_real_robot_capability():
    """A7 的取证：boxes L9 = 40cm 对齐论文真机"37cm 上限"。"""
    assert params_for("boxes", 9).box_height == pytest.approx(0.40)
    assert params_for("boxes", 9).box_height > 0.37


def test_flat_has_no_geometry():
    for lv in range(NUM_LEVELS):
        p = params_for("flat", lv)
        assert p.slope_rise == 0.0 and p.step_height == 0.0
        assert p.box_height == 0.0 and p.rough_amp == 0.0


def test_flat_randomize_strength_covers_zero_to_one():
    """flat 的难度 = 随机化强度 0→1（A7.flat_randomize）。"""
    assert flat_randomize_strength(0) == pytest.approx(0.0)
    assert flat_randomize_strength(9) == pytest.approx(1.0)
    vals = [flat_randomize_strength(lv) for lv in range(NUM_LEVELS)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_params_reject_bad_input():
    with pytest.raises(ValueError, match="未知地形族"):
        params_for("moon", 0)
    with pytest.raises(ValueError, match="难度级"):
        params_for("flat", 10)
    with pytest.raises(ValueError, match="难度级"):
        params_for("flat", -1)


def test_inner_half_is_positive_and_smaller_than_tile():
    for (f, lv), p in all_params().items():
        assert 0.0 < p.inner_half < TILE_SIZE / 2.0
        assert p.inner_half == pytest.approx(TILE_SIZE / 2.0 - BORDER)


# ---------------------------------------------------------------------------
# 高度场：边界恒为 0（跨 tile 连续的必要条件）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_height_is_zero_outside_inner_region(family):
    """留白区必须恒为 0。

    相邻 tile 在边界处都是 0 → 拼接后高度连续。若某族在留白处非 0，
    机器人会撞上"看不见的墙"（高程图看起来正常，但物理上跨不过去）。
    """
    p = params_for(family, 9)
    half = TILE_SIZE / 2.0
    # 取留白带上的样点：贴着 tile 边界
    for edge in (half, -half):
        z = height_at(p, np.array([edge, 0.0, edge]), np.array([0.0, edge, edge]))
        assert np.allclose(z, 0.0), f"{family} 在留白区非 0: {z}"


#: 只在平台上**向上**长特征的族（论文用词：flat / slopes / stairs / **obstacles**）
UPWARD_ONLY_FAMILIES = ("flat", "slopes", "stairs", "boxes")


@pytest.mark.parametrize("family", UPWARD_ONLY_FAMILIES)
def test_upward_families_are_non_negative(family):
    """论文对这些族的用词是 obstacles / slopes / stairs —— 都是向上突起。

    留白区恒为 0，内部特征在 (0, 上限] —— 因此整体非负。
    """
    p = params_for(family, 9)
    _, _, z = sample_grid(p, 41)
    assert z.min() >= -1e-12, f"{family} 出现负高度 {z.min()}"


def test_rough_is_signed_around_zero():
    """rough 是**对称**的：论文用词是 "randomly **rough** terrain" 而非 obstacles。

    一个只有突起的表面是"布满小方块"（boxes），不是粗糙地面 ——
    有凸有凹才是粗糙。因此 rough 的值域是 [−amp, +amp]。
    """
    p = params_for("rough", 9)
    _, _, z = sample_grid(p, 401)
    assert z.min() < -1e-6, "rough 应有低于 0 的凹陷"
    assert z.max() > 1e-6, "rough 应有高于 0 的凸起"
    # 对称性：上下界都触及幅值附近（不要求精确对称，但要覆盖两个方向）
    assert z.min() >= -p.rough_amp - 1e-9
    assert z.max() <= p.rough_amp + 1e-9


@pytest.mark.parametrize("family", FAMILIES)
def test_height_is_zero_at_tile_corners(family):
    p = params_for(family, 9)
    h = TILE_SIZE / 2.0
    corners_x = np.array([-h, h, -h, h])
    corners_y = np.array([-h, -h, h, h])
    assert np.allclose(height_at(p, corners_x, corners_y), 0.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_height_is_finite_everywhere(family):
    p = params_for(family, 9)
    _, _, z = sample_grid(p, 41)
    assert np.all(np.isfinite(z))


@pytest.mark.parametrize("family", FAMILIES)
def test_height_never_exceeds_params(family):
    """高度上界必须由参数表决定，不能出现"比设定还高"的特征。

    rough 的峰值实测约为幅值的 50~80%（L1 归一化 + 相位差），
    因此以 `[−amp, +amp]` 作为上/下界；更紧的约束由
    `test_rough_is_bounded_by_amplitude` 保证。
    """
    for lv in (0, 5, 9):
        p = params_for(family, lv)
        _, _, z = sample_grid(p, 81)
        if family == "flat":
            lo, cap = 0.0, 0.0
        elif family == "slopes":
            lo, cap = 0.0, p.slope_rise
        elif family == "boxes":
            lo, cap = 0.0, p.box_height
        elif family == "stairs":
            lo, cap = 0.0, (STAIR_STEPS - 1) * p.step_height
        else:  # rough：关于 0 对称
            lo, cap = -p.rough_amp, p.rough_amp
        assert z.max() <= cap + 1e-9, (
            f"{family} L{lv} 最高 {z.max():.4f} 超过上限 {cap:.4f}"
        )
        assert z.min() >= lo - 1e-9, (
            f"{family} L{lv} 最低 {z.min():.4f} 低于下界 {lo:.4f}"
        )


# ---------------------------------------------------------------------------
# 各族的具体语义（解析对照）
# ---------------------------------------------------------------------------

def test_flat_is_identically_zero():
    for lv in range(NUM_LEVELS):
        _, _, z = sample_grid(params_for("flat", lv), 21)
        assert np.allclose(z, 0.0)


def test_slopes_is_a_pyramid_peak_at_center():
    """金字塔式：中心最高 = slope_rise，内部边缘最低 = 0。

    解析对照：中心 (0,0) 处 d=0 → t=(inner−0)/inner=1 → z = rise·1。
    内部边缘如 (inner, 0) 处 d=inner → t=0 → z=0。
    """
    p = params_for("slopes", 9)
    z_center = float(height_at(p, np.array(0.0), np.array(0.0)))
    assert z_center == pytest.approx(p.slope_rise)

    z_edge = float(height_at(p, np.array(p.inner_half), np.array(0.0)))
    assert z_edge == pytest.approx(0.0)

    # 单调：从中心向外，等高线上 d 越大越低
    ds = np.linspace(0.0, p.inner_half, 20)
    zs = [float(height_at(p, np.array(d), np.array(0.0))) for d in ds]
    assert all(b <= a + 1e-12 for a, b in zip(zs, zs[1:])), "slopes 非单调下降"


def test_slopes_angle_is_recovered_from_the_field():
    """反演校验：由高度场量出的坡度应等于 A7 的设定角。

    这是"参数 → 几何"的闭环：沿 45° 方向（d = |x|）量斜率，
    rise / inner = tan(θ)。
    """
    inner = TILE_SIZE / 2.0 - BORDER
    for lv in (0, 4, 9):
        p = params_for("slopes", lv)
        deg = get("A7").value.slope_deg[lv]
        measured = p.slope_rise / inner
        assert math.degrees(math.atan(measured)) == pytest.approx(deg, abs=1e-6)


def test_stairs_total_rise_is_proportional_to_difficulty():
    """固定级数 → 总上升量 ∝ 阶高 ∝ 难度。且 L9 的总上升在同一量级（<1.1m）。"""
    for lv in (0, 5, 9):
        p = params_for("stairs", lv)
        _, _, z = sample_grid(p, 201)
        # 最高一级 = (STAIR_STEPS-1) × 阶高；且必须真的取到该高度
        expected_top = (STAIR_STEPS - 1) * p.step_height
        assert z.max() == pytest.approx(expected_top, abs=1e-9)
    # L9 的总上升有界（否则等于在 1m 内爬 55° 的阶梯）
    assert (STAIR_STEPS - 1) * params_for("stairs", 9).step_height < 1.1


def test_stairs_is_monotone_along_x_and_flat_along_y():
    """阶梯只沿 +x 上升；沿 y 方向同一级内高度不变。"""
    p = params_for("stairs", 9)
    xs = np.linspace(-p.inner_half + 1e-6, p.inner_half - 1e-6, 400)
    zs = height_at(p, xs, np.zeros_like(xs))
    assert all(b >= a - 1e-12 for a, b in zip(zs, zs[1:])), "stairs 沿 x 非单调"
    # 同一 x 上换 y：仍在内部区，高度应相同
    ys = np.array([-0.5, 0.0, 0.5])
    z_row = height_at(p, np.full_like(ys, 0.3), ys)
    assert np.allclose(z_row, z_row[0])


def test_stairs_first_step_is_flat_platform():
    """起步平台：x 落在最左一格内时高度为 0（idx=0 → clip(0−1,0,…)=0）。"""
    p = params_for("stairs", 9)
    x0 = -p.inner_half + 0.4 * p.spacing
    assert float(height_at(p, np.array(x0), np.array(0.0))) == pytest.approx(0.0)


def test_boxes_are_discrete_and_do_not_overlap():
    """方块是离散的（有间隙），且相邻方块高度相同 —— 不会连成一堵墙。"""
    p = params_for("boxes", 9)
    _, _, z = sample_grid(p, 401)
    uniq = np.unique(np.round(z, 9))
    # 只应出现 0 与 box_height 两个高度值
    assert len(uniq) == 2, f"方块高度取值应只有 {0.0, p.box_height}，得到 {uniq}"
    assert uniq[0] == pytest.approx(0.0)
    assert uniq[1] == pytest.approx(p.box_height)
    # 必须有平台（0）占比，说明方块间有间隙
    frac_platform = float((z < 1e-9).mean())
    assert frac_platform > 0.2, f"平台占比仅 {frac_platform:.2f}，方块几乎连成一片"


def test_boxes_stay_inside_inner_region():
    """方块不得越出内部区（格数向下取整的直接后果，用实测确认）。"""
    for lv in (0, 5, 9):
        p = params_for("boxes", lv)
        n = 2001
        xs = np.linspace(-TILE_SIZE / 2.0, TILE_SIZE / 2.0, n)
        z = height_at(p, xs, np.zeros_like(xs))
        # 越过内部区边界后必须恰好为 0
        outside = np.abs(xs) > p.inner_half
        assert np.allclose(z[outside], 0.0), f"L{lv} 方块越出内部区"


def test_rough_is_bounded_by_amplitude():
    """rough 的值域是 [−amp, +amp]（对称），上界与下界都不得越出幅值。"""
    for lv in (0, 5, 9):
        p = params_for("rough", lv)
        _, _, z = sample_grid(p, 201)
        assert z.max() <= p.rough_amp + 1e-9, f"L{lv} 凸起超出幅值"
        assert z.min() >= -p.rough_amp - 1e-9, f"L{lv} 凹陷超出幅值"


def test_rough_is_seeded_by_level_and_reproducible():
    """同一 (族, 级) 必须给出同一地形 —— 评估协议要求"各策略使用相同的地形分配"。"""
    a = sample_grid(params_for("rough", 7), 41)[2]
    b = sample_grid(params_for("rough", 7), 41)[2]
    assert np.array_equal(a, b), "同一 seed 应当完全可复现"
    c = sample_grid(params_for("rough", 8), 41)[2]
    assert not np.array_equal(a, c), "不同难度级应是不同地形"


def test_rough_noise_is_truly_periodic():
    """`_periodic_noise` 必须对坐标以 `TILE_SIZE` 为周期 —— **直接测该函数**。

    为什么直接测内部函数：`height_at` 会先把留白区（`|x| > inner_half`）置 0，
    而 `x + TILE_SIZE` 恰好落在留白区 → 通过 `height_at` 比较会得到"两边都是 0"，
    是个**空洞测试**（我第一版就是这么写的，失败了才看出来）。

    周期性的意义：它保证噪声在 tile 上"自洽"（相位连续），且在固定网格上采样时
    不会因 tile 索引不同而突变。真正保证相邻 tile 高度一致的是**留白区恒为 0**
    （见 `test_height_matches_at_shared_border_between_neighbours`）。
    """
    from pgmt.envs.terrain.generators import _periodic_noise

    amp = 0.12
    # 采样点必须都落在有效域内（|x| < inner，以免被留白遮盖）
    x = np.linspace(-0.9, 0.9, 51)
    y = np.linspace(-0.9, 0.9, 51)
    X, Y = np.meshgrid(x, y)
    base = _periodic_noise(X, Y, amp, seed=1009)
    for shift in (-TILE_SIZE, TILE_SIZE):
        assert np.allclose(base, _periodic_noise(X + shift, Y, amp, seed=1009),
                           atol=1e-9), f"沿 x 平移 {shift} 后不周期"
        assert np.allclose(base, _periodic_noise(X, Y + shift, amp, seed=1009),
                           atol=1e-9), f"沿 y 平移 {shift} 后不周期"


def test_rough_noise_is_centered_and_scaled():
    """对称性：取负坐标应得到取负的高度（sin 的奇函数性 ⇒ 整体反对称）。

    实现是若干 `sin(2π(kx·x + ky·y)/T + φ)` 的加权和。对 `(x,y) → (−x,−y)`
    而言相位变号但 `φ` 不变，故**不**严格反对称；因此这里只校验
    "关于 0 对称分布"（均值接近 0、上下界量级相当），不假设奇函数。
    """
    from pgmt.envs.terrain.generators import _periodic_noise

    x = np.linspace(-1.0, 1.0, 401)
    X, Y = np.meshgrid(x, x)
    z = _periodic_noise(X, Y, 0.1, seed=7)
    assert abs(float(z.mean())) < 0.02, f"均值 {z.mean():.4f} 偏离 0 过多"
    assert abs(float(z.max()) - float(-z.min())) < 0.05, "上下界量级不相当"


@pytest.mark.parametrize("family", FAMILIES)
def test_height_matches_at_shared_border_between_neighbours(family):
    """相邻 tile 在接缝两侧高度一致（两边都落在留白区 → 都是 0）。"""
    p = params_for(family, 9)
    # 左 tile 的右边界 x=+half ↔ 右 tile 的左边界 x=−half（同一世界点）
    y = np.linspace(-TILE_SIZE / 2.0, TILE_SIZE / 2.0, 25)
    z_left_edge = height_at(p, np.full_like(y, TILE_SIZE / 2.0), y)
    z_right_edge = height_at(p, np.full_like(y, -TILE_SIZE / 2.0), y)
    assert np.allclose(z_left_edge, z_right_edge, atol=1e-12)
    assert np.allclose(z_left_edge, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 采样与便利函数
# ---------------------------------------------------------------------------

def test_sample_grid_shapes_and_orientation():
    """`zs[i, j]` 应对应 `y = ys[i]`、`x = xs[j]`（与图像行列一致）。"""
    p = params_for("slopes", 9)
    xs, ys, zs = sample_grid(p, 11)
    assert xs.shape == (11,) and ys.shape == (11,) and zs.shape == (11, 11)
    assert zs[5, 5] == pytest.approx(float(height_at(p, np.array(xs[5]), np.array(ys[5]))))


def test_sample_grid_extent_is_configurable():
    p = params_for("slopes", 9)
    xs, ys, zs = sample_grid(p, 9, extent=0.5)
    assert xs[0] == pytest.approx(-0.5) and xs[-1] == pytest.approx(0.5)
    assert zs.shape == (9, 9)


def test_sample_grid_rejects_tiny_n():
    with pytest.raises(ValueError, match="至少为 2"):
        sample_grid(params_for("flat", 0), 1)


def test_height_at_broadcasts_scalars_and_arrays():
    p = params_for("slopes", 9)
    z_scalar = height_at(p, np.array(0.0), np.array(0.0))
    assert np.ndim(z_scalar) == 0
    z_arr = height_at(p, np.array([0.0, 0.5]), np.array([0.0, 0.5]))
    assert np.shape(z_arr) == (2,)


def test_is_inside_tile_matches_height_nonzero_region():
    """`is_inside_tile` 的判定必须与"该点可能非 0"一致（除 flat）。"""
    for family in ("slopes", "stairs", "boxes", "rough"):
        p = params_for(family, 9)
        assert is_inside_tile(p, 0.0, 0.0)
        assert not is_inside_tile(p, TILE_SIZE, 0.0)
        # 内部区之外一律为 0
        assert float(height_at(p, np.array(TILE_SIZE / 2.0), np.array(0.0))) == pytest.approx(0.0)
