"""高程图：与 `glimpse_encoder` 的坐标约定一致、A8 扰动、相对高度语义。

## 这个文件最重要的两条测试

1. **`test_grid_matches_glimpse_samples`** —— 闭环验证。
   `TerrainGlimpseEncoder` 用 `grid_sample` 从高程图上采样 patch。若高程图的格心
   约定与它的 `(center+1)/2*(n−1)` 映射有任何偏差，**patch 会采错位置，但形状
   全对、不报任何错** —— 这是典型的静默 bug。这里把生成的高程图喂给真实的
   encoder，再用"高程图上该点的解析值"核对采样结果。

2. **`test_map_res_matches_glimpse_geometry`** —— A8 的 `map_res` 与 `map_size`
   不是独立参数。给定 `map_size=21`，格距必须等于 `2/(21−1)=0.1`，否则与
   glimpse 采样器的约定冲突。这条把"两个地方的数必须相等"变成断言。
"""

import math

import numpy as np
import pytest
import torch

from pgmt.cfg.assumptions import get
from pgmt.envs.terrain.elevation_map import (
    MAP_SIZE,
    cell_centers,
    corrupt_elevation,
    elevation_extent,
    elevation_stats,
    grid_resolution,
    noise_model_for,
    observe_elevation,
    sample_elevation,
)
from pgmt.envs.terrain.generators import FAMILIES, TILE_SIZE, params_for


# ---------------------------------------------------------------------------
# 与论文 / A8 的规格一致性
# ---------------------------------------------------------------------------

def test_map_size_is_21():
    """论文 §III：M_t ∈ R^{21×21}。"""
    assert MAP_SIZE == 21
    assert get("A8").value.map_size == 21


def test_map_res_matches_glimpse_geometry():
    """**A8 的 map_res 必须等于由 map_size 推出的格距** —— 两者不是独立参数。

    推导：`glimpse_encoder` 把 `[−1, +1] m` 映射到像素 `[0, n−1]`，
    故格距 `= 2/(n−1)`。n=21 → 0.1，正好等于 A8 的 `map_res=0.1`。
    若有人把 map_res 改成 0.05 而不同步 map_size，高程图与采样器就会错位。
    """
    a8 = get("A8").value
    derived = grid_resolution(a8.map_size)
    assert derived == pytest.approx(a8.map_res), (
        f"A8.map_res={a8.map_res} 与由 map_size={a8.map_size} 推出的格距 "
        f"{derived} 不一致 —— 两者必须同步"
    )
    assert grid_resolution(21) == pytest.approx(0.1)


def test_cell_centers_span_exactly_pm_one_metre():
    """n=21、res=0.1 时格心覆盖 ±1.0m（= glimpse 的采样边界）。"""
    c = cell_centers(21)
    assert c.shape == (21,)
    assert c[0] == pytest.approx(-1.0)
    assert c[-1] == pytest.approx(1.0)
    assert c[10] == pytest.approx(0.0), "中间格应在原点"
    assert np.allclose(np.diff(c), 0.1), "格距应均匀为 0.1"
    assert elevation_extent(21) == pytest.approx(1.0)


def test_cell_centers_match_glimpse_mapping():
    """逐格核对：`glimpse_encoder` 的 `(m+1)/2*(n−1)` 必须把格心映到其下标。"""
    c = cell_centers(21)
    n = 21
    px = (c + 1.0) / 2.0 * (n - 1)          # 与 _grid_from_centers 完全同一公式
    assert np.allclose(px, np.arange(n)), f"格心未映到整数下标: {px}"


def test_grid_resolution_rejects_bad_size():
    with pytest.raises(ValueError, match="至少为 2"):
        grid_resolution(1)


# ---------------------------------------------------------------------------
# ★ 闭环：高程图 → glimpse_encoder → 采样值核对
# ---------------------------------------------------------------------------

def test_grid_matches_glimpse_samples():
    """**闭环验证**：`TerrainGlimpseEncoder` 从高程图上采到的 patch 值，
    必须等于高程图在该位置的解析值。

    做法：构造一张线性高程图 `M[i,j] = a·x_j + b·y_i`（解析可算），
    让 encoder 在已知位置取 5×5 patch，再与解析值比较。

    为什么值得单独测：`grid_sample` 的 `align_corners` 约定、`(x,y)` 与
    `(row,col)` 的转置、以及归一化范围，任一处错了 patch 都会"形状正确但内容
    错位"，不报任何错。这是 M3 里最需要闭环验证的一处。
    """
    from pgmt.policy.glimpse_encoder import sample_patches

    n = 21
    c = cell_centers(n)
    a, b = 0.3, -0.7
    # M[i, j] = a·x_j + b·y_i（与高程图的 (行=y, 列=x) 约定一致）
    M = a * c[None, :] + b * c[:, None]
    M_t = torch.tensor(M[None], dtype=torch.float64)   # (1, n, n)

    # 在 (x0, y0) 处取一个 5×5 patch
    x0, y0 = 0.2, -0.3
    centers = torch.tensor([[[x0, y0]]], dtype=torch.float64)   # (1, 1, 2)
    patch = sample_patches(M_t, centers, patch_size=5)[0, 0].numpy()   # (5, 5)

    # 解析期望：(5×5) patch 的中心在 (x0, y0)，格距 0.1
    off = (np.arange(5) - 2) * 0.1
    exp_x = x0 + off[None, :]
    exp_y = y0 + off[:, None]
    expected = a * exp_x + b * exp_y
    assert np.allclose(patch, expected, atol=1e-9), (
        f"glimpse patch 与高程图解析值不符\n实际:\n{patch}\n期望:\n{expected}"
    )


def test_glimpse_reads_the_value_we_wrote():
    """更直白的一致性检查：在格心写一个脉冲，看 encoder 能否在原位读到。"""
    from pgmt.policy.glimpse_encoder import sample_patches

    n = 21
    c = cell_centers(n)
    M = np.zeros((n, n))
    i_y, j_x = 15, 6                      # 任意一个格
    M[i_y, j_x] = 1.0
    M_t = torch.tensor(M[None], dtype=torch.float64)
    centers = torch.tensor([[[c[j_x], c[i_y]]]], dtype=torch.float64)
    patch = sample_patches(M_t, centers, patch_size=5)[0, 0].numpy()
    assert patch[2, 2] == pytest.approx(1.0), "脉冲未落在 patch 中心"
    # 相邻格（±0.1m）应为 0
    assert patch[2, 1] == pytest.approx(0.0)
    assert patch[1, 2] == pytest.approx(0.0)


def test_glimpse_orientation_is_not_transposed():
    """**转置检查**：在 (x 大, y 小) 处放脉冲，encoder 读到的不应是 (x 小, y 大) 处。

    若行列约定搞反，对称的测试全都会过，只有这条能抓住。
    """
    from pgmt.policy.glimpse_encoder import sample_patches

    n = 21
    c = cell_centers(n)
    M = np.zeros((n, n))
    iy, jx = 4, 16
    M[iy, jx] = 1.0
    M_t = torch.tensor(M[None], dtype=torch.float64)
    # 正确位置：(x = c[jx] ≈ +0.6, y = c[iy] ≈ −0.6)
    ok = sample_patches(
        M_t, torch.tensor([[[c[jx], c[iy]]]], dtype=torch.float64), 3)[0, 0, 1, 1]
    assert float(ok) == pytest.approx(1.0)
    # 转置位置：(x = c[iy], y = c[jx]) 应为 0
    bad = sample_patches(
        M_t, torch.tensor([[[c[iy], c[jx]]]], dtype=torch.float64), 3)[0, 0, 1, 1]
    assert float(bad) == pytest.approx(0.0), "高程图被转置读取了"


# ---------------------------------------------------------------------------
# 相对高度语义
# ---------------------------------------------------------------------------

def test_flat_terrain_relative_to_ground_is_zero():
    """平地 + 机器人贴地 → 高程图全 0。"""
    p = params_for("flat", 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    assert M.shape == (21, 21)
    assert np.allclose(M, 0.0)


def test_relative_height_is_measured_from_robot_base():
    """高程图存的是 `h − z_robot`，不是绝对高度。

    验证：同一地形下把 `z_robot` 抬高 0.5m，整张图应整体下降 0.5m。
    """
    p = params_for("slopes", 5)
    M0 = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    M1 = sample_elevation(p, (0.0, 0.0), robot_z=0.5)
    assert np.allclose(M1, M0 - 0.5)


def test_slope_map_shows_a_pyramid_peak():
    """slopes 的高程图应从中心向外单调下降（金字塔），中心最高。"""
    p = params_for("slopes", 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    c = cell_centers(21)
    mid = 10
    assert M[mid, mid] == pytest.approx(p.slope_rise, abs=1e-9)
    # 沿 +x 远离中心：应单调不增
    row = M[mid, mid:]
    assert all(b <= a + 1e-12 for a, b in zip(row, row[1:])), "slopes 图非单调"


def test_map_follows_robot_position():
    """机器人移动到 tile 另一处，高程图内容应随之改变（不是固定图案）。"""
    p = params_for("slopes", 9)
    M_center = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    M_edge = sample_elevation(p, (0.8, 0.0), robot_z=0.0)
    assert not np.allclose(M_center, M_edge)


def _expected_grid(robot_xy, robot_z, yaw, map_size=21):
    """由 `sample_elevation` 的**契约**重算采样坐标（纯 numpy，独立于实现）。

    契约：格点 `(i, j)` 采样 `robot_xy + R(yaw)·(dx, dy)`，其中
    `dx[i,j] = c[j]`（随列变）、`dy[i,j] = c[i]`（随行变），
    返回 `h(x, y) − robot_z`。
    """
    from pgmt.envs.terrain.generators import height_at

    c = cell_centers(map_size)
    dx = np.outer(np.ones_like(c), c)
    dy = np.outer(c, np.ones_like(c))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    gx = robot_xy[0] + cos_y * dx - sin_y * dy
    gy = robot_xy[1] + sin_y * dx + cos_y * dy
    return gx, gy, height_at(params_for("stairs", 9), gx, gy) - robot_z


@pytest.mark.parametrize("yaw", [0.0, math.pi / 4, math.pi / 2, math.pi,
                                 3 * math.pi / 2, 2 * math.pi, -0.7])
def test_grid_follows_the_documented_contract(yaw):
    """**直接核对契约**：每个格点的采样坐标必须等于 `robot + R(yaw)·offset`。

    我第一版写的是"推导 M90 与 M0 的关系"（以为等于转置），结果推导错了、
    测试也错了。教训：**别测两个推导量之间的关系，直接测那个唯一的契约** ——
    采样坐标的构造公式。上面的 `_expected_grid` 是独立实现（纯 numpy），
    与被测代码只共享公式本身。
    """
    p = params_for("stairs", 9)
    robot_xy, robot_z = (-0.3, 0.4), 0.15
    M = sample_elevation(p, robot_xy, robot_z, yaw=yaw)
    _, _, expected = _expected_grid(robot_xy, robot_z, yaw)
    assert np.allclose(M, expected, atol=1e-9), f"yaw={yaw:.3f} 时采样坐标不符契约"


def test_yaw_90_rotates_the_gradient_direction():
    """yaw 旋转的**可观测后果**：stairs 的梯度方向随机器人朝向一起转。

    stairs 只沿 tile 的 +x 方向上升。机器人转 90° 后，其"前方"对准 tile 的 +y：

      yaw=0   → 同列各点 x 相同 → 每**列**常量
      yaw=90° → 同行各点 x 相同 → 每**行**常量

    **只检查严格落在内部区的格点**：`|c| = 1.0` 恰好等于 `inner_half`，是留白区
    的边界；若把边界点一起比，会被留白规则（非内部 → 0）污染，得到"第 0 列是 0"
    这种假失败。这与之前那个"shift 后落进留白区"的空洞测试是同一类瑕疵：
    **必须区分"一定在内部区"与"可能落在留白"的采样点**。
    """
    p = params_for("stairs", 9)
    inside = np.where(np.abs(cell_centers(21)) < p.inner_half - 1e-9)[0]
    assert inside.size > 10, "内部区格点太少，测试无意义"

    M0 = sample_elevation(p, (0.0, 0.0), 0.0, yaw=0.0)
    M90 = sample_elevation(p, (0.0, 0.0), 0.0, yaw=math.pi / 2)

    for j in inside:
        col = M0[inside, j]
        assert np.allclose(col, col[0], atol=1e-9), f"yaw=0 第 {j} 列在内部区非常量"
    for i in inside:
        row = M90[i, inside]
        assert np.allclose(row, row[0], atol=1e-9), f"yaw=90° 第 {i} 行在内部区非常量"

    assert not np.allclose(M0, M90), "旋转 90° 后高程图应改变"


def test_yaw_360_equals_no_rotation():
    """绕一整圈必须回到原样（旋转矩阵的周期性，顺带查符号手误）。"""
    p = params_for("boxes", 7)
    M0 = sample_elevation(p, (-0.3, 0.2), robot_z=0.1, yaw=0.0)
    M360 = sample_elevation(p, (-0.3, 0.2), robot_z=0.1, yaw=2 * math.pi)
    assert np.allclose(M0, M360, atol=1e-9)


def test_yaw_180_flips_the_along_x_gradient():
    """yaw=180°：机器人的"前方"对准 tile 的 −x → stairs 的梯度方向反转。

    推导（把 `c` 代进契约，不凭直觉）：
      - yaw=0   ：`gx = c[j]`。`c` 升序 → `gx` 升序 → stairs 高度沿列**非降**
      - yaw=180°：`gx = −c[j]`。`c` 升序 → `gx` 降序 → 高度沿列**非增**

    只比较内部区格点，避免留白区的 0 干扰。
    """
    p = params_for("stairs", 9)
    inside = np.where(np.abs(cell_centers(21)) < p.inner_half - 1e-9)[0]
    M0 = sample_elevation(p, (0.0, 0.0), 0.0, yaw=0.0)
    M180 = sample_elevation(p, (0.0, 0.0), 0.0, yaw=math.pi)
    row0 = M0[inside.size // 2, inside]
    row180 = M180[inside.size // 2, inside]
    assert all(b >= a - 1e-12 for a, b in zip(row0, row0[1:])), \
        f"yaw=0 沿列应非降（gx=c[j] 升序），得到 {row0}"
    assert all(b <= a + 1e-12 for a, b in zip(row180, row180[1:])), \
        f"yaw=180 沿列应非增（gx=−c[j] 降序），得到 {row180}"


@pytest.mark.parametrize("family", FAMILIES)
def test_map_is_finite_for_all_families(family):
    p = params_for(family, 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.8)
    assert M.shape == (21, 21)
    assert np.all(np.isfinite(M))


def test_map_outside_inner_region_reads_flat():
    """机器人贴近 tile 边缘时，图里 tile 之外的部分应为 0（留白平台）。"""
    p = params_for("slopes", 9)
    # 把机器人放到内部区之外（留白区）
    M = sample_elevation(p, (TILE_SIZE, 0.0), robot_z=0.0)
    assert np.allclose(M, 0.0), "留白区应为平地"


# ---------------------------------------------------------------------------
# A8 扰动
# ---------------------------------------------------------------------------

def test_noise_model_increases_with_level():
    """A8：sigma 与 dropout 概率均随难度线性增长。"""
    sigmas = [noise_model_for(lv).sigma for lv in range(10)]
    drops = [noise_model_for(lv).dropout_prob for lv in range(10)]
    assert sigmas[0] == pytest.approx(get("A8").value.sigma_min)
    assert sigmas[9] == pytest.approx(get("A8").value.sigma_max)
    assert drops[0] == pytest.approx(0.0)
    assert drops[9] == pytest.approx(get("A8").value.dropout_prob_max)
    assert all(b >= a for a, b in zip(sigmas, sigmas[1:]))
    assert all(b >= a for a, b in zip(drops, drops[1:]))


def test_noise_model_rejects_bad_level():
    with pytest.raises(ValueError, match="难度级"):
        noise_model_for(10)
    with pytest.raises(ValueError, match="难度级"):
        noise_model_for(-1)


def test_l0_has_noise_but_zero_dropout():
    """**纠正一个我自己写错的假设**：L0 并非"无扰动"。

    A8 的语义是 `sigma` 从 **`sigma_min = 0.005`** 起线性增长到 `sigma_max`，
    而 `dropout_prob` 从 **0** 起增长。所以 L0 有噪声、无整格丢弃。

    我第一版断言"L0 不应有扰动"，与 A8 矛盾；第二版又用
    "零格集合不变"来证明 dropout 没开 —— 那个检查**没有判别力**，
    因为高斯噪声本身就会让每个格都不再是 0，与 dropout 无关。

    正确做法是**把两件事分开测**：
      1. 本测试：dropout=0 时**不得出现恰好为 0 的格**（噪声不产生 0）
      2. `test_dropout_sets_cells_to_exactly_zero`：dropout>0 时必须出现 0 格
    """
    a8 = get("A8").value
    m0 = noise_model_for(0)
    assert m0.dropout_prob == pytest.approx(0.0), "L0 不应有 dropout"
    assert m0.sigma == pytest.approx(a8.sigma_min), "L0 的 σ 应为 sigma_min"
    assert m0.sigma > 0.0, "L0 仍有 sigma_min 噪声（不是零扰动）"

    p = params_for("slopes", 3)
    M = np.zeros((21, 21))          # 纯平地，干净图全 0
    out = corrupt_elevation(M, m0, np.random.default_rng(0))

    assert not np.allclose(out, M), "L0 应仍有高斯噪声"
    # dropout 关闭 ⇒ 不得有格被**恰好**置 0（连续高斯取到精确 0 的概率为 0）
    assert not np.any(out == 0.0), "dropout=0 时不应出现恰好为 0 的格"

    # 幅度量级：单样本（n=441）的极值对 σ 的倍数，4.5σ 是宽松上界
    # （441 个正态样本的期望最大 |z| ≈ 3.2σ，取 4.5σ 留足余量）
    assert np.abs(out).max() < 4.5 * m0.sigma, (
        f"噪声极值 {np.abs(out).max():.5f} 超出 σ 的合理倍数"
    )
    # **不在此处断言 std**：n=441 时样本标准差的 1σ 相对波动约 3.4%
    # （1/√(2n)），固定种子下可能偏离更多。σ 的准确值由
    # `test_noise_std_matches_sigma`（22050 个样本）负责，避免一处阻塞两处。


def test_dropout_sets_cells_to_exactly_zero():
    """dropout>0 时必须出现**恰好为 0** 的格，且比例接近设定概率。"""
    p = params_for("slopes", 9)
    M = np.full((21, 21), 5.0)
    model = noise_model_for(9)
    assert model.dropout_prob > 0.0
    out = corrupt_elevation(M, model, np.random.default_rng(0))

    zeros = int((out == 0.0).sum())
    assert zeros > 0, "dropout>0 却没有任何格被置 0"
    assert zeros < out.size, "不应整图被丢弃（dropout 是逐格独立）"

    # 置零比例应接近设定概率（二项分布，取 5σ 余量）
    n = out.size
    p_hat = zeros / n
    sigma = math.sqrt(model.dropout_prob * (1 - model.dropout_prob) / n)
    assert abs(p_hat - model.dropout_prob) < 5 * sigma, (
        f"置零比例 {p_hat:.4f} 偏离设定 {model.dropout_prob:.4f}"
    )
    # 未被丢弃的格应是 5 + 小噪声（而非被置 0 或被改动很多）
    kept = out[out != 0.0]
    assert np.all(np.abs(kept - 5.0) < 5 * model.sigma)


def test_zero_sigma_zero_dropout_is_exactly_identity():
    """σ=0 且 dropout=0 时必须是严格恒等 —— 这才是"无扰动"的确切定义。"""
    from pgmt.envs.terrain.elevation_map import ElevationNoiseModel

    p = params_for("slopes", 3)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    out = corrupt_elevation(M, ElevationNoiseModel(0.0, 0.0),
                            np.random.default_rng(0))
    assert np.array_equal(out, M)


def test_corrupt_is_reproducible_with_same_seed():
    """评估协议要求各策略用**完全相同**的观测扰动 → 同 seed 必须逐位一致。"""
    p = params_for("slopes", 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    a = corrupt_elevation(M, noise_model_for(9), np.random.default_rng(42))
    b = corrupt_elevation(M, noise_model_for(9), np.random.default_rng(42))
    assert np.array_equal(a, b)
    c = corrupt_elevation(M, noise_model_for(9), np.random.default_rng(43))
    assert not np.array_equal(a, c)


def test_corrupt_does_not_modify_input():
    p = params_for("slopes", 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    before = M.copy()
    corrupt_elevation(M, noise_model_for(9), np.random.default_rng(0))
    assert np.array_equal(M, before), "corrupt_elevation 不应原地修改输入"


def test_noise_std_matches_sigma():
    """噪声的标准差应接近 A8 的 sigma（大样本）。"""
    p = params_for("slopes", 9)
    M = np.zeros((21, 21))
    model = noise_model_for(9)
    # 用一颗只有噪声、无 dropout 的模型单独验 sigma
    from pgmt.envs.terrain.elevation_map import ElevationNoiseModel

    only_noise = ElevationNoiseModel(sigma=model.sigma, dropout_prob=0.0)
    rng = np.random.default_rng(0)
    samples = np.concatenate([
        corrupt_elevation(M, only_noise, rng).ravel() for _ in range(50)
    ])
    assert samples.std() == pytest.approx(model.sigma, rel=0.05)


def test_observe_elevation_clean_path_when_rng_is_none():
    """`rng=None` 必须走干净路径（不加噪）—— 是否加噪是显式的，不靠默认值。"""
    p = params_for("slopes", 9)
    clean = observe_elevation(p, (0.0, 0.0), 0.0, level=9, rng=None)
    assert np.allclose(clean, sample_elevation(p, (0.0, 0.0), 0.0))

    noisy = observe_elevation(p, (0.0, 0.0), 0.0, level=9,
                              rng=np.random.default_rng(0))
    assert not np.allclose(noisy, clean), "给了 rng 却没有扰动"


def test_elevation_stats_reports_range():
    p = params_for("slopes", 9)
    M = sample_elevation(p, (0.0, 0.0), robot_z=0.0)
    s = elevation_stats(M)
    assert set(s) == {"min", "max", "mean", "std", "range"}
    assert s["range"] == pytest.approx(M.max() - M.min())
    assert s["max"] == pytest.approx(p.slope_rise, abs=1e-9)
