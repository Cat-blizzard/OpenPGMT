"""假设注册表（Assumption Registry）——论文未公开超参的唯一出处。

论文 arXiv:2609.08511v2 未给出全部实现细节。凡需自行拍定的取值，
一律在此登记为编号假设（A1–A15），并附带理由。约定：
  - 所有模块从这里 import 取值，禁止在别处硬编码；
  - 训练启动时调用 `dump()` 写入运行日志，使每个 checkpoint 可追溯到具体假设值；
  - 最终报告以本文件为基准，与实际取值逐项对照并说明偏差。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 类型化取值
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlFreq:
    rate_hz: float = 50.0
    dt: float = 0.02  # s
    episode_s: float = 30.0

    @property
    def steps_per_episode(self) -> int:
        return int(self.episode_s / self.dt)  # 1500


@dataclass(frozen=True)
class FutureRef:
    """C^K：未来参考帧偏移 tau_k = 2^k - 1（帧），k = 0..K-1。"""

    K: int = 6
    offsets: tuple = (0, 1, 3, 7, 15, 31)

    def __post_init__(self):
        assert len(self.offsets) == self.K
        assert all(o == 2**k - 1 for k, o in enumerate(self.offsets))


@dataclass(frozen=True)
class ModelScale:
    token_dim: int = 256
    mhca_heads: int = 4
    activation: str = "elu"  # 策略/编码器 MLP 统一激活函数


@dataclass(frozen=True)
class MlpDims:
    actor: tuple = (512, 256, 128)
    critic: tuple = (512, 256, 128)


@dataclass(frozen=True)
class RoPECfg:
    dim: int = 64  # 每头 16（64 / 4 头）
    base: float = 10000.0


@dataclass(frozen=True)
class PPOCfg:
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 1e-3  # 递减
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class TerrainDifficulty:
    """A7：五族地形 L0→L9 难度参数（线性递增，端点见复现方案 §2）。"""

    slope_deg: tuple = (5.0, 7.78, 10.56, 13.33, 16.11, 18.89, 21.67, 24.44, 27.22, 30.0)
    stairs_h_cm: tuple = (4.0, 6.22, 8.44, 10.67, 12.89, 15.11, 17.33, 19.56, 21.78, 24.0)
    boxes_h_cm: tuple = (5.0, 8.89, 12.78, 16.67, 20.56, 24.44, 28.33, 32.22, 36.11, 40.0)
    rough_amp_cm: tuple = (1.0, 2.22, 3.44, 4.67, 5.89, 7.11, 8.33, 9.56, 10.78, 12.0)
    # flat：无几何，难度 = 随机化强度 0→1
    flat_randomize: tuple = (0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0)


@dataclass(frozen=True)
class ElevationNoise:
    """A8：高程图观测扰动。sigma 随难度线性增长；独立网格 dropout。"""

    sigma_min: float = 0.005  # m, L0
    sigma_max: float = 0.03  # m, L9
    dropout_prob_max: float = 0.1  # L9 时每格独立置 0 的概率
    map_size: int = 21
    map_res: float = 0.1  # m/格，2m×2m yaw 对齐


@dataclass(frozen=True)
class PrivilegedObs:
    """A9：critic 特权观测集（Isaac Gym 标准集）。"""

    items: tuple = (
        "base_lin_vel",
        "base_ang_vel",
        "foot_contact_states",
        "friction_coefficients",
        "terrain_height_at_feet",
        "base_mass_perturbation",
        "com_perturbation",
        "push_perturbation",
        "motor_strength_scale",
    )


@dataclass(frozen=True)
class FallPool:
    """A10：摔倒状态池。termination 时刻记录 [观测, 参考上下文]。"""

    capacity: int = 2048
    init_prob: float = 0.1  # 课程初始：以池初始化 episode 的比例
    prob_max: float = 0.5  # 随 episode 生存率渐进提高到该上限
    survival_window: int = 100  # 生存率滑动窗口（episode 数）


@dataclass(frozen=True)
class CompatibilityRules:
    """A11：behavior–terrain 兼容规则（粗特征分类器）。"""

    # 运动类别 → 禁止采样的 (地形族, 难度区间)
    excluded: tuple = (
        ("lying_prone", ("stairs", "boxes")),  # 躺/趴类不采样于 stairs/boxes 高难度
        ("inverted", ("stairs", "boxes")),  # 倒立类同上
        ("sitting", ("slopes",)),  # 坐姿类不采样于斜坡高难度
    )
    min_level_excluded: int = 5  # 仅 L>=5 的高难度生效


@dataclass(frozen=True)
class Relaxation:
    """A12：terrain-aware tracking relaxation。ẽ = [e − α·χ(κ)·τ(d)]₊"""

    alpha_default: float = 1.0  # α_{h,j}，作用于 lower-body 目标
    chi: Dict[str, float] = field(default_factory=lambda: {
        "flat": 0.0,  # 平地不松弛
        "slopes": 0.6,
        "stairs": 1.0,
        "boxes": 1.0,
        "rough": 0.6,
    })
    tau_saturation: float = 0.05  # τ(d) 线性增长，饱和于目标量级 5%


@dataclass(frozen=True)
class GlobalPosCorrection:
    """A13：v^r += clip(g(‖v^r‖)·λ_pos·e^p, ±v̄)。"""

    gate_v0: float = 0.0  # smoothstep 下界 (m/s)
    gate_v1: float = 0.5  # smoothstep 上界 (m/s)
    lambda_pos: float = 1.0
    clip_v: float = 1.0  # m/s


@dataclass(frozen=True)
class Retargeting:
    """A14：Mixamo(LAFAN1) → G1 重定向管线（沿 OmniH2O 思路）。"""

    skeleton_scale: bool = True
    joint_map: str = "mixamo_to_g1"  # 逐关节旋转映射表（data/retarget_lafan1.py）
    ref_vel_method: str = "finite_diff"
    contact_label: str = "foot_vel_thresh"
    foot_vel_thresh: float = 0.15  # m/s，低于此值判为接触帧


@dataclass(frozen=True)
class GlimpseCfg:
    """A15：Terrain-Glimpse Encoder 结构（论文只给 Ng=4、patch 5×5）。"""

    num_glimpses: int = 4  # 论文 Eq.7 明确 Ng=4
    patch_size: int = 5  # 论文 Eq.8 明确 5×5
    selector_hidden: tuple = (512, 256)  # MLP_φ 隐层，按 A4 风格
    token_hidden: tuple = (256,)  # MLP_ψ 隐层
    loc_extent: float = 1.0  # m：位置输出经 tanh 缩放到地图半宽，保证落图内


@dataclass(frozen=True)
class AdaptiveSamplingCfg:
    """A16：adaptive sampling 权重 = 1 + fail_boost × 失败次数（软加权，保留全覆盖）。"""

    fail_boost: float = 1.0


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumption:
    aid: str
    name: str
    value: Any
    rationale: str


ASSUMPTIONS: Dict[str, Assumption] = {
    "A1": Assumption("A1", "control_frequency", ControlFreq(),
                     "论文未给出控制频率；按 G1 惯例取 50 Hz，30 s episode = 1500 步"),
    "A2": Assumption("A2", "future_ref_frames", FutureRef(),
                     "论文只给偏移公式 τ = 2^k − 1 未给 K；取 K=6（最长 31 帧 ≈ 0.62 s 前瞻）"),
    "A3": Assumption("A3", "model_scale", ModelScale(),
                     "token 维度/头数按 OmniH2O/HOVER 规模"),
    "A4": Assumption("A4", "actor_critic_mlp", MlpDims(),
                     "同 A3；actor 与 critic 同规模"),
    "A5": Assumption("A5", "rope", RoPECfg(),
                     "RoPE 维度 64 = 4 头 × 16/头，常规设置"),
    "A6": Assumption("A6", "ppo", PPOCfg(),
                     "按 rsl-rl 惯例（legged 系默认超参）"),
    "A7": Assumption("A7", "terrain_difficulty", TerrainDifficulty(),
                     "论文未给难度数值；端点对齐真机 37 cm 上限（boxes L9=40 cm）"),
    "A8": Assumption("A8", "elevation_noise", ElevationNoise(),
                     "论文提到 observation corruption 未给细节"),
    "A9": Assumption("A9", "privileged_obs", PrivilegedObs(),
                     "论文未列出；取 Isaac Gym 标准特权观测集"),
    "A10": Assumption("A10", "fall_pool", FallPool(),
                     "沿 RGMT 思路：termination 时刻入池，生存率课程渐进提高池初始化比例"),
    "A11": Assumption("A11", "compatibility_rules", CompatibilityRules(),
                     "论文只说是粗规则；实现为运动特征分类器 + 族/难度排除表"),
    "A12": Assumption("A12", "relaxation", Relaxation(),
                     "论文只给形式；α=1、χ 按族、τ 线性+饱和（5% 量级）"),
    "A13": Assumption("A13", "global_pos_correction", GlobalPosCorrection(),
                     "论文只给形式；smoothstep 门控 + λ_pos=1 + clip 1 m/s"),
    "A14": Assumption("A14", "retargeting", Retargeting(),
                     "沿 OmniH2O 重定向思路（骨骼缩放 + 逐关节旋转映射）"),
    "A15": Assumption("A15", "glimpse", GlimpseCfg(),
                     "论文只给 Ng=4、patch 5×5；MLP 规模按 A4 风格，位置输出 tanh×1m 保证落在地图内"),
    "A16": Assumption("A16", "adaptive_sampling", AdaptiveSamplingCfg(),
                     "论文只说失败频次提高采样概率并保留全覆盖；取线性软加权"),
}


def get(aid: str) -> Assumption:
    """按编号取假设；不存在则报错（拼写保护）。"""
    if aid not in ASSUMPTIONS:
        raise KeyError(f"未知假设编号: {aid}（已登记: {sorted(ASSUMPTIONS)}）")
    return ASSUMPTIONS[aid]


def _json_safe(obj: Any) -> Any:
    """递归转成 JSON 可序列化结构（dataclass→dict, tuple→list）。"""
    if is_dataclass(obj):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (tuple, list)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj


def dump() -> Dict[str, Dict[str, Any]]:
    """序列化全部假设，供训练日志记录（JSON 安全）。"""
    return {aid: _json_safe(a) for aid, a in ASSUMPTIONS.items()}
