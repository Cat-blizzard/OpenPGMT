"""假设注册表（Assumption Registry）——论文未公开超参的唯一出处。

论文 arXiv:2609.08511v2 未给出全部实现细节。凡需自行拍定的取值，
一律在此登记为编号假设（A1–A18），并附带理由。约定：
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


@dataclass(frozen=True)
class DataFilterCfg:
    """A17：训练数据过滤（M1.5c ground 类裁定）——**结论待复核**。

    原裁定理由：ground 类（躺地/翻滚）分解退化（38–116 cm），G1 无脊柱
    且躺地时欧拉分解退化，参考不可用；从训练集排除，摔倒池改用
    fall/push 类（28 cm）。

    ⚠️ 复核状态：该 38–116 cm 观测来自 lafan1_g1_v1_snapshot（v1 版数据）。
    当前 data/processed/quality_report.csv 中 ground 五个序列为
    14.45 / 15.17 / 15.71 / 15.74 / 15.92 cm（均值 15.40，最差但无病态），
    排除理由在当前数据上不成立。注意 eval_retarget 的 FK 误差与 ik_refine
    的优化目标高度重叠（13/18 关键点相同），且只用骨盆平移对齐，属"自证"
    指标 —— 因此**不能仅凭该数字就恢复**，需先用留出关键点 + 绝对位置/
    根朝向误差的独立口径重测。

    在重测完成前保留本过滤行为（保守）；重测通过后应清空
    excluded_prefixes —— 论文的 fall recovery 与 recovery curriculum 正
    需要躺地类动作，排除它们使摔倒池只能依赖 fall/push 类。
    """

    excluded_prefixes: tuple = ("ground",)


@dataclass(frozen=True)
class RewardImpl:
    """A18：奖励核函数与松弛细节（论文 Table I 给了权重，但几处形式未写出）。

    论文明确的（照搬在 `pgmt/rewards/spec.py`，不是假设）：
      - 分组结构 Eq.5 / Eq.9 与全部逐项权重（Table I）
      - 松弛形式 Eq.10 `ẽ = [ e − α·χ(κ)·τ_{mh}(d) ]₊`
      - χ(κ) 只在 slopes/stairs/boxes 激活；TA 只作用于 lower body 的
        link position / link orientation / joint position

    论文**未**写出的（本假设负责拍定）：
      - **指数跟踪奖励的核函数**：论文只说 ẽ "replaces e in the exponential
        tracking reward"，未写形式，且用定冠词 "**the** exponential tracking
        reward" —— 即引用该领域既有约定而非自定义。
        取 **`exp(−e²/σ)`（高斯式，误差的平方）**，依据是 PGMT 明示继承的
        tracking 实现：OmniH2O 论文奖励表把 body position 项写作
        `exp(−0.5‖p−p̂‖²₂)`，其配置注释亦明写
        `tracking reward : exp(-error^2/sigma)`。
        （我最初按"通用形式"写成线性核 `exp(−e/scale)`，是错的，已更正。）
      - **各跟踪目标的 σ 取值**：论文未给 → 见 `pgmt/rewards/spec.py` 的
        `SIGMAS` 表，量级与"上/下半身分离"的结构参照 OmniH2O
        （上半身位置 σ 明显小于下半身，对应论文 §IV "upper-body 保姿态保真 /
        lower-body 让位给平衡与接触"）。σ 量纲是**误差平方**，
        与线性核的"误差尺度"不可直接换算。
      - **τ_{mh}(d) 的斜率与饱和值**：论文只说线性增长并饱和 → 以
        `saturation_value`（单位同**误差**，不是误差平方）表示饱和量；
        A12.tau_saturation=0.05 即"饱和值 ≈ 各目标 5% 量级"。
        注意 Eq.10 的 τ 与误差同量纲，须先钳再平方（见
        `spec.relaxed_tracking_reward` 的单位说明）。
      - **σ 的可用区间**：由"该项不饱和/不消失"这一可验证条件划定，
        见 `data/probe_reward_scales.py` 对真实参考运动的实测。

    这些都属于 A1–A18 假设体系，训练启动时随 `dump()` 写入日志。
    """

    tracking_kernel: str = "exp_neg_squared_error_over_sigma"
    sigma_source: str = "OmniH2O/HOVER tracking 约定（PGMT 明示继承），量级见 spec.SIGMAS"
    tau_linear_from_zero: bool = True  # τ 在 L0 处为 0，线性升至饱和值
    # 松弛预算的饱和量（单位同**误差**，不是误差平方）：A12 的 5% 量级换算
    tau_saturation: Dict[str, float] = field(default_factory=lambda: {
        "ta_link_pos": 0.05,    # m
        "ta_link_ori": 0.1,     # rad（约 5.7°）
        "ta_joint_pos": 0.1,    # rad
    })


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
    "A17": Assumption("A17", "data_filter", DataFilterCfg(),
                     "ground 类重定向退化（待复核，见 DataFilterCfg docstring）；论文未提数据过滤"),
    "A18": Assumption("A18", "reward_impl", RewardImpl(),
                      "Table I 权重与 Eq.10 松弛形式照搬论文；核函数取 exp(−e²/σ)"
                      "（依据 OmniH2O 奖励表 exp(−0.5‖p−p̂‖²) 与其配置注释"
                      "exp(-error^2/sigma)）；σ 取值与 τ 斜率/饱和值见 spec.SIGMAS 与"
                      " data/probe_reward_scales.py 实测"),
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
