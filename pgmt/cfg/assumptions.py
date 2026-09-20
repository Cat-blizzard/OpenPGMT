"""假设注册表（Assumption Registry）——论文未公开超参的唯一出处。

论文 arXiv:2609.08511v2 未给出全部实现细节。凡需自行拍定的取值，
一律在此登记为编号假设（A1–A22），并附带理由。约定：
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
class TerrainTiling:
    """A19：地形 tile 几何（论文只说"five families × ten levels"，未给 tile 规格）。

    论文未写出的部分：
      - **tile 边长** → 必须 > 高程图覆盖范围（2m×2m，论文 §III），否则一张
        高程图会跨越多个 tile，"机器人位于哪个地形族"（Stage 2 松弛的 χ(κ_t)、
        地形课程都依赖它）就没有定义。取 4.0m（约 2 倍余量）。
      - **四周留白** → 相邻 tile 在接缝处高度必须一致，否则机器人会撞上"看不见
        的墙"（高程图看不出异常，物理上跨不过去）。做法是四周留 1.0m 的 z=0 平台，
        几何特征只在内部出现。这是 legged_gym 系 terrain 的通行做法。
      - **stair 级数** → 取固定 4 级，使**总上升量正比于难度**（L9 最大
        ≈ 4×24cm ≈ 0.96m）。若改用"阶距固定"，L9 会在 1m 内爬升 ≈1.4m
        （约 55° 的阶梯），几何上不自洽。
      - **rough 噪声形式** → 取"整数波数正弦叠加"，在 x、y 上以 `tile_size` 为
        周期，从而**跨 tile 无缝**。任意随机场（如逐 tile 独立采样的 Perlin）
        在边界处不连续。
    """

    tile_size: float = 4.0
    border: float = 1.0
    stair_steps: int = 4
    rough_harmonics: int = 4


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
        "slopes": 1.0,
        "stairs": 1.0,
        "boxes": 1.0,
        "rough": 0.0,  # rough 保持严格跟踪
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
    """A17：训练数据过滤（M1.5c ground 类裁定）——**已改判（2026-09-20）：不过滤，保留全集**。

    原裁定理由（已推翻）：ground 类（躺地/翻滚）分解退化（38–116 cm）。
    该观测来自 lafan1_g1_v1_snapshot（v1 版数据），当前数据上无从复现
    （ground 五序列留出误差与 fall 类完全重叠）。

    改判依据（两条独立证据）：
    1. 论文立场：训练设置只说 "motions from LAFAN1 retargeted to a
       humanoid robot"，自适应采样明示 "retaining uniform coverage of the
       full motion dataset"，全文无任何数据过滤表述；且能力主张直接依赖
       躺地/摔倒动作（遥操作 "lying down"、扰动恢复 "fallen configuration
       … upright and fallen states"）。摔倒池只能给"从摔倒状态恢复"的
       经验，"执行躺地参考"需要训练分布里存在躺地参考。
    2. 病根已修复：真正的缺陷是 retarget 的垂直锚定假设"足是最低接触点"
       ——躺地序列整体悬浮 0.42–0.49 m、跨障序列压入地面至踝 −0.286 m
       （实测见 data/probe_heights）。v2 两段式锚定（源侧全身最低体点
       触地 + IK 后刚体校正）修复后，77 序列全集在
       data/processed/lafan1_g1_anchored 重新生成并通过验收，ground 类
       最低踝原点全部落在 0.036–0.108 m。

    excluded_prefixes 保留为机制（MotionDatabase 按前缀过滤），当前为空。
    """

    excluded_prefixes: tuple = ()


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


@dataclass(frozen=True)
class TerminationCfg:
    """A20：终止阈值与容忍区（论文提到 "terrain aware ... terminations" 但未列条件）。

    论文**明确的**：
      - completion 定义（Table II 脚注）：跑满 30s 上限且**未提前终止** = 成功
      - Fig.2 图例把 "terrain aware rewards **& terminations**" 并列，并画出
        "Tolerance zone" 与 "Drift-tolerant tracking" → 终止与奖励一样是
        **地形感知**的：误差落在容忍区内不算失败
      - §V-C：level "jointly controls ... selected termination delays" →
        高难度地形上终止被**延迟**

    论文**未写的**（本假设负责拍定）：
      - 具体有哪些触发条件 → 取三类：参考偏差超容忍区 / 基座过低 / 姿态倾斜过度
      - 容忍区的大小 → 取基准值 + **复用 A12 松弛预算 τ**（使"奖励不罚"与
        "不终止"口径一致，这正是 Fig.2 把两者并列的缘由）
      - 各阈值取值 → 见字段默认值；`root_height_min` 与 `tilt_max_deg` 属
        "可从服务器失败模式校准"的参数，训练启动后应据实际分布复核
      - 延迟随难度增长的系数 → `terrain_delay_scale`（秒/米）
    """

    #: 非地形感知的基准容忍区（米）；χ=0 的族（flat / rough）用它
    ref_deviation_base: float = 0.5
    #: 基座最低高度（米）
    root_height_min: float = 0.35
    #: 基座相对竖直的最大倾角（度）
    tilt_max_deg: float = 70.0
    #: 各原因的基础延迟（秒）
    delay_s: Dict[str, float] = field(default_factory=lambda: {
        "ref_deviation": 0.5,
        "root_low": 0.0,
        "tilted": 0.0,
    })
    #: 地形松弛预算对延迟的放大（秒/米）
    terrain_delay_scale: float = 2.0


@dataclass(frozen=True)
class AuxCfg:
    """A21：辅助项的尺度与阈值（论文 Table I 只给项名与权重）。

    Table I 的 auxiliary 组含两类项，**权重符号已指明区分**：
      - **4 个正向项**（权重 > 0）：`root_ori` 0.5 / `corrected_root_vel` 2.0 /
        `floating_anchor_pos` 1.0 / `recovery_upward_vel` 12.5
        → 用与跟踪组相同的高斯核 `exp(−e²/σ)`
      - **6 个惩罚项**（权重 < 0）：`pelvis_vert_accel` / `ee_accel_mismatch` /
        `action_rate` / `joint_limit` / `undesired_contact` / `head_torso_impact`
        → 返回非负平方代价，负号仅由 Table I 权重携带

    论文未说明各惩罚的**度量方式**与**阈值**，故集中于此：
      - `pelvis_vert_accel`：取加速度**平方**（0 处可导、量纲一致）
      - `ee_accel_mismatch`：取与**参考**比（而非与上一帧比）—— 属 A18 体系里
        标为 UNRESOLVED 的一项，依据不足
      - `joint_limit`：取**越界量的平方**（界内为 0）—— 软约束而非"偏好中位"
      - `undesired_contact`：除允许部位外的接触力超过死区的平方和
      - `head_torso_impact`：⚠️ **论文完全未说"撞击"如何度量**，取"接触力超过
        阈值的部分"仅为连通组合项；`semantics.py` 中该项标为 UNRESOLVED，
        **不得据此声称已复现**
      - `recovery_upward_vel`：**非对称**残差 `max(0, target − 实际上行速度)`,
        只惩罚不足（不该惩罚"起得更快"）
    """

    #: 正向项的 σ（误差**平方**量纲，与跟踪组一致）
    sigma_root_ori: float = 0.5            # rad²
    sigma_corrected_root_vel: float = 1.0  # (m/s)²
    sigma_floating_anchor: float = 0.25    # m²
    sigma_recovery_upward: float = 0.25    # (m/s)²
    #: recovery 项的目标上行速度（m/s）
    recovery_target_upward_vel: float = 0.5
    #: 非期望接触的力死区（N）与允许接触的部位
    contact_force_threshold: float = 1.0
    allowed_contact_bodies: tuple = (
        "left_ankle_roll_link", "right_ankle_roll_link")
    #: 头/躯干撞击阈值（N）—— 依据不足，见 docstring
    head_torso_impact_threshold: float = 50.0


@dataclass(frozen=True)
class TerrainContactCfg:
    """A22：terrain-contact 组的度量与阈值（仅 Stage 2，论文 Eq.9 的第四个值头）。

    论文 §IV-B "Multi-Head Critic Extension" 的**全部**描述只有一段话：

        "To encourage stable contacts, we introduce a new terrain-contact reward
         group r^terrain_t (summarized in Table I). **Local height variation** is
         used to evaluate **touchdown quality**, while **contact labels obtained
         from offline terrain-mesh queries** supervise the consistency between
         reference and simulated contacts. Additional penalties discourage **foot
         slippage, stumbling, rapid contact switching, and excessive contact
         forces**."

    即：6 项的**意图**都点明了，但**一个公式、一个阈值都没给**。本假设负责拍定
    度量方式与尺度，逐项依据强度如下：

    **依据较强的**（有参照实现或论文措辞直接支撑）：
      - `stumble` / `contact_force`：legged_gym 系有同名项，分别取"水平接触力
        超过竖直分量的倍数"与"接触力超过 max_contact_force 的部分"
      - `touchdown_quality`：论文明说用 **local height variation**，故取落点
        附近高度采样的**标准差**
      - `reference_contact_match`：论文明说是参考与仿真接触的**一致性**，
        故取一致率 ∈ [0,1]（**不需要 σ** —— 一致性本身就是归一化奖励）

    **依据较弱的**（论文只有动词，没有度量方式）：
      - `slip`：取接触足的水平速度**平方**和（与仓库其它 `‖·‖²` 代价同族）
      - `contact_switching`：论文只说惩罚"rapid"切换，**未定义 rapid** ——
        取"接触状态未持续满 `contact_switching_min_dwell` 步就再次翻转"，
        代价随已持续步数线性衰减。注意：**不能**直接惩罚"接触状态发生变化"，
        因为正常步态每步都在切换

    ⚠️ **数据源差异（不是参数问题，是管线差异）**：论文的接触标签来自
    "offline terrain-mesh queries"（把参考动作放到地形上查网格），而本仓库的
    `contacts` 字段来自 LAFAN1 足部位置的**速度阈值**（`data/retarget_lafan1.py`）。
    两者不同源，且前者才是论文口径。`reference_contact_match` 的结论因此有
    系统性偏差，需在报告中说明（见 `semantics.py` 该项的 note）。
    """

    #: `touchdown_quality` 的 σ（**m²**，误差平方量纲，与 A18 的核一致）。
    #: √σ = 0.1 m 是"1/e 落足高度变化"：台阶/箱面边缘的落点标准差约 0.1–0.2 m，
    #: 平地约 0 —— 该量级使该项在"平稳落地"与"踩在棱上"之间有区分度
    sigma_touchdown_quality: float = 0.01
    #: 落足质量采样的局部半径（m）。取 0.1 m 约等于 A8 高程图的一格分辨率
    #: （map_res = 0.1 m），使采样点至少覆盖 3×3 格
    touchdown_patch_radius: float = 0.1
    #: `stumble` 判据：水平接触力超过竖直分量的倍数（legged_gym 传统取 5.0）
    stumble_force_ratio: float = 5.0
    #: `contact_switching` 判据：接触状态至少持续多少步才算"稳定"（步）。
    #: 50 Hz 下 5 步 = 0.1 s
    contact_switching_min_dwell: int = 5
    #: `contact_force` 阈值（N）。legged_gym 的 `max_contact_force` 取 500
    contact_force_max: float = 500.0


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
                     "已改判（2026-09-20）：论文支持全集均匀覆盖且能力主张依赖躺地动作；"
                     "原缺陷是垂直锚定（v2 已修复），不再按类过滤（见 DataFilterCfg docstring）"),
    "A18": Assumption("A18", "reward_impl", RewardImpl(),
                      "Table I 权重与 Eq.10 松弛形式照搬论文；核函数取 exp(−e²/σ)"
                      "（依据 OmniH2O 奖励表 exp(−0.5‖p−p̂‖²) 与其配置注释"
                      "exp(-error^2/sigma)）；σ 取值与 τ 斜率/饱和值见 spec.SIGMAS 与"
                      " data/probe_reward_scales.py 实测"),
    "A19": Assumption("A19", "terrain_tiling", TerrainTiling(),
                      "论文只给'5 族 × L0–L9'未给 tile 规格；边长须 > 2m 高程图覆盖"
                      "（否则族/难度无定义），四周留白保证跨 tile 高度连续，"
                      "stair 取固定级数使总上升量正比于难度，rough 取周期正弦保证无缝"),
    "A20": Assumption("A20", "termination", TerminationCfg(),
                      "论文只说 completion = 跑满上限且未提前终止，并提到 terrain aware "
                      "terminations / tolerance zone / selected termination delays，"
                      "但未列触发条件与阈值；取三类触发 + 容忍区复用 A12 松弛预算"),
    "A21": Assumption("A21", "aux_rewards", AuxCfg(),
                      "Table I 的 auxiliary 组含 4 个正向项与 6 个惩罚项（权重符号"
                      "已指明区分）；论文未给度量方式与阈值，集中于此。其中 "
                      "head_torso_impact 的度量完全无依据，semantics.py 标为 UNRESOLVED"),
    "A22": Assumption("A22", "terrain_contact_rewards", TerrainContactCfg(),
                      "Table I 的 terrain-contact 组只有项名与权重；论文 §IV-B 给了"
                      "6 项的定性意图（local height variation → touchdown quality、"
                      "offline terrain-mesh queries → contact consistency、"
                      "slippage/stumbling/rapid contact switching/excessive contact "
                      "forces），但未给任何公式与阈值。度量与尺度集中于此；"
                      "其中 stumble/contact_force 沿用 legged_gym 同名项的传统取值。"
                      "**注意**参考接触标签的数据源与论文不同（速度阈值 vs terrain-mesh 查询）"),
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
