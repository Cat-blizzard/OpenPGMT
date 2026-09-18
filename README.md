# OpenPGMT — PGMT 复现

复现 **PGMT: Perceptive General Motion Tracking for Humanoid Robots**
（[arXiv:2609.08511](https://arxiv.org/abs/2609.08511)，ZJU X-Mechanics + NUS MARMot Lab）。

> **本仓库是独立复现**：与论文作者**无关**，**未获得其官方代码**（论文未开源），
> 完全依据论文正文与插图自行实现。来源、第三方许可与免责声明见
> [`NOTICE.md`](NOTICE.md)。论文 PDF/全文**不包含**在本仓库内。

- 机器人: Unitree G1（29-DoF），动作 = 关节位置目标（PD 控制）
- 训练: 两阶段（Stage 1 平地 tracking 预训练 → Stage 2 感知注入），PPO
- 数据: LAFAN1（Mixamo 骨骼 BVH）→ G1 重定向（已含 IK 精修）
- 评估: 9600 matched episodes（5 地形族 × 10 难度 × 192 集）+ 消融对标 Table II / Fig. 3 / Fig. 4
- 当前状态: **M1 / M1.5 / M2.0 / M2.0b / M3 完成并验证**；M0 待服务器冒烟；Stage 1 的策略包装、多头 PPO 与 rollout 存储已补齐，但 G1 仿真环境和物理训练闭环仍待接入

完整方案见 [`复现方案.md`](复现方案.md)（含每个里程碑的验收标准、假设清单、风险清单）。

## 仓库结构

```
setup/            安装与冒烟脚本（本机 + 服务器两条路线）
pgmt/contracts.py 跨层维度常量唯一出处（OBS_DIM/ACT_DIM/HISTORY_LEN/REF_FRAME_DIM）
pgmt/cfg/         assumptions.py —— 假设清单 A1–A22 的唯一出处
pgmt/policy/      ✅ 全部实现并有单测: RoPE / MHCA / History Encoder / IFM /
                     Glimpse Encoder / Actor / Multi-Head Critic / rotation(6D)
pgmt/envs/        observations.py（观测契约）✅
                  reference_sampler.py（C^K 采样/修正速度/自适应采样）✅
                  terrain/（5 族 × L0–L9 高度场 + 高程图 + 课程 + 兼容规则）✅
                  termination.py（终止条件 + 容忍区，A20）✅
                  g1_env.py（M2，未实现）
pgmt/rewards/     spec.py（Table I 权重 + Eq.10 松弛 + 值域守卫）✅
                  semantics.py（28 项语义对照表）✅
                  tracking / auxiliary / terrain_contact（Table I 的 28 项残差）✅
pgmt/train/       Stage 1 策略包装、多头 PPO 与 rollout 存储（仿真环境适配仍待实现）
data/             ✅ LAFAN1 下载 + BVH 解析 + 重定向 + IK 精修 + 质量评估 + 奖励尺度探针
eval/             基准评估与消融（M5，未实现）；eval/viz/ 为 M1.5 可视化
baselines/        RGMT-Reimpl（M6，尽力而为）
tests/            730 个测试（约 55 个需真实数据，缺失时自动 skip）
```

**不纳入版本控制**（体积大 / 许可约束，需自行生成）：`data/raw/lafan1/`（LAFAN1 原始 BVH）、
`data/processed/lafan1_g1/`（77 个重定向 npz）。见下方「数据准备」。

## 本机开发（Windows，无需 GPU）

```bat
setup\install_local.bat          REM 创建 conda 环境 pgmt-dev、装依赖、跑测试
conda activate pgmt-dev
cd /d D:\PGMT
python -m pytest tests -q        REM 728 passed, 2 skipped（CUDA）（约 55 个需真实数据，缺失时 skip）
```

本机只做纯逻辑开发与单测（torch 2.1.2 **CPU**，版本与服务器对齐以避免数值漂移）；训练在服务器进行。
依赖清单见 `setup/requirements-dev.txt`。

> `setup\install_local.bat` 需要 conda 已对 cmd.exe 初始化（`conda init cmd.exe`），
> 否则 `conda activate` 会失败 —— 脚本会检测并中止，不会污染 base 环境。

## 数据准备

仓库**不含**原始数据与重定向结果，需自行生成两步：

```bat
REM 1) 下载 LAFAN1（约 144MB，研究用途免费；GitHub LFS 需用 media 地址）
bash data/download_lafan1.sh
REM    产物: data/raw/lafan1/*.bvh（77 个序列）

REM 2) Mixamo → G1 重定向（含 IK 精修，CPU、耗时数分钟至一小时）
python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1 --out-dir data/processed/lafan1_g1
REM    产物: data/processed/lafan1_g1/*.npz（77 个）

REM 3)（可选）质量评估与可视化
python -m data.eval_retarget                     REM → data/processed/quality_report.csv
python -m data.viz_retarget data/raw/lafan1/walk1_subject1.bvh --frames 100 600 1200 1800
python -m data.probe_reward_scales               REM 奖励尺度探针（A18 依据）

REM 4)（可选）地形检查图（需 matplotlib）
python -m data.viz_terrain                       REM 五族 × 十级总览 → eval/viz/terrain_overview.png
python -m data.viz_terrain --maps-family stairs --levels 0 5 9
python -m data.viz_terrain --dump stairs 9       REM 无 matplotlib 时打印高程图数值
```

第 1 步之前请先读 [`NOTICE.md`](NOTICE.md)：LAFAN1 与研究用途许可相关，
`data/raw/.external/human2humanoid/` 为上游 **CC-BY-NC-4.0** 内容。

## 本机工具链注意

本仓库开发机曾出现 **PowerShell (`pwsh`) 完全无法启动** —— 进程起步即报
`Fatal error. Your Windows doesn't fully support CET`（exit code `2148734214`，
即 `STATUS_FAIL_FAST_EXCEPTION`），所有 `pwsh` 调用均失败。已确认：

- 与命令无关（连 `echo hello` 也崩），是进程初始化阶段失败
- 系统里 `.NET` 仅 6.0.32，而该版本**没有**这条 CET 检查 → 说明是 harness 自带
  的 pwsh + 自带新版 .NET 与系统 Windows 版本不匹配
- 报错建议的"安装 Windows 更新"具误导性；单独安装 PowerShell 7 也无效
- **cmd.exe 与 Windows PowerShell 5.1 可用**，本项目的全部测试与脚本都在 cmd 下运行

若你也遇到，直接用 cmd / Git Bash 即可，不影响本项目。

## 服务器安装与冒烟（M0 验收）

**训练硬件**: 单机 10× RTX 5880 Ada（sm_89，48GB/卡），驱动 580 / CUDA 13.0。
每卡独立跑一个训练 run（`CUDA_VISIBLE_DEVICES=k`），不做 run 内多卡。

**背景**: Isaac Gym Preview 4 是 cu11.8 时代产物：官方 Python 绑定仅到 py3.8/3.9 且按旧 torch ABI 编译，torch 必须用 2.1 时代版本（cu121）。但 **sm_89（Ada）正是 PP4 的社区黄金架构**（RTX 4090 同代），整个 legged_gym/OmniH2O/PHC 生态都在此架构上验证过。唯一未验证项是本机驱动 580（社区黄金线 525/535，但 headless 训练在 555+ 有先例）——冒烟测试一跑便知。

### 决策树

```
第 0 步（一次性，注册 NVIDIA developer 免费账号）:
  下载 IsaacGym_Preview_4_Package.tar.gz 传到服务器

第 1 步（检查包，1 分钟）:
  bash setup/check_isaacgym.sh <tar 路径>
      ├─ 报告官方绑定支持的 Python 版本（gym_38.so → 3.8；gym_39.so → 3.9）
      └─ 报告 PhysX 是否含本机架构（sm_89）内核/PTX

第 2 步（主路线: 官方 PP4 + Ada 黄金配置）:
  bash setup/install_server_isaacgym.sh --isaacgym <tar 路径> [--python 3.9]
      # python 3.9（自动按绑定选择）+ torch 2.1.2 cu121 + rsl-rl-lib 2.1.2
  python setup/smoke_test.py        # 三阶段: torch / 物理 128 env / PPO 10 iter
  python -m pytest tests -q         # 冒烟通过后再跑一遍单测（可当回归门禁）

第 3 步（兜底: 冒烟失败且无法降驱动时）:
  bash setup/install_server.sh      # Isaac Lab 2.3.2.post1（py3.11 + torch 2.7 cu128）
  python setup/smoke_test_isaaclab.py
  # 若采用 Isaac Lab，M2 环境代码按 Isaac Lab manager-based API 编写
  # （架构不受影响: 观测/奖励/课程逻辑与仿真器解耦）
```

**驱动 580 风险预案**: 若 PP4 在 580 上崩溃（create_sim/PhysX 初始化错误），优先把这台机器（或其中某几张卡对应的训练进程）降驱动到 535/545 档；机器不便降驱动则转第 3 步 Isaac Lab。

### 冒烟测试通过标准（M0 验收）

- 主路线: `smoke_test.py` 三阶段全 PASS（128 env 物理 + 10 iter PPO 闭环）
- 兜底: `smoke_test_isaaclab.py` 两阶段全 PASS（torch + headless 物理仿真）
- 本机: `python -m pytest tests -q` 全绿

## 假设清单

论文未公开的超参集中在 [`pgmt/cfg/assumptions.py`](pgmt/cfg/assumptions.py)（A1–**A22**），训练启动时 `dump()` 写入运行日志，最终报告逐项对照说明偏差。

- **A17（数据过滤）待复核**：其"ground 类重定向退化（38–116cm）"的依据来自旧版数据，当前质量报告显示 ground 五个序列为 14.45–15.92cm（最差但无病态）。需用独立指标重测后决定是否恢复这 5 个躺地/翻滚序列进训练集——详见 `复现方案.md` §M1.5。
- **A18（奖励实现）**：Table I 的逐项权重与 Eq.10 的松弛形式**照搬论文**（见 `pgmt/rewards/spec.py`，并有逐字对照的回归测试）。论文未写出的部分：核函数取 **`exp(−e²/σ)`（高斯式，误差平方）**，依据是 PGMT 明示继承的 tracking 实现（OmniH2O 奖励表 `exp(−0.5‖p−p̂‖²)` 与其配置注释 `exp(-error^2/sigma)`）；σ 取值见 `spec.SIGMAS`。
  - `python -m data.probe_reward_scales` 用真实参考运动暴露量纲错误、给出各 σ 的响应区；**但它量的是参考运动幅度而非跟踪误差，不能用来验证 σ 已标定正确** —— 最终标定须等训练时读到实际误差分布，详见 `复现方案.md` §M2.0b。
  - `pgmt/rewards/semantics.py` 是 Table I 逐项语义对照表（28 条，标注与参照实现的对应关系：identical / approx / PGMT-specific / **待定 4 项**）。待定项为 `head_torso_impact`、`ee_accel_mismatch`、`floating_anchor_pos`、`ta_link_ori` —— 论文或参照实现未给出足够依据，现已按 A18/A21 的显式假设实现，但参照依据仍待核对。标注纪律由测试强制：note 里出现"未确认/未见/未验证"等措辞时只能标待定。
- **A19–A22（M3 / M2 新增）**：`A19` tile 规格（边长 / 留白 / 级数）、`A20` 终止条件与容忍区、`A21` auxiliary 组 10 项的度量与阈值、`A22` terrain-contact 组 6 项的度量与阈值（仅 Stage 2）。后两者论文**只给项名与权重**（Table I 里 terrain-contact 那一段连公式都没有），度量方式全部属本仓库拍定，各项依据强度分级写在对应 dataclass 的 docstring 里。
- **奖励项的符号约定**：Table I 中所有项的值**恒 ≥ 0**，正负完全由权重携带（负权重即惩罚项）。`RewardGroup.sum` 在运行时拒绝负值与 NaN —— 惩罚项若自身返回负值，与负权重相乘会变成**正贡献**（惩罚反转成奖励），这类错误在训练曲线上只表现为"学出怪行为"，极难定位。
- **一处已知的数据源差异**：`reference_contact_match` 的参考接触标签，论文取自 offline terrain-mesh queries，本仓库取 LAFAN1 足部位置的**速度阈值**（`data/retarget_lafan1.py`）。两者不同源，该项数值有系统性偏差。

## 数据管线产物

- `data/raw/lafan1/*.bvh` — 77 个源序列。**不纳入版本控制**，用 `data/download_lafan1.sh` 获取
- `data/processed/lafan1_g1/*.npz` — 77 个精修后序列（qpos/qvel/root_pos/root_rot/contacts）。
  **不纳入版本控制**（数十 MB 二进制 + LAFAN1 衍生数据许可约束），用 `python -m data.retarget_lafan1` 生成
- `data/processed/quality_report.csv` — 质量报告，**含在仓库内**

建议在正式训练前把重定向产物放到独立目录，并运行只读验收：

```bash
python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1 --out-dir data/processed/lafan1_g1_fixed
python -m data.eval_retarget --npz-dir data/processed/lafan1_g1_fixed \
  --out data/processed/quality_report_fixed.csv
python -m data.validate_retarget --npz-dir data/processed/lafan1_g1_fixed \
  --bvh-dir data/raw/lafan1 --quality-report data/processed/quality_report_fixed.csv \
  --out data/processed/acceptance_fixed.json \
  --provenance-out data/processed/provenance_fixed.json
```

`data/validate_retarget.py` 只检查产物完整性、限位、速度差分、四元数和接触标签一致性；
它不等价于物理可执行性证明。旧产物若要盘点而不阻止命令，可加
`--allow-violations`，违规会保留在 JSON 的逐序列诊断中。

现有 CSV 已包含拟合点与留出点的分离指标，但保存的是修复前 IK 的评估结果。
`python -m data.eval_retarget` 现在默认评价磁盘上的实际 NPZ，不再静默从 BVH
重新生成参考。`--regenerate` 用当前算法重算候选，默认写独立的
`quality_report_regenerated.csv`；报告的 `reference_source` 列区分两种来源。
评估过程先保存同目录的独立 `.partial.csv`，全部选中序列成功后才原子替换正式报告；
失败或中断保留原报告并返回非零退出码，打印已保存的部分结果路径。
修复后的 IK 会改变输出，旧 NPZ 不会因代码更新自动变更，正式训练前应另目录生成并复核。

各类指标提供不同的诊断信息，均不能单独证明动作物理可执行：

| 指标 | 含义 | 独立性 |
|---|---|---|
| `fitted_err_cm` | 13 个被 IK 优化过的关键点位置误差 | ❌ IK 训练残差，仅回归监控 |
| `holdout_err_cm` | 4 个**未进入 IK 优化目标**的关键点（左右髋、左右趾）误差 | ⚠️ 泛化检验，非独立 |
| `upright_holdout_err_cm` | 同上 4 点，但把根朝向强制为单位朝向 | ⚠️ 泛化检验，非独立 |
| `min_foot_z` / `spike_pct` / `contact_agree` / `limit_over_pct` | 踝原点高度 / 速度尖峰 / 接触一致 / 限位超限 | 与 IK 位置目标不同的诊断；踝原点不等于足底碰撞几何 |

**为什么"留出"只能算泛化检验**：那 4 个点确实没有目标牵引它们，但
`refine_full` 优化的是 `list(range(29))`——**全部 29 个关节**，改动髋/踝关节角
会直接移动左右髋与左右趾，它们与拟合点共享同一条运动链。所以数值低不足以
证明物理可执行，数值高提示需要检查目标映射、骨架差异和运动链。

`holdout_err_cm` 与 `upright_holdout_err_cm` 的平移部分**完全相同**（都逐帧把
骨盆对到源 Hips），唯一差别是根朝向。后者只是把根朝向强制设为单位旋转的
对照，不能直接当作真实根朝向误差；合理转向、躺倒也会改变它。两者均不衡量累积平移漂移。

> ⚠️ **曾列在报告里的 `root_rot_err_deg` 已确认是恒等式，不是指标**：
> `retarget()` 的 `root_rot = (qw ⊗ grot_src[Hips]) ⊗ Q_MRIG_INV` 与评估里调用的
> `source_root_quat_to_g1_base()` 是同一个式子，故按构造相等 —— 77 个序列实测
> **恒为 0.01°**（float32 舍入），无法区分任何序列，不得用作 A17 证据。它一度
> 被标为"完全独立"，那是**独立但无用**：换算本身的正确性由
> `tests/test_root_frame.py` 对真实数据的闭环测试保证。

> 旧的单一 `fk_err_cm` 与 `ik_refine.refine_full` 的优化目标重叠 13/18 个关键点，
> 且逐帧只用骨盆平移对齐 —— 它**不能**用来判断参考是否可用。A17（排除 ground 类）
> 的原始裁定正是基于这个自证指标，因此需要重裁：先重跑 `data.eval_retarget`，
> 再结合留出点、速度尖峰、关节限位、接触与碰撞几何复核 ground 类。
> `root_rot_err_deg` 不能作为恢复或排除动作的裁定依据。
> 快速复核命令：`python -m data.eval_retarget --only ground`
> （写 `quality_report_ground.csv`，**不会**动完整报告）

## 代码复核（2026-09-16）

本轮修复覆盖以下已确认问题，并增加对应回归测试：

- **IK 与导出**：正则梯度、候选接受使用同一总目标；初始与最终关节限位；有界关节位置直接差分得到速度。
- **参考时空约定**：A2 偏移是控制步，按 `control_dt/source_dt` 换算数据帧，默认前瞻 0.62 s；全部未来速度表达在当前参考锚点系，与位置修正误差同系。辅助速度奖励使用世界系，环境适配时需旋回。
- **网络迁移**：critic 使用真实 PyTorch `(3,H) → (4,H)` 权重布局，保留 device/dtype 与原有三头输出。
- **奖励与终止**：修复参考 yaw 对齐方向；A12 χ 配置与执行统一；自定义终止配置生效；倾角区分直立与倒立；缺少目标 body 显式报错。
- **环境脚本**：包内绑定正确解析为 Python 3.8/3.9 等，支持 `.tar`/`.tar.gz`；检查已有环境版本；冒烟各阶段独立进程，Isaac Gym 先于 PyTorch 导入；GPU 施力使用张量 API；Isaac Lab 箱体启用碰撞。
- **评估来源与保存**：默认读取实际 NPZ，显式 `--regenerate` 才重算；候选报告另存，失败或中断不覆盖已有正式报告。

最终回归：`python -m pytest tests -q -ra -p no:cacheprovider`，**728 passed / 2 skipped**，耗时 147.39 s。另通过全部 Bash 安装脚本语法检查及 Python 3.8 语法解析。

本机验证使用 Python 3.10 / PyTorch 2.7.1 CPU，服务器安装脚本目标仍为
PyTorch 2.1.2 cu121；两者不能视为同环境验证。CUDA 迁移测试在本机跳过，
真实 Isaac Gym / Isaac Lab 仿真仍需服务器冒烟。恢复/倒立动作的终止门控、
未公开奖励定义及 A17 数据过滤仍需通过训练与物理诊断确定。

旧 NPZ/CSV 原样保留。建议以独立目录复核新产物：

```bash
python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1 --out-dir data/processed/lafan1_g1_fixed
python -m data.eval_retarget --npz-dir data/processed/lafan1_g1_fixed --out data/processed/quality_report_fixed.csv
```

## 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境与骨架（本机单测 + 服务器冒烟） | 本机部分完成（工具链问题已修）；待下载 PP4 tar + 服务器冒烟 |
| M1 | LAFAN1 数据管线 + 重定向 | ✅ 完成 |
| M1.5 | IK 精修 + 接触标签统一协议 | ✅ 完成（A17 待复核） |
| M2.0 / M2.0b | 观测契约 / 奖励规格（不依赖仿真器） | ✅ 完成 |
| M3 | 地形系统（5 族 × L0–L9，纯逻辑部分） | ✅ 完成（mesh 注入归 M2） |
| M2 | Stage 1 平地 tracking 预训练 | 🟡 策略/多头 PPO/奖励/终止已就绪；G1 环境、PD 和批量奖励仍未接入 |
| M4 | Stage 2 感知注入 | ⬜ 未开始 |
| M5 | 基准评估 + 消融 | ⬜ 未开始 |
| M6 | 基线与最终报告 | ⬜ 未开始 |

## 参考

- 论文: [arXiv:2609.08511](https://arxiv.org/abs/2609.08511) · 项目主页: https://luyili.github.io/pgmt/
- 基础代码参考: human2humanoid (OmniH2O, https://github.com/OpenRobotLab/human2humanoid)
- 数据: LAFAN1 (https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- 机器人描述: unitree_rl_gym (https://github.com/unitreerobotics/unitree_rl_gym)
- Isaac Gym 兼容性调研: NVIDIA 论坛（PP4 官方答复）、Isaac Lab Releases

## 许可与致谢

本仓库自行编写的代码与文档以 [MIT](LICENSE) 发布；**第三方资源不在该授权范围内**
（LAFAN1 数据、human2humanoid/OmniH2O 的 CC-BY-NC-4.0 内容、Unitree G1 描述文件、
Isaac Gym）。详见 [`NOTICE.md`](NOTICE.md)。
