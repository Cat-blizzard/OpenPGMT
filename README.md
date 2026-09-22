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
- 当前状态（2026-09-22）: **旧的 842-update Stage 1 已归档，不再续跑**。v4 的 16 环境 × 20 更新开/关对照，以及三个种子的 64 环境 × 200 更新均已完成。平均 KL 受控、critic 拟合改善；控制行为仍未通过验收，末段普通 episode 平均仅 1.18 秒，恢复尚未成功，且少数 recovery 状态的 KL 尾部仍需排查。后续执行器与 reset 物理对照见下文。本任务 GPU 占用已释放，最终策略评估按用户要求后置。**尚未进入正式扩规模训练**。见 [v4 实验报告](docs/curriculum_critic_20260921.md)、[v3 复验报告](docs/stability_validation_20260921.md)及[物理接口验收](docs/physics_validation_20260921.md)。

完整方案见 [`复现方案.md`](复现方案.md)（含每个里程碑的验收标准、假设清单、风险清单）。

2026-09-22：完成 URDF/USD 执行器 CPU 审查，将新运行的默认力矩上限从统一 120 Nm
改为资产对应的逐关节限值，接入限量异常快照和 KL 的 CPU 重放。806 项回归通过、2 项跳过。
随后完成 8 场小规模物理对照：新上限在 PhysX 中生效，首步冲击降低，但存活与跟踪未改善。
本任务 GPU 已释放，新的 PPO 与后置评估尚未启动。见[物理对照结果](docs/actuator_physics_20260922.md)
及[执行器与诊断记录](docs/actuator_diagnostics_20260922.md)。

同日完成固定新上限的 6 场 reset 对照：参考初始化使首步冲击降低约 66%、关节 RMSE 降低
约 25%，但平均首次时长从 1.13 s 降至 0.93 s，两组均 12/12 提前终止。保留默认 reset，
候选以显式开关保存；813 项 CPU 回归通过、2 项跳过。本任务 GPU 已释放，尚未启动新的 PPO。
见[reset 对照报告](docs/reset_physics_20260922.md)。

GitHub 可直接查看[实验图表与汇总数据](docs/results/README.md)；后续安排见
[下一阶段计划](docs/next_steps_20260922.md)。原始逐步日志、检查点和完整实验源码快照保留在服务器。

## 仓库结构

```
setup/            Linux/GPU 服务器安装与冒烟脚本
pgmt/contracts.py 跨层维度常量唯一出处（OBS_DIM/ACT_DIM/HISTORY_LEN/REF_FRAME_DIM）
pgmt/cfg/         assumptions.py —— 假设清单 A1–A22 的唯一出处
pgmt/policy/      ✅ 全部实现并有单测: RoPE / MHCA / History Encoder / IFM /
                     Glimpse Encoder / Actor / Multi-Head Critic / rotation(6D)
pgmt/envs/        observations.py（观测契约）✅
                  reference_sampler.py（C^K 采样/修正速度/自适应采样）✅
                  terrain/（5 族 × L0–L9 高度场 + 高程图 + 课程 + 兼容规则）✅
                  termination.py（终止条件 + 容忍区，A20）✅
                  g1_env.py（M2，torch 批量适配器 + Isaac Lab shell）
pgmt/rewards/     spec.py（Table I 权重 + Eq.10 松弛 + 值域守卫）✅
                  semantics.py（28 项语义对照表）✅
                  tracking / auxiliary / terrain_contact（Table I 的 28 项残差）✅
pgmt/train/       Stage 1 策略包装、多头 PPO、rollout 存储与批量 PD/参考/奖励适配器
data/             ✅ LAFAN1 下载 + BVH 解析 + 重定向 + IK 精修 + 质量评估 + 奖励尺度探针
pgmt/eval/        固定 episode 清单与 Isaac 匹配评估器（新路径待物理验收）
eval/             manifests/ 固定清单；viz/ 为 M1.5 可视化；消融待完成
baselines/        RGMT-Reimpl（M6，尽力而为）
tests/            接口、奖励、数据与新版训练协议回归（外部资源缺失时部分 skip）
```

**不纳入版本控制**（体积大 / 许可约束，需自行生成）：`data/raw/lafan1/`（LAFAN1 原始 BVH）、
`data/processed/lafan1_g1/`（77 个重定向 npz）。见下方「数据准备」。

## 运行环境（Linux/GPU 服务器）

当前项目以 Linux GPU 服务器作为唯一受支持的开发、验证和训练环境。服务器上
先完成 Isaac Gym/Isaac Lab 安装，再运行
同一套回归测试和训练命令：

```bash
# 在目标服务器的项目目录中
python -m pytest tests -q

# 纯 Torch 路径只用于协议和奖励链路检查，不包含刚体动力学
python -m pgmt.train.train_stage1 --backend torch --device cpu --dry-run
```

`setup/requirements.txt` 是服务器依赖清单。正式训练必须使用下方的 Isaac Gym
或 Isaac Lab 路径，并在有 GPU 的环境中运行；CPU Torch 结果不能替代物理仿真验收。

## 数据准备

仓库**不含**原始数据与重定向结果，需自行生成两步：

```bash
# 1) 下载 LAFAN1（约 144MB，研究用途免费；GitHub LFS 需用 media 地址）
bash data/download_lafan1.sh
#    产物: data/raw/lafan1/*.bvh（77 个序列）

# 2) Mixamo → G1 重定向（含 IK 精修，CPU、耗时数分钟至一小时）
python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1 --out-dir data/processed/lafan1_g1
#    产物: data/processed/lafan1_g1/*.npz（77 个）

# 3)（可选）质量评估与可视化
python -m data.eval_retarget                     # → data/processed/quality_report.csv
python -m data.viz_retarget data/raw/lafan1/walk1_subject1.bvh --frames 100 600 1200 1800
python -m data.probe_reward_scales               # 奖励尺度探针（A18 依据）

# 4)（可选）地形检查图（需 matplotlib）
python -m data.viz_terrain                       # 五族 × 十级总览 → eval/viz/terrain_overview.png
python -m data.viz_terrain --maps-family stairs --levels 0 5 9
python -m data.viz_terrain --dump stairs 9       # 无 matplotlib 时打印高程图数值
```

第 1 步之前请先读 [`NOTICE.md`](NOTICE.md)：LAFAN1 与研究用途许可相关，
`data/raw/.external/human2humanoid/` 为上游 **CC-BY-NC-4.0** 内容。

## 服务器安装与冒烟（M0 验收）

**训练硬件**: 单机 10× RTX 5880 Ada（sm_89，48GB/卡），驱动 580 / CUDA 13.0。
每卡独立跑一个训练 run（`CUDA_VISIBLE_DEVICES=k`），不做 run 内多卡。

**背景**: Isaac Gym Preview 4 是 cu11.8 时代产物：官方 Python 绑定仅到 py3.8/3.9 且按旧 torch ABI 编译，torch 必须用 2.1 时代版本（cu121）。但 **sm_89（Ada）正是 PP4 的社区黄金架构**（RTX 4090 同代），整个 legged_gym/OmniH2O/PHC 生态都在此架构上验证过。目标服务器驱动 580（社区黄金线 525/535，但 headless 训练在 555+ 有先例）仍需通过冒烟测试确认。

### 决策树

```
第 0 步（一次性，注册 NVIDIA developer 免费账号）:
  下载 IsaacGym_Preview_4_Package.tar.gz 传到服务器

第 1 步（检查包，1 分钟）:
  bash setup/check_isaacgym.sh <tar 路径>
      ├─ 报告官方绑定支持的 Python 版本（gym_38.so → 3.8；gym_39.so → 3.9）
      └─ 报告 PhysX 是否含目标架构（sm_89）内核/PTX

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

Isaac Sim 5.1 的 Python 扩展仍使用 `warp.types.array`。安装脚本默认固定
`warp-lang==1.12.1`；当前环境若显示 `warp-lang 1.14+` 或启动时报
`AttributeError: module 'warp.types' has no attribute 'array'`，先执行：

```bash
pip install --force-reinstall --no-deps 'warp-lang==1.12.1'
```

### 冒烟测试通过标准（M0 验收）

- 主路线: `smoke_test.py` 三阶段全 PASS（128 env 物理 + 10 iter PPO 闭环）
- 兜底: `smoke_test_isaaclab.py` 两阶段全 PASS（torch + headless 物理仿真）
- 服务器回归: `python -m pytest tests -q` 全绿

### CUDA P2P / IOMMU 告警处置

**2026-09-21 本机状态更新**：GPU 0 故障会使未隔离的 CUDA 初始化失败。本轮已经
使用 GPU UUID 隔离及显式 Vulkan 映射完成物理验证，具体参数见
[物理验收记录](docs/physics_validation_20260921.md#运行环境与本轮代码调整)。下述未隔离方式
是故障发生前的记录，不能直接套用到当前机器；本轮可重放命令保存在各 run 的
`manifest.json` 和 `run.sh` 中。

这台 10 卡服务器启用了 IOMMU，Isaac Sim 的 `carb.cudainterop` 会在启动时对可见的
物理卡做 P2P 带宽/延迟探测，并可能打印 `cudaErrorTooManyPeers`。Stage 1/2 和冒烟入口
都显式使用 `multi_gpu=False`、关闭渲染多 GPU，并设置 `renderer.multiGpu.maxGpuCount=1`；
GPU 9 上的 20 步箱体物理仍通过（z=0.250）。这组参数不能阻止底层全局拓扑探测，因而
偶尔仍会看到 IOMMU/P2P 输出，但不会改变 PhysX 实际使用的 `cuda:9`。

不要在 Isaac Sim 路径使用 `CUDA_VISIBLE_DEVICES=9` 代替 `--device cuda:9`：隔离测试虽
隐藏了 P2P 输出，却使 GPU foundation 进入 CUDA bad state 并导致物理冒烟失败。除非要
由管理员统一调整主机 IOMMU/驱动配置，否则无需改内核参数；该告警目前属于启动诊断，
不构成已验证单卡训练的失败条件。

### G1 资产与 Stage 1 入口

仓库不重新分发第三方 G1 网格或 USD。服务器上已有授权的 ProtoMotions
资产时，先做只读验收并保存 manifest：

```bash
python setup/check_g1_asset.py \
  --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf \
  --mjcf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/mjcf/g1.xml \
  --usd /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/usd/g1.usd \
  --json data/processed/g1_asset_manifest.json
```

如果希望在服务器工作区形成一个带授权文件的入口，可生成不进入 git 的
符号链接目录（也可把 `--mode` 改为 `copy`）：

```bash
python setup/prepare_g1_asset.py \
  --source-root /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets \
  --destination data/raw/g1/external/protomotions
```

URDF/MJCF 的 29 个关节、29 个执行器、body 名称和网格路径会被静态核验；
USD 的关节顺序必须在 Isaac Sim 进程中核验：

```bash
python setup/probe_g1_isaaclab.py --device cuda:9 \
  --asset /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/usd/g1.usd \
  --json data/processed/g1_usd_runtime_probe.json
```

Isaac Lab/Kit 使用 `--device cuda:k` 选择卡；不要只依赖
`CUDA_VISIBLE_DEVICES` 限制 Vulkan 的设备探测。纯 Torch 训练仍可使用
`CUDA_VISIBLE_DEVICES=9 --device cuda:0` 的单卡写法。

启动一次不依赖 Isaac Sim 的 Stage 1 PPO 训练（使用本仓库已有重定向 NPZ，
PD、参考动作和三头批量奖励均走 `G1Env`）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pgmt.train.train_stage1 \
  --backend torch --device cuda:0 --num-envs 256 --steps-per-env 24 --updates 1000 \
  --reference-data data/processed/lafan1_g1_mesh_v2 \
  --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf \
  --checkpoint runs/stage1_g1.pt
```

`--backend torch` 现在使用 URDF FK、真实 body-level tracking residual、PD、A13
位置修正和批量 PPO；它不包含刚体动力学。`--backend isaaclab` 会强制要求
`--asset`、`--urdf`、`--reference-data` 和物理采集的 `--fall-pool`，不会静默退回 torch。`--dry-run` 或
`--backend mock` 可在无 GPU/无 Isaac Sim 时验证协议、超时 bootstrap 和 checkpoint。

Stage 2 的协议路径也已接通（21×21 elevation、Terrain Glimpse、terrain-contact
奖励和四头 critic）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pgmt.train.train_stage2 --backend torch --device cuda:0 \
  --reference-data data/processed/lafan1_g1_mesh_v2 \
  --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf \
  --stage1-checkpoint runs/stage1_g1.pt \
  --num-envs 4 --steps-per-env 24 --updates 1
```

`--updates` 表示本次训练的**绝对停止迭代**；`--lr-schedule-updates` 独立指定完整
学习率日程（默认 1000），续跑必须继承原日程。短试验不会将学习率提前衰减至零。
`--critic-completion` 可启用达到 actor KL 预算后的 critic 独立补充更新，默认关闭；
其调度和课程公式属于论文未公开细节的工程假设，详见 v4 报告。
从 checkpoint 恢复时只运行剩余迭代，
不会重复已完成的 PPO 更新。新版 checkpoint 保存环境可见状态与课程，但 PhysX 内部缓存不保证逐比特复演。checkpoint 同时保存环境的
参考帧、观测历史、接触历史、恢复池和自适应采样状态。

真实 Isaac Lab Stage 1 训练需要显式指定 USD、URDF、参考数据和物理卡；`--metrics` 可把最终更新和设备信息写到 JSON，避免 Kit 关闭时终端结果被吞掉：

```bash
python -m pgmt.train.train_stage1 --backend isaaclab --device cuda:9 \
  --num-envs 256 --steps-per-env 24 --updates 1000 \
  --asset /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/usd/g1.usd \
  --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf \
  --reference-data data/processed/lafan1_g1_mesh_v2 \
  --checkpoint runs/stage1_g1_isaaclab.pt \
  --metrics runs/stage1_g1_isaaclab.metrics.json
```

最小物理闭环已用 `--num-envs 2 --steps-per-env 2 --updates 1` 成功保存一个
17 MiB checkpoint；它是启动和接口验收，不代表训练收敛或论文指标。

Isaac Lab 启动器显式关闭渲染多 GPU，并设置 `renderer.multiGpu.maxGpuCount=1`；这只能
限制渲染路径，不能改变主机 CUDA/Vulkan 的全局设备枚举。最近一次代码审计已修复 Isaac Lab 接触力字段、base-frame 速度观测、末帧参考速度、
fallback 加速度历史、超时边界和物理 reset 写回等接口问题。`python -m pytest -q`
当前为 764 个测试全通过；GPU 9 的 CUDA 合约检查、USD 29-DoF 探针和箱体物理冒烟
也已通过。使用 `CUDA_VISIBLE_DEVICES=9` 虽能消除 P2P/IOMMU 输出，但会让 Isaac Sim
的 GPU foundation 进入 CUDA bad state，箱体冒烟失败，因此不作为 Isaac Sim 的运行方式；
正式路径继续使用不设置该变量的 `--device cuda:9`。真实 G1 最小物理 PPO 更新已经
保存 checkpoint，不能把它等同于训练收敛或论文指标复现。

## 接下来做什么

新版详细操作顺序见 [`docs/alignment_20260921.md`](docs/alignment_20260921.md)：
先验收物理 fall pool 采集、Stage 1 smoke 和五地形 Stage 2 smoke，再从零开始新的
Stage 1。旧 842-update 结果只作归档，不进入新训练或 Stage 2。

`setup/run_revised.sh` 每次只执行显式选择的一个步骤，不会自动启动下一阶段。
CPU mock/torch 检查不能代替物理接触和训练收敛验收。

## 假设清单

论文未公开的超参集中在 [`pgmt/cfg/assumptions.py`](pgmt/cfg/assumptions.py)（A1–**A22**），训练启动时 `dump()` 写入运行日志，最终报告逐项对照说明偏差。

- **A17（数据过滤）已改判（2026-09-20）**：原"ground 类重定向退化"的裁定依据被推翻；论文证据（"retaining uniform coverage of the **full motion dataset**"、遥操作含 lying down、全文无任何数据过滤表述）支持**保留全部 77 序列**。原病根是 `retarget()` 的垂直锚定假设"足是最低接触点"——躺地（锚到最像站立的帧，参考悬浮 0.42–0.49 m）与跨障（髋低于足，参考压入地面至踝 −0.286 m）两类动作失效。已修复为**两段式 v2 锚定**：① `retarget()` 源侧按"全序列全身最低体点触地"；② `_finalize` 在 IK 后按 G1 自身 FK 最低体点做刚体 z 校正（IK 目标随 root 刚体移动，残差与 qpos 不变；最低体点为踝时锚到静止踝高 0.036，否则锚到 0）。全集已重生成于 `data/processed/lafan1_g1_anchored`（验收零违例），新版训练使用其轨迹不变、接触标签重建的 `lafan1_g1_mesh_v2`；旧 `lafan1_g1_continuous`（42 序列、来历无记录）废弃。
- **A18（奖励实现）**：Table I 的逐项权重与 Eq.10 的松弛形式**照搬论文**（见 `pgmt/rewards/spec.py`，并有逐字对照的回归测试）。论文未写出的部分：核函数取 **`exp(−e²/σ)`（高斯式，误差平方）**，依据是 PGMT 明示继承的 tracking 实现（OmniH2O 奖励表 `exp(−0.5‖p−p̂‖²)` 与其配置注释 `exp(-error^2/sigma)`）；σ 取值见 `spec.SIGMAS`。
  - `python -m data.probe_reward_scales` 用真实参考运动暴露量纲错误、给出各 σ 的响应区；**但它量的是参考运动幅度而非跟踪误差，不能用来验证 σ 已标定正确** —— 最终标定须等训练时读到实际误差分布，详见 `复现方案.md` §M2.0b。
  - `pgmt/rewards/semantics.py` 是 Table I 逐项语义对照表（28 条，标注与参照实现的对应关系：identical / approx / PGMT-specific / **待定 4 项**）。待定项为 `head_torso_impact`、`ee_accel_mismatch`、`floating_anchor_pos`、`ta_link_ori` —— 论文或参照实现未给出足够依据，现已按 A18/A21 的显式假设实现，但参照依据仍待核对。标注纪律由测试强制：note 里出现"未确认/未见/未验证"等措辞时只能标待定。
- **A19–A22（M3 / M2 新增）**：`A19` tile 规格（边长 / 留白 / 级数）、`A20` 终止条件与容忍区、`A21` auxiliary 组 10 项的度量与阈值、`A22` terrain-contact 组 6 项的度量与阈值（仅 Stage 2）。后两者论文**只给项名与权重**（Table I 里 terrain-contact 那一段连公式都没有），度量方式全部属本仓库拍定，各项依据强度分级写在对应 dataclass 的 docstring 里。
- **奖励项的符号约定**：Table I 中所有项的值**恒 ≥ 0**，正负完全由权重携带（负权重即惩罚项）。`RewardGroup.sum` 在运行时拒绝负值与 NaN —— 惩罚项若自身返回负值，与负权重相乘会变成**正贡献**（惩罚反转成奖励），这类错误在训练曲线上只表现为"学出怪行为"，极难定位。
- **参考接触标签（2026-09-21）**：新版 `data.build_mesh_contacts` 使用 URDF 碰撞球与显式参考网格离线查询。作者参考地形未公开，当前使用 z=0 平面参考网格及 2cm 容差，仍属工程假设；也可输入外部高度场。原足速阈值标签保留以供比较。

## 数据管线产物

- `data/raw/lafan1/*.bvh` — 77 个源序列。**不纳入版本控制**，用 `data/download_lafan1.sh` 获取
- `data/processed/lafan1_g1_anchored/*.npz` — **77 个 v2 锚定精修序列（新版 mesh-contact 数据的输入）**。
  **不纳入版本控制**（体积 + LAFAN1 衍生数据许可），用 `python -m data.retarget_lafan1
  --out-dir data/processed/lafan1_g1_anchored` 生成
- `data/processed/quality_report_anchored.csv` — 当前管线的质量报告（拟合 14.80 / 留出 12.67cm）
- `data/processed/quality_report.csv` — 旧版基线（修复前 IK + v1 锚定，保留作对比）
- 旧产物（`lafan1_g1`、`lafan1_g1_continuous`、`lafan1_g1_continuous_full`）保留作对比，不再用于训练

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

## 代码复核（2026-09-19）

本轮修复覆盖以下已确认问题，并增加对应回归测试：

- **IK 与导出**：正则梯度、候选接受使用同一总目标；初始与最终关节限位；有界关节位置直接差分得到速度。
- **参考时空约定**：A2 偏移是控制步，按 `control_dt/source_dt` 换算数据帧，默认前瞻 0.62 s；全部未来速度表达在当前参考锚点系，与位置修正误差同系。辅助速度奖励使用世界系，环境适配时需旋回。
- **网络迁移**：critic 使用真实 PyTorch `(3,H) → (4,H)` 权重布局，保留 device/dtype 与原有三头输出。
- **奖励与终止**：修复参考 yaw 对齐方向；A12 χ 配置与执行统一；自定义终止配置生效；倾角区分直立与倒立；缺少目标 body 显式报错。
- **环境脚本**：包内绑定正确解析为 Python 3.8/3.9 等，支持 `.tar`/`.tar.gz`；检查已有环境版本；冒烟各阶段独立进程，Isaac Gym 先于 PyTorch 导入；GPU 施力使用张量 API；Isaac Lab 箱体启用碰撞。
- **评估来源与保存**：默认读取实际 NPZ，显式 `--regenerate` 才重算；候选报告另存，失败或中断不覆盖已有正式报告。

当前 Linux 回归：`python -m pytest tests -q`，**764 passed**。纯 Torch PPO
路径也已完成 Stage 1/Stage 2 的最小闭环，用于检查环境、奖励、采样器和 checkpoint
接口；这些结果不包含刚体动力学。真实 Isaac Gym / Isaac Lab 仿真仍需在目标 GPU
服务器上完成冒烟和训练验证。恢复/倒立动作的终止门控、未公开奖励定义及 A17
数据过滤仍需通过训练与物理诊断确定。

旧 NPZ/CSV 原样保留。建议以独立目录复核新产物：

```bash
python -m data.retarget_lafan1 --bvh-dir data/raw/lafan1 --out-dir data/processed/lafan1_g1_fixed
python -m data.eval_retarget --npz-dir data/processed/lafan1_g1_fixed --out data/processed/quality_report_fixed.csv
```

## 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境与骨架（服务器回归 + 物理冒烟） | 代码回归完成；待下载 PP4 tar + 服务器冒烟 |
| M1 | LAFAN1 数据管线 + 重定向 | ✅ 完成 |
| M1.5 | IK 精修 + 接触标签统一协议 | ✅ 完成（A17 待复核） |
| M2.0 / M2.0b | 观测契约 / 奖励规格（不依赖仿真器） | ✅ 完成 |
| M3 | 地形系统（5 族 × L0–L9，纯逻辑部分） | ✅ 完成（mesh 注入归 M2） |
| M2 | Stage 1 平地 tracking 预训练 | 🟡 URDF-FK torch body reward、PD、A13、recovery/adaptive sampling 和 PPO 已接入；USD 运行时映射与 Isaac Lab 物理闭环待空闲 GPU 验收 |
| M4 | Stage 2 感知注入 | 🟡 Torch 协议路径、elevation、terrain-contact reward 和四头 PPO 已接入；真实 terrain physics 待验收 |
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
