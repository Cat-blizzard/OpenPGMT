# PGMT 复现

复现 **PGMT: Perceptive General Motion Tracking for Humanoid Robots**（arXiv:2609.08511v2，ZJU X-Mechanics + NUS MARMot Lab）。
官方代码未开源，本项目自行实现。完整方案见 [`复现方案.md`](复现方案.md)。

- 机器人: Unitree G1（29-DoF），动作 = 关节位置目标（PD 控制）
- 训练: 两阶段（Stage 1 平地 tracking 预训练 → Stage 2 感知注入），PPO
- 数据: LAFAN1（Mixamo 骨骼 BVH）→ G1 重定向
- 评估: 9600 matched episodes（5 地形族 × 10 难度 × 192 集）+ 消融对标 Table II / Fig. 3 / Fig. 4

## 仓库结构

```
setup/            安装与冒烟脚本（见下）
pgmt/cfg/         assumptions.py —— 假设清单 A1–A14 的唯一出处
pgmt/policy/      rope.py（RoPE，已实现+单测）；后续: History Encoder/IFM/Glimpse/Actor/Critic
pgmt/envs/        G1 环境与地形系统（M2–M3）
pgmt/rewards/     奖励实现（M2/M4）
pgmt/train/       训练入口（M2/M4）
eval/             基准评估与消融（M5）
baselines/        RGMT-Reimpl（M6，尽力而为）
data/             LAFAN1 下载与重定向（M1）
tests/            纯逻辑单测（Windows 本机可跑）
```

## 本机开发（Windows，无需 GPU）

```bat
setup\install_local.bat          REM 创建 conda 环境 pgmt-dev 并装依赖
conda activate pgmt-dev
pytest tests -v                  REM 预期 20 passed
```

本机只做纯逻辑开发与单测（torch CPU）；训练在服务器进行。

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
  bash setup/install_server_isaacgym.sh --isaacgym <tar 路径>
      # python 3.9（自动按绑定选择）+ torch 2.1.2 cu121 + rsl-rl-lib 2.1.2
  python setup/smoke_test.py        # 三阶段: torch / 物理 128 env / PPO 10 iter

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
- 本机: `pytest tests -v` 全绿（20 passed）

## 假设清单

论文未公开的超参集中在 [`pgmt/cfg/assumptions.py`](pgmt/cfg/assumptions.py)（A1–A14），训练启动时 `dump()` 写入运行日志，最终报告逐项对照说明偏差。

## 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 环境与骨架（本机单测 + 服务器冒烟） | 本机部分完成；待下载 PP4 tar + 服务器冒烟 |
| M1 | LAFAN1 数据管线 + 重定向 | 未开始 |
| M2 | Stage 1 平地 tracking 预训练 | 未开始 |
| M3 | 地形系统（5 族 × L0–L9） | 未开始 |
| M4 | Stage 2 感知注入 | 未开始 |
| M5 | 基准评估 + 消融 | 未开始 |
| M6 | 基线与最终报告 | 未开始 |

## 参考

- 论文主页: https://luyili.github.io/pgmt/
- 基础代码参考: human2humanoid (OmniH2O, https://github.com/OpenRobotLab/human2humanoid)
- 数据: LAFAN1 (https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- Isaac Gym 兼容性调研: NVIDIA 论坛（PP4 × Blackwell 官方答复）、Isaac Lab Releases（Blackwell 支持自 2.1.1）
