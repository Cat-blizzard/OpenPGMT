# 执行器修正与异常诊断：CPU 阶段

2026-09-22。已完成资产对照、力矩配置候选、被动诊断、CPU 回归和手动物理验证入口。
此 CPU 阶段结束时尚未启动 GPU。随后用户授权的小规模物理对照已完成，见
[物理对照结果](actuator_physics_20260922.md)：首步冲击降低，但存活与跟踪未改善。
GPU 训练与后置评估仍未启动。

## 证据与改动

直接使用 Isaac Sim 随附的 `pxr` 库解析 USD；没有启动 SimulationApp 或加载物理场景。
29 个关节的名字、父子刚体、轴、角度边界、effort 和 velocity 与配套 URDF 一致。
这验证了所审查的执行器/运动学规格，不代表已对全部网格、惯量或其他机器人版本作等价证明。

| 关节组 | 数量 | URDF/USD 力矩上限 Nm | v4 配置 Nm | 新候选 Nm | 资产速度上限 rad/s |
|---|---:|---:|---:|---:|---:|
| 髋部、waist yaw | 7 | 88 | 120 | 88 | 32 |
| 膝 | 2 | 139 | 120 | 139 | 20 |
| 踝、waist roll/pitch | 6 | 50 | 120 | 50 | 37 |
| 肩、肘、wrist roll | 10 | 25 | 120 | 25 | 37 |
| wrist pitch/yaw | 4 | 5 | 120 | 5 | 22 |

默认配置改为 `asset_effort_v1`。训练入口及物理探针保留
`--actuator-profile legacy_uniform120`，用于新建同条件对照。
逐关节限值按规范名字映射到 Isaac 的运行时顺序；使用 `effort_limit_sim`。
CPU PD 适配器使用同一组力矩值，但其简化积分不能代替物理验证。

只改变力矩上限。kp=80 Nm/rad、kd=2 Nm·s/rad、零关节 reset、随机化、动作延迟、
奖励、终止、恢复课程和 PPO 规则保持原设置。USD 的默认 gain 换算后约为 kp=100、kd=1；
训练明确覆盖为 80/2，这属于独立复现的控制器选择，不因与资产默认不同就判为代码错误。
速度仍继承 USD，尚未作真实 PhysX 读取验收；没有把 45 rad/s 超速终止阈值提高。

USD 角度/角速度以 degree 为单位，力矩为 Nm，角驱动 gain 以每 degree 计。
因此角度/速度乘 π/180，gain 乘 180/π。单位来源：
[OpenUSD DriveAPI](https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html)、
[NVIDIA PhysxJointAPI](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_joint_a_p_i.html)。
当前 USD 的 metersPerUnit 和 kilogramsPerUnit 均为 1。

腕部 roll 的动作范围常量为 ±1.9722 rad，资产为 ±1.972222054 rad，相差约
0.000022054 rad，动作范围略窄；此次不修改动作边界或重定向数据。
首次审查使用 1e-6 rad 容差，因此这两项未通过；原始失败报告保留。
最终报告明确使用与四位小数精度对应的 5e-5 rad 容差，没有将真实资产差异掩盖为舍入。

审查文件（本地 runs，不提交外部资产）：

- `runs/actuator_diagnostics_20260922/actuator_audit_v2.json`：原始属性、转换值、逐项检查。
- `runs/actuator_diagnostics_20260922/actuator_audit_v2.csv`：29 行控制器对照表。
- 原始 `actuator_audit.json` / `.csv`：保留的初次严格容差结果。
- URDF SHA256：`0dc078a30278f5e51f1cdffe30e2f898e6654c60e2c25a0431891b7b45475e48`。
- USD SHA256：`ccb3de2c206fb331106c8f3931045817820b95dc541c94d3e702931db1d7e3d7`。

## 诊断记录

Stage 1/2 通过 `--diagnostics-dir <新目录>` 显式开启。默认每类最多 4 个事件，
每事件最多 4 个状态，同类最少间隔 25 个控制步或 PPO 更新。
同一目录重新开启也计入已有文件，不覆盖事件。记录器不抽随机数，不改变环境或优化器。
KL 样本选择分块保留最严重的有限个状态，不复制整批观测。

| 事件 | 保存内容 |
|---|---|
| actuators | PhysX effort/velocity/kp/kd、规范名字到运行时索引、完整环境配置；effort 不匹配则报错 |
| first_action | reset 后且动作应用前的 q/qd/root/速度/历史/队列，及首个策略动作、延迟后的实际目标 |
| joint_speed | 超过既有 45 rad/s 阈值的关节名对应 mask、q/qd/target、力矩估计及估计饱和比例 |
| ee_acceleration | 四末端分别记录机器人/参考加速度、误差、当前/上步速度、dt、序列/帧、episode 年龄 |
| kl | 单状态 KL > 0.1 时保存观测、Normal 的 old/new 参数、逐关节 KL、普通/恢复标签、rollout 索引、更新前后模型 |

加速度触发值为任一末端误差范数 > 100 m/s²。0.1 与 100 仅为记录触发值，
不是新增的优化约束或终止阈值。`first_action` 的 before 位于该步动作执行前，
此时 reset 观测已构建，历史缓存是否为零以实际记录为准。

Isaac 隐式执行器的 `applied_torque` 是最后物理子步的 PD 估计，**并非求解器测量力矩**。
快照保留明确的 `torque_source` 标签；饱和比例同样只是估计，不能据此声称测得真实力矩。
来源为本机 IsaacLab `ImplicitActuator.compute` 实现；没有为获取测量值启动 GPU。

CPU 重放命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/jxc/envs/pbfm_isaaclab/bin/python -m setup.replay_kl_diagnostic \
  <diagnostics目录>/kl_000.pt
```

重放校验该事件的模型输出与解析 KL，报告观测尺度和误差。
它不是原始 minibatch/Adam 更新轨迹的重演；完整优化消融仍需要额外保存训练批次。
v4 seed 1 第 179 轮的历史异常没有样本快照，仍无法从聚合日志准确还原。

## CPU 验证

- 资产审查：29 个关节所有明确列出的静态检查通过；`runtime_verified=false`。
- 针对性环境/PPO 回归：34 passed。
- 完整测试：806 passed、2 skipped，47.69 秒。日志 `runs/actuator_diagnostics_20260922/cpu_regression.log`。
- Stage 1 和 Stage 2 诊断开/关：相同 seed、两次更新，模型、训练指标和 Torch RNG 逐位一致。
- KL 事件在 CPU 上重放通过；故意修改保存的 Normal 参数会使重放校验失败。
- 覆盖反序运行时关节映射、错误限值拒绝、跨启动数量限制、首步延迟、reset 前异常状态保留。
- 手动 GPU 入口只检查了语法、导入和 `--help`，未执行物理模拟。
- 分块限量采集修改后的最终诊断回归：8 passed，3.33 秒，日志 `cpu_diagnostic_final.log`。
- 保存了 `cpu_replay_example/kl_000.pt` 和 `replay_result.json`：mock 样本输出误差为零。
  此例降低记录阈值并使用人工分组标签，仅验证链路，不是新捕获的物理异常。

## 下一次手动物理对照

`setup/run_actuator_check.sh` 每次只运行一个实验，结束即退出，不串联评估。
先按当前机器状态选 GPU UUID 与 Vulkan index，不能把上一轮的空闲状态视为当前授权。
调用格式：

```bash
PGMT_RENDER_GPU=<当前Vulkan索引> bash setup/run_actuator_check.sh \
  neutral legacy_uniform120 <已安排的GPU_UUID> <新的输出目录>
```

按以下顺序分别执行并审查结果：

1. `neutral`：4 环境、150 控制步、无学习、普通 reset；两种 profile 同 seed=17 对照。
2. `reference`：同条件直接参考目标播放。边界投影幅度显式记录；播放失败不能单独归因于 PPO。
3. 配置读取正确且异常可解释后，再执行 `ppo`：16 环境 × 24 步 × 20 更新，即 7680 transitions。
   完整参考库、既有 256 状态离线 pool、critic completion 开启、LR horizon=1000，两种 profile 同 seed。

主要看超速关节/时点、末端加速度的来源、力矩估计饱和，以及短 PPO 是否有限、每轮是否有有效
actor 更新。随后才决定是否研究 PD 或参考 reset，并另作单变量实验。
此入口尚未加入参考状态 reset 候选；待力矩对照结果后再实现，避免同时改变多个因素。

新的 checkpoint 保存实际环境配置，resume 会拒绝悄悄更换力矩、PD 或默认关节姿态。
评估结果身份新增 actuator profile 与执行器实现 hash，不允许混接不同物理条件的输出行。
v4 的最终评估继续使用 `learning_source_snapshot.tar.gz` 冻结源码，与原 initial 基线配对；
仅选择 legacy profile 不等于恢复完整旧协议。原始训练、pool、checkpoint 和延后评估脚本均保留。

## 论文复现边界

这是已核实资产规格的工程修正，不能称为论文作者的执行器配置，也不能承诺带来学习收益。
保留论文的关节位置目标 + PD、共享 IFM、多头 critic、奖励权重与两阶段流程。
当前不加恢复辅助外力，不放宽终止，不改奖励尺度或 KL 预算。
正式扩规模 Stage 1 仍等待物理和学习验收。
