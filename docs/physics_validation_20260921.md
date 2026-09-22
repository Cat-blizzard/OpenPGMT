# 新版小规模物理验证（2026-09-21）

**结论：本轮覆盖的物理接口通过；训练稳定性不通过，暂不扩大 Stage 1。**
使用全部 77 个 mesh-contact 参考序列、真实 G1 USD 和 IsaacLab/PhysX。
旧 842-update checkpoint 未参与本次实验。没有启动正式训练或 9600 集评测。

## 实测范围

| 检查 | 配置 | 结果 |
|---|---|---|
| 跌倒池采集 | 8 环境，3 批，每批 3 秒零刚度沉降 | 16 个合格状态；root z=0.058–0.115m，最大线速度 0.021m/s |
| Stage 1 PPO | 从零开始，8 环境 × 24 步 × 2 更新 | 384 条 transition，24 次提前终止，6 个 recovery transition；保存完整 checkpoint |
| Stage 2 PPO | 从本轮 Stage 1 smoke 初始化，5 环境 × 24 步 × 2 更新 | 240 条 transition，14 次提前终止，真实地形网格与四头奖励运行完成 |
| 恢复机制 | 强制 8 环境从真实跌倒池 reset，180 个控制步 | fallen pose 写入 PhysX；参考 root z=0.793m；前 149 步无终止，第 150 步起允许终止 |
| 动作接口 | 两阶段各检查采样/重算 log-prob 及实际延迟 target | log-prob 误差 0；2 步延迟后，articulation target 与原目标逐元素相等 |
| 动力学随机化 | 从 PhysX 读回质量、COM、摩擦和关节刚度 | 与 critic 对应参数一致；Stage 1 质量比例 0.932–1.075，PD 比例 0.852–1.148 |
| 碰撞地形 | 5 族 × 10 级，600 次 PhysX raycast | 全部击中 `/World/terrain`；与高度查询最大误差 0.0000425m；含负高度，无覆盖 GroundPlane |
| 地形课程 | 人工触发 timeout 和越界失败 | 真实 auto-reset 后 L0→L1→L0；写回位置与 tile 坐标一致 |
| 机器人接触 | 5 个 L9 地形内部，160 个物理步 | 每族都检测到真实接触力；参考采样通过 compatibility 筛选 |

上表中的 timeout/失败是为了验证课程接线而人工触发，不能作为策略成功率。
恢复检查使用默认关节目标，不代表策略已学会起身。16 状态的池仅供 smoke 使用，
尚未采集正式 2048 状态池。匹配评测入口与 reference contact 标签准确率不在本轮验收范围内。

## 未通过的训练稳定性

| 指标 | Stage 1 更新 1 | Stage 1 更新 2 |
|---|---:|---:|
| 更新后采样 KL 估计 | 97.46 | 3228.95 |
| clip fraction | 0.931 | 0.835 |
| upper / lower 平均奖励 | 2.045 / 2.199 | 1.897 / 2.023 |
| auxiliary 平均奖励 | −204.13 | −4432.45 |
| auxiliary value loss | 1.83×10⁶ | 2.78×10⁹ |

Stage 2 的更新后采样 KL 为 8807.97、21.39，同样不支持扩大训练。
这些 KL 是当前实现对最后一个 minibatch 的估计，不是完整分布的精确 KL。
所有 PPO loss/gradient 和保存的策略参数保持有限；**有限且正常退出并不等于训练稳定**。

为定位原因，新增加权辅助奖励均值/峰值与动作饱和比例日志，使用相同 seed 和参数
重跑 Stage 1 两次更新。原有更新指标完全重现，分项结果为：

- 第 1 轮 `ee_accel_mismatch` 加权均值 **−203.91**，主导 auxiliary。
- 第 2 轮 `undesired_contact` 加权均值 **−4195.79**，单个 transition 的绝对峰值
  **287678.53**；`ee_accel_mismatch` 均值仍为 −236.06。
- 两轮 `|latent|>3` 的比例均为 **0**，没有接近 tanh 边界的动作。
- 第 2 轮共有 6 个 recovery transition；当前非足部接触项按超阈值力的平方惩罚，
  恢复接触的具体贡献还需要按 episode 类型拆分，不能仅凭本表证明全部来自恢复样本。

下一步应先核对参考/物理末端加速度的量纲、差分与尖峰，以及非足部接触惩罚在
恢复过程中的定义和尺度；再用短对照实验检查更小学习率和 KL 约束。
目前没有修改奖励公式、权重或 PPO 默认学习率来让本轮结果表面通过。

## 运行环境与本轮代码调整

本机 GPU 0 requires-reset 导致未隔离的 CUDA 初始化失败。经实际 Kit 表确认，
物理 GPU 9 的 UUID 对应当前 Vulkan index 8；使用
`CUDA_VISIBLE_DEVICES=GPU-d86528b1-67d4-d67f-db2c-ccae580a1267`、`--device cuda:0`、
`PGMT_RENDER_GPU=8` 完成上述运行。编号仅适用于本机当前状态，不应复制到别的机器。
GPU 9 原有约 600MB 的轻负载任务保持运行；未终止其他任务或重置设备。

新增 `pgmt/envs/isaac_app.py` 统一单 GPU 启动，并允许显式 Vulkan index；不设置
`PGMT_RENDER_GPU` 时保留 AppLauncher 默认选择。渲染间隔与控制 decimation 一致。
首次探针错误选择 Vulkan 9 的日志保留为 `device_probe.log`，不计入通过结果；
后续六个采集、训练、诊断进程均正常退出，exit code=0。

新增 `setup/validate_revised_physics.py` 可独立重做机制检查，并逐项写入结果。
相关 CPU 回归本轮为 **21 passed**；修改前完整 CPU 基线为 782 passed / 2 skipped。

## 本机产物

- `runs/revised_v2/physics_validation/verification.json`：汇总，`ready_for_scaled_training=false`。
- `runs/revised_v2/physics_validation/commands.json`：完整 argv、环境变量及超时设置。
- `runs/revised_v2/physics_validation/stage{1,2}_checks.json`：物理诊断细项。
- `runs/revised_v2/collect-smoke/fall_pool.pt`：16 状态物理池及数据指纹。
- `runs/revised_v2/{stage1-smoke,stage2-smoke,stage1-trace}/`：日志、metrics、checkpoint、退出码。

本轮 checkpoint 全部仅用于接口诊断，不能当成已经训练好的 Stage 1 prior。
