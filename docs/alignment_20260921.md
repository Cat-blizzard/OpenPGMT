# 新版复现流程（2026-09-21）

旧的 `stage1_scale_20260920_gpu8_anchored` 实验停在 842/1000 更新，按用户要求归档。
不再续跑，也不用于 Stage 2 初始化；原 checkpoint 保留用于问题诊断。
旧续跑脚本已禁用。新版 PPO checkpoint 使用 `joint_targets_tanh_v2` 版本，拒绝旧动作接口。

## 实现变化

- Actor 包装器输出 **29 个有界绝对关节目标（rad）**，由 URDF 对应的关节限位仿射映射
  `tanh(z)` 得到。rollout 保存 `z`，log-prob 包含 tanh 与缩放的 Jacobian；饱和时
  不从 float32 目标角度反求 `atanh`。环境不再额外裁剪/缩放目标。探索初始标准差
  为 latent 空间 0.3，log-std 限制在 [-5, 1]，熵使用当前分布重参数采样估计。
- 正式 Isaac Stage 1 必须提供非空、带物理采集来源的 recovery pool。池内姿态、
  序列顺序和 pose fingerprint 绑定数据；Isaac 分支实际接收该池并更新成功率及
  自适应采样统计。起身参考保持正常朝向/高度，恢复前 3 秒免除低高度、倾斜和
  参考偏差终止。mock/fallback 的普通 episode 不再得到 recovery 常数奖励。
- `collect_fall_pool` 在真实物理中用零刚度、弱阻尼及初始速度扰动采集跌倒状态，
  不训练 policy。仅保留低高度、倾斜且速度有限的状态，附 URDF/资产路径、种子及
  参考数据指纹。未执行此采集前，不会假造一个“已准备好”的 pool。
- Stage 2 把 `TerrainAtlas.mesh()` 写入 Isaac 的全局碰撞网格，移除覆盖粗糙地形
  凹处的平面。机器人 reset 到对应 family/level 的实际世界原点并加地表高度；
  高度查询与碰撞使用相同的三角形插值。5 类地形按环境轮流分配，成功升一级、
  失败降一级；阈值与延迟同步更新，采样接入已有 A11 粗兼容规则。
- 物理场景每个 family/level 默认占 32m×32m：将原 4m 模块重复 8×8，网格分辨率
  0.1m。几何保持可查，台阶边缘是一个网格宽的坡面。这些是本项目选择，不是
  作者参数。越出被分配区域按失败计，并单独报告；可配置更大的区域，禁止将
  跨到另一族/难度的运动当作原分配的成功 episode。
- 正式 Stage 2 必须从新版物理 Stage 1 checkpoint 或 Stage 2 resume 启动。
  从零开始必须明确指定 `--allow-scratch-ablation`；mock 流程只验证协议。
- `data.build_mesh_contacts` 根据 URDF 足部碰撞球和参考地形网格离线查询标签，
  不再使用足速阈值。当前显式采用 **z=0 平面参考网格**（作者未提供参考地形）；
  也支持外部规则高度场 NPZ。2cm 接触容差是实现假设。标签来自参考场景，
  不随独立采样的 Stage 2 地形更换，从而保持动作与地形不配对的设定。
- 动力学随机化实际写入 PhysX：骨盆质量/惯量/COM、碰撞材质摩擦、PD 增益；
  critic 读取对应已应用的数值。动作延迟为 0–2 个控制步，噪声作用于实际观测
  和高度图。范围见 `pgmt/envs/randomization.py`，并随 checkpoint/评测清单保存。
  摩擦离散为 32 档，限制持续 reset 带来的 PhysX 材质数量。
- 两阶段逐轮保存指标与 checkpoint，使用临时文件原子替换；保留分头奖励、
  termination/timeout 计数。保存随机状态、课程、延迟队列、恢复池统计和仿真
  可见姿态。PhysX 接触求解器内部缓存不能序列化，恢复不保证逐比特复演。

## 对此前判断的更正

单凭“Gaussian 动作在环境中裁剪”不能证明 PPO off-policy：如果 rollout 保留
原始采样和对应 log-prob，裁剪可以作为环境的确定性映射。本次修订的依据是
明确目标角度范围、消除多对一裁剪和改善饱和处数值处理，不是已经证明原 PPO
公式错误。论文没有指定 tanh 分布；这是复现中的可审查选择。

## 匹配评测

`pgmt.eval.manifest` 固定每个 episode 的动作序列、起始帧、family/level、随机种子、
30s horizon、随机化配置、数据文件与 G1 USD/URDF 的 SHA-256。
`eval/manifests/matched_9600_v2.json` 含 5×10×192=9600 集，其中 L9 共 960 集。

`pgmt.eval.run` 只使用 IsaacLab，不提供可误认为物理结果的 mock 路径。
逐集写 JSONL，支持同 checkpoint/manifest 恢复；报告整体、各格和 L9 completion、
Wilson 95% 区间、joint/body/root RMSE、contact F1。失败 episode 保留在分母内，
RMSE 以实际存活时间内的逐坐标均方误差统计。它是我们的明确口径，不能认为
所有度量细节都已从论文还原。body RMSE 在各自 root yaw 坐标系中计算。

评估固定为单环境串行，确保其他 episode 的早停不改变随机流；速度低于批量评估。
只采样兼容且还剩至少 30s 的源帧，筛选规则写入 manifest；评测不运行升降级课程。
不同策略使用同一 manifest、同一 reset/噪声流和同一物理配置，Stage 1 策略也在
Stage 2 环境中评测，只忽略其高度图。消融模型和论文数值复现尚未完成。

## 已生成的数据与验证

- `data/processed/lafan1_g1_mesh_v2`：77 个文件；qpos/qvel/root pose/frame_time
  与 anchored 数据逐数组一致，原启发式 contacts 另存为 `contacts_heuristic`。
- 993,344 个足-帧标签中 129,651 个改变；旧/新 contact 比例分别约 8.88%/9.01%。
  这是标签定义变化统计，不是接触准确率证明；容差和参考网格仍需物理验收。
- CPU mock 的 Stage 1→Stage 2 参数迁移及各 2 次 PPO 更新通过。
- 使用全部 77 个新参考文件的 torch Stage 1/Stage 2 各 1 次更新通过，Stage 2
  同时覆盖 5 个地形 family；这些结果均保存在 `runs/revised_v2_cpu_smoke`。
- 新增回归覆盖有界分布/Jacobian/饱和概率、延迟执行、网格插值与 reset 原点、
  升降级、恢复参考和宽限期、随机化写入、状态恢复、接触标签和评测分母。
- 完整 CPU 回归：**782 passed / 2 skipped**（两项 CUDA 用例因本机 GPU 0 requires-reset 而在 CPU 检查中跳过）。
- **后续实测更新**：16 状态物理池已采集，新版 Stage 1/2 各 2 次物理 PPO 更新
  和机制诊断已完成。50 个碰撞地形格子、随机化读回、动作延迟、恢复宽限期及
  课程 reset 检查通过；但 KL 与辅助惩罚尺度异常，**暂不扩大训练**。
  GPU 0 故障通过 GPU 9 UUID 隔离及显式 Vulkan 映射绕开，未重启或终止其他任务。
  结果、适用范围和精确命令见 [物理验收报告](physics_validation_20260921.md)。

## 下一次运行顺序

先用 `setup/run_revised.sh` 显式选择步骤与空闲 GPU。脚本检查占用，记录完整命令、
日志和退出码；请在 tmux 会话中运行，避免终端断开中断进程。
该通用脚本要求未隔离且健康的 GPU 枚举。本次故障机器上的 UUID/Vulkan 映射
采用单独记录的命令，详见物理验收报告及 `commands.json`。

```bash
# 以下三个 smoke 步骤已完成；产物保留，不能覆盖重跑。
bash setup/run_revised.sh collect-smoke 8   # 采集 16 个真实跌倒状态
bash setup/run_revised.sh stage1-smoke 8    # 新接口下的 8 环境 / 2 更新
bash setup/run_revised.sh stage2-smoke 8    # 5 地形 / 2 更新，来自新的 Stage 1 smoke

# 当前训练稳定性未通过；修复并复验后才考虑以下正式步骤：
bash setup/run_revised.sh collect 8         # 2048 个真实跌倒状态
bash setup/run_revised.sh stage1 8          # 从零训练，256 环境 / 1000 更新
bash setup/run_revised.sh stage2 8          # 来自新版 Stage 1
bash setup/run_revised.sh eval-stage1 8
bash setup/run_revised.sh eval-stage2 8
```

256 环境/1000 更新仍是工程规模，不能等同论文训练预算。以上脚本不会自动进入
下一个阶段。已有输出的步骤不会被覆盖；需要续跑时应指定新的输出目录并显式
传入新版 checkpoint。原 842-update checkpoint 不在任何新命令中。

其他机器重新生成数据和清单：

```bash
python -m data.build_mesh_contacts --input data/processed/lafan1_g1_anchored \
  --output data/processed/lafan1_g1_mesh_v2 --urdf /path/to/g1.urdf
python -m pgmt.eval.manifest --reference-data data/processed/lafan1_g1_mesh_v2 \
  --asset /path/to/g1.usd --urdf /path/to/g1.urdf --output eval/manifests/local_v2.json
```

来源：[PGMT v2](https://arxiv.org/html/2609.08511v2)、
[有界高斯策略的变换公式](https://spinningup.openai.com/en/latest/algorithms/sac.html)。
物理 API 按本机 IsaacLab v2.3.2 源码核对，本轮已完成上述范围的实际新路径检查。
