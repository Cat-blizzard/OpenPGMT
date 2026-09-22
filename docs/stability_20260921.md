# 奖励尺度与 PPO 更新幅度：诊断与实现

本轮完成三次真实物理采样和固定 batch 上的优化器对照。
**以下四项现已接入正式训练代码，且从零完成 16 环境、20 次更新在线复验。**
本页保留修改前的诊断数据；在线结果另记于
[v3 复验报告](stability_validation_20260921.md)。暂不扩大训练。

## 建议采用的最小改动

1. **先减小初始动作的突变，保留 EE acceleration mismatch 的公式及 −1e−3 权重。**
   新策略的末层 bias 设置为默认站姿经过关节范围反变换后的 latent，末层 weight
   缩小到初始化值的 0.01 倍，初始 latent std 从 0.3 改为 0.1。
   映射仍为有界绝对关节目标，已有 checkpoint 加载时覆盖初始化参数。
2. **undesired_contact 从超阈值接触力平方和改为超阈值的非足部接触部位数。**
   保留 −0.1 权重、1N 阈值和独立的 head/torso impact 强度惩罚。
   不把 recovery 时所有惩罚关闭。接触计数是新的 A21 工程假设，不是论文公开公式。
3. **对每个 critic head 的 value loss 分别缩放。**
   第一版采用冻结于本次 rollout 的 `s_j=max(std(R_j),1)`，最小化
   `((V_j-R_j)/s_j)^2`；value clipping 也在同一缩放单位下计算。
   Actor 所用 GAE、bootstrap 和对外 value 保持原始 reward 单位；现有逐头 advantage
   normalization 保留。不裁剪整组 reward，不用一个全局缩放数混合所有 heads。
   每轮同时记录 raw value error 与 scaled loss，避免缩小 loss 数字掩盖拟合失败。
4. **在上述初始化下，PPO 初始 LR 用 1e−4，并加入整批 KL 检查。**
   以行为策略为固定参照，计算整个 rollout 上 latent Normal 的解析 KL（先对 29
   个动作维度求和，再对状态求平均）。相同可逆 tanh/affine 映射不改变该 KL。
   候选步超过 0.02 时撤回参数及 Adam 状态、减半 LR；接近预算（≥0.018）后结束
   本轮更新。5 epochs 是上限，不再强制跑满。保留 PPO clip=0.2 和 grad clip=1。

这组选择已接入正式实现并通过回归，随后进行 16 环境/20 更新的在线复验。
复验重点包括实际学习进展、raw value error、普通/恢复 episode 分项、接触冲击及
动作变化，不能只用“KL 小”和“进程正常退出”判定训练成功。

## 物理测量：EE 惩罚主要来自机器人的运动

使用 16 环境、24 个控制步（每个环境 0.48 秒）；刻意将 8 个环境设为普通 reset，
另 8 个设为真实 recovery reset。每次共 384 条 transition，两类各 192 条。
三种初始化的初始观测、历史、参考和 privileged 输入逐元素相等。

基线普通 transition 的末端加速度范数中位数：物理 **173.8m/s²**，参考 **3.66m/s²**。
相应加权平方能量分别约 −167.79 和 −0.65，实际 mismatch 为 −168.89。
因此本批数据不支持“参考加速度异常主导惩罚”的猜测，也不支持直接把该权重缩小
100 倍作为首选修复。reset 边界和参考导数仍可另查，本测量不是所有动作的证明。

| 初始化 | EE 加权均值（全 batch） | 普通 / 恢复 EE 均值 | 提前终止数 |
|---|---:|---:|---:|
| 原初始化，std=0.3 | −184.53 | −168.89 / −200.17 | 8 |
| 原初始化，std=0.1 | −67.02 | −51.19 / −82.85 | 8 |
| 默认站姿附近、末层 weight×0.01，std=0.1 | −25.77 | −16.40 / −35.13 | 0 |

这是初始化和动作平滑性的短期结果，不是训练成功率。恢复仍在 3 秒宽限期中，
所以 recovery 未终止不能当作成功起身。
候选初始化的恢复接触冲击仍然较大，原始总奖励并没有改善：约 −4250→−7567。
它降低了 EE 惩罚，却没有解决接触力平方项，后者必须单独处理。

## 接触惩罚：明确区分“发生接触”和“撞击强度”

在基线 recovery transition 中，当前 undesired_contact 加权均值约 **−8146**，
相同接触状态若按部位计数则约 **−0.0484**；站姿附近初始化时两者分别约
**−15102** 和 **−0.0958**。平方力把允许存在身体支撑的恢复过程变成极大的负回报。

论文 Table I 分别列出 undesired contact（−0.1）与 head/torso impact（−1e−5），
没有给出前者必须按牛顿平方计算的定义。用计数表达接触违规、用独立撞击项表达
强度，是本项目建议的明确解释，不能声称就是作者实现。
本机 IsaacLab v2.3.2 的 `envs/mdp/rewards.py::undesired_contacts` 也使用超阈值
接触数量；这仅提供实现参照，不构成 PGMT 作者采用它的证据。

## 固定数据上的 PPO 对照

各候选使用相同初始权重、完整 rollout、minibatch 顺序和种子。
接触计数候选在保存的 transition 上重算 reward/GAE，尚未产生新的在线学习轨迹。
这里报告的 KL 是整批状态上的解析平均值，与上一轮最后一个 minibatch 的采样
KL 估计不是同一个统计量，不能直接对比数值大小。

原初始化 batch：

| 更新配置 | 20 个梯度步后的整批 KL |
|---|---:|
| 原 PPO，LR=1e−3 | 19.786 |
| 只更新 actor，LR=1e−3 | 16.602 |
| 只更新 critic，LR=1e−3 | 0.01697 |
| 原 PPO，LR=1e−4 | 0.4958 |
| 原 PPO，LR=1e−5 | 0.02150 |
| 分头 value loss 缩放，LR=1e−4 | 0.6885 |

共享编码器的初始 critic / actor 梯度范数为 1332.9 / 0.353，确有尺度失衡；
但 actor-only 仍然大幅移动，说明不能把大 KL 全归因于 critic。
critic-only 的长期效果也不能由这 20 步推断。

默认站姿附近、std=0.1 的 batch：

| 更新配置 | 整批 KL | 实际接受的梯度步数 |
|---|---:|---:|
| 原 PPO，LR=1e−3 | 1.4886 | 20 |
| 接触计数 + value loss 缩放 + LR=1e−4 | 0.03416 | 20 |
| 上一行 + 整批 KL 约束/提前结束 | **0.01996** | **13** |

候选方案中 critic / actor 共享梯度范数约 0.0277 / 0.0102。
这支持将 LR、value loss 尺度和 KL 控制一起处理，但不证明长期价值拟合、恢复
或 tracking 已改善。只缩小 LR 也不能修正原始 reward 的语义。

## 可复查产物

- `setup/probe_training_stability.py`：采集与 CPU 固定 batch 对照入口。
- `runs/stability_20260921/batch{,_noise01,_neutral01}.pt`：物理 batch、初始 policy、逐步诊断。
- `runs/stability_20260921/initialization_comparison.json`：初始化对照及普通/恢复拆分。
- `runs/stability_20260921/comparison_v2.json`：原初始化 batch 的优化器对照。
- `runs/stability_20260921/comparison_neutral01.json`：候选初始化 batch 的优化器对照。
- 初版 `comparison.json` 也保留：它在 KL 达到上限后仍反复缩步，最终 LR 接近
  零。该方式已从候选方案排除；v2 在预算耗尽时结束本轮，避免无效更新。

## 正式实现的边界

- A4/A6/A21 统一记录新初始化、PPO 参数及接触计数假设。训练 checkpoint 使用
  `pgmt_count_contact_kl_v3` 契约；恢复和 Stage 2 初始化拒绝旧奖励版本。
- KL 约束是整批状态的 **平均** `KL(old || new)`，不保证每个状态都小于 0.02；
  另存 `exact_kl_max_state` 暴露状态间差异。`approx_kl_post` 仍是最后一个
  minibatch 上的采样估计，只作诊断，不作为更新接受条件。
- 每个候选梯度步最多回退 8 次；每次撤回参数以及 Adam moments/step。
  回退加载使用独立副本，避免 Adam 原位更新污染保存的回退点。没有可接受步时
  结束本轮，并记录实际接受步数（可以为零）。
- 每轮开始按原线性日程设置 LR，轮内超限才减半；下一轮重新从其日程 LR 开始。
  达到预算的 90% 后停止本轮，避免在剩余预算不足时反复缩步。
- value scale 固定于本轮 rollout，不改变 reward、return、GAE 或 bootstrap。
  同时保存所有头的原始 MSE、缩放 loss、explained variance 和实际 optimizer 步数。
- 普通/恢复 episode 按 reset 前的 recovery 标志拆分，记录奖励分项、跟踪误差、
  接触力峰值和已终止 episode 的时长。未结束 episode 不计入结束时长均值。
- 历史诊断脚本显式保留“力平方基线”采样模式，用于复查上方消融；它不是生产
  奖励配置。旧 JSON 是当时版本的结果，不能当作当前在线复验数据。

依据：[PGMT v2 §IV-A 与 Table I](https://arxiv.org/html/2609.08511v2)。
