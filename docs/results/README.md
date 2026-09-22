# 2026-09-22 进度同步：图表与汇总

此目录提供可在 GitHub 上直接查看的实验统计。原始逐步日志、检查点、参考数据、
机器人资产和冻结源码快照保留在服务器的各实验目录，不包含在这里。
数字来自已完成实验，不代表已通过正式扩规模或论文匹配评估。

| 实验 | 范围与结论 | 图表 | 数据 |
|---|---|---|---|
| [三种子站立学习与参考 v4](../standing_multiseed_20260922.md) | 六场各 100 更新，共 230400 transitions，600 次 CPU 复核通过；训练未完成 10 s，匹配评估后置，GPU 已释放；v4 支撑改善但继续隔离 | [训练 PNG](standing_multiseed_20260922.png) / [v4 PNG](reference_support_v4_20260922.png) | [训练 JSON](standing_multiseed_20260922.json) / [v4 JSON](reference_support_v4_20260922.json) / [验证](standing_multiseed_verification_20260922.json) |
| [同动作回放与参考 v3](../standing_replay_reference_20260922.md) | 9600 transitions、0 PPO，峰值与跌倒触地同步；独立 v3 部分改善，支撑和插值仍未通过 | [回放 PNG](standing_replay_20260922.png) / [v3 PNG](reference_support_v3_20260922.png) | [回放 JSON](standing_replay_20260922.json) / [v3 JSON](reference_support_v3_20260922.json) / [验证](standing_replay_reference_verification_20260922.json) |
| [有界站立学习与八片段病因](../standing_learning_20260922.md) | 两条件各 20 更新，共 15360 transitions，接口复核通过；踝子步峰值及参考支撑问题仍在，GPU 已释放 | [PNG](standing_learning_20260922.png) | [训练 JSON](standing_learning_20260922.json) / [速度事件](standing_speed_events_20260922.json) / [参考病因与 v2](reference_causes_v2_20260922.json) / [验证](standing_learning_verification_20260922.json) |
| [资产一致参考与载荷候选](../asset_standing_validation_20260922.md) | 新 8 片段仍需质量修正；四组 1596 transitions，固定 PD 平均首次存活 1.47→2.29 s，仍 0/4 完成 10 s | [PNG](asset_standing_validation_20260922.png) | [JSON](asset_standing_validation_20260922.json) / [导数验证](reference_derivatives_20260922.json) / [最终验证](asset_standing_verification_20260922.json) |
| [v3 稳定性复验](../stability_validation_20260921.md) | 16 环境 × 20 更新，数值检查通过，行为未达标 | [PNG](stability_v3_20260921.png) | 详细统计见报告 |
| [v4 课程与 critic](../curriculum_critic_20260921.md) | 3 种子 × 64 环境 × 200 更新，critic 改善，控制行为未一致改善 | [PNG](curriculum_v4_20260921.png) | [训练内汇总 JSON](curriculum_v4_20260921.json) |
| [执行器对照](../actuator_physics_20260922.md) | 8 场无学习物理运行，新上限生效，首步冲击降低 | [PNG](actuator_comparison_20260922.png) | [JSON](actuator_comparison_20260922.json) / [CSV](actuator_comparison_20260922.csv) |
| [reset 对照](../reset_physics_20260922.md) | 6 场无学习物理运行，参考初始化降低冲击，存活与身体跟踪未一致改善 | [PNG](reset_comparison_20260922.png) | [JSON](reset_comparison_20260922.json) / [逐 episode CSV](reset_episodes_20260922.csv) |
| [固定片段 PPO](../fixed_clip_ppo_20260922.md) | 两种 reset 各 16 环境 × 20 更新；数值检查通过，真实首步冲击和左踝超速仍需定位 | [PNG](fixed_clip_ppo_20260922.png) | [JSON：配对校验、逐轮数据和事件摘要](fixed_clip_ppo_20260922.json) |
| [PhysX 子步修正](../physics_learning_20260922.md) | 七场同动作回放，原始速度峰值 195.42→39.01 rad/s；登记 A23 | [PNG](substep_physics_20260922.png) | [JSON](substep_physics_20260922.json) |
| [单动作学习](../physics_learning_20260922.md) | 三种子 × 16 环境 × 100 更新；匹配短时检查未显示学习收益 | [PNG](simple_learning_20260922.png) | [配对 JSON](simple_learning_20260922.json) / [验证记录](physics_learning_validation_20260922.json) |
| [参考、静态支撑与真实 rollout](../reference_control_audit_20260922.md) | 8 片段 CPU 复算；三组静态控制共 860 transitions，固定 PD 基准 1.50 s；GAE/CPU 更新核验通过 | [PNG](reference_control_audit_20260922.png) | [JSON](reference_control_audit_20260922.json) / [参考放置核查](reference_placement_audit_20260922.json) |

早期无学习物理对照每场 4 环境、3 秒，主要统计首次 episode；跟踪误差使用每对环境共同存活的时间段。
不能把这些短时播放结果作为经过训练的策略分数或 30 秒成功率。v4 最终策略评估继续后置。
固定片段 PPO 图表为单种子的训练内统计，5 秒片段，没有执行冻结策略的匹配评测。
此前单动作诊断单独完成 initial/final 冻结策略的 5 秒上限检查；它仍不是 v4 的
30 秒评估或全库泛化结果。确定性同起点的 16 个环境副本不能视为 16 个独立场景。
参考核查见[单片段审查](single_clip_reference_20260922.json)和[候选限位占比](fixed_clip_limits_20260922.json)。
此前静态控制每组最多 4 环境 × 10 s，所有首次 episode 结束即退出；实际各 300/296/264
transitions，均未达到预定静态门槛，当时后续物理学习未启动。一次 CPU PPO 更新只核验采集样本，
不部署策略、不代表行为改善。三组共享一个种子和四个指定起点，不能当作独立多种子结果。

先前同步的图片直接复制既有产物；CSV 仅统一换行符，JSON 仅选取聚合字段，原始数值不变。
[发布来源清单](publication_manifest_20260922.json)记录服务器来源、选取字段及 SHA256；
[reset 验证记录](reset_validation_20260922.json)对应上一次实验完成时的验证状态，
包括 813 项 CPU 回归通过、2 项跳过，以及随后 6 项分析检查通过。
新增固定片段 PPO JSON 内含本轮证据及分析源码 SHA-256；图表由同名分析器从原始逐轮指标生成。

当前决策：保留逐关节上限与默认 reset，候选参考初始化作为显式实验开关。
此前载荷候选四组分别 300/300/500/496 transitions，无训练更新。
此前两组站立学习共 15360 transitions，平均已结束 episode 时长 1.268/1.532 s，
但并未执行冻结策略前后评估，不能宣称训练收益。每环境总预算 9.6 s，真实 10 s timeout 不可达。
20 更新及其后的 9600 transitions 回放已完成；轨迹/读回一致，峰值与触地同步。
随后六场站立训练已完成：原目标/候选逐种子平均 episode 时长的均值约 1.251/1.403 s，
均没有 10 s timeout，尚无一致训练内改善；不能据此替代冻结策略前后评估。
新 v4 联合修正根/腿支撑，50 Hz 接触查询一致，仍有残余支撑/连续性及物理验证问题，继续隔离。
匹配评估的 24 个任务已完成 CPU 准备，实际物理评估继续后置。
安排见[下一阶段计划](../next_steps_20260922.md)。
