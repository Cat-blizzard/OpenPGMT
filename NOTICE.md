# NOTICE — 来源、许可与免责声明

本仓库是论文 **PGMT: Perceptive General Motion Tracking for Humanoid Robots**
（arXiv:2609.08511v2）的**独立复现**，由第三方（[@Cat-blizzard](https://github.com/Cat-blizzard)）
自行实现。

## 与论文作者的关系

本仓库**与论文作者无关**，且**未获得其官方代码** —— 论文主页与 arXiv 均未公开实现，
本项目完全依据论文正文与插图自行编写。任何实现细节上的偏差都属于本项目的责任，
不代表原论文的方法或结果。

论文的版权归原作者所有。**本仓库不包含论文 PDF 或全文文本**；引用请以 arXiv 为准：
<https://arxiv.org/abs/2609.08511>

## 第三方资源与许可

| 资源 | 用途 | 许可 / 来源 |
|---|---|---|
| **LAFAN1**（Ubisoft LaForge Animation Dataset） | 训练与评估用的运动捕捉数据（Mixamo 骨骼 BVH） | 研究用途免费。见 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset> |
| **human2humanoid**（OmniH2O，LeCAR-Lab） | 重定向思路与参照奖励约定的参考；`data/raw/.external/human2humanoid/` 下的文件为其上游内容 | **CC-BY-NC-4.0**（非商业） |
| **Unitree G1** 描述文件（`data/raw/g1/*.xml`） | 机器人运动学/关节限位 | Unitree Robotics 公开发布 |
| **ProtoMotions G1 asset**（服务器外部 `protomotions/data/assets/{usd,urdf,mjcf,mesh/G1}`） | 完整 USD/URDF/MJCF 与网格候选 | ProtoMotions 根目录 NVIDIA 非商业研究/评估许可；`mesh/G1/LICENSE` 为 Unitree BSD-3-Clause；不随本仓库重新分发 |
| **Isaac Gym Preview 4** | 训练用仿真器（需自行从 NVIDIA 下载，**不在本仓库内**） | NVIDIA 许可 |

### 本仓库不包含的内容（需自行获取或生成）

- **LAFAN1 原始数据**（`data/raw/lafan1/`、`lafan1.zip`，约 144MB）
- **重定向后的训练数据**（`data/processed/lafan1_g1/`，77 个 `.npz`）
- **完整 G1 USD/URDF/MJCF 与网格**（第三方二进制，需由使用者自行取得并保留两份上游许可）

这两项体积较大且受上述许可约束，未纳入版本控制。生成方式见
[README「数据准备」](README.md#数据准备)。

`data/processed/quality_report.csv` 与 `eval/viz/*.png` **包含在内** —— 它们是本
项目自身的评估产物（质量报告与骨架对比渲染），用于佐证重定向质量。

## 许可

本仓库**自行编写**的代码与文档以 [MIT License](LICENSE) 发布。

注意：该 MIT 授权**不覆盖**上表中的第三方资源，也不覆盖由 LAFAN1 衍生的数据 ——
若你重新生成 `data/processed/lafan1_g1/`，其使用仍受 LAFAN1 的许可约束
（研究用途、需署名、非商业）。

## 免责声明

本仓库为学术复现工作，**不提供任何担保**。用于真实机器人前请自行评估安全性。
