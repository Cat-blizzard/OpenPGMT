"""跨层共享的维度常量（策略层与环境层的唯一出处）。

此前 `_OBS_DIM = 96` 在 `pgmt/policy/history_encoder.py` 与
`pgmt/policy/actor.py` 各硬编码了一份，`_ACT_DIM = 29` 亦同 —— 一旦观测
布局变化，两处会静默不一致。

本模块放在 `pgmt/` 顶层（不依赖 `pgmt.envs` / `pgmt.policy`），
使策略层可以引用共享维度而不必反向依赖环境层。

96 维的内部分段定义见 `pgmt/envs/observations.py` 的 `OBS_LAYOUT`；
`tests/test_observations.py` 校验两者一致。
"""

from __future__ import annotations

#: G1 29-DoF 关节数（= 动作维度 = `q_t`/`qd_t`/`a_prev` 维度）
ACT_DIM = 29

#: `o_t` 维度（论文 §III）：e_t(6) + ω_t(3) + q_t(29) + q̇_t(29) + a_{t−1}(29)
OBS_DIM = 6 + 3 + ACT_DIM + ACT_DIM + ACT_DIM  # 96

#: `H_t` 帧数（论文 §III：ten-frame proprioception history）
HISTORY_LEN = 10

#: `C^K` 每帧维度（论文 Eq.1）：q^r(29) + q̇^r(29) + ṽ^r(3)
REF_FRAME_DIM = ACT_DIM + ACT_DIM + 3  # 61
