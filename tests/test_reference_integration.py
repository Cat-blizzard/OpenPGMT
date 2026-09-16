"""M2 契约的"无仿真器"集成冒烟：真实参考数据 → 观测 → 策略前向。

这是 M2 环境的**前置验证**：在完全不需要 Isaac Gym 的前提下，把
"运动数据库 → C^K 参考帧 → H_t 历史 → o_t → HistoryEncoder/IFM/Actor"
这条链路真跑一遍。它能在本机（或服务器 CPU）暴露接口错配，而不必等到
搭好 `g1_env.py` 才发现。

与 `test_policy_integration.py` 的区别：后者用随机张量只验证**形状与梯度**；
本文件用**真实重定向序列**验证**数据语义**（e_t 在参考与机器人朝向一致时
为恒等、H_t 的时间顺序、C^K 的偏移递增等）。

需要 `data/raw/lafan1` 与 `data/processed/lafan1_g1`；缺失时整体跳过。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM
from pgmt.envs.observations import (
    OBS_LAYOUT,
    PRIV_DIM,
    HistoryBuffer,
    build_obs_from_state,
)
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.policy.actor import Actor
from pgmt.policy.history_encoder import HistoryEncoder
from pgmt.policy.ifm import IFM
from pgmt.policy.multi_head_critic import HEAD_ORDER_STAGE2, MultiHeadCritic
from pgmt.policy.rotation import quat_to_mat

NPZ_DIR = os.path.join("data", "processed", "lafan1_g1")
_HAS = os.path.isdir(NPZ_DIR) and any(
    f.endswith(".npz") for f in os.listdir(NPZ_DIR)) if os.path.isdir(NPZ_DIR) else False

pytestmark = pytest.mark.skipif(not _HAS, reason="需要 data/processed/lafan1_g1")


@pytest.fixture(scope="module")
def db():
    return MotionDatabase(NPZ_DIR)


def test_can_build_obs_from_real_reference(db):
    """o_t 可由真实参考帧 + 关节状态拼出，且各段落在正确位置。"""
    seq, t = 0, 30.0
    q, qd = db.ref_at(seq, t)
    ref_rot = db.ref_rot(seq, t)
    omega = np.zeros(3, dtype=np.float32)
    a_prev = np.zeros(ACT_DIM, dtype=np.float32)

    o = build_obs_from_state(robot_quat=ref_rot, ref_quat=ref_rot,
                             omega_t=omega, q_t=q, qd_t=qd, a_prev=a_prev)
    assert o.shape == (OBS_DIM,)

    # 机器人朝向 == 参考朝向 → e_t 必须是 6D 恒等
    assert np.allclose(o[OBS_LAYOUT.index("e_t")], [1, 0, 0, 0, 1, 0], atol=1e-5)
    # q_t / qd_t 段应逐字等于参考
    assert np.allclose(o[OBS_LAYOUT.index("q_t")], q, atol=1e-6)
    assert np.allclose(o[OBS_LAYOUT.index("qd_t")], qd, atol=1e-6)


def test_obs_e_t_is_valid_rotation_for_real_refs(db):
    """对一段真实轨迹，e_t 解回矩阵必须是合法旋转（det=+1、正交）。"""
    from pgmt.policy.rotation import rot6d_to_rotmat

    for t in np.linspace(0.0, 100.0, 11):
        ref_rot = db.ref_rot(0, float(t))
        # 让机器人朝向偏离参考：绕 z 转 0.4 rad
        w = np.cos(0.2)
        robot = np.array([w, 0.0, 0.0, np.sin(0.2)])
        q, qd = db.ref_at(0, float(t))
        o = build_obs_from_state(robot_quat=robot, ref_quat=ref_rot,
                                 omega_t=np.zeros(3, dtype=np.float32),
                                 q_t=q, qd_t=qd,
                                 a_prev=np.zeros(ACT_DIM, dtype=np.float32))
        R = rot6d_to_rotmat(o[OBS_LAYOUT.index("e_t")].astype(np.float64))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_ref_rot_trajectory_is_continuous_over_real_data(db):
    """整段真实轨迹上，相邻 1/50 s 的参考朝向变化必须有界。

    这是最短弧插值的验收：若半球对齐缺失，`q`/`−q` 翻转点会出现接近
    180–360° 的单步跳变。阈值取 30°（远大于真实运动在 20 ms 内的最大
    角速度，又远小于假跳变的量级）。
    """
    T = db.seq_len(0)
    worst = 0.0
    prev = db.ref_rot(0, 0.0)
    for k in range(1, min(T * 2, 400)):   # 控制频率扫描：每 0.5 源帧
        cur = db.ref_rot(0, k * 0.5)
        R = quat_to_mat(prev).T @ quat_to_mat(cur)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        worst = max(worst, float(ang))
        prev = cur
    assert worst < 30.0, f"参考朝向单步最大跳变 {worst:.2f}° 过大（疑 360° 假旋转）"


def test_history_over_real_reference_trajectory(db):
    """按控制频率推进后，H_t 应装满、时间正序、不含当前帧。

    时序（第一版写错的点）：循环里先 push 再读，所以读完 12 帧后
    `as_tensor()[-1]` 是**最后推入的那一帧** `obs_list[-1]`，而不是 `-2`。
    "不含当前帧"体现为：末行 == 最后推入的观测，且**下一个尚未推入的**
    观测不在其中。
    """
    hb = HistoryBuffer()
    dt_frames = 30.0 / 50.0  # 30 fps 源 → 50 Hz 控制
    obs_list = []
    n_steps = HISTORY_LEN + 2
    for k in range(n_steps):
        q, qd = db.ref_at(0, k * dt_frames)
        ref_rot = db.ref_rot(0, k * dt_frames)
        o = build_obs_from_state(robot_quat=ref_rot, ref_quat=ref_rot,
                                 omega_t=np.zeros(3, dtype=np.float32),
                                 q_t=q, qd_t=qd,
                                 a_prev=np.zeros(ACT_DIM, dtype=np.float32))
        obs_list.append(o)
        hb.push(o)

    assert hb.full
    H = hb.as_tensor()
    assert H.shape == (HISTORY_LEN, OBS_DIM)
    # 缓冲存的是最近 10 帧：obs_list[2..11]，末行 = obs_list[-1]
    assert np.allclose(H[-1], obs_list[-1], atol=1e-6)
    assert np.allclose(H[0], obs_list[n_steps - HISTORY_LEN], atol=1e-6)
    # 时间正序：每行都应能对上 obs_list 的对应元素
    for i in range(HISTORY_LEN):
        assert np.allclose(H[i], obs_list[n_steps - HISTORY_LEN + i], atol=1e-6), f"行 {i} 顺序错"


def test_history_excludes_the_frame_not_yet_pushed(db):
    """“不含当前帧”的直接验证：用 observe_then_advance 走真实轨迹。

    第 k 步的返回值必须**不含**本步的观测 —— 这是 H_t 定义的核心。
    """
    hb = HistoryBuffer()
    dt_frames = 30.0 / 50.0
    prev_obs = None
    for k in range(HISTORY_LEN + 2):
        q, qd = db.ref_at(0, k * dt_frames)
        ref_rot = db.ref_rot(0, k * dt_frames)
        o = build_obs_from_state(robot_quat=ref_rot, ref_quat=ref_rot,
                                 omega_t=np.zeros(3, dtype=np.float32),
                                 q_t=q, qd_t=qd,
                                 a_prev=np.zeros(ACT_DIM, dtype=np.float32))
        H = hb.observe_then_advance(o)
        if prev_obs is not None:
            # 末行必须是上一帧，绝不是本帧
            assert np.allclose(H[-1], prev_obs, atol=1e-6), f"第 {k} 步的 H_t 含入当前帧"
            assert not np.allclose(H[-1], o, atol=1e-6) or np.allclose(prev_obs, o, atol=1e-6)
        prev_obs = o


def test_full_policy_forward_on_real_reference(db):
    """真实数据驱动的完整策略前向（Stage 1 与 Stage 2），含梯度。"""
    B = 4
    dt_frames = 30.0 / 50.0
    obs_list, cs = [], []
    for b in range(B):
        seq = b % db.num_sequences
        q, qd = db.ref_at(seq, 10.0 + b * dt_frames)
        ref_rot = db.ref_rot(seq, 10.0 + b * dt_frames)
        obs_list.append(build_obs_from_state(
            robot_quat=ref_rot, ref_quat=ref_rot,
            omega_t=np.zeros(3, dtype=np.float32), q_t=q, qd_t=qd,
            a_prev=np.zeros(ACT_DIM, dtype=np.float32)))
        cs.append(db.future_refs(seq, 10.0 + b * dt_frames))
    o = torch.from_numpy(np.stack(obs_list))
    C = torch.from_numpy(np.stack(cs)).float()
    H = o.unsqueeze(1).repeat(1, HISTORY_LEN, 1)

    hist, ifm, actor = HistoryEncoder(), IFM(), Actor()
    s_hist = hist(o, H)
    s_int = ifm(s_hist, C)
    a = actor(o, s_int)
    assert a.shape == (B, ACT_DIM)

    # Stage 2：注入地形 token + 四头 critic
    from pgmt.policy.glimpse_encoder import TerrainGlimpseEncoder
    glimpse = TerrainGlimpseEncoder()
    M = torch.zeros(B, 21, 21)
    z, r, _ = glimpse(M, s_hist, C)
    s_int2 = ifm(s_hist, C, z)
    a2 = actor(o, s_int2)
    crit = MultiHeadCritic(priv_dim=PRIV_DIM, head_names=HEAD_ORDER_STAGE2)
    v = crit(torch.zeros(B, PRIV_DIM), s_int2)
    assert a2.shape == (B, ACT_DIM) and v.shape == (B, 4)

    (a2.mean() + v.sum() + z.sum()).backward()
    for name, mod in [("hist", hist), ("ifm", ifm), ("glimpse", glimpse),
                      ("actor", actor), ("critic", crit)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in mod.parameters()), f"{name} 未收到梯度"


def test_future_refs_offsets_increase_over_real_data(db):
    """C^K 的时间偏移必须递增（A2: τ = 2^k − 1），反映在参考量随 k 变化。"""
    C = db.future_refs(0, 20.0)
    assert C.shape[0] == 6
    # 关节位置在 k=4(τ=15) 与 k=5(τ=31) 上应不同（走/跑类动作 16 帧内必有变化）
    assert not np.allclose(C[4, :ACT_DIM], C[5, :ACT_DIM]), "远帧参考不应完全相同"


def test_ref_at_and_ref_rot_agree_on_sequence_bounds(db):
    """越界时刻：q 与 root_rot 都必须截断到末帧（两者行为一致）。"""
    T = db.seq_len(0)
    q_end, _ = db.ref_at(0, 10 * T)
    r_end = db.ref_rot(0, 10 * T)
    q_last, _ = db.ref_at(0, T - 1)
    assert np.allclose(q_end, q_last, atol=1e-6)
    assert np.isclose(np.linalg.norm(r_end), 1.0, atol=1e-6)
