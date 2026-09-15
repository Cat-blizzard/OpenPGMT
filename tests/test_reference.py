"""reference_sampler.py：C^K 采样 / A13 修正速度 / A16 adaptive sampling。"""

import numpy as np
import pytest

from pgmt.cfg.assumptions import get
from pgmt.envs.reference_sampler import (
    AdaptiveSampler,
    MotionDatabase,
    correct_anchor_velocity,
)


@pytest.fixture
def db(tmp_path):
    """合成 2 序列数据库（帧率 30，qpos 可辨识的线性序列）。"""
    for name, T in [("a", 100), ("b", 40)]:
        qpos = np.zeros((T, 29))
        qpos[:, 0] = np.arange(T) * 0.01  # 关节 0 线性增长
        qvel = np.zeros((T, 29))
        qvel[:, 0] = 0.01 * 30
        root_pos = np.zeros((T, 3))
        root_pos[:, 0] = np.arange(T) * 0.1
        np.savez(tmp_path / f"{name}.npz", qpos=qpos.astype(np.float32),
                 qvel=qvel.astype(np.float32), root_pos=root_pos.astype(np.float32),
                 root_rot=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T, 1)),
                 contacts=np.zeros((T, 2), dtype=bool),
                 frame_time=np.float32(1 / 30), joint_names=np.zeros(29))
    return MotionDatabase(str(tmp_path))


def test_db_load_and_lengths(db):
    assert db.num_sequences == 2
    assert list(db.lengths()) == [100, 40]


def test_ref_at_interpolation(db):
    q, qd = db.ref_at(0, 10.5)
    assert q[0] == pytest.approx(0.105)  # 帧 10 与 11 的中点
    assert qd[0] == pytest.approx(0.3)
    # 越界截断
    q, _ = db.ref_at(1, 999.0)
    assert q[0] == pytest.approx(39 * 0.01)


def test_anchor_velocity(db):
    v = db.anchor_velocity(0, 10.0)
    assert v[0] == pytest.approx(0.1 * 30)  # 0.1 m/帧 × 30 fps
    assert v[2] == pytest.approx(0.0)  # 平面速度，z=0


def test_future_refs_offsets_and_layout(db):
    cfg = get("A2").value
    C = db.future_refs(0, 50.0)
    assert C.shape == (cfg.K, 61)
    # τ=0 → 当前帧 q = 0.5；τ=1 → 0.51
    assert C[0, 0] == pytest.approx(0.50)
    assert C[1, 0] == pytest.approx(0.51)
    # 布局：q(29) + q̇(29) + ṽ(3)；q̇[0]=0.3，ṽ = 根速度 (3, 0, 0)
    assert C[0, 29] == pytest.approx(0.3)
    assert C[0, 58] == pytest.approx(3.0, rel=1e-5)
    assert C[0, 60] == pytest.approx(0.0)


def test_future_refs_clamp_at_sequence_end(db):
    C = db.future_refs(1, 39.0)  # 序列 b 末帧
    assert C[0, 0] == pytest.approx(39 * 0.01)
    assert np.allclose(C[1:, 0], 39 * 0.01)  # 越界帧全部截断到末帧


def test_correct_anchor_velocity_gate_zero_at_rest():
    # 静止参考：门控归零 → 不修正
    v = correct_anchor_velocity(np.array([0.0, 0.0]), np.array([1.0, 0.5]))
    assert np.allclose(v, [0.0, 0.0], atol=1e-12)


def test_correct_anchor_velocity_clip():
    v = correct_anchor_velocity(np.array([2.0, 0.0]), np.array([10.0, 0.0]))
    cfg = get("A13").value
    # 高速时门控=1，修正 = clip(λ·e, ±v̄) = ±1.0
    assert v[0] == pytest.approx(2.0 + cfg.clip_v)
    assert v[1] == pytest.approx(0.0)


def test_correct_anchor_velocity_partial_gate():
    # 中速：门控在 (0,1)，修正被部分应用
    v0 = np.array([0.25, 0.0])
    e = np.array([2.0, 0.0])
    v = correct_anchor_velocity(v0, e)
    assert 0.25 < v[0] < 0.25 + 2.0


def test_adaptive_sampler_uniform_initially(db):
    rng = np.random.default_rng(0)
    sampler = AdaptiveSampler(db, seg_len=20)
    n = 4000
    counts = {}
    for _ in range(n):
        c = sampler.sample(rng)
        counts[c] = counts.get(c, 0) + 1
    # 无失败时近似均匀（候选段: seq a 4 段 + seq b 2 段）
    fracs = np.array(list(counts.values())) / n
    assert fracs.max() - fracs.min() < 0.05
    assert sampler.failure_rate() == 0.0


def test_adaptive_sampler_boosts_failed_segments(db):
    rng = np.random.default_rng(0)
    sampler = AdaptiveSampler(db, seg_len=20)
    bad = (0, 0)
    for _ in range(10):
        sampler.update(*bad, failed=True)
    sampler.update(0, 40, failed=True)
    # 失败段的采样概率应高于未失败段
    n = 4000
    counts = {}
    for _ in range(n):
        c = sampler.sample(rng)
        counts[c] = counts.get(c, 0) + 1
    p_bad = counts[bad] / n
    others = [v for k, v in counts.items() if k not in (bad, (0, 40))]
    assert p_bad > max(others) / n, f"失败段概率 {p_bad:.3f} 应高于未失败段"
    # 全覆盖保留：所有候选段仍被采样到
    assert set(counts) == set(sampler._candidates())
    assert 0.0 < sampler.failure_rate() <= 1.0
