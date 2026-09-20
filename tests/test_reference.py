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


def test_db_excludes_ground_sequences(tmp_path):
    """A17（2026-09-20 改判后）：不再按类过滤——ground 序列保留进训练集。

    原排除理由（v1 数据的 38–116 cm 退化）已被推翻；论文 "retaining
    uniform coverage of the full motion dataset" 且能力主张依赖躺地动作；
    真正的缺陷是垂直锚定，已由 v2 两段式锚定修复（见 DataFilterCfg）。
    """
    for name, T in [("ground1_subject1", 30), ("walk1_subject1", 30)]:
        qpos = np.zeros((T, 29))
        np.savez(tmp_path / f"{name}.npz", qpos=qpos, qvel=qpos,
                 root_pos=np.zeros((T, 3)),
                 root_rot=np.tile(np.array([1, 0, 0, 0]), (T, 1)),
                 contacts=np.zeros((T, 2), dtype=bool),
                 frame_time=np.float32(1 / 30), joint_names=np.zeros(29))
    db2 = MotionDatabase(str(tmp_path))
    assert db2.num_sequences == 2
    assert {s["name"] for s in db2.seqs} == {"walk1_subject1", "ground1_subject1"}


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
    # τ=0 → 当前源帧 q=0.5；一个50Hz控制步=0.6个30fps源帧。
    assert C[0, 0] == pytest.approx(0.50)
    assert C[1, 0] == pytest.approx(0.506)
    assert C[-1, 0] == pytest.approx(0.686)  # 31步=0.62s=18.6源帧
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


# ---------------------------------------------------------------------------
# ref_rot：参考锚点朝向的四元数最短弧插值（o_t 的 e_t 依赖此路径）
# ---------------------------------------------------------------------------

def _quat_z(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _db_with_rotation(quats, frame_time=1 / 30):
    """构造只关心 root_rot 的合成库（T = len(quats)）。"""
    T = len(quats)
    return MotionDatabase.from_sequences([{
        "qpos": np.zeros((T, 29), dtype=np.float32),
        "qvel": np.zeros((T, 29), dtype=np.float32),
        "root_pos": np.zeros((T, 3), dtype=np.float32),
        "root_rot": np.asarray(quats, dtype=np.float32),
        "frame_time": np.float32(frame_time),
        "name": "rot",
    }])


def test_ref_rot_endpoints_exact():
    db = _db_with_rotation([_quat_z(0.0), _quat_z(0.8), _quat_z(1.6)])
    assert np.allclose(db.ref_rot(0, 0.0), _quat_z(0.0), atol=1e-6)
    assert np.allclose(db.ref_rot(0, 1.0), _quat_z(0.8), atol=1e-6)


def test_ref_rot_fractional_frame_matches_analytic_halfway():
    """0° → 90° 的半帧处应为 45°（解析对照，非自证）。"""
    from pgmt.policy.rotation import quat_to_mat
    db = _db_with_rotation([_quat_z(0.0), _quat_z(np.pi / 2)])
    q = db.ref_rot(0, 0.5)
    v = quat_to_mat(q) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0], atol=1e-6)


def test_ref_rot_clamps_outside_sequence():
    db = _db_with_rotation([_quat_z(0.0), _quat_z(0.5)])
    assert np.allclose(db.ref_rot(0, 999.0), _quat_z(0.5), atol=1e-6)
    assert np.allclose(db.ref_rot(0, -5.0), _quat_z(0.0), atol=1e-6)


def test_ref_rot_takes_shortest_arc_across_sign_flip():
    """相邻帧四元数符号翻转（同一旋转的两种表示）时不得跳出 360° 假旋转。

    这是 LAFAN1 30 fps → 50 Hz 重采样的真实路径：缺半球对齐就会在
    翻转点产生一次假的大幅转动。
    """
    from pgmt.policy.rotation import quat_to_mat
    a = _quat_z(0.2)
    db = _db_with_rotation([a, -a])  # −a 与 a 表示同一旋转
    for w in (0.1, 0.25, 0.5, 0.75, 0.9):
        q = db.ref_rot(0, w)
        assert np.allclose(quat_to_mat(q), quat_to_mat(a), atol=1e-6), f"w={w}"


def test_ref_rot_output_is_unit_quaternion():
    db = _db_with_rotation([_quat_z(0.1), _quat_z(0.9), _quat_z(2.0)])
    for t in np.linspace(0.0, 2.0, 9):
        q = db.ref_rot(0, float(t))
        assert q.shape == (4,)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_ref_rot_missing_field_raises():
    db = _db_with_rotation([_quat_z(0.0), _quat_z(0.3)])
    del db.seqs[0]["root_rot"]
    with pytest.raises(KeyError, match="root_rot"):
        db.ref_rot(0, 0.0)


# ---------------------------------------------------------------------------
# MotionDatabase.from_sequences（内存构造，M2 单测用）
# ---------------------------------------------------------------------------

def test_from_sequences_builds_usable_db():
    db = _db_with_rotation([_quat_z(0.0), _quat_z(0.4), _quat_z(0.8)])
    assert db.num_sequences == 1
    assert db.seq_len(0) == 3
    assert list(db.lengths()) == [3]
    assert db.future_refs(0, 0.0).shape == (6, 61)


def test_from_sequences_validates_required_fields():
    with pytest.raises(ValueError, match="缺字段"):
        MotionDatabase.from_sequences([{"qpos": np.zeros((3, 29))}])


def test_from_sequences_a17_filter_is_opt_in():
    T = 3
    base = {
        "qpos": np.zeros((T, 29), dtype=np.float32),
        "qvel": np.zeros((T, 29), dtype=np.float32),
        "root_pos": np.zeros((T, 3), dtype=np.float32),
        "root_rot": np.tile(_quat_z(0.0), (T, 1)).astype(np.float32),
        "frame_time": np.float32(1 / 30),
    }
    seqs = [dict(base, name="ground1_subject1"), dict(base, name="walk1_subject1")]
    assert MotionDatabase.from_sequences(seqs).num_sequences == 2  # 默认不过滤
    # A17 改判后 excluded_prefixes 为空：opt-in 的过滤机制仍在，但当前
    # 注册表配置下不再排除任何类（改判依据见 DataFilterCfg docstring）。
    assert MotionDatabase.from_sequences(seqs, apply_a17_filter=True).num_sequences == 2


# ---------------------------------------------------------------------------
# 采样热路径：缓存与向量化必须与原逐项语义一致
# ---------------------------------------------------------------------------

def test_lengths_is_cached_and_stable():
    db = _db_with_rotation([_quat_z(0.0)] * 7)
    a = db.lengths()
    b = db.lengths()
    assert np.array_equal(a, [7])
    assert a is b, "lengths() 应返回缓存数组（采样热路径每步都调用）"


def test_lengths_correct_for_multiple_sequences():
    T = 4
    base = {
        "qpos": np.zeros((T, 29), dtype=np.float32),
        "qvel": np.zeros((T, 29), dtype=np.float32),
        "root_pos": np.zeros((T, 3), dtype=np.float32),
        "root_rot": np.tile(_quat_z(0.0), (T, 1)).astype(np.float32),
        "frame_time": np.float32(1 / 30),
    }
    db = MotionDatabase.from_sequences([dict(base, name="a"), dict(base, name="b")])
    assert list(db.lengths()) == [4, 4]


def _ref_candidates(db, seg_len):
    """逐项重建候选段（参照实现，用于对齐向量化版本）。"""
    cands = []
    for i, L in enumerate(db.lengths()):
        for s in range(0, max(L - seg_len + 1, 1), seg_len):
            cands.append((i, s))
    return cands


def test_candidates_match_reference_enumeration():
    """候选段枚举必须与逐项参照实现一致，且每个段都能完整放进序列。

    L=25、seg_len=10 → range(0, 25-10+1=16, 10) = [0, 10]。
    起点 20 需要帧 20..29，超出序列（越界），**不是**合法候选 —— 我第一版
    把字面量写成了 [(0,0),(0,10),(0,20)]，是断言错了而非实现错了。

    已知性质（非缺陷）：不足一整段的尾部帧不会被任何候选段覆盖。以 L=25、
    seg_len=10 为例，帧 20..24 不在任何段内。这是网格枚举的固有结果；论文
    未规定段长，"retaining uniform coverage of the full motion dataset"
    按段覆盖理解。若要覆盖尾部需改成 `start = min(grid, L - seg_len)`，
    但这会与网格键（起点对齐 seg_len）冲突，故保持现状。
    """
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    assert s._candidates() == _ref_candidates(db, 10)
    assert s._candidates() == [(0, 0), (0, 10)]
    # 每个候选段必须能完整容纳 seg_len 帧
    for _, start in s._candidates():
        assert start + 10 <= 25, f"起点 {start} 的段越界"


def test_candidates_cover_full_length_when_divisible():
    """恰好整除时，全部帧都被候选段覆盖（尾部不丢）。"""
    db = _db_with_rotation([_quat_z(0.0)] * 30)
    s = AdaptiveSampler(db, seg_len=10)
    assert s._candidates() == [(0, 0), (0, 10), (0, 20)]
    covered = set()
    for _, start in s._candidates():
        covered |= set(range(start, start + 10))
    assert covered == set(range(30)), "整除时不得有帧被漏掉"


def test_candidates_are_cached_not_rebuilt():
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    assert s._candidates() is s._candidates(), "候选段应缓存"


def test_candidates_handles_sequence_shorter_than_segment():
    """序列比段还短时至少要给出一个候选，否则采样无解。"""
    db = _db_with_rotation([_quat_z(0.0)] * 3)
    s = AdaptiveSampler(db, seg_len=100)
    assert s._candidates() == [(0, 0)]


def test_sampler_rejects_nonpositive_seg_len():
    db = _db_with_rotation([_quat_z(0.0)] * 10)
    with pytest.raises(ValueError):
        AdaptiveSampler(db, seg_len=0)


def test_weights_match_reference_formula():
    """`_weights()` 必须逐项等于 1 + β·fails[key]（向量化前的原公式）。"""
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    # 给第 0 段 3 次失败、第 2 段 1 次失败、一个不存在于候选集的键
    s.update(0, 0, failed=True)
    s.update(0, 0, failed=True)
    s.update(0, 0, failed=True)
    s.update(0, 20, failed=True)
    s.fails[(9, 3)] = 5.0  # 越界键，应被忽略

    cands = s._candidates()
    ref = np.array([1.0 + s.fail_boost * s.fails.get(s._key(*c), 0.0) for c in cands])
    assert np.allclose(s._weights(), ref), f"权重不一致: {s._weights()} vs {ref}"


def test_weights_are_cached_and_invalidated_on_update():
    """权重缓存必须随 update() 失效，否则采样会一直用陈旧权重。"""
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    w0 = s._weights()
    assert s._weights() is w0, "无更新时应命中缓存"

    s.update(0, 0, failed=True)
    w1 = s._weights()
    assert w1 is not w0, "update 后应重算"
    assert w1[0] == pytest.approx(1.0 + s.fail_boost)
    assert w1[1] == pytest.approx(1.0), "其他段权重不变"


def test_failed_segment_weight_grows_monotonically():
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    prev = 1.0
    for _ in range(4):
        s.update(0, 10, failed=True)
        cur = float(s._weights()[1])
        assert cur > prev, "失败次数增加，权重必须单调上升"
        prev = cur


def test_sampler_never_returns_out_of_range_segment():
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    s.update(0, 0, failed=True)
    rng = np.random.default_rng(7)
    cands = set(s._candidates())
    for _ in range(500):
        assert s.sample(rng) in cands


def test_sampler_falls_back_to_uniform_on_bad_weights():
    """极端防御：权重异常时退回均匀采样而不是崩溃（NaN/inf 场景）。

    注意：这里直接改 `fails` 绕过了公开 API，必须手动失效权重缓存 ——
    否则读到的是改之前的缓存，测试就测不到 fallback 分支。
    """
    db = _db_with_rotation([_quat_z(0.0)] * 25)
    s = AdaptiveSampler(db, seg_len=10)
    s.fails[(0, 0)] = float("nan")
    s._invalidate_weights()
    assert not np.isfinite(s._weights().sum()), "前置条件：权重应为非有限"

    rng = np.random.default_rng(0)
    cands = set(s._candidates())
    seen = {s.sample(rng) for _ in range(200)}
    assert seen <= cands
    assert len(seen) > 1, "退回均匀采样后不应总取同一个段"


# Reference velocities and position correction share the current reference frame.
def _moving_db(yaw=0.0, n_frames=40):
    """世界 +x 匀速 1m/s，给定参考朝向。"""
    db = _db_with_rotation([_quat_z(yaw)] * n_frames, frame_time=0.1)
    db.seqs[0]["root_pos"][:, 0] = np.arange(n_frames) * 0.1
    return db


def test_anchor_velocity_rotates_world_displacement_into_reference_frame():
    db = _moving_db(yaw=np.pi / 2)
    # 世界 +x 在朝向 +y 的参考系里为 -y，而非仍然 +x。
    assert np.allclose(db.anchor_velocity(0, 2.5), [0.0, -1.0, 0.0], atol=1e-6)
    refs = db.future_refs(0, 2.5, e_p=np.array([0.25, 0.0]), lambda_pos=1.0)
    assert np.allclose(refs[:, -3:], [0.25, -1.0, 0.0], atol=1e-5)


def test_anchor_velocity_uses_full_reference_rotation_before_planar_projection():
    # 参考绕 y 轴转 +90°：世界 +z 对应参考 -x。
    quat = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    db = _db_with_rotation([quat] * 3, frame_time=0.1)
    db.seqs[0]["root_pos"][:, 2] = [0.0, 0.1, 0.2]
    assert np.allclose(db.anchor_velocity(0, 0.5), [-1.0, 0.0, 0.0], atol=1e-6)


def test_future_velocities_share_current_anchor_when_reference_turns():
    db = _moving_db()
    db.seqs[0]["root_rot"][1:] = _quat_z(np.pi / 2)
    # 当前 t=0 朝向 +x；即使未来朝向 +y，全部 token 仍表达在当前系。
    assert np.allclose(db.future_refs(0, 0.0)[:, -3:], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(db.anchor_velocity(0, 1.0), [0.0, -1.0, 0.0], atol=1e-6)
    assert np.allclose(db.anchor_velocity(0, 1.0, anchor_t=0.0), [1.0, 0.0, 0.0], atol=1e-6)


def test_corrected_future_refs_are_invariant_to_global_yaw_and_translation():
    from pgmt.policy.rotation import quat_to_mat

    db = _moving_db(yaw=0.3)
    moved = _moving_db(yaw=0.3 + 1.2)
    rotation = quat_to_mat(_quat_z(1.2))
    shift = np.array([3.0, -2.0, 1.0])
    moved.seqs[0]["root_pos"] = db.seqs[0]["root_pos"] @ rotation.T + shift
    ref_pos = db.seqs[0]["root_pos"][3]
    robot_pos = ref_pos - np.array([0.2, 0.1, 0.0])
    error = (quat_to_mat(db.ref_rot(0, 3.0)).T @ (ref_pos - robot_pos))[:2]
    moved_ref, moved_robot = rotation @ ref_pos + shift, rotation @ robot_pos + shift
    moved_error = (quat_to_mat(moved.ref_rot(0, 3.0)).T @ (moved_ref - moved_robot))[:2]
    assert np.allclose(error, moved_error, atol=1e-6)
    assert np.allclose(db.future_refs(0, 3.0, e_p=error),
                       moved.future_refs(0, 3.0, e_p=moved_error), atol=1e-5)


def test_single_frame_reference_has_zero_velocity():
    db = _moving_db(yaw=1.0, n_frames=1)
    assert np.array_equal(db.anchor_velocity(0, 10.0), np.zeros(3))
    assert np.array_equal(db.future_refs(0, 0.0)[:, -3:], np.zeros((6, 3)))


@pytest.mark.parametrize("key,value", [
    ("qpos", np.zeros((0, 29))),
    ("qvel", np.zeros((2, 29))),
    ("root_pos", np.zeros((3, 2))),
    ("root_rot", np.zeros((3, 4))),
    ("frame_time", 0.0),
    ("frame_time", np.nan),
])
def test_database_rejects_invalid_sampling_inputs_from_both_load_paths(tmp_path, key, value):
    seq = dict(_db_with_rotation([_quat_z(0.0)] * 3).seqs[0])
    seq[key] = value
    with pytest.raises(ValueError):
        MotionDatabase.from_sequences([seq])
    np.savez(tmp_path / "bad.npz", **seq)
    with pytest.raises(ValueError):
        MotionDatabase(str(tmp_path))


@pytest.mark.parametrize("seg_len", [0, -1, 1.5])
def test_samplers_reject_invalid_segment_length(seg_len):
    db = _moving_db()
    with pytest.raises(ValueError, match="seg_len"):
        db.sample_segment(np.random.default_rng(0), seg_len)
    with pytest.raises(ValueError, match="seg_len"):
        AdaptiveSampler(db, seg_len)


@pytest.mark.parametrize("boost", [-1.0, np.inf, np.nan])
def test_adaptive_sampler_rejects_invalid_failure_boost(boost):
    with pytest.raises(ValueError, match="fail_boost"):
        AdaptiveSampler(_moving_db(), seg_len=10, fail_boost=boost)


def test_velocity_correction_rejects_broadcasting_between_coordinate_shapes():
    with pytest.raises(ValueError, match="形状"):
        correct_anchor_velocity(np.zeros(3), np.zeros(2))


def test_future_refs_represent_same_physical_times_at_30_and_50_fps():
    """独立物理轨迹 q0(t)=t；50Hz控制下最远token应是当前+0.62秒。"""
    references = []
    for fps in (30, 50):
        times = np.arange(2 * fps + 1, dtype=np.float64) / fps
        qpos = np.zeros((len(times), 29))
        qpos[:, 0] = times
        qvel = np.zeros_like(qpos)
        qvel[:, 0] = 1.0
        root_pos = np.zeros((len(times), 3))
        root_pos[:, 0] = 2.0 * times
        db = MotionDatabase.from_sequences([{
            "qpos": qpos, "qvel": qvel, "root_pos": root_pos,
            "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (len(times), 1)),
            "frame_time": 1.0 / fps,
        }])
        C = db.future_refs(0, t=0.4 * fps)
        np.testing.assert_allclose(C[:, 0], [0.4, 0.42, 0.46, 0.54, 0.70, 1.02])
        np.testing.assert_allclose(C[:, -3:], np.tile([2.0, 0.0, 0.0], (6, 1)))
        references.append(C)
    np.testing.assert_allclose(references[0], references[1], atol=1e-12)
