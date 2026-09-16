"""观测契约：96 维布局、拼装器、历史环形缓冲（H_t 语义）、A9 特权维度。

这一层是 M2 `g1_env.py` 的接口定义，必须在不依赖 Isaac Gym 的前提下锁定，
否则环境层会"边写边猜"。本文件覆盖：
  - `OBS_LAYOUT` 分段与 `pgmt.contracts.OBS_DIM` 的一致性
  - 拼装器按名称切片可复原（写-读往返）
  - **`H_t` 不含 `o_t`** 的语义约定 + 冷启动填充 + 环形回绕
  - A9 特权观测维度表可配置、切片不重叠
"""

import numpy as np
import pytest

from pgmt.contracts import ACT_DIM, HISTORY_LEN, OBS_DIM, REF_FRAME_DIM
from pgmt.envs.observations import (
    OBS_LAYOUT,
    OBS_SEGMENTS,
    PRIV_DIM,
    PRIV_LAYOUT,
    HistoryBuffer,
    ObsLayout,
    PrivilegedLayout,
    build_obs,
    build_obs_from_state,
)


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------

def test_obs_dim_matches_paper_decomposition():
    """96 = e_t(6) + ω_t(3) + q(29) + q̇(29) + a_{t−1}(29)（论文 §III）。"""
    assert OBS_DIM == 96
    assert ACT_DIM == 29
    assert HISTORY_LEN == 10
    assert REF_FRAME_DIM == 61  # 论文 Eq.1
    assert 6 + 3 + 29 + 29 + 29 == OBS_DIM


def test_layout_segments_sum_to_total():
    assert sum(d for _, d, _ in OBS_SEGMENTS) == OBS_DIM
    assert OBS_LAYOUT.total_dim == OBS_DIM


def test_layout_names_and_order_follow_paper():
    assert OBS_LAYOUT.names == ("e_t", "omega_t", "q_t", "qd_t", "a_prev")


def test_layout_slices_are_contiguous_and_non_overlapping():
    covered = np.zeros(OBS_DIM, dtype=int)
    for seg in OBS_LAYOUT:
        assert seg.slice.stop - seg.slice.start == seg.dim
        covered[seg.slice] += 1
    assert np.all(covered == 1), "每维必须恰好被一个段覆盖"


def test_layout_offsets():
    assert OBS_LAYOUT.index("e_t") == slice(0, 6)
    assert OBS_LAYOUT.index("omega_t") == slice(6, 9)
    assert OBS_LAYOUT.index("q_t") == slice(9, 38)
    assert OBS_LAYOUT.index("qd_t") == slice(38, 67)
    assert OBS_LAYOUT.index("a_prev") == slice(67, 96)


def test_layout_rejects_duplicate_names():
    with pytest.raises(ValueError, match="重复"):
        ObsLayout((("a", 3, "x"), ("a", 3, "y")))


def test_layout_rejects_nonpositive_dim():
    with pytest.raises(ValueError, match="必须为正"):
        ObsLayout((("a", 0, "x"),))


def test_layout_unknown_segment_raises():
    with pytest.raises(KeyError):
        OBS_LAYOUT.index("nope")


def test_custom_layout_total_is_sum():
    lay = ObsLayout((("x", 2, "a"), ("y", 5, "b")))
    assert lay.total_dim == 7
    assert lay.index("y") == slice(2, 7)


# ---------------------------------------------------------------------------
# 拼装器
# ---------------------------------------------------------------------------

def _fake_parts(batch: int | None = None):
    rng = np.random.default_rng(0)
    shape = (batch,) if batch is not None else ()
    return {
        "e_t": rng.normal(size=shape + (6,)),
        "omega_t": rng.normal(size=shape + (3,)),
        "q_t": rng.normal(size=shape + (29,)),
        "qd_t": rng.normal(size=shape + (29,)),
        "a_prev": rng.normal(size=shape + (29,)),
    }


def test_build_obs_single_shape_and_slice_roundtrip():
    p = _fake_parts()
    o = build_obs(**p)
    assert o.shape == (OBS_DIM,)
    for name in OBS_LAYOUT.names:
        assert np.allclose(o[OBS_LAYOUT.index(name)], p[name]), f"{name} 段错位"


def test_build_obs_batched_shape_and_slice_roundtrip():
    p = _fake_parts(batch=5)
    o = build_obs(**p)
    assert o.shape == (5, OBS_DIM)
    for name in OBS_LAYOUT.names:
        assert np.allclose(o[:, OBS_LAYOUT.index(name)], p[name])


def test_build_obs_is_float32():
    assert build_obs(**_fake_parts()).dtype == np.float32


def test_build_obs_rejects_wrong_segment_dim():
    p = _fake_parts()
    p["q_t"] = np.zeros(OBS_DIM)  # 故意错
    with pytest.raises(ValueError, match="q_t 维度"):
        build_obs(**p)


def test_build_obs_from_state_uses_identity_relative_rotation_when_equal():
    """机器人朝向 == 参考朝向时，e_t 必须是 6D 恒等 [1,0,0,0,1,0] 且落在 e_t 段内。"""
    q = np.array([np.cos(0.35), 0.0, 0.0, np.sin(0.35)])  # 绕 z 的任意朝向
    p = _fake_parts()
    o = build_obs_from_state(robot_quat=q, ref_quat=q, omega_t=p["omega_t"],
                             q_t=p["q_t"], qd_t=p["qd_t"], a_prev=p["a_prev"])
    assert o.shape == (OBS_DIM,)
    assert np.allclose(o[OBS_LAYOUT.index("e_t")], [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_build_obs_from_state_batched():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    p = _fake_parts(batch=4)
    o = build_obs_from_state(robot_quat=q, ref_quat=q, omega_t=p["omega_t"],
                             q_t=p["q_t"], qd_t=p["qd_t"], a_prev=p["a_prev"])
    assert o.shape == (4, OBS_DIM)


# ---------------------------------------------------------------------------
# 历史环形缓冲 H_t
# ---------------------------------------------------------------------------

def _obs_with_marker(value: float) -> np.ndarray:
    """构造一帧可辨识的 o_t：所有元素都等于 value。"""
    return np.full(OBS_DIM, value, dtype=np.float32)


def test_history_cold_start_is_all_fill():
    hb = HistoryBuffer()
    h = hb.as_tensor()
    assert h.shape == (HISTORY_LEN, OBS_DIM)
    assert np.all(h == 0.0)
    assert hb.count == 0 and not hb.full


def test_history_excludes_current_frame_and_orders_oldest_first():
    """**核心约定**：推进到 t 后，H_t = [o_{t-10}, ..., o_{t-1}]，不含 o_t。

    这里只 push 到 o_2（共 3 帧），H_t 应把 o_0..o_2 放在最后 3 行、
    前面 7 行是填充；末行 == o_2 且**不等于**尚未 push 的 o_3。
    """
    hb = HistoryBuffer()
    for v in (1.0, 2.0, 3.0):
        hb.push(_obs_with_marker(v))
    h = hb.as_tensor()
    assert np.all(h[:7] == 0.0), "前端应为冷启动填充"
    assert np.allclose([h[7, 0], h[8, 0], h[9, 0]], [1.0, 2.0, 3.0]), "时间正序"
    assert np.allclose(h[-1], _obs_with_marker(3.0)), "末行 = 最近一帧 o_2"


def test_history_full_excludes_the_frame_not_yet_pushed():
    """推满 10 帧后，H_t = 这 10 帧；再 push 第 11 帧时最旧的必须被丢弃。"""
    hb = HistoryBuffer()
    for i in range(HISTORY_LEN):
        hb.push(_obs_with_marker(float(i)))
    h = hb.as_tensor()
    assert hb.full
    assert np.allclose(h[:, 0], np.arange(HISTORY_LEN)), "正序 0..9，无填充"

    hb.push(_obs_with_marker(10.0))
    h2 = hb.as_tensor()
    assert np.allclose(h2[:, 0], np.arange(1, 11)), "环形回绕：丢掉 o_0"
    assert h2.shape == (HISTORY_LEN, OBS_DIM)


def test_history_len_matches_paper_default():
    assert HistoryBuffer().length == HISTORY_LEN
    assert HistoryBuffer().as_tensor().shape[0] == 10


def test_history_rejects_wrong_obs_shape():
    hb = HistoryBuffer()
    with pytest.raises(ValueError, match="形状"):
        hb.push(np.zeros(OBS_DIM + 1))
    with pytest.raises(ValueError, match="形状"):
        hb.push(np.zeros((1, OBS_DIM)))


def test_history_rejects_bad_config():
    with pytest.raises(ValueError):
        HistoryBuffer(length=0)
    with pytest.raises(ValueError):
        HistoryBuffer(obs_dim=-1)


def test_history_reset_clears():
    hb = HistoryBuffer()
    hb.push(_obs_with_marker(1.0))
    hb.reset()
    assert hb.count == 0
    assert np.all(hb.as_tensor() == 0.0)


def test_history_custom_fill_value():
    hb = HistoryBuffer(fill=-1.0)
    hb.push(_obs_with_marker(5.0))
    h = hb.as_tensor()
    assert np.all(h[: HISTORY_LEN - 1] == -1.0)
    assert np.allclose(h[-1], _obs_with_marker(5.0))


def test_history_custom_length():
    hb = HistoryBuffer(length=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        hb.push(_obs_with_marker(v))
    h = hb.as_tensor()
    assert h.shape == (3, OBS_DIM)
    assert np.allclose(h[:, 0], [2.0, 3.0, 4.0])


def test_history_as_batch_repeats_and_shapes():
    hb = HistoryBuffer()
    hb.push(_obs_with_marker(1.0))
    hb.push(_obs_with_marker(2.0))
    b = hb.as_batch(4)
    assert b.shape == (4, HISTORY_LEN, OBS_DIM)
    assert np.allclose(b[0], b[3]), "共享同一历史时各 batch 行应相同"
    assert b[0].shape == hb.as_tensor().shape
    assert np.allclose(b[0], hb.as_tensor())


def test_history_len_dunder():
    hb = HistoryBuffer()
    assert len(hb) == 0
    hb.push(_obs_with_marker(0.0))
    assert len(hb) == 1


# ---------------------------------------------------------------------------
# 别名安全（环境层会复用 obs_buf 并按行切片 push，这是真实用法）
# ---------------------------------------------------------------------------

def test_history_push_copies_input_so_later_mutation_is_safe():
    """**回归防护**：push 必须复制。

    环境层的典型写法是一个复用的 `obs_buf` 逐帧重填，然后 `hb.push(obs_buf[i])`。
    若 push 只做 `np.asarray`（对 float32 是零拷贝），缓冲区里存的就是同一个
    底层内存的视图 —— 下一次重填会篡改"历史"，症状是策略输入悄悄退化为当前帧
    的重复。这里显式验证：push 之后再改原数组，历史内容不得变化。
    """
    hb = HistoryBuffer()
    buf = np.zeros(OBS_DIM, dtype=np.float32)
    for v in (1.0, 2.0, 3.0):
        buf[:] = v
        hb.push(buf)          # 传的是同一个 buf（视图/同一对象）
    buf[:] = 999.0            # 模拟下一帧重填

    h = hb.as_tensor()
    assert np.allclose(h[-3:, 0], [1.0, 2.0, 3.0]), "历史被后续写入篡改"
    assert not np.allclose(h[-1, 0], 999.0)


def test_history_push_copies_slice_views():
    """按行切片 push（env 里最常见的形式）同样必须安全。"""
    hb = HistoryBuffer()
    buf = np.zeros((4, OBS_DIM), dtype=np.float32)
    for i, v in enumerate((5.0, 6.0, 7.0, 8.0)):
        buf[i] = v
        hb.push(buf[i])
    buf[:] = -1.0
    h = hb.as_tensor()
    assert np.allclose(h[-4:, 0], [5.0, 6.0, 7.0, 8.0])


def test_history_push_accepts_list_and_float64():
    """非 float32 输入本来就会新建数组，也必须正常工作。"""
    hb = HistoryBuffer(obs_dim=3)
    hb.push([1.0, 2.0, 3.0])
    hb.push(np.array([4.0, 5.0, 6.0], dtype=np.float64))
    h = hb.as_tensor()
    assert h.shape == (HISTORY_LEN, 3)
    assert np.allclose(h[-1], [4.0, 5.0, 6.0])


def test_build_obs_returns_fresh_array_not_a_view():
    """拼装结果必须独立于入参 —— 否则 push 进环形缓冲后会被上游改写。"""
    p = _fake_parts()
    o = build_obs(**p)
    assert not any(np.shares_memory(o, v) for v in p.values())


def test_build_obs_batched_does_not_alias_inputs():
    p = _fake_parts(batch=3)
    o = build_obs(**p)
    assert not any(np.shares_memory(o, v) for v in p.values())


# ---------------------------------------------------------------------------
# observe_then_advance：把"先取值再推进"的顺序固定下来
# ---------------------------------------------------------------------------

def test_observe_then_advance_excludes_current_frame():
    """**核心顺序契约**：第 k 步拿到的 H 必须是前 k 帧，不含本步的 o_k。"""
    hb = HistoryBuffer()
    seen = []
    for k in range(HISTORY_LEN + 3):
        o = _obs_with_marker(float(k))
        seen.append(hb.observe_then_advance(o))
    # 第 0 步：还没推入任何帧 → 全填充
    assert np.all(seen[0] == 0.0)
    # 第 1 步：H 应只含 o_0
    assert np.allclose(seen[1][-1], _obs_with_marker(0.0))
    # 第 3 步：H 尾三行应为 o_0,o_1,o_2（不含 o_3）
    assert np.allclose(seen[3][-3:, 0], [0.0, 1.0, 2.0])
    # 第 11 步（已满）：H 尾行应为 o_10，绝不等于本步的 o_11
    assert np.allclose(seen[11][-1], _obs_with_marker(10.0))
    assert not np.allclose(seen[11][-1], _obs_with_marker(11.0)), "H_t 含入了当前帧"


def test_observe_then_advance_matches_manual_order():
    hb1, hb2 = HistoryBuffer(), HistoryBuffer()
    for k in range(6):
        o = _obs_with_marker(float(k))
        a = hb1.observe_then_advance(o)
        b = hb2.as_tensor()      # 手动顺序：先取
        hb2.push(o)              # 再推进
        assert np.allclose(a, b)
    assert hb1.count == hb2.count


def test_observe_then_advance_returns_correct_shape_and_validates():
    hb = HistoryBuffer()
    h = hb.observe_then_advance(_obs_with_marker(1.0))
    assert h.shape == (HISTORY_LEN, OBS_DIM)
    with pytest.raises(ValueError):
        hb.observe_then_advance(np.zeros(OBS_DIM + 1))


def test_observe_then_advance_does_not_mutate_returned_history():
    """返回的 H 必须是**快照**：后续 push 不得改写已返回的数组。

    注意时序（这是我第一版写错的地方）：`observe_then_advance(o_0)` 的**返回值**
    反映的是 push *之前*的缓冲（本帧不含在内），所以第 0 步返回的全是填充行，
    真正的 `o_0` 从第 1 步的返回值开始出现。断言的正确对象是 h2（第 1 步的
    返回值），而不是 h1。
    """
    hb = HistoryBuffer()
    h1 = hb.observe_then_advance(_obs_with_marker(1.0))
    h2 = hb.observe_then_advance(_obs_with_marker(2.0))

    assert np.allclose(h1, 0.0), "第 0 步的 H 应为全填充（不含 o_0）"
    # h2 是"推入 o_0 之后"的快照：末行 = o_0，其余为填充
    assert np.allclose(h2[-1], _obs_with_marker(1.0))
    assert np.allclose(h2[-2], 0.0)

    # 关键：再推一帧不得改写已经返回的 h2
    hb.observe_then_advance(_obs_with_marker(3.0))
    assert np.allclose(h2[-1], _obs_with_marker(1.0)), "先前返回的 H 被后续 push 改写"
    assert np.allclose(h1, 0.0), "第 0 步的 H 被后续 push 改写"
    assert not np.shares_memory(h1, h2)


# ---------------------------------------------------------------------------
# 冷启动填充：零填充 vs e_t 恒等分段填充
# ---------------------------------------------------------------------------

def test_e_t_identity_fill_shape_and_segments():
    from pgmt.envs.observations import E6_IDENTITY, e_t_identity_fill

    f = e_t_identity_fill()
    assert f.shape == (OBS_DIM,)
    assert np.allclose(f[OBS_LAYOUT.index("e_t")], [1, 0, 0, 0, 1, 0])
    assert np.allclose(f[OBS_LAYOUT.index("e_t")], E6_IDENTITY)
    # 其余段必须为 0
    for name in ("omega_t", "q_t", "qd_t", "a_prev"):
        assert np.allclose(f[OBS_LAYOUT.index(name)], 0.0), f"{name} 应为 0"


def test_zero_fill_e_t_is_not_a_valid_rotation_encoding():
    """记录零填充的副作用：6D 零向量解出的不是合法旋转。

    这条测试不是在断言"零填充是错的"，而是把该副作用**固化成已知事实**，
    避免将来有人误以为零填充给出的是中性/合法的 e_t。
    """
    from pgmt.policy.rotation import rot6d_to_rotmat

    zero6 = np.zeros((1, 6))
    R = rot6d_to_rotmat(zero6)[0]
    # 退化：第二基向量无法确定（靠 eps 兜底），列不构成正交旋转
    assert not np.allclose(R @ R.T, np.eye(3), atol=1e-6), (
        "6D 零向量竟然解出了合法旋转 —— 本测试的前提已变，请复核 e_t 语义"
    )


def test_segmented_fill_is_used_in_cold_start():
    from pgmt.envs.observations import e_t_identity_fill

    hb = HistoryBuffer(fill=e_t_identity_fill())
    h = hb.as_tensor()
    assert h.shape == (HISTORY_LEN, OBS_DIM)
    assert np.allclose(h[:, OBS_LAYOUT.index("e_t")], [1, 0, 0, 0, 1, 0])
    assert np.allclose(h[:, OBS_LAYOUT.index("q_t")], 0.0)


def test_segmented_fill_only_affects_padding_rows():
    from pgmt.envs.observations import e_t_identity_fill

    hb = HistoryBuffer(fill=e_t_identity_fill())
    hb.push(_obs_with_marker(2.0))
    h = hb.as_tensor()
    assert np.allclose(h[:-1, OBS_LAYOUT.index("e_t")], [1, 0, 0, 0, 1, 0])
    # 真实帧的 e_t 段应来自观测本身（marker 全 2.0），不被填充污染
    assert np.allclose(h[-1, OBS_LAYOUT.index("e_t")], 2.0)


def test_fill_accepts_scalar_and_matches_legacy_behaviour():
    hb = HistoryBuffer(fill=-1.0)
    hb.push(_obs_with_marker(5.0))
    h = hb.as_tensor()
    assert np.all(h[: HISTORY_LEN - 1] == -1.0)
    assert np.allclose(h[-1], _obs_with_marker(5.0))
    assert hb.fill == -1.0


def test_fill_rejects_wrong_length_vector():
    with pytest.raises(ValueError, match="fill"):
        HistoryBuffer(fill=np.zeros(OBS_DIM + 3, dtype=np.float32))


def test_full_buffer_has_no_padding_rows():
    """缓冲满了以后不得再出现填充行（分段填充也不能泄漏进来）。"""
    from pgmt.envs.observations import e_t_identity_fill

    hb = HistoryBuffer(fill=e_t_identity_fill())
    for k in range(HISTORY_LEN):
        hb.push(_obs_with_marker(float(k) + 1.0))
    h = hb.as_tensor()
    assert np.allclose(h[:, 0], np.arange(1, HISTORY_LEN + 1))
    assert not np.any(np.isclose(h[:, 0], 0.0))


# ---------------------------------------------------------------------------
# 特权观测（A9）
# ---------------------------------------------------------------------------

def test_priv_layout_segments_match_a9_items():
    """A9 注册的 9 个项目必须与特权布局一一对应。"""
    from pgmt.cfg.assumptions import get

    a9_items = set(get("A9").value.items)
    assert set(PRIV_LAYOUT.names) == a9_items, "特权布局与 A9 清单不一致"


def test_priv_layout_slices_contiguous():
    covered = np.zeros(PRIV_DIM, dtype=int)
    for seg in PRIV_LAYOUT:
        s = PRIV_LAYOUT.index(seg.name)
        covered[s] += 1
    assert np.all(covered == 1)


def test_priv_dim_positive_and_total_matches_sum():
    assert PRIV_DIM == sum(seg.dim for seg in PRIV_LAYOUT)
    assert PRIV_DIM > 0


def test_priv_dynamics_scaling_grows_with_n_envs():
    one = PrivilegedLayout(1)
    four = PrivilegedLayout(4)
    # 按 env 展开的量应线性增长，全局量不变
    assert four.dim("motor_strength_scale") == 4 * one.dim("motor_strength_scale")
    assert four.dim("friction_coefficients") == 4 * one.dim("friction_coefficients")
    assert four.dim("push_perturbation") == one.dim("push_perturbation")
    assert four.total_dim > one.total_dim


def test_priv_layout_rejects_bad_n_envs():
    with pytest.raises(ValueError):
        PrivilegedLayout(0)
    with pytest.raises(ValueError):
        PrivilegedLayout(-3)


def test_priv_layout_unknown_raises():
    with pytest.raises(KeyError):
        PRIV_LAYOUT.index("nope")


def test_base_lin_vel_is_privileged_not_in_obs():
    """论文里基座线速度只给 critic —— o_t 布局中不得出现。"""
    assert "base_lin_vel" in PRIV_LAYOUT.names
    assert "base_lin_vel" not in OBS_LAYOUT.names
