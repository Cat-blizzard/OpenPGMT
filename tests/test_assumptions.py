"""assumptions.py 结构校验：编号齐全、取值自洽、可序列化。"""

import json

import pytest

from pgmt.cfg import assumptions as A
from pgmt.cfg.assumptions import ASSUMPTIONS, dump, get


def test_all_assumptions_registered_in_order():
    ids = list(ASSUMPTIONS)
    assert ids == [f"A{i}" for i in range(1, 18)], "A1–A17 应齐全且有序"


def test_duplicate_ids_impossible():
    assert len(ASSUMPTIONS) == len({a.aid for a in ASSUMPTIONS.values()})


def test_A1_control_frequency():
    v = get("A1").value
    assert v.dt == pytest.approx(0.02)
    assert v.steps_per_episode == 1500  # 30 s episode


def test_A2_future_ref_offsets():
    v = get("A2").value
    assert v.K == 6
    assert list(v.offsets) == [2**k - 1 for k in range(6)] == [0, 1, 3, 7, 15, 31]


def test_A3_A4_model_scale():
    assert get("A3").value.token_dim == 256
    assert get("A3").value.mhca_heads == 4
    assert list(get("A4").value.actor) == [512, 256, 128]
    assert list(get("A4").value.critic) == [512, 256, 128]


def test_A5_rope_consistent_with_heads():
    rope = get("A5").value
    assert rope.dim % 2 == 0
    assert rope.dim % get("A3").value.mhca_heads == 0  # 每头 16 维


def test_A6_ppo():
    v = get("A6").value
    assert v.clip_param == pytest.approx(0.2)
    assert v.gamma == pytest.approx(0.99)
    assert v.lam == pytest.approx(0.95)
    assert v.num_steps_per_env == 24


def test_A7_terrain_difficulty_monotonic_and_bounded():
    v = get("A7").value
    families = {
        "slope_deg": (5.0, 30.0),
        "stairs_h_cm": (4.0, 24.0),
        "boxes_h_cm": (5.0, 40.0),
        "rough_amp_cm": (1.0, 12.0),
        "flat_randomize": (0.0, 1.0),
    }
    for name, (lo, hi) in families.items():
        vals = getattr(v, name)
        assert len(vals) == 10, f"{name} 应有 L0–L9 共 10 级"
        assert vals[0] == pytest.approx(lo) and vals[-1] == pytest.approx(hi)
        assert all(a <= b for a, b in zip(vals, vals[1:])), f"{name} 应单调不减"


def test_A8_elevation_noise_sane():
    v = get("A8").value
    assert 0 < v.sigma_min <= v.sigma_max
    assert 0 <= v.dropout_prob_max < 1
    assert v.map_size == 21 and v.map_res == pytest.approx(0.1)


def test_A10_A12_A13_bounds():
    pool = get("A10").value
    assert pool.capacity > 0 and 0 < pool.init_prob <= pool.prob_max <= 1

    relax = get("A12").value
    assert all(0.0 <= c <= 1.0 for c in relax.chi.values())
    assert relax.chi["flat"] == 0.0  # 平地不松弛
    assert relax.tau_saturation > 0

    gpc = get("A13").value
    assert gpc.gate_v1 > gpc.gate_v0 >= 0
    assert gpc.clip_v > 0 and gpc.lambda_pos > 0


def test_A15_glimpse():
    v = get("A15").value
    assert v.num_glimpses == 4 and v.patch_size == 5
    assert v.loc_extent > 0
    assert all(d > 0 for d in v.selector_hidden)
    assert all(d > 0 for d in v.token_hidden)


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("A99")


def test_dump_json_serializable():
    d = dump()
    assert set(d) == set(ASSUMPTIONS)
    json.dumps(d)  # 不抛异常即通过
    assert d["A1"]["value"]["rate_hz"] == 50.0
    assert d["A7"]["value"]["stairs_h_cm"] == [4.0, 6.22, 8.44, 10.67, 12.89,
                                               15.11, 17.33, 19.56, 21.78, 24.0]
