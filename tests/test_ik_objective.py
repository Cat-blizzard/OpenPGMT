"""Analytic IK regressions independent of optional motion assets."""

import numpy as np
import pytest

import data.ik_refine as ik
from data.retarget_lafan1 import G1_JOINT_LIMITS, G1_JOINT_NAMES


def _data(value=0.0):
    qpos = np.zeros((1, 29))
    qpos[0, 0] = value
    return {"qpos": qpos, "root_pos": np.zeros((1, 3)),
            "root_rot": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "frame_time": np.float32(1 / 30)}


def _scalar_problem(monkeypatch, position, derivative):
    def fk(qpos, root_pos, root_rot):
        point = np.zeros((len(qpos), 3))
        point[:, 0] = position(qpos[:, 0])
        return {"point": point}, {"theta": qpos[:, 0]}

    def jacobian(pos, quat, bodies, names):
        result = np.zeros((len(pos["point"]), 3, 1))
        result[:, 0, 0] = derivative(quat["theta"])
        return result

    monkeypatch.setattr(ik, "_batch_fk", fk)
    monkeypatch.setattr(ik, "_batch_jacobian", jacobian)


def test_prior_converges_to_analytic_regularized_optimum(monkeypatch):
    # (theta - 1)^2 + (theta - 0)^2 has its unique optimum at 0.5.
    _scalar_problem(monkeypatch, lambda x: x, lambda x: np.ones_like(x))
    result = ik._refine(_data(), np.array([[[1.0, 0.0, 0.0]]]),
                        [0], ["point"], np.ones(1), lambda_smooth=1.0,
                        max_iter=10)
    assert result[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_acceptance_includes_prior_not_only_position_error(monkeypatch):
    # Exact nonlinear Jacobian: the first LM proposal slightly improves the
    # position residual but worsens the total regularized objective.
    _scalar_problem(monkeypatch,
                    lambda x: x * np.exp(-5 * x),
                    lambda x: (1 - 5 * x) * np.exp(-5 * x))
    result = ik._refine(_data(), np.array([[[1.0, 0.0, 0.0]]]),
                        [0], ["point"], np.ones(1), lambda_smooth=1.0,
                        max_iter=1)
    assert result[0, 0] == 0.0


@pytest.mark.parametrize("max_iter", [0, 1, 8])
def test_initial_limit_violation_is_projected_without_an_accepted_step(monkeypatch, max_iter):
    _scalar_problem(monkeypatch, np.zeros_like, np.zeros_like)
    original = _data(3.1)
    result = ik._refine(original, np.zeros((1, 1, 3)), [0], ["point"],
                        np.ones(1), max_iter=max_iter)
    assert result[0, 0] == G1_JOINT_LIMITS["left_hip_pitch"][1]
    assert original["qpos"][0, 0] == 3.1


def test_finalize_bounds_all_joints_and_recomputes_consistent_velocity():
    data = _data()
    data["qpos"] = np.zeros((3, 29))
    data["root_pos"] = np.zeros((3, 3))
    data["root_rot"] = np.tile(data["root_rot"], (3, 1))
    candidate = np.zeros((3, 29))
    candidate[0] = -10.0
    candidate[2] = 10.0
    out = ik._finalize(data, candidate)
    lo = np.array([G1_JOINT_LIMITS[n][0] for n in G1_JOINT_NAMES])
    hi = np.array([G1_JOINT_LIMITS[n][1] for n in G1_JOINT_NAMES])
    assert np.all(out["qpos"] >= lo - 1e-6)
    assert np.all(out["qpos"] <= hi + 1e-6)
    expected = np.diff(out["qpos"].astype(float), axis=0) / float(data["frame_time"])
    np.testing.assert_allclose(out["qvel"][1:], expected, atol=1e-4)
    assert not out["qvel"][0].any()
    assert out["contacts"].shape == (3, 2)
