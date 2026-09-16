"""Finite hinge coordinates must never take a shortcut through a hard stop."""

from types import SimpleNamespace

import numpy as np
import pytest

import data.retarget_lafan1 as retargeting
from data.ik_refine import _finalize


def _export(entry, candidate, dt, monkeypatch):
    frames = len(candidate)
    if entry == "refine":
        data = {"qpos": candidate.copy(), "frame_time": np.float32(dt),
                "root_pos": np.zeros((frames, 3)),
                "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))}
        return _finalize(data, candidate)
    names = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
             "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
             "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
             "RightShoulder", "RightArm", "RightForeArm", "RightHand",
             "Neck", "Spine", "Spine2"]
    parents = np.zeros(len(names), dtype=int)
    parents[0] = -1
    bvh = SimpleNamespace(names=names, parents=parents, num_frames=frames,
                          frame_time=dt, joint_index=names.index)
    positions = np.zeros((frames, len(names), 3))
    rotations = np.zeros((frames, len(names), 4))
    rotations[..., 0] = 1.0
    bvh.fk = lambda unit_scale=1.0: (positions.copy(), rotations.copy())
    monkeypatch.setattr(retargeting, "_estimate_scale", lambda positions, bvh: 0.01)
    # Isolate export semantics from the unrelated source-rotation decomposition.
    monkeypatch.setattr(retargeting, "_gimbal_protect", lambda positions: candidate.copy())
    return retargeting.retarget(bvh)


@pytest.mark.parametrize("entry", ["retarget", "refine"])
def test_full_limit_traversal_uses_exported_coordinate_difference(entry, monkeypatch):
    lo = np.array([retargeting.G1_JOINT_LIMITS[n][0] for n in retargeting.G1_JOINT_NAMES])
    hi = np.array([retargeting.G1_JOINT_LIMITS[n][1] for n in retargeting.G1_JOINT_NAMES])
    assert (hi - lo > np.pi).any(), "the regression must exercise unwrap's wrong branch"
    out = _export(entry, np.stack([lo, hi]), 0.02, monkeypatch)
    expected = (hi.astype(np.float32).astype(float) - lo.astype(np.float32).astype(float)) / float(out["frame_time"])
    np.testing.assert_allclose(out["qvel"][1], expected, rtol=1e-6)
    # Independent literal oracle for left_hip_pitch: +5.4105 rad, not -0.8727.
    assert out["qvel"][1, 0] == pytest.approx(270.525, rel=1e-6)
    assert np.all(out["qvel"][1] > 0)
    assert not out["qvel"][0].any()


@pytest.mark.parametrize("entry", ["retarget", "refine"])
def test_out_of_range_positions_are_clipped_not_wrapped(entry, monkeypatch):
    candidate = np.stack([np.full(29, -7.0), np.full(29, 7.0)])
    out = _export(entry, candidate, 0.02, monkeypatch)
    lo = np.array([retargeting.G1_JOINT_LIMITS[n][0] for n in retargeting.G1_JOINT_NAMES])
    hi = np.array([retargeting.G1_JOINT_LIMITS[n][1] for n in retargeting.G1_JOINT_NAMES])
    np.testing.assert_allclose(out["qpos"][0], lo, atol=1e-6)
    np.testing.assert_allclose(out["qpos"][1], hi, atol=1e-6)
    np.testing.assert_array_equal(candidate[0], np.full(29, -7.0))
