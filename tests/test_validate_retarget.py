"""Artifact-only checks for the stored LAFAN1 → G1 references."""

import json

import numpy as np

from data.retarget_lafan1 import G1_JOINT_LIMITS, G1_JOINT_NAMES
from data.validate_retarget import validate_artifacts


def _write_artifact(directory, name="walk_demo", frames=3, **overrides):
    data = {
        "joint_names": np.array(G1_JOINT_NAMES),
        "qpos": np.zeros((frames, 29), dtype=np.float32),
        "qvel": np.zeros((frames, 29), dtype=np.float32),
        "root_pos": np.tile([0.0, 0.0, 0.8], (frames, 1)).astype(np.float32),
        "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)).astype(np.float32),
        # Zero-pose G1 feet are below the contact height threshold, so the
        # stored labels for this synthetic static sequence are true.
        "contacts": np.ones((frames, 2), dtype=bool),
        "frame_time": np.float32(1 / 30),
        "scale": np.float32(0.01),
    }
    data.update(overrides)
    path = directory / f"{name}.npz"
    np.savez(path, **data)
    return path


def test_artifact_only_validation_and_ground_filter(tmp_path):
    _write_artifact(tmp_path, "walk_demo")
    _write_artifact(tmp_path, "ground1_subject1", frames=2)

    result = validate_artifacts(tmp_path)

    assert result["passed"]
    assert result["checked_sequences"] == 2
    ground = result["ground_filter"]
    assert ground["ground_sequences"] == 1
    assert ground["retained_sequences"] == 1
    assert ground["ground_frames_checked"] == 2
    assert ground["retained_frames_checked"] == 3


def test_schema_errors_are_failures_even_when_violations_are_allowed(tmp_path):
    _write_artifact(tmp_path, qpos=np.zeros((3, 28), dtype=np.float32))

    result = validate_artifacts(tmp_path, allow_violations=True)

    assert not result["passed"]
    assert any("qpos must have shape" in failure for failure in result["failures"])


def test_allow_violations_accepts_limit_and_velocity_diagnostics(tmp_path):
    qpos = np.zeros((3, 29), dtype=np.float32)
    qpos[0, 0] = G1_JOINT_LIMITS[G1_JOINT_NAMES[0]][1] + 0.2
    qvel = np.zeros((3, 29), dtype=np.float32)
    qvel[1, 0] = 4.0
    _write_artifact(tmp_path, qpos=qpos, qvel=qvel)

    strict = validate_artifacts(tmp_path)
    permitted = validate_artifacts(tmp_path, allow_violations=True)

    assert not strict["passed"]
    assert permitted["passed"]
    sequence = permitted["sequences"]["walk_demo"]
    assert "joint_limits" in sequence["failed_checks"]
    assert "qvel_direct_difference" in sequence["failed_checks"]


def test_cli_json_output_is_read_only(tmp_path, monkeypatch):
    _write_artifact(tmp_path)
    output = tmp_path / "validation.json"
    before = (tmp_path / "walk_demo.npz").read_bytes()
    import data.validate_retarget as module

    monkeypatch.setattr("sys.argv", ["validate_retarget", "--npz-dir", str(tmp_path),
                                      "--out", str(output)])
    assert module.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"]
    assert (tmp_path / "walk_demo.npz").read_bytes() == before
