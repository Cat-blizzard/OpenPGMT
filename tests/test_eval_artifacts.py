"""Evaluation must measure the artifact consumed by training."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import data.eval_retarget as evaluation
from data.retarget_lafan1 import G1_JOINT_NAMES


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    names = list(dict.fromkeys(src for src, _, _ in evaluation.KEYPOINTS))
    bvh = SimpleNamespace(num_frames=3, frame_time=1 / 30,
                          joint_index=names.index)
    source_pos = np.zeros((3, len(names), 3))
    source_rot = np.zeros((3, len(names), 4))
    source_rot[..., 0] = 1.0
    bvh.fk = lambda unit_scale=1.0: (source_pos.copy(), source_rot.copy())
    data = {"joint_names": np.array(G1_JOINT_NAMES),
            "qpos": np.zeros((3, 29), dtype=np.float32),
            "qvel": np.zeros((3, 29), dtype=np.float32),
            "root_pos": np.tile([0.0, 0.0, 0.8], (3, 1)).astype(np.float32),
            "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)).astype(np.float32),
            "contacts": np.zeros((3, 2), dtype=bool),
            "frame_time": np.float32(bvh.frame_time), "scale": np.float32(0.01)}
    path = tmp_path / "walk_test.npz"
    np.savez(path, **data)
    monkeypatch.setattr(evaluation, "NPZ_DIR", str(tmp_path))
    monkeypatch.setattr(evaluation, "load_bvh", lambda path: bvh)
    return bvh, data, path


def test_default_evaluation_reads_npz_without_running_retarget(artifact, monkeypatch):
    _, data, path = artifact
    def forbidden(*args, **kwargs):
        pytest.fail("default evaluation silently regenerated the reference")
    monkeypatch.setattr(evaluation, "retarget", forbidden)
    before = evaluation.evaluate("walk_test")
    data["root_pos"][:, 2] += 0.25
    np.savez(path, **data)
    after = evaluation.evaluate("walk_test")
    assert before["reference_source"] == after["reference_source"] == "stored_npz"
    assert after["min_foot_z"] - before["min_foot_z"] == pytest.approx(0.25)


def test_missing_artifact_does_not_fall_back_to_regeneration(artifact, monkeypatch):
    bvh, _, path = artifact
    path.unlink()
    monkeypatch.setattr(evaluation, "retarget", lambda bvh: pytest.fail("regenerated"))
    with pytest.raises(FileNotFoundError):
        evaluation.load_reference("walk_test", bvh)


def test_regeneration_is_explicit_and_records_its_source(artifact, monkeypatch):
    import data.ik_refine as ik
    _, data, path = artifact
    path.unlink()
    monkeypatch.setattr(evaluation, "retarget", lambda bvh: data)
    monkeypatch.setattr(ik, "refine_full", lambda raw, bvh: raw)
    row = evaluation.evaluate("walk_test", regenerate=True)
    assert row["reference_source"] == "regenerated"
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("joint_names", np.array(G1_JOINT_NAMES[::-1])),
    ("qpos", np.zeros((2, 29))),
    ("qvel", np.full((3, 29), np.nan)),
    ("scale", np.float32(0.0)),
    ("frame_time", np.float32(1 / 60)),
])
def test_incompatible_artifacts_are_rejected(artifact, field, value):
    bvh, data, path = artifact
    data[field] = value
    np.savez(path, **data)
    with pytest.raises(ValueError, match=field):
        evaluation.load_reference("walk_test", bvh)


def test_regeneration_report_does_not_overwrite_artifact_report():
    assert evaluation.resolve_out_path(None, None, regenerate=True) == (
        "data/processed/quality_report_regenerated.csv")
    assert evaluation.resolve_out_path("ground", None, regenerate=True) == (
        "data/processed/quality_report_regenerated_ground.csv")
    with pytest.raises(ValueError):
        evaluation.resolve_out_path(None, evaluation.FULL_REPORT_PATH, regenerate=True)


def test_stored_contact_label_consistency_is_reported(artifact):
    _, data, path = artifact
    before = evaluation.evaluate("walk_test")
    data["contacts"] = ~data["contacts"]
    np.savez(path, **data)
    after = evaluation.evaluate("walk_test")
    assert before["contact_label_agree"] + after["contact_label_agree"] == pytest.approx(100.0)
    assert before["contact_label_agree"] != after["contact_label_agree"]


def test_explicit_directories_do_not_mutate_module_defaults(artifact, monkeypatch):
    bvh, _, path = artifact
    seen = []
    original_source = evaluation.SRC_DIR
    monkeypatch.setattr(evaluation, "NPZ_DIR", "nonexistent_default")
    monkeypatch.setattr(evaluation, "load_bvh", lambda p: seen.append(p) or bvh)
    row = evaluation.evaluate("walk_test", bvh_dir="custom_source", npz_dir=str(path.parent))
    assert row["reference_source"] == "stored_npz"
    assert seen == [str(Path("custom_source") / "walk_test.bvh")]
    assert evaluation.NPZ_DIR == "nonexistent_default"
    assert evaluation.SRC_DIR == original_source


def test_cli_passes_custom_directories_to_evaluation(artifact, tmp_path, monkeypatch):
    _, data, _ = artifact
    expected = evaluation.evaluate("walk_test")
    source = tmp_path / "source"
    fixed = tmp_path / "fixed"
    source.mkdir()
    fixed.mkdir()
    np.savez(fixed / "walk_test.npz", **data)
    output = tmp_path / "report.csv"
    calls = []
    def evaluate_spy(name, **kwargs):
        calls.append((name, kwargs))
        return expected
    monkeypatch.setattr(evaluation, "evaluate", evaluate_spy)
    monkeypatch.setattr("sys.argv", ["eval_retarget", "--bvh-dir", str(source),
                                   "--npz-dir", str(fixed), "--out", str(output)])
    evaluation.main()
    assert calls == [("walk_test", {"regenerate": False,
                                    "bvh_dir": str(source), "npz_dir": str(fixed)})]
    assert output.exists()
