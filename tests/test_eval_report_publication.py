"""A failed or interrupted audit must not replace the previous complete report."""

import csv
from pathlib import Path

import pytest

import data.eval_retarget as evaluation


def _row(name):
    return {"seq": name, "type": "walk", "frames": 2,
            "reference_source": "stored_npz", "fk_err_cm": 1.0,
            "fitted_err_cm": 1.0, "holdout_err_cm": 2.0,
            "upright_holdout_err_cm": 3.0, "root_rot_err_deg": 0.0,
            "min_foot_z": 0.01, "spike_pct": 0.0, "contact_agree": 100.0,
            "worst_holdout_kp": "LeftToe", "limit_over_pct": 0.0}


@pytest.fixture
def report_run(tmp_path, monkeypatch):
    inputs = tmp_path / "motions"
    inputs.mkdir()
    for name in ("walk_a", "walk_b"):
        (inputs / (name + ".npz")).touch()
    output = tmp_path / "quality_report.csv"
    output.write_bytes(b"previous complete report\n")
    argv = ["eval_retarget", "--npz-dir", str(inputs), "--out", str(output)]
    monkeypatch.setattr("sys.argv", argv)
    return inputs, output, argv


def _partials(output):
    return list(output.parent.glob(output.name + ".*.partial.csv"))


def _read_rows(path):
    with path.open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("all_fail", [False, True])
def test_failure_keeps_previous_report_and_partial_results(report_run, monkeypatch, capsys, all_fail):
    _, output, _ = report_run
    before = output.read_bytes()
    def evaluate(name, **kwargs):
        if all_fail or name == "walk_b":
            raise ValueError("bad motion")
        return _row(name)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(evaluation.os, "replace", lambda *args: pytest.fail("published partial run"))
    assert evaluation.main() == 1
    assert output.read_bytes() == before
    partials = _partials(output)
    assert len(partials) == 1
    assert len(_read_rows(partials[0])) == (0 if all_fail else 1)
    assert str(partials[0]) in capsys.readouterr().out


def test_keyboard_interrupt_preserves_report_and_completed_rows(report_run, monkeypatch, capsys):
    _, output, _ = report_run
    before = output.read_bytes()
    def evaluate(name, **kwargs):
        if name == "walk_b":
            raise KeyboardInterrupt
        return _row(name)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    assert evaluation.main() == 130
    assert output.read_bytes() == before
    partials = _partials(output)
    assert len(partials) == 1
    assert [row["seq"] for row in _read_rows(partials[0])] == ["walk_a"]
    assert str(partials[0]) in capsys.readouterr().out


def test_complete_run_publishes_once_with_atomic_replace(report_run, monkeypatch):
    _, output, _ = report_run
    before = output.read_bytes()
    observed = []
    def evaluate(name, **kwargs):
        assert output.read_bytes() == before
        observed.append(name)
        return _row(name)
    replace = evaluation.os.replace
    replacements = []
    def publish(source, destination):
        assert output.read_bytes() == before
        assert observed == ["walk_a", "walk_b"]
        assert [row["seq"] for row in _read_rows(Path(source))] == observed
        replacements.append((source, destination))
        replace(source, destination)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(evaluation.os, "replace", publish)
    assert evaluation.main() == 0
    assert len(replacements) == 1
    assert Path(replacements[0][1]) == output
    assert [row["seq"] for row in _read_rows(output)] == observed
    assert not _partials(output)


@pytest.mark.parametrize("empty_input", [False, True])
def test_no_selected_sequences_fails_without_touching_report(report_run, monkeypatch, empty_input):
    inputs, output, argv = report_run
    before = output.read_bytes()
    if empty_input:
        for path in inputs.iterdir():
            path.unlink()
    else:
        argv.extend(["--only", "ground"])
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: pytest.fail("unexpected evaluation"))
    assert evaluation.main() == 1
    assert output.read_bytes() == before
    assert not _partials(output)
