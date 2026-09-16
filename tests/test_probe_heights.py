"""Height diagnostics must not infer whole-body contact from one ankle height."""

import numpy as np

from data.probe_heights import report


def test_negative_minimum_reports_existing_frames_not_entire_sequence(capsys):
    values = np.array([-0.2, 0.3, 0.4])
    report({"seq": "example", "T": 3, "reference_source": "stored_npz",
            "src_hip_z": values, "src_foot_z": values, "foot_rel": values,
            "argmin_foot_rel": 0, "foot_at_argmin_above_lowest": 0.0,
            "g1_pelvis_z": values, "g1_foot_z": values})
    output = capsys.readouterr().out
    assert "存在踝原点低于零平面的帧" in output
    assert "整段穿地" not in output
    assert "stored_npz" in output
