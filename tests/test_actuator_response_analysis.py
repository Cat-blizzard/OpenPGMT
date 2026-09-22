"""First-episode analysis must not treat resets or censoring as successes."""
import numpy as np

from setup.analyze_actuator_response import prefix_steps, summarize


def probe():
    run = {"environment_config": {"control_dt": .02, "torque_limit": [5.]*29},
           "initial": {"joint_names": [str(i) for i in range(29)]},
           "ee_names": [str(i) for i in range(4)], "steps": [],
           "first_episode_end": [{"step": 2, "terminated": True, "reasons": ["joint_speed"]}, None]}
    for t in range(4):
        qvel = np.zeros((2, 29))
        qvel[0, 0] = 50 if t == 1 else 0
        if t > 1:
            qvel[0, 0] = 999  # belongs to a later episode, must be excluded
        actual, reference = np.zeros((2, 4, 3)), np.zeros((2, 4, 3))
        actual[:, :, 0], reference[:, :, 0] = 3, 1
        run["steps"].append({"pre_reset": {
            "qvel": qvel, "root_pos": np.ones((2, 3)),
            "state_ee_accel": actual, "reference_ee_accel": reference,
            "estimated_torque_nm": np.zeros((2, 29)), "episode_steps": [t+1]*2,
            "reference_seq_idx": [0, 1], "reference_frame": [t, t]},
            "per_env_tracking": {k: [4, 9] for k in ("joint_mse", "body_mse", "root_mse")}})
    return run


def test_first_episode_excludes_reset_outlier_and_preserves_censoring():
    run = probe()
    assert prefix_steps(run).tolist() == [2, 4]
    result = summarize(run)
    assert result["failures"] == 1
    assert result["censored_at_budget"] == 1
    assert result["observed_first_episode_s"] == [.04, .08]
    assert result["max_joint_speed_rad_s"] == 50
    assert result["overspeed_transition_count"] == 1
    assert result["max_state_ee_accel_m_s2"] == 3
    assert result["max_reference_ee_accel_m_s2"] == 1
    assert result["max_ee_error_m_s2"] == 2
    assert result["joint_rmse"] == np.sqrt((2*4+4*9)/6)


def test_common_prefix_comparison_uses_equal_transition_budget():
    result = summarize(probe(), np.array([1, 2]))
    assert result["observed_first_episode_s"] == [.02, .04]
    assert result["overspeed_transition_count"] == 0
    assert result["max_joint_speed_rad_s"] == 0
