import torch

from pgmt.envs.recovery import FallRecoveryPool


def _state():
    return {
        "qpos": torch.zeros(29), "qvel": torch.ones(29),
        "root_pos": torch.tensor([0., 0., .4]), "root_quat": torch.tensor([1., 0., 0., 0.]),
        "root_lin_vel": torch.zeros(3), "root_ang_vel": torch.zeros(3),
    }


def test_recovery_pool_bounds_and_curriculum_probability():
    pool = FallRecoveryPool(capacity=2, init_prob=.1, prob_max=.5, survival_window=2)
    pool.add(_state(), seq_idx=3, frame=4.5)
    pool.add(_state(), seq_idx=4, frame=2.)
    pool.add(_state(), seq_idx=5, frame=1.)
    assert len(pool) == 2
    assert pool.probability == .1
    pool.record_outcome(False)
    assert pool.probability > .1
    sample = pool.sample(3)
    assert sample["qpos"].shape == (3, 29)
    assert sample["seq_idx"].shape == (3,)
