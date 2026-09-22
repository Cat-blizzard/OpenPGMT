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
    assert pool.probability == .1  # recovery failure must not increase difficulty
    sample = pool.sample(3)
    assert sample["qpos"].shape == (3, 29)
    assert sample["seq_idx"].shape == (3,)


def test_curriculum_requires_ordinary_samples_and_limits_probability_changes():
    pool = FallRecoveryPool()
    for _ in range(100):
        pool.record_outcome(False)
    assert pool.probability == .1
    for _ in range(19):
        pool.record_ordinary_episode(30., 30.)
    assert pool.probability == .1
    pool.record_ordinary_episode(30., 30.)
    assert abs(pool.probability - .12) < 1e-8
    for _ in range(1000):
        old = pool.probability
        pool.record_ordinary_episode(30., 30.)
        assert 0 <= pool.probability - old <= .020000001
    assert pool.probability == .5


def test_curriculum_resume_preserves_progress_and_separate_outcomes():
    pool = FallRecoveryPool()
    for _ in range(27):
        pool.record_ordinary_episode(15., 30.)
    pool.record_outcome(False)
    restored = FallRecoveryPool()
    restored.load_state_dict(pool.state_dict())
    assert restored.curriculum_metrics() == pool.curriculum_metrics()
    for _ in range(31):
        pool.record_ordinary_episode(30., 30.)
        restored.record_ordinary_episode(30., 30.)
    assert restored.curriculum_metrics() == pool.curriculum_metrics()
