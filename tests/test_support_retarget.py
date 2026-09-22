"""Meaningful constraints of the opt-in v3 and no-learning replay budget."""
import numpy as np
import pytest
import torch

from data.support_retarget import solve_support_height, contiguous_runs
from setup.replay_standing_events import validate_tape


def test_root_height_respects_nonfoot_collision_over_incompatible_stance():
    # Foot targets would request lowering by 10 cm; another body permits only 2 cm.
    feet = np.full((40, 2), .10)
    delta, _ = solve_support_height(feet, np.full(40, .02), np.ones((40, 2), bool), 1/30)
    assert np.min(delta+.02) >= 0
    assert np.min(feet+delta[:,None]) > .079


def test_root_height_supports_without_rewriting_source_joint_motion():
    t = np.arange(80)/30
    feet = np.repeat((.05+.01*np.sin(t))[:,None], 2, axis=1)
    delta, _ = solve_support_height(feet, feet[:,0], np.ones((80,2),bool), 1/30)
    assert np.max(feet+delta[:,None]) < .02
    assert np.min(feet+delta[:,None]) >= 0
    assert np.max(abs(np.diff(delta))) < .005


def test_support_bouts_keep_real_flight_gaps():
    assert contiguous_runs([True,True,False,False,True,False]) == [(0,2),(4,5)]


@pytest.mark.parametrize('tape', [torch.zeros(301,16,29), torch.zeros(300,4,29),
                                 torch.full((300,16,29),float('nan'))])
def test_bad_replay_budget_rejected_before_gpu(tape):
    with pytest.raises(ValueError):
        validate_tape({'actions':tape})
