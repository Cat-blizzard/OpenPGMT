"""Reset ablations permit initial state differences, never different targets."""
import copy

import pytest

from setup.analyze_reset_response import matched_prefix


def pair():
    run = {'environment_config': {'reset_mode':'nominal', 'stiffness':80., 'control_dt':.02},
           'initial': {'reference_seq_idx':[1], 'reference_frame':[0.], 'reference_qpos':[[.2]],
                       'reference_qvel':[[.3]], 'sequence_names':['motion'], 'qpos':[[0.]]},
           'first_episode_end': [{'step':2}],
           'steps':[{'pre_reset':{'target':[[.2]], 'reference_seq_idx':[1], 'reference_frame':[i+1.]}}
                    for i in range(3)]}
    other = copy.deepcopy(run)
    other['environment_config']['reset_mode'] = 'reference_state'
    other['initial']['qpos'] = [[.2]]
    other['first_episode_end'] = [None]
    return run, other


def test_initial_pose_and_post_reset_changes_are_allowed():
    old, new = pair()
    old['steps'][2]['pre_reset']['target'] = [[999.]]
    assert matched_prefix(old, new).tolist() == [2]


@pytest.mark.parametrize('change', ['config','reference','action'])
def test_rejects_unmatched_reference_or_pd_or_executed_target(change):
    old, new = pair()
    if change == 'config':
        new['environment_config']['stiffness'] = 90.
    elif change == 'reference':
        new['initial']['reference_frame'] = [1.]
    else:
        new['steps'][1]['pre_reset']['target'] = [[.3]]
    with pytest.raises(ValueError, match='unmatched'):
        matched_prefix(old, new)
