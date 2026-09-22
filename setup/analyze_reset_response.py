"""Paired ordinary-reset analysis, holding the verified actuator profile fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from setup.analyze_actuator_response import arrays, prefix_steps, summarize


def matched_prefix(nominal, candidate):
    for key, value in nominal['environment_config'].items():
        if key != 'reset_mode' and candidate['environment_config'].get(key) != value:
            raise ValueError(f'unmatched configuration: {key}')
    for key in ('reference_seq_idx', 'reference_frame', 'reference_qpos', 'reference_qvel', 'sequence_names'):
        if nominal['initial'][key] != candidate['initial'][key]:
            raise ValueError(f'unmatched initial reference: {key}')
    if len(nominal['steps']) != len(candidate['steps']):
        raise ValueError('unmatched probe budgets')
    common = np.minimum(prefix_steps(nominal), prefix_steps(candidate))
    mask = np.arange(len(nominal['steps']))[:, None] < common[None]
    left, right = arrays(nominal), arrays(candidate)
    for key in ('target', 'reference_seq_idx', 'reference_frame'):
        if not np.array_equal(left[key][mask], right[key][mask]):
            raise ValueError(f'unmatched input before first reset: {key}')
    return common


def analyze(root):
    runs, failed = {}, []
    for path in sorted(root.glob('*/response.json')):
        run = json.loads(path.read_text())
        if run['status'] != 'completed':
            failed.append({'path': str(path), 'status': run['status'], 'error': run.get('error')})
            continue
        args = run['arguments']
        if (args['actuator_profile'] != 'asset_effort_v1' or args['target_mode'] != 'reference'
                or len(run['steps']) != args['steps'] or (path.parent/'exit_code').read_text().strip() != '0'):
            raise ValueError(f'invalid probe: {path}')
        key = (args['seed'], args['reset_mode'])
        if key in runs:
            raise ValueError(f'duplicate completed condition: {key}')
        readback = torch.load(path.with_suffix('.diagnostics')/'actuators_000.pt', map_location='cpu', weights_only=False)
        for name, expected in (('effort_nm', torch.tensor(run['environment_config']['torque_limit'])),
                               ('stiffness_nm_rad', 80.), ('damping_nm_s_rad', 2.)):
            torch.testing.assert_close(readback[name], torch.as_tensor(expected).expand_as(readback[name]))
        if args['reset_mode'] == 'reference_state':
            audit = run['reset_audit']
            if max(audit['state_max_abs_error'].values()) > 2e-4 or min(audit['collision_min_height_m']) < -2e-4:
                raise ValueError(f'failed reset readback: {path}')
        runs[key] = (run, path, readback)
    pairs = []
    for seed in sorted({key[0] for key in runs}):
        keys = [(seed, mode) for mode in ('nominal', 'reference_state')]
        if not all(key in runs for key in keys):
            continue
        (old, old_path, old_rb), (new, new_path, new_rb) = [runs[key] for key in keys]
        old_source = json.loads((old_path.parent/'manifest.json').read_text())['source_sha256']
        new_source = json.loads((new_path.parent/'manifest.json').read_text())['source_sha256']
        for name, digest in old_source.items():
            if name.startswith('pgmt/') or name in ('setup/check_actuator_response.py', 'setup/run_actuator_check.sh'):
                if digest != new_source.get(name):
                    raise ValueError(f'unmatched experiment source: {name}')
        torch.testing.assert_close(old_rb['velocity_rad_s'], new_rb['velocity_rad_s'])
        common = matched_prefix(old, new)
        pairs.append({'seed': seed, 'nominal_path': str(old_path), 'candidate_path': str(new_path),
            'reference_and_inputs_match': True, 'experiment_source_matches': True,
            'nominal': summarize(old), 'candidate': summarize(new), 'common_prefix_steps': common.tolist(),
            'common_prefix_nominal': summarize(old, common), 'common_prefix_candidate': summarize(new, common),
            'candidate_reset_audit': new['reset_audit']})
    groups = {}
    for mode in ('nominal', 'candidate'):
        metrics = {
            'mean_observed_first_episode_s': [np.mean(p[mode]['observed_first_episode_s']) for p in pairs],
            'common_joint_rmse_rad': [p['common_prefix_'+mode]['joint_rmse'] for p in pairs],
            'common_body_rmse_m': [p['common_prefix_'+mode]['body_rmse'] for p in pairs],
            'first_step_peak_state_ee_accel_m_s2': [p[mode]['first_step']['max_state_ee_accel_m_s2'] for p in pairs],
            'common_p95_ee_error_m_s2': [p['common_prefix_'+mode]['p95_ee_error_m_s2'] for p in pairs]}
        groups[mode] = {
            'seed_summary': {name: {'mean': float(np.mean(v)), 'sample_sd': float(np.std(v, ddof=1)) if len(v)>1 else None,
                                    'values': v} for name, v in metrics.items()} if pairs else {},
            'first_episode_failures': sum(p[mode]['failures'] for p in pairs),
            'censored_at_3s': sum(p[mode]['censored_at_budget'] for p in pairs),
            'first_episode_count': sum(len(p[mode]['observed_first_episode_s']) for p in pairs),
            'overspeed_transitions': sum(p[mode]['overspeed_transition_count'] for p in pairs),
            'max_joint_speed_rad_s': max((p[mode]['max_joint_speed_rad_s'] for p in pairs), default=None)}
    return {'scope': 'ordinary reference PD playback, no learning; first episodes only, 3-second censoring',
        'candidate': 'reference joint/root pose+velocity, complete command buffers, upward-only collision clearance',
        'seeds': [p['seed'] for p in pairs], 'groups': groups, 'pairs': pairs, 'failed_attempts': failed,
        'completed_runs': len(runs), 'unpaired_completed_runs': len(runs)-2*len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'seeds': result['seeds'], 'groups': result['groups'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
