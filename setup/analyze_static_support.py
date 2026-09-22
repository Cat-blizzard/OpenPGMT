"""Summarize first-episode standing controls, without counting resets as success."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry


def run(root, urdf, output):
    if output.with_suffix('.json').exists() or output.with_suffix('.png').exists():
        raise FileExistsError(output)
    geometry = SupportGeometry(urdf)
    rows, series = [], []
    initial_states = []
    reference = json.loads((root / 'reference_audit_v1/summary.json').read_text())
    learning = json.loads((root / 'rollout_audit_v1/summary.json').read_text())
    for mode in ('fixed', 'deterministic', 'stochastic'):
        folder = root / f'static_{mode}_v1'
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics['status'] != 'completed':
            raise ValueError(f'incomplete run: {mode}')
        frames = torch.load(folder / 'physical_trace.pt', weights_only=False, map_location='cpu')['frames']
        def stack(key):
            return torch.stack([f[key] for f in frames]).numpy()
        active = np.array([f['first_episode_active'] for f in frames])
        q, root_pos, root_quat = stack('qpos'), stack('root_pos'), stack('root_quat')
        target, torque = stack('target'), stack('estimated_torque')
        contact, foot_vel = stack('foot_contact'), stack('foot_vel')
        geom = geometry.compute(q.reshape(-1, 29), root_pos.reshape(-1, 3), root_quat.reshape(-1, 4))
        com = geom['com'].reshape(len(frames), 4, 3)
        feet = geom['foot_points'].reshape(len(frames), 4, 8, 3)
        # A deliberately generous geometric bound: all sphere centers of both
        # feet, irrespective of actual active contact or friction constraints.
        margin_x = np.minimum(com[..., 0]-feet[..., 0].min(-1), feet[..., 0].max(-1)-com[..., 0])
        tilt = np.rad2deg(np.arccos(np.clip(1-2*(root_quat[..., 1:3]**2).sum(-1), -1, 1)))
        limits = np.array(metrics['environment_config']['torque_limit'])
        force_ratio = abs(torque) / limits
        aux = {k: torch.stack([f['weighted_aux_terms'][k] for f in frames]).numpy()
               for k in frames[0]['weighted_aux_terms']}
        rows.append({'mode': mode, 'transitions_including_post_first_resets': metrics['transitions'],
            'first_episode_transitions': int(active.sum()),
            'first_episode_ends': metrics['first_episode_ends'],
            'completed_10s': sum(not e['terminated'] and e['duration_s'] >= 9.999 for e in metrics['first_episode_ends']),
            'substeps': metrics['substeps'], 'initial_action_rms_rad': float(np.sqrt((target[0]**2).mean())),
            'first_episode_joint_rmse_rad': float(np.sqrt((q[active]**2).mean())),
            'first_episode_contact_fraction': contact[active].mean(0).tolist(),
            'first_episode_contact_link_xy_speed_m_s': float(np.linalg.norm(foot_vel[..., :2], axis=-1)[contact & active[..., None]].mean()),
            'first_episode_peak_estimated_torque_limit_ratio': float(force_ratio[active].max()),
            'first_episode_estimated_torque_saturation_fraction': float((force_ratio[active] >= .99).mean()),
            'first_episode_weighted_aux_mean': {k: float(v[active].mean()) for k, v in aux.items()},
            'benchmark_com_outside_both_feet_x_bound_at_s': next((float((i+1)*.02) for i in range(len(frames))
                                                               if active[i, 0] and margin_x[i, 0] < 0), None),
            'initial_policy_sha256': sha256(folder / 'initial_policy.pt'),
            'metrics_sha256': sha256(folder / 'metrics.json'), 'physical_trace_sha256': sha256(folder / 'physical_trace.pt')})
        series.append({'time': (np.arange(len(frames)) + 1) * .02, 'root_z': root_pos[:, 0, 2],
                       'tilt': tilt[:, 0], 'com_margin_x': margin_x[:, 0]})
        initial_states.append(metrics['initial'])
    for key in ('qpos', 'qvel', 'root_pos', 'root_quat', 'reference_qpos', 'reference_qvel', 'reference_frame'):
        if not all(np.array_equal(initial_states[0][key], s[key]) for s in initial_states[1:]):
            raise ValueError(f'control initial states differ: {key}')
    weights = [torch.load(root / f'static_{mode}_v1/initial_policy.pt', weights_only=True, map_location='cpu')
               for mode in ('fixed', 'deterministic', 'stochastic')]
    if not all(all(torch.equal(w[k], weights[0][k]) for k in weights[0]) for w in weights[1:]):
        raise ValueError('initial policy weights differ')
    summary = {'status': 'completed', 'scope': 'three static controls, one seed and four prescribed starts; not training evaluation',
        'source_root': str(root.resolve()), 'source_sha256': sha256(__file__), 'controls': rows,
        'paired_initial_states_and_weights_passed': True, 'static_gate_passed': False,
        'physical_ppo_updates_run': 0, 'reference_audit': reference, 'rollout_audit': learning,
        'gpu_release': json.loads((root / 'gpu_release_v1.json').read_text()),
        'notes': ['Contact-link speed includes rolling; it is not exact contact-point slip.',
                  'URDF COM/foot-bound comparison is geometric, not a complete contact stability criterion.',
                  'Implicit applied_torque is an estimate, not a full constraint-solver torque measurement.',
                  'The three trajectories share a seed and four prescribed starts; do not report as independent seeds.']}
    benchmark = rows[0]['first_episode_ends'][0]
    summary['static_gate_passed'] = not benchmark['terminated'] and benchmark['duration_s'] >= 9.999
    _write_metrics(output.with_suffix('.json'), summary)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7), constrained_layout=True)
    colors = ['#386cb0', '#7a4b9a', '#e08214']
    for row, s, color in zip(rows, series, colors):
        for ax, key in zip(axes[0], ('root_z', 'tilt', 'com_margin_x')):
            ax.plot(s['time'], s[key], label=row['mode'], color=color)
            ax.set_xlabel('Time (s)'); ax.grid(alpha=.2)
    axes[0, 0].set(title='Upright start: root height', ylabel='m')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Upright start: root tilt', ylabel='degrees')
    axes[0, 2].set(title='COM vs both-foot x bound (geometric)', ylabel='margin (m)')
    axes[0, 2].axhline(0, color='black', lw=.7)
    x = np.arange(4)
    for i, (row, color) in enumerate(zip(rows, colors)):
        axes[1, 0].bar(x+(i-1)*.24, [e['duration_s'] for e in row['first_episode_ends']], .24, color=color, label=row['mode'])
    axes[1, 0].set_xticks(x, ['upright', 'roll +1', 'roll -1', 'pitch +1'])
    completions = ', '.join(str(row['completed_10s'])+'/4' for row in rows)
    longest = max(e['duration_s'] for row in rows for e in row['first_episode_ends'])
    axes[1, 0].set(title=f'First episodes: {completions} reach 10 s', ylabel='seconds', ylim=(0, max(1.8, longest*1.15)))
    cosine = np.array(learning['gradient_cosine'])
    axes[1, 1].imshow(cosine, vmin=-1, vmax=1, cmap='coolwarm')
    axes[1, 1].set_xticks(range(3), ['upper', 'lower', 'aux'])
    axes[1, 1].set_yticks(range(3), ['upper', 'lower', 'aux'])
    axes[1, 1].set_title('Actor/shared gradient cosine (264 samples)')
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{cosine[i,j]:.2f}', ha='center', va='center')
    clip_names = [c['clip_id'].replace('low_motion_', 'low ').replace('slow_walk_', 'walk ') for c in reference['clips']]
    axes[1, 2].barh(clip_names, [100*c['ik_pitch_at_limit_fraction'] for c in reference['clips']], color='#386cb0')
    axes[1, 2].set(title='Saved reference: waist at +0.52 rad', xlabel='frames (%)', xlim=(0, 105))
    fig.suptitle('Reference and static-control audit | A23, kp=80 / kd=2 | no GPU learning', fontsize=14)
    fig.savefig(output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ('status', 'static_gate_passed', 'paired_initial_states_and_weights_passed')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'urdf', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.urdf, a.output)
