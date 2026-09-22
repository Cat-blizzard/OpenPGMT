"""Extract transient joint-speed events from saved 5 ms traces on CPU."""
import argparse
import json
from pathlib import Path

import torch

from pgmt.envs.g1_env import G1_JOINT_NAMES
from pgmt.train.fixed_clip_diagnostic import sha256


def run(source, output, threshold):
    if output.exists():
        raise FileExistsError(output)
    events = []
    for condition in ('control', 'candidate'):
        offset = 0
        for path in sorted((source / condition / 'captures').glob('update_*_before.pt')):
            capture = torch.load(path, map_location='cpu', weights_only=False)
            raw = capture['substep_qvel']
            if raw is None:
                raise ValueError('missing full physical substep trace')
            for s, env, joint in (raw.abs() > threshold).nonzero().tolist():
                tick, substep = divmod(s, 4)
                frame = capture['frames'][tick]
                physical = frame['physics']
                state = physical['state']
                events.append({
                    'condition': condition, 'update': capture['update'],
                    'control_step': offset + tick + 1, 'substep': substep + 1,
                    'env_id': env, 'joint': G1_JOINT_NAMES[joint],
                    'peak_signed_rad_s': float(raw[s, env, joint]),
                    'episode_step': int(physical['episode_steps'][env]),
                    'control_tick_joint_pos_rad': float(state['joint_pos'][env, joint]),
                    'control_tick_joint_vel_rad_s': float(state['joint_vel'][env, joint]),
                    'target_rad': float(physical['target'][env, joint]),
                    'estimated_torque_nm': float(physical['estimated_torque'][env, joint]),
                    'joint_limit_rad': [float(state[key][joint]) for key in ('joint_low', 'joint_high')],
                    'within_tick_velocities_rad_s': raw[tick * 4:tick * 4 + 4, env, joint].tolist(),
                    'pre_reset_root_height_m': float(state['root_pos'][env, 2]),
                    'pre_reset_root_vertical_velocity_m_s': float(state['root_lin_vel'][env, 2]),
                    'terminated': bool(frame['terminated'][env]),
                    'source_file': str(path), 'source_sha256': sha256(path),
                })
            offset += len(capture['frames'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'status': 'completed', 'threshold_rad_s': threshold, 'events': events,
        'indexing': 'control_step/substep/episode_step are one-based; env_id is zero-based',
        'interpretation': 'Transient peaks between control ticks; causal attribution to controller '
                          'versus contact solver is not established. Torque and position are '
                          'end-of-control-tick observations, not measurements at the peak substep.',
        'source_sha256': sha256(__file__),
    }, indent=2) + '\n')
    print(f'{len(events)} joint-substep events saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=45.0)
    args = parser.parse_args()
    run(args.source, args.output, args.threshold)
