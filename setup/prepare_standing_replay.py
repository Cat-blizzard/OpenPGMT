"""CPU event alignment and bounded action tapes from the completed standing run."""
import argparse
import json
from pathlib import Path

import torch

from pgmt.envs.g1_env import G1_JOINT_NAMES, REQUIRED_BODY_NAMES
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for condition, event_step, eid in [('control', 54, 11), ('candidate', 274, 5)]:
        folder = source / condition
        frames, actions, raw, fingerprints = [], [], [], {}
        for update in range(1, 14):
            path = folder / 'captures' / f'update_{update:02d}_before.pt'
            b = torch.load(path, map_location='cpu', weights_only=False)
            frames.extend(b['frames']); actions.append(b['storage']['actions']); raw.append(b['substep_qvel'])
            fingerprints[str(path)] = sha256(path)
        actions, raw, frames = torch.cat(actions)[:300], torch.cat(raw)[:1200], frames[:300]
        assert actions.shape == (300, 16, 29) and raw.shape == (1200, 16, 29)
        expected = {k: torch.stack([f['physics']['state'][k] for f in frames]) for k in
                    ('joint_pos', 'joint_vel', 'root_pos', 'root_quat', 'root_lin_vel', 'contact_forces')}
        expected.update({k: torch.stack([f[k] for f in frames]) for k in ('terminated', 'timeouts')})
        expected['episode_steps'] = torch.stack([f['physics']['episode_steps'] for f in frames])
        torch.save({'actions': actions, 'expected': expected, 'substep_qvel': raw,
                    'source_sha256': fingerprints, 'source_condition': condition}, output / (condition + '.pt'))
        j = G1_JOINT_NAMES.index('left_ankle_roll')
        context = []
        for t in range(event_step - 6, event_step + 5):
            p = frames[t]['physics']; s = p['state']; force = s['contact_forces'][eid].norm(dim=-1)
            context.append({'control_step': t + 1, 'episode_step': int(p['episode_steps'][eid]),
                'target_rad': float(actions[t, eid, j]),
                'target_delta_rad': float(actions[t, eid, j] - actions[t-1, eid, j]),
                'joint_pos_rad': float(s['joint_pos'][eid, j]), 'joint_vel_rad_s': float(s['joint_vel'][eid, j]),
                'substep_velocities_rad_s': raw[t*4:t*4+4, eid, j].tolist(),
                'root_height_m': float(s['root_pos'][eid, 2]),
                'root_vertical_velocity_m_s': float(s['root_lin_vel'][eid, 2]),
                'estimated_torque_nm': float(p['estimated_torque'][eid, j]),
                'contacts_n': {name: float(force[k]) for k, name in enumerate(REQUIRED_BODY_NAMES) if float(force[k]) > 1},
                'terminated': bool(frames[t]['terminated'][eid])})
        rows.append({'condition': condition, 'event_step': event_step, 'env_id': eid, 'context': context})
    _write_metrics(output / 'summary.json', {'status': 'completed', 'events': rows,
        'replay_needed': True, 'missing': ['substep positions surrounding the peak',
            'substep contacts', 'substep torque provenance and readback'],
        'tape_steps_per_condition': 300, 'maximum_physical_transitions': 9600,
        'source_sha256': sha256(__file__)})
    print('CPU context and two 300-step action tapes complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
