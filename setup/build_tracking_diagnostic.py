"""Freeze kinematically screened short clips before any learning comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pgmt.envs.reference_sampler import MotionDatabase


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(database, horizon=5.):
    rows = []
    for seq in database.seqs:
        name = seq['name']
        if not name.startswith(('aiming', 'walk')):
            continue
        category = 'low_motion' if name.startswith('aiming') else 'slow_walk'
        dt = seq['frame_time']
        window = int(np.ceil(horizon/dt)) + 1
        lookahead = int(np.ceil(.62/dt)) + 2
        velocity = np.gradient(seq['root_pos'], dt, axis=0)
        speed = np.linalg.norm(velocity[:, :2], axis=-1)
        rms = np.sqrt(np.mean(seq['qvel']**2, axis=-1))
        quat = seq['root_rot']/np.linalg.norm(seq['root_rot'], axis=-1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1-2*(quat[:,1:3]**2).sum(-1), -1, 1)))
        candidates = []
        for start in range(0, len(rms)-window-lookahead, max(1, round(1/dt))):
            sl = slice(start, start+window)
            metrics = dict(mean_root_speed_m_s=float(speed[sl].mean()),
                joint_speed_rms_p95_rad_s=float(np.quantile(rms[sl], .95)),
                max_root_tilt_deg=float(tilt[sl].max()),
                root_z_range_m=float(np.ptp(seq['root_pos'][sl,2])),
                max_joint_speed_rad_s=float(np.abs(seq['qvel'][sl]).max()))
            v, q95 = metrics['mean_root_speed_m_s'], metrics['joint_speed_rms_p95_rad_s']
            common = metrics['max_root_tilt_deg']<=15 and metrics['root_z_range_m']<=.12 and metrics['max_joint_speed_rad_s']<=15
            eligible = (v<=.25 and q95<=1.) if category=='low_motion' else (.2<=v<=.8 and q95<=1.5)
            if common and eligible:
                candidates.append(dict(sequence=name, start_frame=start, horizon_s=horizon,
                    category=category, screening=metrics, score=q95+v))
        if candidates:
            rows.append(min(candidates, key=lambda x:(x['score'], x['start_frame'])))
    clips=[]
    for category in ('low_motion','slow_walk'):
        ranked=sorted((r for r in rows if r['category']==category), key=lambda r:(r['score'],r['sequence'],r['start_frame']))
        if len(ranked)<4:
            raise ValueError(f'need four distinct sequences for {category}, found {len(ranked)}')
        for rank, row in enumerate(ranked[:4]):
            clips.append(dict(row, clip_id=f'{category}_{rank}', split='train' if rank%2==0 else 'check'))
    return clips, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-data',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh manifest; do not overwrite a comparison')
    db=MotionDatabase(str(a.reference_data))
    clips, candidates=screen(db)
    manifest={'schema':'pgmt_fixed_clip_diagnostic_v1','control_dt':.02,'sim_dt':.005,'horizon_s':5.,
      'selection':{'basis':'kinematics only; no physics or policy outcomes used',
        'low_motion':'aiming sequences, mean root xy speed <=0.25 m/s, p95 per-frame joint-speed RMS <=1.0 rad/s; not stationary standing',
        'slow_walk':'walk sequences, mean root xy speed 0.2..0.8 m/s, p95 joint-speed RMS <=1.5 rad/s',
        'common':'max root tilt <=15 deg, root z range <=0.12 m, max absolute joint speed <=15 rad/s',
        'ranking':'one best 5s window per sequence, score=qvel RMS p95 + root speed; start grid=1s',
        'split':'four best distinct sequences per category; ranks 0,2 train; ranks 1,3 reserved check',
        'future_padding_s':.62, 'scope':'engineering diagnostic subset, not paper evaluation or guaranteed dynamically feasible motion'},
      'clips':clips, 'screened_sequence_candidates':candidates,
      'reference_files':{f.name:sha256(f) for f in sorted(a.reference_data.glob('*.npz'))},
      'sequence_names':[s['name'] for s in db.seqs], 'asset_sha256':sha256(a.asset),'urdf_sha256':sha256(a.urdf)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(clips,indent=2))


if __name__=='__main__':main()
