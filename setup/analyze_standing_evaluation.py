"""CPU comparison of the deferred 24 jobs, using matched survival windows."""
import argparse
import json
from pathlib import Path

import torch

from pgmt.train.train_stage1 import _write_metrics
from setup.standing_multiseed_protocol import jobs,EVAL_TRANSITIONS


def common_window(initial,final):
    count=min(len(initial),len(final));rows=[]
    for eid in range(4):
        keep=[t for t in range(count) if initial[t]['active'][eid] and final[t]['active'][eid]]
        if not keep:raise ValueError('no common first-episode sample')
        row={'start_index':eid,'steps':len(keep),'duration_s':len(keep)*.02}
        for key in ('tilt_deg','joint_mse','body_mse'):
            means=[torch.stack([frames[t][key][eid] for t in keep]).mean() for frames in (initial,final)]
            label=key
            if key.endswith('_mse'):
                means=[v.sqrt() for v in means];label=key.replace('_mse','_rmse')
            row[label+'_initial_final']=[float(v) for v in means]
        rows.append(row)
    return rows


def physics_tail(trajectory,metrics):
    frames=trajectory['frames'];active=torch.stack([f['active'] for f in frames]);raw=trajectory['substep_qvel']
    if raw.shape!=(len(frames)*4,4,29):raise ValueError('incomplete evaluation substep trace')
    velocity=raw[active.repeat_interleave(4,dim=0)].abs().amax(-1)
    torque=torch.stack([f['physics']['estimated_torque'] for f in frames])[active].abs()
    ratio=torque/torch.tensor(metrics['environment_config']['torque_limit'])
    return {'first_episode_substep_speed_p99_max_rad_s':[float(velocity.quantile(.99)),float(velocity.max())],
            'first_episode_estimated_torque_peak_limit_fraction':float(ratio.max()),
            'first_episode_estimated_torque_saturated_fraction':float((ratio>=.99).float().mean())}


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    root=a.training_root/'deferred_evaluation';rows=[];total=0;plan_hashes=set()
    for job in jobs():
        for mode in ('deterministic','stochastic'):
            reports=[];trajectories=[];initials=[]
            for when in ('initial','final'):
                folder=root/f"{job['name']}_{when}_{mode}"
                m=json.loads((folder/'metrics.json').read_text())
                if m['status']!='completed' or m['job']['job']!=folder.name:raise ValueError('incomplete/misidentified evaluation job')
                traj=torch.load(folder/'trajectory.pt',map_location='cpu',weights_only=False)
                if m['executed_transitions']!=len(traj['frames'])*4 or not 0<m['executed_transitions']<=2000:
                    raise ValueError('evaluation transition accounting mismatch')
                total+=m['executed_transitions'];plan_hashes.add(m['plan_sha256'])
                reports.append(m);trajectories.append(traj)
                initials.append(torch.load(folder/'initial_evidence.pt',map_location='cpu',weights_only=False))
            if reports[0]['environment_config']!=reports[1]['environment_config']:raise ValueError('unmatched environment configuration')
            for key in initials[0]['observations']:
                torch.testing.assert_close(initials[0]['observations'][key],initials[1]['observations'][key],atol=0,rtol=0)
            if not torch.equal(initials[0]['latent_noise'],initials[1]['latent_noise']):raise ValueError('unmatched evaluation noise')
            rows.append({**job,'mode':mode,'initial_final_episodes':[m['episodes'] for m in reports],
                'completed_10s_initial_final':[m['completed_10s'] for m in reports],
                'common_window_per_start':common_window(*(t['frames'] for t in trajectories)),
                'physics_initial_final':[physics_tail(t,m) for t,m in zip(trajectories,reports)]})
    if len(plan_hashes)!=1 or total>EVAL_TRANSITIONS:raise ValueError('mixed evaluation plans or excessive budget')
    _write_metrics(a.output,{'status':'completed','pairs':rows,'actual_transitions':total,'budget_transitions':EVAL_TRANSITIONS,
        'scope':'three training seeds reported individually; four starts are not independent training seeds'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('training-root','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
