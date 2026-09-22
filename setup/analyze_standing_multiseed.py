"""CPU audits and training-only summaries of the six frozen standing runs."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmt.envs.g1_env import G1_JOINT_NAMES,REQUIRED_BODY_NAMES
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_standing_double_values import audit_run
from setup.standing_multiseed_protocol import jobs,check_pair,TRAIN_TRANSITIONS


def event_context(folder):
    events=[]
    for u in range(1,101):
        b=torch.load(folder/'captures'/f'update_{u:02d}_before.pt',map_location='cpu',weights_only=False)
        v=b['substep_qvel'];frames=b['frames']
        for sid,eid in (v.abs().amax(-1)>45).nonzero().tolist():
            t=sid//4;p=frames[t]['physics'];s=p['state'];j=int(v[sid,eid].abs().argmax())
            quat=s['root_quat'][eid];tilt=float(torch.rad2deg(torch.acos((1-2*quat[1:3].square().sum()).clamp(-1,1))))
            forces=s['contact_forces'][eid].norm(dim=-1)
            events.append({'update':u,'control_step':(u-1)*24+t+1,'substep':sid%4+1,'env_id':eid,
                'joint':G1_JOINT_NAMES[j],'velocity_rad_s':float(v[sid,eid,j]),
                'all_substeps_rad_s':v[t*4:t*4+4,eid,j].tolist(),
                'episode_step':int(p['episode_steps'][eid]),'root_height_m':float(s['root_pos'][eid,2]),
                'root_vertical_velocity_m_s':float(s['root_lin_vel'][eid,2]),'tilt_deg':tilt,
                'terminated_this_control_step':bool(frames[t]['terminated'][eid]),
                'control_tick_contact_n':{k:float(forces[i]) for i,k in enumerate(REQUIRED_BODY_NAMES) if forces[i]>1},
                'control_tick_estimated_torque_nm':float(p['estimated_torque'][eid,j]),
                'note':'raw velocity is substep evidence; contact/torque here are control-tick readback, not impact-time measurements'})
    return events


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    while True:
        ledger=json.loads((a.source/'execution_ledger.json').read_text())
        for job in jobs():
            folder=a.source/job['name'];dest=a.output/job['name']
            if (dest/'summary.json').exists():continue
            row=next((r for r in ledger['runs'] if r['name']==job['name']),None)
            if row is None or row.get('status')!='completed':continue
            audit_run(folder,dest)
            _write_metrics(dest/'speed_events.json',{'events':event_context(folder)})
        if ledger['status']!='running' or not a.watch:break
        time.sleep(5)
    if ledger['status']!='completed':
        _write_metrics(a.output/'status.json',{'status':'incomplete','training_ledger_status':ledger['status']});return
    rows=[]
    for job in jobs():
        s=json.loads((a.output/job['name']/'summary.json').read_text());s.update(job)
        m=json.loads((a.source/job['name']/'metrics.json').read_text())
        ends=json.loads((a.source/job['name']/'episode_evidence.json').read_text())['ended_episodes']
        s['training_windows']={}
        for label,first,last in [('first20',1,20),('last20',81,100)]:
            updates=m['updates'][first-1:last];eps=[e for e in ends if (first-1)*24<e['step']<=last*24]
            s['training_windows'][label]={'ended_episode_count':len(eps),
                'ended_duration_mean_s':float(np.mean([e['duration_s'] for e in eps])) if eps else None,
                'actor_steps_mean':float(np.mean([r['optimizer_steps'] for r in updates])),
                'kl_mean':float(np.mean([r['exact_kl'] for r in updates])),
                'kl_p99_mean':float(np.mean([r['kl_all_p99'] for r in updates])),
                'aux_terms_mean':{k:float(np.mean([r['weighted_aux_terms_mean'][k] for r in updates]))
                                  for k in updates[0]['weighted_aux_terms_mean']}}
        s['speed_events']=json.loads((a.output/job['name']/'speed_events.json').read_text())['events']
        rows.append(s)
    pairs=[check_pair(a.source,seed) for seed in (0,1,2)]
    release=json.loads((a.source/'gpu_release.json').read_text())
    if release['task_pids_alive']:raise ValueError('training GPU children remain alive')
    summary={'status':'completed','scope':'training rollouts and CPU interface audits; frozen-policy evaluation deferred',
        'value_replay':'float64 CPU forward, cast to stored float32; original comparison tolerances unchanged',
        'runs':rows,'pairs':pairs,'total_transitions':sum(r['transitions'] for r in rows),
        'budget_transitions':TRAIN_TRANSITIONS,'gpu_release':release,'evaluation_run':False}
    if summary['total_transitions']!=TRAIN_TRANSITIONS:raise ValueError('incorrect batch budget')
    _write_metrics(a.output/'summary.json',summary)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row in rows:
        u=row['updates'];x=[r['update'] for r in u]
        for ax,key in zip(axs.flat,('exact_kl','actor_steps','joint_rmse_rad','body_rmse_m')):
            ax.plot(x,[r[key] for r in u],label=row['name'],alpha=.8)
    for ax,title in zip(axs.flat,('Mean policy KL','Accepted actor optimizer steps','Rollout joint RMSE (rad)','Rollout body RMSE (m)')):
        ax.set_title(title);ax.set_xlabel('PPO update');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Three-seed standing diagnostic | training rollouts, evaluation deferred')
    fig.savefig(a.output/'summary.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--watch',action='store_true');run(p.parse_args())
