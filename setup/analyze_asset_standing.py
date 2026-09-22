"""Paired analysis of the frozen four static-load comparisons; no GPU use."""
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


def run(root, output):
    if output.with_suffix('.json').exists() or output.with_suffix('.png').exists():raise FileExistsError(output)
    rows=[];all_frames=[];initial=[];weights=[];series=[]
    for condition,mode in [('control','fixed'),('control','deterministic'),('candidate','fixed'),('candidate','deterministic')]:
        folder=root/f'{condition}_{mode}_v1'
        m=json.loads((folder/'metrics.json').read_text())
        if m['status']!='completed':raise ValueError(f'incomplete run {folder}')
        fs=torch.load(folder/'physical_trace.pt',map_location='cpu',weights_only=False)['frames'];all_frames.append(fs)
        def stack(key):return torch.stack([f[key] for f in fs]).numpy()
        active=np.array([f['first_episode_active'] for f in fs]);q=stack('qpos');rq=stack('root_quat')
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(rq[...,1:3]**2).sum(-1),-1,1)))
        ratio=abs(stack('estimated_torque'))/np.array(m['environment_config']['torque_limit'])
        cost=-.001*((stack('ee_accel')-stack('reference_ee_accel'))**2).sum((-1,-2))
        saved=torch.stack([f['weighted_aux_terms']['ee_accel_mismatch'] for f in fs]).numpy()
        np.testing.assert_allclose(cost,saved,rtol=2e-6,atol=2e-6)
        # GPU reciprocal multiplication and NumPy division can differ by one ULP.
        np.testing.assert_allclose(stack('ee_accel'),(stack('ee_lin_vel')-stack('previous_ee_lin_vel'))/.02,rtol=2e-7,atol=2e-6)
        history_error=0.;reset_error=0.;reset_count=0
        for i,f in enumerate(fs):
            first=f['episode_steps'].numpy()==1
            if first.any():
                reset_count+=int(first.sum());reset_error=max(reset_error,float(abs(f['previous_ee_lin_vel'][first]).max()))
            if i:
                keep=~first
                if keep.any():history_error=max(history_error,float(abs(f['previous_ee_lin_vel'][keep]-fs[i-1]['ee_lin_vel'][keep]).max()))
        if history_error or reset_error:raise ValueError('EE velocity history crosses time/reset boundaries')
        contact=stack('foot_contact');forces=stack('contact_forces')
        row={'condition':condition,'mode':mode,'first_episode_ends':m['first_episode_ends'],
             'mean_duration_s':float(np.mean([e['duration_s'] for e in m['first_episode_ends']])),
             'completed_10s':sum(not e['terminated'] and e['duration_s']>=9.999 for e in m['first_episode_ends']),
             'transitions':m['transitions'],'first_episode_transitions':int(active.sum()),
             'first_episode_joint_rmse_rad':float(np.sqrt((q[active]**2).mean())),
             'first_episode_mean_tilt_deg':float(tilt[active].mean()),'first_episode_max_tilt_deg':float(tilt[active].max()),
             'first_episode_contact_fraction':contact[active].mean(0).tolist(),
             'first_episode_peak_contact_force_n':float(np.linalg.norm(forces,axis=-1)[active].max()),
             'first_episode_peak_torque_fraction':float(ratio[active].max()),
             'first_episode_torque_saturation_fraction':float((ratio[active]>=.99).mean()),
             'first_episode_ee_accel_cost_mean':float(saved[active].mean()),
             'initial_target_deviation_from_frozen_rms_rad':float(np.sqrt(((stack('target')[0]-np.array(m['frozen_target']))**2).mean())),
             'substeps':m['substeps'],'runtime_fk_readback':m['runtime_fk_readback'],
             'ee_audit':{'history_error_m_s':history_error,'reset_history_error_m_s':reset_error,
                'first_step_snapshots':reset_count,'reward_reconstruction_passed':True,
                'static_reference_max_accel_m_s2':float(abs(stack('reference_ee_accel')).max()),
                'actual_peak_accel_m_s2':float(abs(stack('ee_accel')[active]).max())},
             'metrics_sha256':sha256(folder/'metrics.json'),'trace_sha256':sha256(folder/'physical_trace.pt')}
        rows.append(row);initial.append(m['initial'])
        weights.append(torch.load(folder/'initial_policy.pt',map_location='cpu',weights_only=True))
        stop=m['first_episode_ends'][0]['step']
        series.append({'time':np.arange(1,stop+1)*.02,'tilt':tilt[:stop,0],
            'root_z':stack('root_pos')[:stop,0,2],'ee_cost':saved[:stop,0]})
    for key in ('qpos','qvel','root_pos','root_quat','reference_qpos','reference_qvel','reference_frame'):
        if not all(np.array_equal(initial[0][key],s[key]) for s in initial[1:]):raise ValueError(f'unpaired initial {key}')
    differences=[[k for k in weights[0] if not torch.equal(weights[0][k],w[k])] for w in weights]
    if differences[1] or differences[2]!=differences[3] or len(differences[2])!=1 or not differences[2][0].startswith('actor.') or not differences[2][0].endswith('.bias'):
        raise ValueError('policy changed beyond the pre-registered actor target bias')
    common=[min(r['first_episode_ends'][e]['step'] for r in rows) for e in range(4)]
    for row,fs in zip(rows,all_frames):
        selected=[f for f in fs]
        total=sum(common)
        sq=sum(float((selected[t]['qpos'][e]**2).sum()) for e in range(4) for t in range(common[e]))
        tilt=[];ee=[]
        for e in range(4):
            for t in range(common[e]):
                rq=selected[t]['root_quat'][e].numpy()
                tilt.append(np.rad2deg(np.arccos(np.clip(1-2*(rq[1:3]**2).sum(),-1,1))))
                ee.append(float(selected[t]['weighted_aux_terms']['ee_accel_mismatch'][e]))
        row['common_window']={'joint_rmse_rad':float(np.sqrt(sq/(29*total))),
            'mean_tilt_deg':float(np.mean(tilt)),'ee_accel_cost_mean':float(np.mean(ee))}
    summary={'status':'completed','scope':'four prescribed starts, seed 0; no PPO or training benefit claim',
        'controls':rows,'total_transitions':sum(r['transitions'] for r in rows),'budget':8000,
        'paired_initial_states_passed':True,'policy_differences_from_control':differences,
        'common_window_steps_per_start':common,'physical_ppo_updates':0,
        'candidate':json.loads((root/'fixture_v4/candidate.json').read_text()),
        'asset_audit':json.loads((root/'fixture_v4/asset_audit.json').read_text()),
        'reference_generation':json.loads((root/'reference_v1/summary.json').read_text()),
        'reference_runtime':json.loads((root/'reference_runtime_audit_v1.json').read_text()),
        'gpu_release':json.loads((root/'gpu_release_v1.json').read_text()),
        'notes':['Implicit applied_torque is an actuator estimate, not a complete solver torque measurement.',
                 'Substep velocities were read every 5 ms; peaks/counts and threshold events are saved, not a full substep time series.',
                 'Full first-episode means have unequal survival windows; use the paired common-window metrics.',
                 'One static contact force distribution is feasible; unilateral/contact dynamics need not realize it.'],
        'source_script_sha256':sha256(__file__),'source_root':str(root.resolve())}
    if summary['total_transitions']>8000 or summary['gpu_release']['run_pids_alive']:raise ValueError('budget/release failure')
    _write_metrics(output.with_suffix('.json'),summary)
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for row,s in zip(rows,series):
        label=row['condition']+' / '+row['mode']
        axs[0,0].plot(s['time'],s['tilt'],label=label)
        axs[0,1].plot(s['time'],s['root_z'],label=label)
    axs[0,0].set(title='Upright start: tilt',xlabel='s',ylabel='deg');axs[0,0].legend(fontsize=8)
    axs[0,1].set(title='Upright start: root height',xlabel='s',ylabel='m')
    x=np.arange(4)
    for i,row in enumerate(rows):axs[1,0].bar(x+(i-1.5)*.2,[e['duration_s'] for e in row['first_episode_ends']],.2,label=row['condition']+'/'+row['mode'])
    axs[1,0].set_xticks(x,['upright','roll +1','roll -1','pitch +1'])
    axs[1,0].set(title='First episodes (all below 10 s)',ylabel='s')
    clips=summary['reference_generation']['clips']
    axs[1,1].bar(np.arange(8),[c['foot_min_clearance_m']*100 for c in clips])
    axs[1,1].set_xticks(np.arange(8),['low0','low1','low2','low3','walk0','walk1','walk2','walk3'])
    axs[1,1].set(title='New clips: minimum foot clearance',ylabel='cm')
    axs[1,1].axhline(2,color='red',lw=1,label='contact tolerance');axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Asset-bound references and static-load target | PD 80/2 | no learning')
    fig.savefig(output.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps({'transitions':summary['total_transitions'],'controls':[
        {k:r[k] for k in ('condition','mode','mean_duration_s','completed_10s','common_window')} for r in rows]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
