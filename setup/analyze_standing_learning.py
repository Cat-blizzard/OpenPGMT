"""CPU replay of every saved standing rollout; no optimizer or simulator calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import kl_divergence

from pgmt.envs.g1_env import REQUIRED_BODY_NAMES,G1_JOINT_NAMES
from pgmt.rewards.batched import BatchedRewardComputer
from pgmt.rewards.spec import AUX
from pgmt.train.policy import Stage1Policy
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_captured_rollout import independent_gae


def audit_run(folder, output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    m=json.loads((folder/'metrics.json').read_text())
    if m['status']!='completed':raise ValueError('incomplete physical run')
    policy=Stage1Policy().eval();computer=BatchedRewardComputer(REQUIRED_BODY_NAMES,G1_JOINT_NAMES)
    errors={k:0. for k in ('action','log_prob','value','next_value','gae','advantage','reward','ee_reward','ee_history','kl')}
    rows=[];substeps=[];trace=[];previous_velocity=None;previous_after=None
    for u in range(1,m['completed_updates']+1):
        before_path=folder/'captures'/f'update_{u:02d}_before.pt';after_path=folder/'captures'/f'update_{u:02d}_after.pt'
        b=torch.load(before_path,map_location='cpu',weights_only=False);a=torch.load(after_path,map_location='cpu',weights_only=False)
        s=b['storage'];fs=b['frames'];T,N=s['actions'].shape[:2]
        if previous_after is not None and any(not torch.equal(previous_after[k],v) for k,v in b['policy_before'].items()):
            raise ValueError('policy chain is discontinuous')
        previous_after=a['policy_after'];policy.load_state_dict(b['policy_before'])
        def check(name,actual,expected,atol=2e-4,rtol=2e-5):
            error=float((actual-expected).abs().max());errors[name]=max(errors[name],error)
            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        with torch.no_grad():
            obs={k:v.flatten(0,1) for k,v in s['observations'].items()}
            value=policy.evaluate_actions(obs,s['actions'].flatten(0,1),latent_actions=s['latent_actions'].flatten(0,1))
            check('action',policy._target(s['latent_actions']),s['actions'],2e-6,1e-6)
            check('log_prob',value.log_probs,s['log_probs'].flatten())
            check('value',value.values,s['values'].flatten(0,1))
            dist=policy.latent_distribution(obs)
            for t,f in enumerate(fs):
                nv=policy.value(f['next_observations']);mask=f['timeouts']&~f['terminated']
                if mask.any():
                    terminal=f['terminal_observation']
                    if terminal is None:raise ValueError('timeout missing pre-reset observation')
                    nv[mask]=policy.value({k:v[mask] if len(v)==N else v for k,v in terminal.items()})
                check('next_value',nv,s['next_values'][t])
                torch.testing.assert_close(f['physics']['target'],s['actions'][t],atol=0,rtol=0)
                torch.testing.assert_close(f['reward'],s['rewards'][t],atol=0,rtol=0)
                assert torch.equal(f['terminated'],s['terminated'][t]) and torch.equal(f['timeouts'],s['timeouts'][t])
            policy.load_state_dict(a['policy_after']);new_dist=policy.latent_distribution(obs)
            kl=kl_divergence(dist,new_dist).sum(-1)
            check('kl',kl.mean(),torch.tensor(a['metrics']['exact_kl']),2e-5,2e-5)
        gae=independent_gae(*(s[k].double().numpy() for k in ('rewards','values','next_values')),
            s['terminated'].numpy(),s['timeouts'].numpy(),m['ppo_config']['gamma'],m['ppo_config']['lam'])
        check('gae',s['advantages'],torch.tensor(gae,dtype=torch.float32))
        flat=gae.reshape(-1,3);normalized=(flat-flat.mean(0))/(flat.std(0)+1e-8)
        check('advantage',s['policy_advantages'].flatten(),torch.tensor(normalized.mean(-1),dtype=torch.float32),2e-5,2e-5)
        def combine(key):
            examples=[f['physics'][key] for f in fs]
            return {k:(examples[0][k] if k in ('joint_low','joint_high') else torch.cat([x[k] for x in examples]))
                    for k in examples[0] if isinstance(examples[0][k],torch.Tensor)}
        kwargs={k:torch.cat([f['physics'][k] for f in fs]) for k in ('action','previous_action','corrected_velocity','recovery_mask')}
        reward,terms=computer.compute(combine('state'),combine('reference'),previous_state=combine('previous_state'),**kwargs)
        check('reward',reward,s['rewards'].flatten(0,1))
        for k,w in AUX.terms:
            if k=='ee_accel_mismatch':
                if k in fs[0]['weighted_aux_terms']:
                    check('ee_reward',terms['aux'][k]*w,torch.cat([f['weighted_aux_terms'][k] for f in fs]))
                elif m['arguments']['backend']=='isaaclab':raise ValueError('missing physical auxiliary reward evidence')
        for f in fs:
            p=f['physics'];current=p['state']['body_lin_vel'];previous=p['previous_state']['body_lin_vel'];first=p['episode_steps']==1
            if first.any():check('ee_history',previous[first],torch.zeros_like(previous[first]),0,0)
            if previous_velocity is not None and (~first).any():check('ee_history',previous[~first],previous_velocity[~first],0,0)
            previous_velocity=current
            trace.append({k:p[k] for k in ('state','reference','foot_contact','estimated_torque','episode_steps')})
        raw=b['substep_qvel']
        if raw is not None:
            if raw.shape!=(T*4,N,29) or not torch.isfinite(raw).all():raise ValueError('incomplete/non-finite 5 ms trace')
            # The last substep of each control tick must match pre-reset readback.
            torch.testing.assert_close(raw[3::4],torch.stack([f['physics']['state']['joint_vel'] for f in fs]),atol=2e-5,rtol=1e-5)
            substeps.append(raw)
        row={'update':u,'actor_steps':a['metrics']['optimizer_steps'],'critic_only_steps':a['metrics']['critic_only_steps'],
             'exact_kl':float(kl.mean()),'max_state_kl':float(kl.max()),'reward_mean':s['rewards'].mean((0,1)).tolist(),
             'terminations':int(s['terminated'].sum()),'timeouts':int((s['timeouts']&~s['terminated']).sum()),
             'joint_rmse_rad':float(torch.cat([f['tracking']['joint_mse'] for f in fs]).mean().sqrt()),
             'body_rmse_m':float(torch.cat([f['tracking']['body_mse'] for f in fs]).mean().sqrt()),
             'before_sha256':sha256(before_path),'after_sha256':sha256(after_path)}
        if row['actor_steps']<1:raise ValueError('no accepted actor update')
        rows.append(row);print(folder.name,'CPU audited',u,flush=True)
    qvel=torch.cat(substeps) if substeps else None
    q=torch.stack([f['state']['joint_pos'] for f in trace]);rq=torch.stack([f['state']['root_quat'] for f in trace])
    tilt=torch.rad2deg(torch.acos((1-2*rq[...,1:3].square().sum(-1)).clamp(-1,1)))
    torque=torch.stack([f['estimated_torque'] for f in trace]);ratio=torque.abs()/torch.tensor(m['environment_config']['torque_limit'])
    ends=json.loads((folder/'episode_evidence.json').read_text())['ended_episodes']
    episodes=[e['duration_s'] for e in ends]
    result={'status':'completed','scope':'training-rollout interface audit; no frozen-policy evaluation',
        'condition':m['standing_fixture']['condition'],'updates':rows,'max_abs_errors':errors,
        'transitions':sum(u['steps'] for u in m['updates']),'actor_updates_total':sum(r['actor_steps'] for r in rows),
        'mean_exact_kl':float(np.mean([r['exact_kl'] for r in rows])),'maximum_state_kl':max(r['max_state_kl'] for r in rows),
        'episode_count':len(ends),'ended_episode_duration_mean_s':float(np.mean(episodes)) if episodes else None,
        'ended_episode_duration_max_s':max(episodes) if episodes else None,'physical_timeouts':sum(r['timeouts'] for r in rows),
        'termination_reasons':{k:sum(k in e['reasons'] for e in ends) for k in sorted({k for e in ends for k in e['reasons']})},
        'mean_tilt_deg':float(tilt.mean()),'mean_joint_rmse_rad':float(q.square().mean().sqrt()),
        'torque_estimate_peak_limit_fraction':float(ratio.max()),'torque_estimate_saturation_fraction':float((ratio>=.99).float().mean()),
        'contact_fraction':torch.stack([f['foot_contact'] for f in trace]).float().mean((0,1)).tolist(),
        'raw_substep_shape':None if qvel is None else list(qvel.shape),
        'raw_substep_peak_rad_s':None if qvel is None else float(qvel.abs().max()),
        'raw_over45_env_substeps':None if qvel is None else int((qvel.abs().amax(-1)>45).sum()),
        'initial_evidence_sha256':sha256(folder/'initial_evidence.pt'),'final_checkpoint_sha256':sha256(folder/'policy.pt'),
        'source_sha256':sha256(__file__)}
    _write_metrics(output/'summary.json',result)
    return result


def pair(root,output):
    if output.with_suffix('.json').exists():raise FileExistsError(output)
    rows=[json.loads((root/(c+'_cpu_audit')/'summary.json').read_text()) for c in ('control','candidate')]
    m=[json.loads((root/c/'metrics.json').read_text()) for c in ('control','candidate')]
    initial=[torch.load(root/c/'initial.pt',map_location='cpu',weights_only=False) for c in ('control','candidate')]
    weights=[v['diagnostic_ppo']['policy'] for v in initial]
    differences=[k for k in weights[0] if not torch.equal(weights[0][k],weights[1][k])]
    if differences!=['actor.mlp.net.6.bias']:raise ValueError('initial policies differ beyond target bias')
    evidence=[torch.load(root/c/'initial_evidence.pt',map_location='cpu',weights_only=False) for c in ('control','candidate')]
    for key in evidence[0]['observations']:
        torch.testing.assert_close(evidence[0]['observations'][key],evidence[1]['observations'][key],atol=0,rtol=0)
    torch.testing.assert_close(evidence[0]['latent_std'],evidence[1]['latent_std'],atol=0,rtol=0)
    if m[0]['ppo_config']!=m[1]['ppo_config'] or m[0]['environment_config']!=m[1]['environment_config']:raise ValueError('unpaired training configuration')
    release=json.loads((root/'gpu_release.json').read_text())
    if release['task_pids_alive']:raise ValueError('GPU child remains alive')
    result={'status':'completed','conditions':rows,'total_transitions':sum(r['transitions'] for r in rows),
        'budget':15360,'paired_initial_state_and_parameters':True,'initial_parameter_differences':differences,
        'gpu_release':release,'physical_evaluation_run':False,'source_sha256':sha256(__file__),'source_root':str(root.resolve())}
    if result['total_transitions']!=15360:raise ValueError('unexpected transition count')
    _write_metrics(output.with_suffix('.json'),result)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row in rows:
        u=row['updates'];x=[r['update'] for r in u]
        for ax,key in zip(axs.flat,('exact_kl','max_state_kl','joint_rmse_rad','body_rmse_m')):
            ax.plot(x,[r[key] for r in u],label=row['condition']);ax.set_xlabel('PPO update');ax.grid(alpha=.2)
    for ax,title in zip(axs.flat,('Mean policy KL','Maximum state KL','Rollout joint RMSE (rad)','Rollout body RMSE (m)')):ax.set_title(title)
    axs[0,0].legend();fig.suptitle('20-update standing diagnostic | training rollouts, not policy evaluation')
    fig.savefig(output.with_suffix('.png'),dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair',action='store_true');a=p.parse_args()
    pair(a.source,a.output) if a.pair else audit_run(a.source,a.output)
