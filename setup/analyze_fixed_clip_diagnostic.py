"""CPU-only paired interface audit; on-policy training statistics are not evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pgmt.train.fixed_clip_diagnostic import finite_tree, sha256
from pgmt.envs.g1_env import G1_JOINT_NAMES
from setup.replay_kl_diagnostic import replay


def read(path):return json.loads(Path(path).read_text())


def tensor_tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def window(rows):
    groups=[r['episode_groups']['ordinary'] for r in rows]
    n=sum(g['steps'] for g in groups)
    ended=sum(g['terminations']+g['timeouts'] for g in groups)
    return {'transitions':n,'terminations':sum(g['terminations'] for g in groups),
        'timeouts':sum(g['timeouts'] for g in groups),
        'ended_episode_mean_s':sum(g.get('ended_episode_duration_mean_s',0)*(g['terminations']+g['timeouts']) for g in groups)/ended if ended else None,
        'joint_rmse_rad':float(np.sqrt(sum(g['mean']['tracking/joint_mse']*g['steps'] for g in groups)/n)),
        'body_rmse_m':float(np.sqrt(sum(g['mean']['tracking/body_mse']*g['steps'] for g in groups)/n))}


def first_stats(rows):
    target=np.asarray([r['requested_target'] for r in rows]);q=np.asarray([r['qpos'] for r in rows])
    actual=np.asarray([r['state_ee_accel'] for r in rows]);ref=np.asarray([r['reference_ee_accel'] for r in rows])
    per_joint=np.sqrt(np.mean((target-q)**2,axis=0))
    return {'count':len(rows),'target_minus_q_rms_rad':float(np.sqrt(np.mean((target-q)**2))),
        'target_minus_q_max_abs_rad':float(np.abs(target-q).max()),
        'largest_joint_target_gaps':[{'joint':G1_JOINT_NAMES[i],'rms_rad':float(per_joint[i])}
            for i in np.argsort(per_joint)[-5:][::-1]],
        'state_ee_accel_peak_m_s2':float(np.linalg.norm(actual,axis=-1).max()),
        'state_ee_accel_p95_m_s2':float(np.quantile(np.linalg.norm(actual,axis=-1),.95)),
        'ee_accel_error_p95_m_s2':float(np.quantile(np.linalg.norm(actual-ref,axis=-1),.95))}


def analyze(root):
    raw={};out={};weights={};initials={};readbacks={};sources={}
    for mode in ('nominal','reference_state'):
        path=root/mode;m=read(path/'metrics.json');e=read(path/'episode_evidence.json')
        manifest=read(path/'manifest.json');a=m['arguments'];cfg=m['environment_config'];rows=m['updates']
        if (m['status']!='completed' or m['completed_updates']!=20 or len(rows)!=20
                or a['num_envs']!=16 or a['steps_per_env']!=24 or a['seed']!=0 or a['backend']!='isaaclab'
                or cfg['reset_mode']!=mode or (root/f'{mode}_exit_code').read_text().strip()!='0'):
            raise ValueError(f'incomplete/wrong physical diagnostic: {path}')
        if any(cfg[k] for k in ('enable_adaptive_sampling','randomize_dynamics','corrupt_observations','max_action_delay')):
            raise ValueError('uncontrolled curriculum/randomization')
        if (not finite_tree(m) or not finite_tree(e)
                or any(r['optimizer_steps']<1 or r['recovery_steps'] or r['steps']!=384 for r in rows)
                or any(r['exact_kl']>m['ppo_config']['target_kl']+1e-6 for r in rows)):
            raise ValueError('failed numeric/update/KL guard')
        train=[c for c in manifest['clips'] if c['split']=='train']
        per_env={i:[] for i in range(16)}
        for record in e['reset_schedule']:
            i=record['env_id'];expected=train[(i+len(per_env[i]))%len(train)]
            if (record['episode']!=len(per_env[i]) or record['clip_id']!=expected['clip_id']
                    or record['start_frame']!=expected['start_frame']
                    or manifest['sequence_names'][record['sequence_index']]!=expected['sequence']):
                raise ValueError('reset deviated from frozen clip schedule')
            per_env[i].append(record)
        if any(r['requested_target']!=r['executed_target'] for r in e['first_steps']):
            raise ValueError('policy action replaced')
        if (sum(v['steps'] for v in e['clip_totals'].values())!=7680
                or len(e['ended_episodes'])!=sum(r['terminations']+r['timeouts'] for r in rows)
                or sorted(r['env_id'] for r in e['first_steps'] if r['episode']==0)!=list(range(16))):
            raise ValueError('incomplete episode evidence')
        for filename,count in (('initial.pt',0),('policy.pt',20)):
            state=torch.load(path/filename,map_location='cpu',weights_only=False)
            if state['schema']!='pgmt_fixed_clip_diagnostic_v1' or 'ppo' in state or not finite_tree(state) or state['diagnostic_ppo']['update_count']!=count:
                raise ValueError('invalid checkpoint')
            if count==0:weights[mode]=state['diagnostic_ppo']
        initial=torch.load(path/'initial_evidence.pt',map_location='cpu',weights_only=False)
        reset=initial['reset_state']
        if mode=='reference_state':
            for key in ('qpos','qvel'):
                torch.testing.assert_close(reset[key],reset['reference_'+key],atol=2e-5,rtol=1e-5)
        else:
            torch.testing.assert_close(reset['qpos'],torch.zeros_like(reset['qpos']),atol=2e-5,rtol=0)
            torch.testing.assert_close(reset['qvel'],torch.zeros_like(reset['qvel']),atol=2e-5,rtol=0)
        rb=torch.load(path/'diagnostics/actuators_000.pt',map_location='cpu',weights_only=False)
        for key,val in (('effort_nm',cfg['torque_limit']),('stiffness_nm_rad',80.),('damping_nm_s_rad',2.)):
            torch.testing.assert_close(rb[key],torch.tensor(val).expand_as(rb[key]))
        replays=[dict(file=str(p),**replay(p)) for p in sorted((path/'diagnostics').glob('kl_*.pt'))]
        speeds=[]
        for snapshot in sorted((path/'diagnostics').glob('joint_speed_*.pt')):
            event=torch.load(snapshot,map_location='cpu',weights_only=False)
            for k,eid in enumerate(event['env_ids'].tolist()):
                j=int(event['qvel'][k].abs().argmax())
                speeds.append({'snapshot':str(snapshot),'step':event['step'],'env_id':eid,
                    'episode_s':float(event['episode_length_buf'][k]*event['control_dt']),
                    'sequence':event['sequence_names'][k],'joint':event['joint_names'][j],
                    'velocity_rad_s':float(event['qvel'][k,j]),'asset_velocity_limit_rad_s':float(rb['velocity_rad_s'][0,j]),
                    'joint_pos_rad':float(event['qpos'][k,j]),'target_rad':float(event['target'][k,j]),
                    'estimated_torque_nm':float(event['estimated_torque_nm'][k,j]),
                    'effort_limit_nm':float(event['effort_limit_nm'][j]),
                    'torque_source':event['torque_source'],'root_height_m':float(event['root_pos'][k,2])})
        per_clip={k:{'transitions':v['steps'],'joint_rmse_rad':float(np.sqrt(v['joint_mse_sum']/v['steps'])),
            'body_rmse_m':float(np.sqrt(v['body_mse_sum']/v['steps']))} for k,v in e['clip_totals'].items()}
        out[mode]={'first5':window(rows[:5]),'last5':window(rows[-5:]),'all':window(rows),
            'initial_episode_first_action':first_stats([r for r in e['first_steps'] if r['episode']==0]),
            'all_episode_first_actions':first_stats(e['first_steps']),
            'actor_steps_min':min(r['optimizer_steps'] for r in rows),'actor_steps_total':sum(r['optimizer_steps'] for r in rows),
            'critic_steps_total':sum(r['value_optimizer_steps'] for r in rows),
            'mean_kl_max':max(r['exact_kl'] for r in rows),'max_state_kl':max(r['exact_kl_max_state'] for r in rows),
            'p99_kl_max':max(r['kl_all_p99'] for r in rows),
            'termination_reasons':{reason:sum(r['episode_groups']['ordinary']['termination_reasons'].get(reason,0) for r in rows)
                for reason in rows[-1]['episode_groups']['ordinary']['termination_reasons']},
            'per_clip':per_clip,'kl_replays':replays,
            'overspeed_snapshots':speeds,'overspeed_snapshot_scope':'at most four events; not necessarily every overspeed episode or the whole-run maximum',
            'update_series':[{'update':i+1,**window([r]),'actor_steps':r['optimizer_steps'],
                'critic_steps':r['value_optimizer_steps'],'mean_kl':r['exact_kl'],
                'max_state_kl':r['exact_kl_max_state'],'p99_kl':r['kl_all_p99']} for i,r in enumerate(rows)],
            'evidence_sha256':{name:sha256(path/name) for name in ('metrics.json','episode_evidence.json',
                'manifest.json','source_manifest.json','initial.pt','policy.pt','initial_evidence.pt','diagnostics/actuators_000.pt')},
            'started_at':m['started_at'],'finished_at':m['finished_at']}
        raw[mode]=m;initials[mode]=initial;readbacks[mode]=rb;sources[mode]=read(path/'source_manifest.json')['source_sha256']
    left,right=[raw[m] for m in ('nominal','reference_state')]
    for key in ('manifest_sha256','initial_reference','ppo_config'):
        if left[key]!=right[key]:raise ValueError(f'unmatched {key}')
    for key,val in left['environment_config'].items():
        if key!='reset_mode' and right['environment_config'][key]!=val:raise ValueError(f'unmatched env {key}')
    for name,digest in sources['nominal'].items():
        if name.startswith('pgmt/') and sources['reference_state'].get(name)!=digest:raise ValueError(f'unmatched source {name}')
    if not tensor_tree_equal(weights['nominal'],weights['reference_state']):raise ValueError('initial weights/optimizer/RNG not identical')
    torch.testing.assert_close(readbacks['nominal']['velocity_rad_s'],readbacks['reference_state']['velocity_rad_s'])
    for key in ('reference_seq_idx','reference_frame','reference_qpos','reference_qvel'):
        torch.testing.assert_close(initials['nominal']['reset_state'][key],initials['reference_state']['reset_state'][key],rtol=0,atol=0)
    return {'scope':'single-seed, fixed-clip ordinary PPO interface diagnostic; on-policy statistics, not matched evaluation or convergence',
        'pair_checks_passed':True,'identical_initial_policy_optimizer_rng':True,'manifest_sha256':left['manifest_sha256'],
        'analysis_source_sha256':sha256(__file__),
        'total_transitions':15360,'runs':out,'reserved_check_evaluated':False,'gpu_evaluation_chained':False}


def plot(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():raise FileExistsError(output)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for mode,color in (('nominal','#2766a2'),('reference_state','#bd5519')):
        rows=read(root/mode/'metrics.json')['updates'];x=np.arange(1,len(rows)+1)
        groups=[r['episode_groups']['ordinary'] for r in rows]
        values=[np.sqrt([g['mean']['tracking/joint_mse'] for g in groups]),
            np.sqrt([g['mean']['tracking/body_mse'] for g in groups]),
            [g.get('ended_episode_duration_mean_s',float('nan')) for g in groups],
            [r['exact_kl_max_state'] for r in rows]]
        for ax,y in zip(axes.flat,values):ax.plot(x,y,label=mode,color=color,marker='.',linewidth=1.3)
    titles=['Joint RMSE (rad)','Body RMSE (m)','Mean duration of ended episodes (s)','Maximum state KL per update']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xlabel('PPO update');ax.grid(alpha=.2);ax.set_xticks([1,5,10,15,20])
    axes[0,0].legend();fig.suptitle('Fixed-clip PPO diagnostic: seed 0, 16 envs, 20 updates\nOn-policy training statistics; not matched evaluation')
    fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',type=Path)
    a=p.parse_args();result=analyze(a.root)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if a.plot:plot(a.root,a.plot)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
