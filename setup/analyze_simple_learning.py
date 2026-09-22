"""Audit a bounded single-motion learning check; aggregate by training seed."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pgmt.train.fixed_clip_diagnostic import finite_tree,sha256
from setup.analyze_fixed_clip_diagnostic import first_stats, window
from setup.replay_kl_diagnostic import replay


def read(p):return json.loads(Path(p).read_text())


def seed_stats(values):
    return {'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}


def reward_window(rows):
    keys=('reward_upper','reward_lower','reward_aux','reward_mean')
    return {**{k:float(np.mean([r[k] for r in rows])) for k in keys},
        'aux_terms':{k:float(np.mean([r['weighted_aux_terms_mean'][k] for r in rows]))
                     for k in rows[0]['weighted_aux_terms_mean']}}


def analyze(root):
    pairs=[]
    source_hashes=None
    for seed in (0,1,2):
        train=root/f'seed{seed}_train';m=read(train/'metrics.json');rows=m['updates']
        if m['status']!='completed' or m['completed_updates']!=100 or len(rows)!=100:
            raise ValueError('incomplete learning budget')
        if not finite_tree(m) or any(r['optimizer_steps']<1 or r['steps']!=384 or r['recovery_steps'] for r in rows):
            raise ValueError('invalid numerical/update evidence')
        if any(r['exact_kl']>.020001 for r in rows):raise ValueError('KL budget exceeded')
        cfg=m['environment_config']
        if any(cfg[k] for k in ('enable_adaptive_sampling','randomize_dynamics','corrupt_observations','max_action_delay')):
            raise ValueError('unexpected curriculum or randomization')
        if (cfg['physics_external_forces_every_iteration'] is not True or cfg['physics_min_velocity_iterations']!=4
                or cfg['reset_mode']!='reference_state' or m['actor_initialization']['mode']!='clip_start'):
            raise ValueError('wrong physics/initialization condition')
        if m['substep_physics']['environment_substeps']!=153600:raise ValueError('incomplete substep monitoring')
        if read(root/f'seed{seed}_train_exit.json')['exit_code']!=0:raise ValueError('training process failed')
        sources=read(train/'source_manifest.json')['source_sha256']
        if source_hashes is not None and source_hashes!=sources:raise ValueError('training sources differ between seeds')
        source_hashes=sources
        for name,digest in sources.items():
            if sha256(root/'source_snapshot'/name)!=digest:raise ValueError(f'source snapshot mismatch: {name}')
            if name.startswith(('pgmt/','data/')) or name=='setup/evaluate_simple_diagnostic.py':
                if sha256(name)!=digest:raise ValueError(f'executed source changed: {name}')
        evidence=read(train/'episode_evidence.json')
        clips=[c for c in read(train/'manifest.json')['clips'] if c['split']=='train']
        if len(clips)!=1:raise ValueError('not a single-clip diagnostic')
        clip=clips[0]
        if any(r['clip_id']!=clip['clip_id'] or r['start_frame']!=clip['start_frame'] for r in evidence['reset_schedule']):
            raise ValueError('reference starts changed')
        if any(r['requested_target']!=r['executed_target'] for r in evidence['first_steps']):
            raise ValueError('policy action replaced')
        if (sum(c['steps'] for c in evidence['clip_totals'].values())!=38400
                or len(evidence['ended_episodes'])!=sum(r['terminations']+r['timeouts'] for r in rows)):
            raise ValueError('incomplete episode evidence')
        for name,count in (('initial.pt',0),('policy.pt',100)):
            checkpoint=torch.load(train/name,map_location='cpu',weights_only=False)
            if not finite_tree(checkpoint) or checkpoint['diagnostic_ppo']['update_count']!=count or 'ppo' in checkpoint:
                raise ValueError('invalid diagnostic checkpoint')
        pair={'seed':seed,'train_first10':window(rows[:10]),'train_last10':window(rows[-10:]),
            'rewards_first10':reward_window(rows[:10]),'rewards_last10':reward_window(rows[-10:]),
            'actor_steps_total':sum(r['optimizer_steps'] for r in rows),'actor_steps_min':min(r['optimizer_steps'] for r in rows),
            'max_mean_kl':max(r['exact_kl'] for r in rows),'max_state_kl':max(r['exact_kl_max_state'] for r in rows),
            'max_substep_speed':m['substep_physics']['max_abs_velocity_rad_s'],
            'overspeed_env_substeps':m['substep_physics']['over45_env_substeps'],
            'overspeed_control_terminations':sum('joint_speed' in r['reasons'] for r in evidence['ended_episodes']),
            'first_actions_initial':first_stats([r for r in evidence['first_steps'] if r['episode']==0]),
            'first_actions_all_resets':first_stats(evidence['first_steps']),
            'kl_replay':[{'file':str(p.relative_to(root)),**replay(p)} for p in sorted((train/'diagnostics').glob('kl_*.pt'))],
            'actor_initialization':m['actor_initialization'],'evaluation':{},
            'training_windows':[{'last_update':i,**window(rows[i-10:i])} for i in range(10,101,10)],
            'evidence_sha256':{name:sha256(train/name) for name in ('metrics.json','initial.pt','policy.pt','source_manifest.json','episode_evidence.json')}}
        evals={}
        for when,count,filename in (('initial',0,'initial.pt'),('final',100,'policy.pt')):
            path=root/f'seed{seed}_{when}_eval';e=read(path/'metrics.json');evals[when]=e
            if (e['status']!='completed' or e['checkpoint_update']!=count
                    or e['checkpoint_sha256']!=sha256(train/filename) or e['manifest_sha256']!=m['manifest_sha256']
                    or e['eval_seed']!=10000+seed or not finite_tree(e)
                    or read(root/f'seed{seed}_{when}_eval_exit.json')['exit_code']!=0):
                raise ValueError('invalid evaluation')
            pair['evidence_sha256'][when+'_eval.json']=sha256(path/'metrics.json')
        if evals['initial']['environment_config']!=evals['final']['environment_config']:
            raise ValueError('evaluation environment conditions differ')
        if evals['initial']['environment_config']!=cfg:
            raise ValueError('evaluation physics/reset differs from training')
        for mode in ('stochastic','deterministic'):
            left,right=[evals[t]['passes'][mode] for t in ('initial','final')]
            for key in ('qpos','qvel','root_pos','root_quat','root_lin_vel','root_ang_vel','reference_qpos','reference_qvel',
                        'reference_seq_idx','reference_frame','history','target','_action_queue','_action_delay'):
                np.testing.assert_allclose(left['initial'][key],right['initial'][key],rtol=0,atol=2e-5,err_msg=key)
            if len(left['episodes'])!=16 or len(right['episodes'])!=16:raise ValueError('missing evaluation episodes')
            pair['evaluation'][mode]={t:{k:evals[t]['passes'][mode][k] for k in
                ('duration_mean_s','completed_5s','joint_rmse_transition_weighted','body_rmse_transition_weighted','overspeed_episodes','episodes')}
                for t in ('initial','final')}
        pair['eval_max_substep_speed']={t:evals[t]['substep_physics']['max_abs_velocity_rad_s'] for t in ('initial','final')}
        pair['eval_overspeed_env_substeps']={t:evals[t]['substep_physics']['over45_env_substeps'] for t in ('initial','final')}
        pairs.append(pair)
    groups={}
    for mode in ('stochastic','deterministic'):
        groups[mode]={}
        for when in ('initial','final'):
            values=[p['evaluation'][mode][when] for p in pairs]
            groups[mode][when]={k:seed_stats([r[k] for r in values]) for k in
                ('duration_mean_s','joint_rmse_transition_weighted','body_rmse_transition_weighted')}
            groups[mode][when].update(completed_5s=sum(v['completed_5s'] for v in values),episodes=48,
                overspeed_episodes=sum(v['overspeed_episodes'] for v in values))
    return {'scope':'one fixed walking clip; three training seeds; matched 5s first-episode checks; not full-library or paper evaluation',
        'pair_checks_passed':True,'seeds':[0,1,2],'total_training_transitions':115200,
        'total_monitored_training_env_substeps':460800,'groups':groups,'pairs':pairs,
        'analysis_source_sha256':sha256(__file__),'source_snapshot_verified':True,
        'evaluation_source_sha256':source_hashes['setup/evaluate_simple_diagnostic.py'],
        'note':'deterministic replicas share the same reference/reset and are not 48 independent trials; uncertainty is across 3 training seeds'}


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if path.exists():raise FileExistsError(path)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for p in result['pairs']:
        rows=p['training_windows'];x=[r['last_update'] for r in rows]
        axes[0,0].plot(x,[r['joint_rmse_rad'] for r in rows],label=f'seed {p["seed"]}')
        axes[0,1].plot(x,[r['ended_episode_mean_s'] for r in rows],label=f'seed {p["seed"]}')
    for ax,mode in zip(axes[1],('stochastic','deterministic')):
        for p in result['pairs']:
            ax.plot([0,1],[p['evaluation'][mode][w]['duration_mean_s'] for w in ('initial','final')],marker='o',label=f'seed {p["seed"]}')
        ax.set_xticks([0,1],['Initial','After 100 updates']);ax.set_ylim(0,5.3)
        ax.set_title(f'Matched 5s duration: {mode}');ax.set_ylabel('Seconds')
    axes[0,0].set_title('Training joint RMSE (rad), 10-update windows');axes[0,1].set_title('Training ended-episode duration (s)')
    for ax in axes[0]:ax.set_xlabel('Update');ax.legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Single-motion diagnostic: corrected physics, fixed initial pose bias\nThree seeds; 16 environments; 100 updates per seed')
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();result=analyze(a.root)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if a.plot:plot(result,a.plot)
    print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
