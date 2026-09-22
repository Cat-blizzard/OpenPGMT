"""Fresh-process, matched five-second learning check, separate from paper evaluation.

Only first episodes count. Initial/final policies receive the same reference
starts and evaluation seed. Stochastic and deterministic results stay separate.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import traceback

import torch

from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, ObservedDiagnosticEnv, SubstepVelocityMonitor, load_manifest, finite_tree, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.train_stage1 import build_env, _write_metrics, _seed_everything


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('schema')!='pgmt_fixed_clip_diagnostic_v1':raise ValueError('requires an explicit diagnostic checkpoint')
    manifest=load_manifest(args.manifest,args.reference_data,args.urdf,args.asset)
    if sha256(args.manifest)!=saved['manifest_sha256']:raise ValueError('checkpoint/reference mismatch')
    if manifest['horizon_s']!=5.:raise ValueError('this learning check is limited to five-second clips')
    args.output.mkdir(parents=True)
    report={'status':'running','scope':'5s first-episode diagnostic, not 30s paper evaluation',
        'checkpoint_sha256':sha256(args.checkpoint),'checkpoint_update':saved['diagnostic_ppo']['update_count'],
        'manifest_sha256':sha256(args.manifest),'eval_seed':args.eval_seed,'passes':{},
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    app=env=None
    try:
        _seed_everything(args.eval_seed)
        from pgmt.envs.isaac_app import launch_isaac_app
        app=launch_isaac_app(args.device)
        cfg=dict(saved['environment_config'])
        for key in ('num_envs','device','asset_path','reference_data_dir','reference_urdf_path'):cfg.pop(key)
        env=build_env(args.num_envs,torch.device(args.device),backend='isaaclab',app=app,
            asset_path=args.asset,reference_urdf=args.urdf,reference_data=args.reference_data,env_options=cfg)
        c=env.env.core;schedule=FixedClipSchedule(c,manifest);recorder=attach_diagnostics(env,args.output/'diagnostics')
        policy=Stage1Policy().to(c.device).eval();policy.load_state_dict(saved['diagnostic_ppo']['policy'])
        report['environment_config']=asdict(c.cfg)
        for mode in ('stochastic','deterministic'):
            # Fresh logical episodes and identical reference schedule in both
            # checkpoints. PhysX contact caches are not claimed bitwise equal.
            schedule.episodes[:]=0;obs=env.reset();_seed_everything(args.eval_seed)
            if mode=='stochastic':
                wrapped=ObservedDiagnosticEnv(env,schedule)
                monitor=SubstepVelocityMonitor(env)
            active=torch.ones(args.num_envs,dtype=torch.bool,device=c.device)
            counts=torch.zeros(args.num_envs,device=c.device)
            joint=torch.zeros_like(counts);body=torch.zeros_like(counts)
            ends=[None]*args.num_envs
            initial=recorder.context(c,torch.arange(c.num_envs,device=c.device))
            for step in range(1,c.cfg.max_episode_steps+1):
                with torch.no_grad():a=policy.act(obs,deterministic=mode=='deterministic').actions
                obs,_,terminated,timeout,info=wrapped.step(a)
                joint+=info['tracking']['joint_mse']*active;body+=info['tracking']['body_mse']*active;counts+=active
                for eid in ((terminated|timeout)&active).nonzero().flatten().tolist():
                    ends[eid]={'duration_s':float(info['episode_elapsed_s'][eid]),'timeout':bool(timeout[eid]),
                        'terminated':bool(terminated[eid]),'reasons':[k for k,v in info['termination_reasons'].items() if bool(v[eid])]}
                active &= ~(terminated|timeout)
                if not active.any():break
            if active.any():raise RuntimeError('five-second episode did not report termination/timeout')
            report['passes'][mode]=cpu_copy({'initial':initial,'episodes':ends,'steps':counts,
                'joint_rmse_per_env':(joint/counts).sqrt(),'body_rmse_per_env':(body/counts).sqrt(),
                'duration_mean_s':sum(x['duration_s'] for x in ends)/len(ends),
                'completed_5s':sum(x['timeout'] and not x['terminated'] for x in ends),
                'joint_rmse_transition_weighted':(joint.sum()/counts.sum()).sqrt(),
                'body_rmse_transition_weighted':(body.sum()/counts.sum()).sqrt(),
                'overspeed_episodes':sum('joint_speed' in x['reasons'] for x in ends)})
            if not finite_tree(report):raise RuntimeError('non-finite evaluation report')
            _write_metrics(args.output/'metrics.json',report)
        report['substep_physics']=monitor.summary()
        report['status']='completed';_write_metrics(args.output/'metrics.json',report)
        print({mode:{k:v for k,v in result.items() if k in ('duration_mean_s','completed_5s','overspeed_episodes')}
            for mode,result in report['passes'].items()},flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());_write_metrics(args.output/'metrics.json',report)
        traceback.print_exc();raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','reference-data','asset','urdf','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--eval-seed',type=int,default=10000)
    run(p.parse_args())


if __name__=='__main__':main()
