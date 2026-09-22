"""Deferred matched 10-second standing evaluation; one fresh process per job.

Run only when evaluation is requested. Outputs live under the training batch's
unique deferred_evaluation directory; each of the 24 jobs can execute once.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import torch

from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, ObservedDiagnosticEnv, SubstepVelocityMonitor, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.standing_diagnostic import FrozenStandingFixture, StandingCapture
from pgmt.train.train_stage1 import build_env, _seed_everything, _write_metrics, _atomic_output
from setup.standing_multiseed_protocol import validate_plan, evaluation_jobs, TRAIN_TRANSITIONS
from setup.standing_evaluation_integrity import (serialized_configuration,validate_runtime_source,
                                                validate_checkpoint_file,validate_runtime_configuration)


def select_checkpoint(plan_path, training_root, job_name):
    plan=validate_plan(plan_path)
    matches=[j for j in evaluation_jobs() if j['job']==job_name]
    if len(matches)!=1:raise ValueError('unknown evaluation job')
    j=matches[0];root=Path(training_root)
    ledger=json.loads((root/'execution_ledger.json').read_text())
    if (ledger['status']!='completed' or ledger['total_transitions']!=TRAIN_TRANSITIONS
            or ledger['plan_sha256']!=sha256(plan_path)):
        raise ValueError('evaluation requires this completed training batch')
    validate_runtime_source(root)
    folder=root/j['name'];m=json.loads((folder/'metrics.json').read_text())
    path=folder/('initial.pt' if j['checkpoint']=='initial' else 'policy.pt')
    validate_checkpoint_file(root,path,sha256(plan_path))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    manifest_path=Path(plan_path).parent/f"seed{j['seed']}.json"
    if (m['completed_updates']!=100 or m['status']!='completed' or saved['seed']!=j['seed']
            or saved['manifest_sha256']!=sha256(manifest_path)
            or saved['actor_initialization']['mode']!='standing_'+j['condition']
            or serialized_configuration(saved['environment_config'])!=m['environment_config']):
        raise ValueError('checkpoint does not match standing evaluation job')
    expected=0 if j['checkpoint']=='initial' else 100
    if saved['diagnostic_ppo']['update_count']!=expected:raise ValueError('incorrect initial/final checkpoint')
    return plan,j,path,saved,json.loads(manifest_path.read_text())


def first_episode_summary(frames, ends):
    # Frames retain active masks so a later CPU comparison can use the exact
    # initial/final common survival window, without survival-selection bias.
    counts=torch.stack([f['active'] for f in frames]).sum(0)
    if (counts==0).any() or any(e is None for e in ends):raise ValueError('incomplete first episodes')
    tilt=torch.stack([f['tilt_deg']*f['active'] for f in frames]).sum(0)/counts
    mse=torch.stack([f['joint_mse']*f['active'] for f in frames]).sum(0)/counts
    return {'episodes':ends,'active_steps_per_start':counts.tolist(),
            'mean_tilt_per_start_deg':tilt.tolist(),'joint_rmse_per_start_rad':mse.sqrt().tolist(),
            'completed_10s':sum(e['timeout'] and not e['terminated'] for e in ends),
            'executed_transitions':len(frames)*4,'scored_transitions':int(counts.sum())}


def run(a):
    plan,j,path,saved,manifest=select_checkpoint(a.plan,a.training_root,a.job)
    output=a.training_root/'deferred_evaluation'/j['job']
    output.mkdir(parents=True,exist_ok=False)
    fixture=FrozenStandingFixture(a.plan.parent/'fixture',manifest,j['condition'])
    noise_seed=plan['evaluation']['noise_seed_by_training_seed'][str(j['seed'])]
    noise=torch.randn(500,4,29,generator=torch.Generator().manual_seed(noise_seed))
    report={'status':'running','job':j,'checkpoint_sha256':sha256(path),'plan_sha256':sha256(a.plan),
            'budget_transitions':2000,'executed_transitions':0,'noise_seed':noise_seed,
            'source_sha256':sha256(__file__),
            'scope':'10-second standing diagnostic; separate from paper evaluation'}
    frames=[];ends=[None]*4;app=env=monitor=None;capture=None
    _write_metrics(output/'metrics.json',report)
    try:
        _seed_everything(j['seed'])
        from pgmt.envs.isaac_app import launch_isaac_app
        app=launch_isaac_app(a.device)
        cfg=dict(saved['environment_config'])
        for key in ('num_envs','device','asset_path','reference_data_dir','reference_urdf_path'):cfg.pop(key)
        env=build_env(4,torch.device(a.device),backend='isaaclab',app=app,asset_path=Path(plan['asset']),
            reference_urdf=Path(plan['urdf']),reference_data=a.plan.parent/'fixture'/'reference',env_options=cfg)
        core=env.env.core
        validate_runtime_configuration(saved['environment_config'],asdict(core.cfg))
        fixture.apply(core);schedule=FixedClipSchedule(core,manifest)
        recorder=attach_diagnostics(env,output/'diagnostics');obs=env.reset()
        capture=StandingCapture(ObservedDiagnosticEnv(env,schedule),core)
        monitor=SubstepVelocityMonitor(env,retain_raw=True)
        policy=Stage1Policy().to(core.device).eval();policy.load_state_dict(saved['diagnostic_ppo']['policy'])
        report['environment_config']=asdict(core.cfg);report['fixture']=fixture.evidence
        torch.save(cpu_copy({'observations':obs,'state':recorder.context(core,torch.arange(4,device=core.device)),
                             'latent_noise':noise}),output/'initial_evidence.pt')
        active=torch.ones(4,dtype=torch.bool,device=core.device)
        for step in range(500):
            with torch.no_grad():
                dist=policy.latent_distribution(obs)
                latent=dist.loc if j['mode']=='deterministic' else dist.loc+dist.scale*noise[step].to(core.device)
                action=policy._target(latent)
            obs,_,terminated,timeout,info=capture.step(action)
            frame=capture.take()[0];p=frame['physics'];q=p['state']['root_quat']
            frames.append(cpu_copy({'active':active,'tilt_deg':torch.rad2deg(torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1,1))),
                'joint_mse':frame['tracking']['joint_mse'],'body_mse':frame['tracking']['body_mse'],
                'action':action,'physics':p,'terminated':terminated,'timeout':timeout}))
            for eid in ((terminated|timeout)&active).nonzero().flatten().tolist():
                ends[eid]={'duration_s':float(info['episode_elapsed_s'][eid]),'terminated':bool(terminated[eid]),
                    'timeout':bool(timeout[eid]),'reasons':[k for k,v in info['termination_reasons'].items() if bool(v[eid])]}
            active &= ~(terminated|timeout)
            report['executed_transitions']=(step+1)*4
            if not active.any():break
        if active.any():raise RuntimeError('10-second timeout branch did not terminate the first episode')
        report.update(first_episode_summary(frames,ends));report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        try:
            if capture is not None:report['executed_transitions']=capture.total_steps*4
            with _atomic_output(output/'trajectory.pt') as stream:
                torch.save({'frames':frames,'ends':ends,'substep_qvel':None if monitor is None else monitor.take_raw()},stream)
            _write_metrics(output/'metrics.json',report)
        finally:
            try:
                if env is not None:env.close()
            finally:
                if app is not None:app.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','training-root'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--job',choices=[j['job'] for j in evaluation_jobs()],required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
