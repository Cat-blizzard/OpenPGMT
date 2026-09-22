"""Explicit fixed-clip diagnostic; production Stage 1 requirements stay intact.

Fresh runs only. Its checkpoints use a separate diagnostic_ppo key and are
intentionally not accepted as production Stage 1/2 resume checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from pgmt.cfg.assumptions import get, dump as dump_assumptions
from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import build_env, _seed_everything, _write_metrics, _atomic_output


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path, reference_data, urdf, asset=None):
    m=json.loads(Path(path).read_text())
    if m['schema']!='pgmt_fixed_clip_diagnostic_v1' or m['control_dt']!=.02 or m['sim_dt']!=.005:
        raise ValueError('incompatible diagnostic manifest')
    actual={p.name:sha256(p) for p in sorted(Path(reference_data).glob('*.npz'))}
    if actual!=m['reference_files'] or sha256(urdf)!=m['urdf_sha256'] or (asset and sha256(asset)!=m['asset_sha256']):
        raise ValueError('diagnostic data/assets differ from frozen manifest')
    return m


class FixedClipSchedule:
    """Per-environment episode order, independent of reset timing and policy RNG."""
    def __init__(self, core, manifest):
        if (core.cfg.stage!=1 or core.cfg.terrain_family!='flat' or core.recovery_pool is not None or core.cfg.enable_adaptive_sampling
                or core.cfg.randomize_dynamics or core.cfg.corrupt_observations or core.cfg.max_action_delay):
            raise ValueError('fixed clip diagnostic requires ordinary, unrandomized flat Stage 1')
        self.core=core
        self.clips=[c for c in manifest['clips'] if c['split']=='train']
        if not self.clips or len({c['clip_id'] for c in manifest['clips']})!=len(manifest['clips']):
            raise ValueError('missing training clips or duplicate clip id')
        names=[s['name'] for s in core.reference_database.seqs]
        if names!=manifest['sequence_names']:
            raise ValueError('reference sequence ordering differs')
        self.pairs=[]
        for c in self.clips:
            i=names.index(c['sequence']); seq=core.reference_database.seqs[i]
            if c['horizon_s']!=core.cfg.episode_length_s or c['start_frame']<0:
                raise ValueError('clip/environment horizon mismatch or invalid start')
            end=c['start_frame']+(c['horizon_s']+.62)/seq['frame_time']
            if end>=len(seq['qpos'])-1:
                raise ValueError('clip lacks episode/future padding')
            self.pairs.append((i,c['start_frame']))
        self.episodes=np.zeros(core.num_envs,dtype=np.int64)
        self.current=np.zeros(core.num_envs,dtype=np.int64)
        self.records=[]
        self.original=core._sample_reference
        core._sample_reference=self.sample

    def sample(self, ids):
        indices=ids.tolist()
        for eid in indices:
            clip=int((eid+self.episodes[eid])%len(self.clips))
            self.current[eid]=clip
            self.core._forced_reference[eid]=self.pairs[clip]
        # The production sampler still executes before forced references are
        # installed. Its unused draws must not perturb policy exploration RNG.
        devices=[self.core.device] if self.core.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.original(ids)
        for eid in indices:
            c=int(self.current[eid]); seq,frame=self.pairs[c]
            if int(self.core.reference_seq_idx[eid])!=seq or float(self.core.reference_frame[eid])!=frame:
                raise RuntimeError('scheduled reset did not select its frozen reference')
            self.records.append({'env_id':eid,'episode':int(self.episodes[eid]),'clip_id':self.clips[c]['clip_id'],
                                 'sequence_index':seq,'start_frame':frame})
            self.episodes[eid]+=1


class ObservedDiagnosticEnv:
    """Record the executed first action and pre-reset physics, without changing them."""
    def __init__(self, env, schedule):
        self.env=env; self.core=schedule.core; self.schedule=schedule
        self.num_envs=env.num_envs if hasattr(env,'num_envs') else self.core.num_envs
        self.device=self.core.device
        self.first_steps=[]; self.ends=[]; self.clip_sums={}; self.step_count=0
        self.first_physics={}
        self.ee_ids=[self.core._reward_computer.bi[n] for n in
            ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        self.reward_impl=self.core._reward
        self.core._reward=self.reward

    def reward(self):
        self.pre_reset_tracking={k:v.clone() for k,v in self.core.tracking_metrics().items()}
        ids=(self.core.episode_length_buf==1).nonzero().flatten()
        self.first_physics={}
        if len(ids):
            c=self.core
            actual=(c.body_lin_vel[ids][:,self.ee_ids]-c._previous_body_lin_vel[ids][:,self.ee_ids])/c.cfg.control_dt
            ref=c.reference_body['body_accel'][ids][:,self.ee_ids]
            self.first_physics={int(eid):cpu_copy({'state_ee_accel':actual[k], 'reference_ee_accel':ref[k],
                'executed_target':c.target[eid],
                'max_joint_speed_rad_s':c.qvel[eid].abs().max()}) for k,eid in enumerate(ids.tolist())}
        return self.reward_impl()

    def step(self, actions):
        c=self.core; clips=self.schedule.current.copy(); episodes=self.schedule.episodes.copy()-1
        ids=(c.episode_length_buf==0).nonzero().flatten()
        before=cpu_copy({'ids':ids,'qpos':c.qpos[ids],'qvel':c.qvel[ids],
            'reference_qpos':c.reference_qpos[ids],'requested_target':actions[ids]})
        result=self.env.step(actions)
        self.step_count+=1
        _,_,terminated,truncated,info=result
        # The CPU protocol adapter omits tracking extras; capture them before
        # auto-reset just as the Isaac adapter does. Never read reset states.
        if 'tracking' not in info:
            info={**info,'tracking':self.pre_reset_tracking}
            result=(*result[:4],info)
        for k,eid in enumerate(before['ids'].tolist()):
            physics=self.first_physics[eid]
            if not torch.equal(physics['executed_target'],before['requested_target'][k]):
                raise RuntimeError('first policy action was changed before execution')
            self.first_steps.append({'step':self.step_count,'env_id':eid,'episode':int(episodes[eid]),
                'clip_id':self.schedule.clips[int(clips[eid])]['clip_id'],
                **{name:value[k] for name,value in before.items() if name!='ids'}, **physics})
        tracking=cpu_copy(info['tracking'])
        for eid in range(self.num_envs):
            key=self.schedule.clips[int(clips[eid])]['clip_id']
            agg=self.clip_sums.setdefault(key,{'steps':0,'joint_mse_sum':0.,'body_mse_sum':0.})
            agg['steps']+=1
            for name in ('joint_mse','body_mse'):
                agg[name+'_sum']+=float(tracking[name][eid])
        for eid in (terminated|truncated).nonzero().flatten().tolist():
            self.ends.append({'step':self.step_count,'env_id':eid,'episode':int(episodes[eid]),
                'clip_id':self.schedule.clips[int(clips[eid])]['clip_id'],
                'duration_s':float(info['episode_elapsed_s'][eid]),'terminated':bool(terminated[eid]),
                'timeout':bool(truncated[eid]),
                'reasons':[name for name,mask in info['termination_reasons'].items() if bool(mask[eid])]})
        return result


class SubstepVelocityMonitor:
    """Observe raw PhysX velocities at 5 ms, including between control ticks."""
    def __init__(self, env, *, retain_raw=False):
        self.env=env;self.core=env.env.core;self.original=env.env.scene.update
        self.samples=0;self.over45_env_substeps=0;self.peak=0.;self.events=[]
        self.retain_raw=retain_raw;self.raw=[]
        env.env.scene.update=self.update

    def update(self, dt):
        self.original(dt)
        c=self.core;v=c.articulation.root_physx_view.get_dof_velocities()[:,c.joint_ids]
        if not torch.isfinite(v).all():raise RuntimeError('non-finite raw substep velocity')
        if self.retain_raw:self.raw.append(cpu_copy(v))
        self.samples+=c.num_envs;peak=float(v.abs().max());self.peak=max(self.peak,peak)
        over=v.abs().amax(-1)>45;self.over45_env_substeps+=int(over.sum())
        if over.any() and len(self.events)<4:
            ids=over.nonzero().flatten()[:4]
            self.events.append(cpu_copy({'control_step':c._diagnostic_step,'env_ids':ids,
                'qvel':v[ids],'qpos':c.articulation.root_physx_view.get_dof_positions()[ids][:,c.joint_ids],
                'target':c.target[ids],'reference_seq_idx':c.reference_seq_idx[ids],
                'reference_frame':c.reference_frame[ids]}))

    def summary(self):
        return {'environment_substeps':self.samples,'max_abs_velocity_rad_s':self.peak,
            'over45_env_substeps':self.over45_env_substeps,'events':self.events,
            'scope':'raw substep observation only; no clipping or altered termination'}

    def take_raw(self):
        raw=torch.stack(self.raw) if self.raw else torch.empty((0,self.core.num_envs,29))
        self.raw=[]
        return raw


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return bool(np.isfinite(value))
    return True


def diagnostic_initial_targets(core, schedule, mode):
    """A fixed actor bias for one clip; never substitute an executed action."""
    if mode=='zero':return None
    if mode!='clip_start' or len(schedule.clips)!=1 or core.cfg.reset_mode!='reference_state':
        raise ValueError('clip_start initialization requires one fixed clip and reference_state reset')
    q=core.reference_qpos[0].detach().clone()
    if not torch.allclose(core.reference_qpos,q.expand_as(core.reference_qpos),atol=1e-6,rtol=0):
        raise ValueError('initial targets must be constant across environments')
    # A reference can lie on a hard bound (waist pitch here). A tanh mean must
    # be inside the interval; retain a recorded 1% half-range margin.
    mid=(core.joint_low+core.joint_high)/2;half=(core.joint_high-core.joint_low)/2
    return mid+half*((q-mid)/half).clamp(-.99,.99)


def run(args):
    if args.output.exists():raise FileExistsError('use a fresh diagnostic output directory')
    if (args.num_envs<=0 or args.steps_per_env<=0 or not 0<args.updates<=args.lr_schedule_updates
            or args.learning_epochs<=0 or args.mini_batches<=0):
        raise ValueError('invalid diagnostic budget')
    manifest=load_manifest(args.manifest,args.reference_data,args.urdf,args.asset)
    standing=None
    if getattr(args,'standing_fixture',None) is not None:
        from pgmt.train.standing_diagnostic import FrozenStandingFixture
        if args.actor_init!='zero':raise ValueError('standing fixture owns the actor target initialization')
        standing=FrozenStandingFixture(args.standing_fixture,manifest,args.standing_condition)
        if args.backend=='isaaclab':
            budget=manifest['learning_budget']
            if any(getattr(args,k)!=budget[b] for k,b in (('num_envs','envs'),('steps_per_env','steps_per_update'),
                ('updates','updates'),('seed','seed'),('lr_schedule_updates','lr_schedule_updates'))):
                raise ValueError('standing learning budget differs from the frozen protocol')
    elif getattr(args,'standing_condition',None) is not None:
        raise ValueError('standing condition requires a frozen fixture')
    args.output.mkdir(parents=True)
    result={'status':'running','scope':'fixed-clip ordinary tracking interface diagnostic, not formal training/evaluation',
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'manifest_sha256':sha256(args.manifest),'completed_updates':0,'updates':[],
        'started_at':datetime.now(timezone.utc).isoformat()}
    sources=sorted(p for folder in ('pgmt','setup','data') for p in Path(folder).rglob('*.py')
                   if 'raw' not in p.parts and '.external' not in p.parts)
    _write_metrics(args.output/'source_manifest.json',{'source_sha256':{str(p):sha256(p) for p in sources},
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'vulkan_index':os.environ.get('PGMT_RENDER_GPU'),
        'gpu_evaluation_chained':False})
    _write_metrics(args.output/'manifest.json',manifest)
    _write_metrics(args.output/'metrics.json',result)
    env=app=observed=schedule=capture=monitor=None
    pending_capture=None
    try:
        _seed_everything(args.seed)
        if args.backend=='isaaclab':
            from pgmt.envs.isaac_app import launch_isaac_app
            app=launch_isaac_app(args.device)
        env=build_env(args.num_envs,torch.device(args.device),backend=args.backend,app=app,
            reference_data=args.reference_data,asset_path=args.asset,reference_urdf=args.urdf,
            env_options={'seed':args.seed,'reset_mode':args.reset_mode,'episode_length_s':manifest['horizon_s'],
                'enable_adaptive_sampling':False,'randomize_dynamics':False,'corrupt_observations':False,'max_action_delay':0})
        core=getattr(getattr(env,'env',env),'core',env)
        standing_target=None if standing is None else standing.apply(core)
        schedule=FixedClipSchedule(core,manifest)
        recorder=attach_diagnostics(env,args.output/'diagnostics')
        recorder.config=replace(recorder.config,min_interval=3)
        observations=env.reset()
        observed=ObservedDiagnosticEnv(env,schedule)
        monitor=SubstepVelocityMonitor(env,retain_raw=standing is not None) if args.backend=='isaaclab' else None
        if standing is not None:
            from pgmt.train.standing_diagnostic import StandingCapture, capture_storage
            capture=StandingCapture(observed,core)
            result['standing_fixture']=standing.evidence
            (args.output/'captures').mkdir()
        result['environment_config']=asdict(core.cfg)
        result['initial_reference']={'sequence_indices':core.reference_seq_idx.tolist(),'frames':core.reference_frame.tolist()}
        # Re-seed after environment/reset construction so weights and the PPO
        # sampling stream do not depend on reset implementation internals.
        _seed_everything(args.seed)
        initial_targets=(standing_target if standing is not None else
                         diagnostic_initial_targets(core,schedule,getattr(args,'actor_init','zero')))
        policy=Stage1Policy(initial_joint_targets=None if initial_targets is None else initial_targets.cpu()).to(core.device)
        result['actor_initialization']={'mode':('standing_'+standing.condition if standing is not None else getattr(args,'actor_init','zero')),
            'fixed_mean_target':None if initial_targets is None else initial_targets.tolist(),
            'max_reference_offset_rad':None if initial_targets is None else float((initial_targets-core.reference_qpos[0]).abs().max())}
        config=replace(get('A6').value,num_steps_per_env=args.steps_per_env,
            num_learning_epochs=args.learning_epochs,num_mini_batches=args.mini_batches,
            complete_critic_epochs=True,lr_schedule_updates=args.lr_schedule_updates)
        ppo=PPO(policy,config,total_updates=args.lr_schedule_updates,diagnostics=recorder)
        result['ppo_config']=asdict(config)
        with torch.no_grad():
            dist=policy.latent_distribution(observations)
            initial=cpu_copy({'observations':observations,'latent_mean':dist.loc,'latent_std':dist.scale,
                'mean_target':policy._target(dist.loc),'reset_state':recorder.context(core,torch.arange(core.num_envs,device=core.device))})
        torch.save(initial,args.output/'initial_evidence.pt')

        def checkpoint(name):
            state={'schema':'pgmt_fixed_clip_diagnostic_v1','diagnostic_ppo':ppo.state_dict(),
                'scope':result['scope'],'environment_config':asdict(core.cfg),'assumptions':dump_assumptions(),
                'manifest_sha256':result['manifest_sha256'],'schedule':{'episodes':schedule.episodes.tolist()},
                'actor_initialization':result['actor_initialization'],
                'env':env.state_dict(),'seed':args.seed,'backend':args.backend}
            if not finite_tree(state):raise RuntimeError('non-finite diagnostic checkpoint')
            with _atomic_output(args.output/name) as stream:torch.save(state,stream)
        checkpoint('initial.pt')
        for _ in range(args.updates):
            if capture is not None:pending_capture={'policy_before':cpu_copy(policy.state_dict()),'update':ppo.update_count+1}
            observations,collected=ppo.collect_rollout(capture or observed,observations)
            if capture is not None:
                pending_capture.update(storage=capture_storage(ppo.storage),frames=capture.take(),
                    substep_qvel=(monitor.take_raw() if monitor else None))
                with _atomic_output(args.output/'captures'/f"update_{pending_capture['update']:02d}_before.pt") as path:
                    torch.save(pending_capture,path)
            metrics=ppo.update(); metrics.update(collected)
            metrics['timestamp']=datetime.now(timezone.utc).isoformat()
            if not finite_tree(metrics) or metrics['optimizer_steps']<1:
                raise RuntimeError('non-finite metrics or no accepted actor update')
            result['updates'].append(metrics); result['completed_updates']=ppo.update_count
            if monitor is not None:result['substep_physics']=monitor.summary()
            checkpoint('policy.pt')
            if capture is not None:
                with _atomic_output(args.output/'captures'/f"update_{pending_capture['update']:02d}_after.pt") as path:
                    torch.save({'policy_after':cpu_copy(policy.state_dict()),'metrics':metrics},path)
                pending_capture=None
            _write_metrics(args.output/'metrics.json',result)
            _write_metrics(args.output/'episode_evidence.json',{'reset_schedule':schedule.records,
                'first_steps':observed.first_steps,'ended_episodes':observed.ends,'clip_totals':observed.clip_sums})
            print(json.dumps({'update':ppo.update_count,'exact_kl':metrics['exact_kl'],
                'max_state_kl':metrics['exact_kl_max_state'],'actor_steps':metrics['optimizer_steps']}),flush=True)
        result['status']='completed'; result['finished_at']=datetime.now(timezone.utc).isoformat()
        _write_metrics(args.output/'metrics.json',result)
    except BaseException:
        result['status']='failed'; result['error']=traceback.format_exc()
        _write_metrics(args.output/'metrics.json',result)
        if observed is not None:
            _write_metrics(args.output/'episode_evidence_partial.json',{'reset_schedule':schedule.records,
                'first_steps':observed.first_steps,'ended_episodes':observed.ends,'clip_totals':observed.clip_sums})
        if capture is not None:
            torch.save({'pending':pending_capture,'frames':capture.frames,'substep_qvel':None if monitor is None else monitor.take_raw(),
                        'executed_transitions':capture.total_steps*core.num_envs},args.output/'partial_capture.pt')
            result['executed_transitions']=capture.total_steps*core.num_envs
            _write_metrics(args.output/'metrics.json',result)
        traceback.print_exc();raise
    finally:
        if env is not None and hasattr(env,'close'):env.close()
        if app is not None:app.close()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','reference-data','asset','urdf','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reset-mode',choices=('nominal','reference_state'),required=True)
    p.add_argument('--actor-init',choices=('zero','clip_start'),default='zero')
    p.add_argument('--standing-fixture',type=Path)
    p.add_argument('--standing-condition',choices=('control','candidate'))
    p.add_argument('--backend',choices=('torch','isaaclab'),default='isaaclab')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps-per-env',type=int,default=24)
    p.add_argument('--updates',type=int,default=20);p.add_argument('--lr-schedule-updates',type=int,default=1000)
    p.add_argument('--learning-epochs',type=int,default=5);p.add_argument('--mini-batches',type=int,default=4)
    run(p.parse_args())


if __name__=='__main__':main()
