"""Bounded PhysX substep observation and identical-action solver comparisons.

No PPO updates. Recording does not change targets, termination, or dynamics.
Raw PhysX and cached articulation data are checked before every automatic reset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
from pathlib import Path
import traceback

import torch

from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, load_manifest, finite_tree, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.train_stage1 import _seed_everything, _write_metrics


def build_probe_env(args, manifest):
    import pgmt.envs.g1_env as module
    module=importlib.reload(module)
    options=module.G1EnvConfig(num_envs=args.num_envs,device=args.device,seed=args.seed,
        reference_data_dir=str(args.reference_data),reference_urdf_path=str(args.urdf),
        asset_path=str(args.asset),reset_mode=args.reset_mode,episode_length_s=manifest['horizon_s'],
        enable_adaptive_sampling=False,randomize_dynamics=False,corrupt_observations=False,max_action_delay=0,
        physics_external_forces_every_iteration=args.external_forces_every_iteration,
        physics_min_velocity_iterations=args.velocity_iterations)
    cfg=module.G1DirectRLEnvCfg(robot_cfg=module.make_g1_articulation_cfg(str(args.asset),options),pgmt_cfg=options)
    cfg.scene.num_envs=args.num_envs;cfg.scene.env_spacing=2.5
    cfg.sim.physx.enable_external_forces_every_iteration=args.external_forces_every_iteration
    cfg.sim.physx.min_velocity_iteration_count=args.velocity_iterations
    return module.IsaacLabPPOAdapter(module.IsaacLabG1Env(cfg))


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<=0 or args.num_envs<=0 or args.velocity_iterations<0:raise ValueError('invalid probe budget')
    manifest=load_manifest(args.manifest,args.reference_data,args.urdf,args.asset)
    args.output.mkdir(parents=True)
    result={'status':'running','scope':'frozen policy/action replay physics probe; no learning',
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'manifest_sha256':sha256(args.manifest),'steps':[]}
    _write_metrics(args.output/'metrics.json',result)
    app=env=None;frames=[];actions=[]
    try:
        _seed_everything(args.seed)
        from pgmt.envs.isaac_app import launch_isaac_app
        app=launch_isaac_app(args.device)
        env=build_probe_env(args,manifest);c=env.env.core;robot=env.env.robot;view=robot.root_physx_view
        schedule=FixedClipSchedule(c,manifest);recorder=attach_diagnostics(env,args.output/'diagnostics')
        obs=env.reset()
        result['environment_config']=asdict(c.cfg)
        result['physics_config']={k:getattr(env.env.cfg.sim.physx,k) for k in
            ('solver_type','min_position_iteration_count','min_velocity_iteration_count','enable_external_forces_every_iteration')}
        result['initial']=recorder.context(c,torch.arange(c.num_envs,device=c.device))
        result['runtime_joint_names']=list(robot.joint_names)
        result['runtime_body_names']=list(robot.body_names)
        result['runtime_joint_ids']=c.joint_ids.tolist()
        result['asset_properties']=cpu_copy({'masses':view.get_masses(),'inertias':view.get_inertias(),
            'coms':view.get_coms(),'armatures':view.get_dof_armatures(),
            'effort_limits':view.get_dof_max_forces(),'velocity_limits':view.get_dof_max_velocities()})
        # Preserve authored articulation settings and all four ankle inertias.
        from pxr import Usd
        stage=Usd.Stage.Open(str(args.asset))
        result['asset_ankle_and_solver_properties']={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes()
            if a.GetName().startswith(('physics:mass','physics:diagonalInertia','physics:principalAxes','physics:centerOfMass','physxArticulation:'))}
            for p in stage.Traverse() if ('ankle' in p.GetName() and p.GetTypeName()=='Xform') or p.HasAttribute('physxArticulation:solverPositionIterationCount')}
        policy=Stage1Policy().to(c.device).eval()
        state=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)['diagnostic_ppo']
        policy.load_state_dict(state['policy'])
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state(state['rng_device'],c.device)
        tape=None if args.action_tape is None else torch.load(args.action_tape,map_location=c.device,weights_only=False)['actions']
        if tape is not None and tape.shape!=(args.steps,args.num_envs,29):raise ValueError('action tape shape mismatch')
        original_update=env.env.scene.update
        previous_q=None;step=0;substep=0
        peak_velocity=0.;max_cache_q_error=0.;max_cache_v_error=0.

        def observe(dt):
            nonlocal previous_q,substep,peak_velocity,max_cache_q_error,max_cache_v_error
            original_update(dt)
            q=view.get_dof_positions().clone()[:,c.joint_ids]
            v=view.get_dof_velocities().clone()[:,c.joint_ids]
            cache_q=robot.data.joint_pos[:,c.joint_ids];cache_v=robot.data.joint_vel[:,c.joint_ids]
            eq=float((q-cache_q).abs().max());ev=float((v-cache_v).abs().max())
            max_cache_q_error=max(max_cache_q_error,eq);max_cache_v_error=max(max_cache_v_error,ev)
            if eq>1e-6 or ev>1e-6:raise RuntimeError('raw PhysX and articulation cache disagree')
            substep+=1
            frame=cpu_copy({'step':step,'substep':substep,'episode_step_before':c.episode_length_buf,
                'qpos':q,'qvel':v,'qpos_fd':(q-previous_q)/dt,'target':c.target,
                'physx_target':view.get_dof_position_targets()[:,c.joint_ids],
                'estimated_drive_torque':robot.data.applied_torque[:,c.joint_ids],
                'contact_forces':env.env.scene['contact_forces'].data.net_forces_w,
                'root_pos':robot.data.root_link_pos_w,'root_quat':robot.data.root_link_quat_w,
                'body_lin_vel':robot.data.body_link_lin_vel_w,'body_ang_vel':robot.data.body_link_ang_vel_w,
                'reference_seq_idx':c.reference_seq_idx,'reference_frame':c.reference_frame})
            if not finite_tree(frame):raise RuntimeError('non-finite physics trace')
            if not torch.equal(frame['target'],frame['physx_target']):raise RuntimeError('PhysX target differs from policy command')
            frames.append(frame);previous_q=q;peak_velocity=max(peak_velocity,float(v.abs().max()))

        env.env.scene.update=observe
        first_ends=[None]*args.num_envs
        for step in range(1,args.steps+1):
            previous_q=view.get_dof_positions().clone()[:,c.joint_ids];substep=0
            with torch.no_grad():action=policy.act(obs).actions if tape is None else tape[step-1]
            actions.append(cpu_copy(action))
            obs,reward,terminated,timeout,info=env.step(action)
            if substep!=c.cfg.decimation:raise RuntimeError('unexpected substep count')
            if not torch.isfinite(reward).all():raise RuntimeError('non-finite reward')
            ended=[]
            for eid in (terminated|timeout).nonzero().flatten().tolist():
                row={'env_id':eid,'duration_s':float(info['episode_elapsed_s'][eid]),
                    'reasons':[k for k,v in info['termination_reasons'].items() if bool(v[eid])],
                    'timeout':bool(timeout[eid]),'step':step}
                ended.append(row)
                if first_ends[eid] is None:first_ends[eid]=row
            result['steps'].append({'step':step,'ended':ended,'joint_mse':info['tracking']['joint_mse'].tolist(),
                'body_mse':info['tracking']['body_mse'].tolist()})
            if step%50==0:
                _write_metrics(args.output/'metrics.json',result)
                print({'step':step,'max_substep_velocity':peak_velocity},flush=True)
        result.update(status='completed',first_episode_ends=first_ends,reset_schedule=schedule.records,
            max_substep_velocity=peak_velocity,max_cache_q_error=max_cache_q_error,max_cache_v_error=max_cache_v_error)
        torch.save({'schema':'pgmt_substep_trace_v1','joint_names':list(recorder.context(c,torch.tensor([0],device=c.device))['joint_names']),
            'frames':frames},args.output/'substeps.pt')
        torch.save({'actions':torch.stack(actions),'manifest_sha256':result['manifest_sha256']},args.output/'actions.pt')
        _write_metrics(args.output/'metrics.json',result)
        print('COMPLETED',args.output,flush=True)
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());_write_metrics(args.output/'metrics.json',result)
        if frames:torch.save({'frames':frames},args.output/'partial_substeps.pt')
        traceback.print_exc();raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','reference-data','urdf','asset','initial-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--action-tape',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=150);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--reset-mode',choices=('nominal','reference_state'),default='reference_state')
    p.add_argument('--external-forces-every-iteration',action='store_true')
    p.add_argument('--velocity-iterations',type=int,default=0)
    run(p.parse_args())


if __name__=='__main__':main()
