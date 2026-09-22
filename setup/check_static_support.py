"""Bounded standing controls and real rollout capture, without PPO updates.

The synthetic fixture is a diagnostic, not a replacement for motion data.
Four starts are fixed before simulation: upright, roll +/-1 deg, pitch +1 deg.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, SubstepVelocityMonitor, load_manifest, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.train_stage1 import _seed_everything, _write_metrics, build_env
from setup.audit_reference_control import SupportGeometry


def prepare(output, urdf, asset):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = output / 'reference'
    data.mkdir()
    n = 600  # 12 seconds, including future-reference padding.
    np.savez_compressed(data / 'static_neutral.npz', qpos=np.zeros((n, 29), np.float32),
        qvel=np.zeros((n, 29), np.float32), root_pos=np.tile([0, 0, .793], (n, 1)).astype(np.float32),
        root_rot=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        contacts=np.ones((n, 2), bool), frame_time=np.array(.02),
        contact_source=np.array('synthetic_static_diagnostic'))
    quat = np.tile([1., 0, 0, 0], (4, 1))
    angle = np.deg2rad(1.) / 2
    for row, axis, sign in ((1, 1, 1), (2, 1, -1), (3, 2, 1)):
        quat[row, 0] = np.cos(angle)
        quat[row, axis] = sign * np.sin(angle)
    root = np.tile([0., 0, .793], (4, 1))
    geometry = SupportGeometry(urdf)
    before = geometry.compute(np.zeros((4, 29)), root, quat)
    lift = np.maximum(-before['clearance'].min((1, 2)), 0)
    root[:, 2] += lift
    after = geometry.compute(np.zeros((4, 29)), root, quat)
    torch.save({'root_quat': torch.tensor(quat, dtype=torch.float32),
                'root_height': torch.tensor(root[:, 2], dtype=torch.float32)}, output / 'starts.pt')
    _write_metrics(output / 'geometry.json', {
        'scope': 'URDF foot-sphere clearance and COM; not dynamic stability',
        'starts': ['upright', 'roll_plus_1deg', 'roll_minus_1deg', 'pitch_plus_1deg'],
        'height_lift_m': lift.tolist(), 'root_pos': root.tolist(), 'root_quat': quat.tolist(),
        'com_world': after['com'].tolist(), 'foot_points': after['foot_points'].tolist(),
        'foot_clearance': after['clearance'].tolist(), 'mass_kg': after['mass_kg']})
    _write_metrics(output / 'manifest.json', {
        'schema': 'pgmt_fixed_clip_diagnostic_v1', 'control_dt': .02, 'sim_dt': .005, 'horizon_s': 10.,
        'scope': 'synthetic zero-joint standing; no source-motion replacement',
        'clips': [{'clip_id': 'static_neutral', 'sequence': 'static_neutral', 'start_frame': 0,
                   'horizon_s': 10., 'split': 'train'}], 'sequence_names': ['static_neutral'],
        'reference_files': {p.name: sha256(p) for p in data.glob('*.npz')},
        'asset_sha256': sha256(asset), 'urdf_sha256': sha256(urdf)})


def run(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = load_manifest(a.fixture / 'manifest.json', a.fixture / 'reference', a.urdf, a.asset)
    starts = torch.load(a.fixture / 'starts.pt', map_location='cpu', weights_only=True)
    if a.condition is not None:
        if a.mode not in ('fixed', 'deterministic'):
            raise ValueError('static-load comparison only permits fixed or mean-policy control')
        if sha256(a.fixture / 'candidate.json') != manifest['candidate_sha256']:
            raise ValueError('candidate changed after protocol freeze')
    a.output.mkdir(parents=True)
    result = {'status': 'running', 'mode': a.mode, 'scope': 'static control diagnostic; no learning',
              'condition': a.condition, 'starts_sha256': sha256(a.fixture / 'starts.pt'),
              'budget': {'envs': 4, 'maximum_steps': 500, 'maximum_transitions': 2000},
              'manifest_sha256': sha256(a.fixture / 'manifest.json'), 'source_sha256': sha256(__file__)}
    _write_metrics(a.output / 'metrics.json', result)
    app = env = None
    trace, rollout = [], []
    executed_steps = 0
    try:
        _seed_everything(a.seed)
        from pgmt.envs.isaac_app import launch_isaac_app
        app = launch_isaac_app(a.device)
        env = build_env(4, torch.device(a.device), backend='isaaclab', app=app,
            reference_data=str(a.fixture / 'reference'), asset_path=str(a.asset), reference_urdf=str(a.urdf),
            env_options={'seed': a.seed, 'episode_length_s': 10., 'enable_adaptive_sampling': False,
                         'randomize_dynamics': False, 'corrupt_observations': False,
                         'max_action_delay': 0, 'reset_mode': 'nominal'})
        c = env.env.core
        if a.condition is not None:
            c.default_q.copy_(starts['pose'].to(c.device).expand_as(c.default_q))
        c.default_root_quat.copy_(starts['root_quat'].to(c.device))
        c.default_root_pos[:, 2] = starts['root_height'].to(c.device)
        schedule = FixedClipSchedule(c, manifest)
        recorder = attach_diagnostics(env, a.output / 'diagnostics')
        obs = env.reset()
        monitor = SubstepVelocityMonitor(env)
        _seed_everything(a.seed)  # identical weights and sampling start across controls
        fixed_target = (c.default_q.clone() if a.condition is None else
                        starts[a.condition+'_target'].to(c.device).expand_as(c.default_q).clone())
        policy = Stage1Policy(initial_joint_targets=None if a.condition is None else fixed_target[0].cpu()).to(c.device).eval()
        torch.save(cpu_copy(policy.state_dict()), a.output / 'initial_policy.pt')
        result['initial'] = recorder.context(c, torch.arange(4, device=c.device))
        result['initial_body_lin_vel'] = cpu_copy(c.body_lin_vel)
        result['initial_previous_body_lin_vel'] = cpu_copy(c._previous_body_lin_vel)
        result['environment_config'] = asdict(c.cfg)
        result['frozen_target'] = fixed_target[0].cpu().tolist()
        expected = c._reference_motion.kinematics(c.qpos, c.qvel, c.root_pos, c.root_quat,
                                                 c.root_lin_vel, c.root_ang_vel)
        ids = [c._reference_motion.body_names.index(n) for n in c._reward_computer.body_names]
        ep = (c.body_pos-expected['body_pos'][:,ids]).abs().max().item()
        eq = torch.minimum((c.body_quat-expected['body_quat'][:,ids]).abs().amax(-1),
                           (c.body_quat+expected['body_quat'][:,ids]).abs().amax(-1)).max().item()
        result['runtime_fk_readback'] = {'position_max_error_m':ep,'quaternion_max_error':eq,
            'tolerance_position_m':2e-4,'tolerance_quaternion':2e-4}
        if ep > 2e-4 or eq > 2e-4:
            raise RuntimeError('runtime poses disagree with the asset-bound reference model')
        result['physics_config'] = {k: getattr(env.env.cfg.sim.physx, k) for k in
            ('solver_type', 'min_velocity_iteration_count', 'enable_external_forces_every_iteration')}
        reward_impl = c._reward
        evidence = {}
        foot_ids = [c._reward_computer.bi[n] for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        ee_ids = [c._reward_computer.bi[n] for n in ('left_ankle_roll_link', 'right_ankle_roll_link',
                                                   'left_wrist_yaw_link', 'right_wrist_yaw_link')]

        def observe_reward():
            evidence.clear()
            evidence.update(cpu_copy({
                'qpos': c.qpos, 'qvel': c.qvel, 'target': c.target,
                'root_pos': c.root_pos, 'root_quat': c.root_quat,
                'foot_pos': c.body_pos[:, foot_ids], 'foot_vel': c.body_lin_vel[:, foot_ids],
                'foot_contact': c.foot_contact, 'contact_forces': c.contact_forces[:, foot_ids],
                'ee_lin_vel':c.body_lin_vel[:,ee_ids],
                'previous_ee_lin_vel':c._previous_body_lin_vel[:,ee_ids],
                'ee_accel':(c.body_lin_vel[:,ee_ids]-c._previous_body_lin_vel[:,ee_ids])/c.cfg.control_dt,
                'reference_ee_lin_vel':c.reference_body['body_lin_vel'][:,ee_ids],
                'reference_ee_accel':c.reference_body['body_accel'][:,ee_ids],
                'estimated_torque': c.articulation.data.applied_torque[:, c.joint_ids],
                'reference_frame': c.reference_frame, 'episode_steps': c.episode_length_buf}))
            return reward_impl()

        c._reward = observe_reward
        ends = [None] * 4
        for step in range(500):
            current = {k: v.clone() for k, v in obs.items()}
            before_frame = c.reference_frame.clone()
            with torch.no_grad():
                output = policy.act(current, deterministic=a.mode != 'stochastic')
                action = fixed_target if a.mode == 'fixed' else output.actions
                obs, reward, terminated, timeout, info = env.step(action)
                executed_steps += 1
                next_value = policy.value(obs)
                terminal = info.get('terminal_observation')
                mask = timeout & ~terminated
                if mask.any():
                    if terminal is None:
                        raise RuntimeError('missing timeout observation')
                    selected = {k: v[mask] if v.shape[0] == 4 else v for k, v in terminal.items()}
                    next_value[mask] = policy.value(selected)
            # Reward hook ran after physics/reference advance, before auto-reset.
            torch.testing.assert_close(evidence['reference_frame'], before_frame.cpu() + 1, atol=1e-4, rtol=0)
            torch.testing.assert_close(evidence['target'], action.cpu(), atol=0, rtol=0)
            if not torch.isfinite(reward).all():
                raise RuntimeError('non-finite reward')
            trace.append(dict(evidence, step=step + 1, first_episode_active=[v is None for v in ends],
                              reward=cpu_copy(reward), tracking=cpu_copy(info['tracking']),
                              weighted_aux_terms=cpu_copy(info.get('weighted_aux_terms', {}))))
            if a.mode == 'stochastic':
                rollout.append(cpu_copy({'observations': current, 'actions': action,
                    'latent_actions': output.latent_actions, 'log_probs': output.log_probs,
                    'values': output.values, 'rewards': reward, 'next_observations': obs,
                    'next_values': next_value, 'terminated': terminated, 'timeouts': timeout,
                    'terminal_observation': terminal}))
            for eid in (terminated | timeout).nonzero().flatten().tolist():
                if ends[eid] is None:
                    ends[eid] = {'step': step + 1, 'duration_s': float(info['episode_elapsed_s'][eid]),
                                 'terminated': bool(terminated[eid]), 'timeout': bool(timeout[eid]),
                                 'reasons': [k for k, v in info['termination_reasons'].items() if bool(v[eid])]}
            if (step + 1) % 50 == 0:
                print({'step': step + 1, 'first_ends': ends}, flush=True)
            if all(v is not None for v in ends):
                break
        for eid, end in enumerate(ends):
            if end is None:
                ends[eid] = {'step': 500, 'duration_s': 10., 'terminated': False, 'timeout': False,
                             'reasons': [], 'censored_at_horizon': True}
        torch.save({'frames': trace}, a.output / 'physical_trace.pt')
        if rollout:
            torch.save({'schema': 'pgmt_static_rollout_v1', 'frames': rollout,
                        'head_names': policy.head_names}, a.output / 'rollout.pt')
        result.update(status='completed', first_episode_ends=ends, steps=step + 1,
                      transitions=4 * (step + 1), reset_schedule=schedule.records,
                      substeps=monitor.summary(), action_and_reference_timing_passed=True)
        _write_metrics(a.output / 'metrics.json', result)
        print('COMPLETED', a.output, flush=True)
    except BaseException:
        result.update(status='failed', error=traceback.format_exc(),
                      steps=executed_steps, transitions=4 * executed_steps)
        _write_metrics(a.output / 'metrics.json', result)
        if trace:
            torch.save({'frames': trace}, a.output / 'partial_physical_trace.pt')
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'fixed', 'deterministic', 'stochastic'), required=True)
    for name in ('output', 'urdf', 'asset'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--condition', choices=('control','candidate'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.output, a.urdf, a.asset)
    elif a.fixture is None:
        p.error('--fixture is required for physics')
    else:
        run(a)


if __name__ == '__main__':
    main()
