"""Bounded replay of saved standing actions with passive 5 ms observations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import torch

from pgmt.train.diagnostics import attach_diagnostics, cpu_copy
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, load_manifest, finite_tree, sha256
from pgmt.train.standing_diagnostic import FrozenStandingFixture
from pgmt.train.train_stage1 import build_env, _seed_everything, _write_metrics


def validate_tape(tape):
    actions = tape['actions']
    if actions.shape != (300, 16, 29) or not torch.isfinite(actions).all():
        raise ValueError('replay requires the frozen 300 x 16 x 29 finite tape')
    return actions


def run(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    original_metrics = json.loads((a.original / 'metrics.json').read_text())
    args = original_metrics['arguments']
    manifest = load_manifest(a.fixture / 'manifest.json', a.fixture / 'reference', args['urdf'], args['asset'])
    tape = torch.load(a.tape, map_location='cpu', weights_only=False)
    actions = validate_tape(tape)
    condition = original_metrics['standing_fixture']['condition']
    if tape['source_condition'] != condition:
        raise ValueError('wrong action tape condition')
    for name, value in tape['source_sha256'].items():
        if sha256(name) != value:
            raise ValueError('original capture fingerprint changed')
    fixture = FrozenStandingFixture(a.fixture, manifest, condition)
    a.output.mkdir(parents=True)
    result = {'status': 'running', 'condition': condition, 'ppo_updates': 0, 'executed_steps': 0,
              'transitions': 0, 'maximum_transitions': 4800, 'tape_sha256': sha256(a.tape),
              'steps': [], 'source_sha256': sha256(__file__)}
    app = env = None
    frames = []
    try:
        _seed_everything(0)
        from pgmt.envs.isaac_app import launch_isaac_app
        app = launch_isaac_app('cuda:0')
        env = build_env(16, torch.device('cuda:0'), backend='isaaclab', app=app,
            reference_data=args['reference_data'], asset_path=args['asset'], reference_urdf=args['urdf'],
            env_options={'seed': 0, 'reset_mode': 'nominal', 'episode_length_s': manifest['horizon_s'],
                'enable_adaptive_sampling': False, 'randomize_dynamics': False,
                'corrupt_observations': False, 'max_action_delay': 0})
        core = env.env.core; robot = env.env.robot; view = robot.root_physx_view
        fixture.apply(core); schedule = FixedClipSchedule(core, manifest)
        attach_diagnostics(env, a.output / 'diagnostics')
        obs = env.reset()
        initial = torch.load(a.original / 'initial_evidence.pt', map_location='cpu', weights_only=False)
        for key, value in obs.items():
            torch.testing.assert_close(value.cpu(), initial['observations'][key], atol=2e-6, rtol=0)
        config = json.loads(json.dumps(asdict(core.cfg), default=str))
        if config != original_metrics['environment_config']:
            raise ValueError('environment configuration differs from original training')
        result.update(initial_observations_match=True, environment_config=config,
            sensor_body_names=list(env.env.scene['contact_forces'].body_names),
            physics_config={k: getattr(env.env.cfg.sim.physx, k) for k in
                ('solver_type', 'min_position_iteration_count', 'min_velocity_iteration_count', 'enable_external_forces_every_iteration')},
            torque_scope='Isaac implicit actuator pre-substep PD estimate; not measured solver drive torque')
        sensor = env.env.scene['contact_forces']
        original_update = env.env.scene.update
        step = substep = 0
        previous_q = previous_v = previous_stamp = None

        def observe(dt):
            nonlocal substep, previous_q, previous_v, previous_stamp
            original_update(dt); substep += 1
            q = view.get_dof_positions().clone()[:, core.joint_ids]
            v = view.get_dof_velocities().clone()[:, core.joint_ids]
            torch.testing.assert_close(q, robot.data.joint_pos[:, core.joint_ids], atol=1e-6, rtol=0)
            torch.testing.assert_close(v, robot.data.joint_vel[:, core.joint_ids], atol=1e-6, rtol=0)
            forces = sensor.data.net_forces_w
            raw_forces = sensor.contact_physx_view.get_net_contact_forces(dt=dt).view_as(forces)
            torch.testing.assert_close(forces, raw_forces, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(sensor._timestamp, sensor._timestamp_last_update, atol=1e-6, rtol=0)
            torch.testing.assert_close(sensor._timestamp - previous_stamp,
                                       torch.full_like(previous_stamp, dt), atol=1e-6, rtol=0)
            target = view.get_dof_position_targets()[:, core.joint_ids]
            torch.testing.assert_close(core.target, target, atol=0, rtol=0)
            evidence = cpu_copy({'step': step, 'substep': substep,
                'qpos': q, 'qvel': v, 'qpos_before': previous_q, 'qvel_before': previous_v,
                'qpos_fd': (q-previous_q)/dt, 'target': target,
                'estimated_drive_torque': robot.data.applied_torque[:, core.joint_ids],
                'contact_forces': forces, 'sensor_timestamp': sensor._timestamp,
                'root_pos': robot.data.root_link_pos_w, 'root_quat': robot.data.root_link_quat_w,
                'root_lin_vel': robot.data.root_link_lin_vel_w,
                'body_pos': robot.data.body_link_pos_w, 'body_quat': robot.data.body_link_quat_w})
            if not finite_tree(evidence):
                raise RuntimeError('non-finite physical evidence')
            frames.append(evidence)
            previous_q, previous_v, previous_stamp = q, v, sensor._timestamp.clone()

        env.env.scene.update = observe
        actions = actions.to(core.device)
        for step in range(1, 301):
            previous_q = view.get_dof_positions().clone()[:, core.joint_ids]
            previous_v = view.get_dof_velocities().clone()[:, core.joint_ids]
            previous_stamp = sensor._timestamp.clone(); substep = 0
            # Count attempted control steps conservatively even if observation fails.
            result.update(executed_steps=step, transitions=step*16)
            obs, reward, terminated, timeout, info = env.step(actions[step-1])
            if substep != 4 or not torch.isfinite(reward).all():
                raise RuntimeError('invalid control step')
            expected = tape['expected']; last = frames[-1]
            qerr = float((last['qpos']-expected['joint_pos'][step-1]).abs().max())
            rerr = float((last['root_pos']-expected['root_pos'][step-1]).abs().max())
            verr = float((torch.stack([f['qvel'] for f in frames[-4:]])-tape['substep_qvel'][(step-1)*4:step*4]).abs().max())
            reset_match = torch.equal(terminated.cpu(), expected['terminated'][step-1]) and torch.equal(timeout.cpu(), expected['timeouts'][step-1])
            result['steps'].append({'step': step, 'max_q_error_rad': qerr, 'max_root_error_m': rerr,
                'max_substep_velocity_error_rad_s': verr, 'reset_match': reset_match,
                'peak_speed_rad_s': float(torch.stack([f['qvel'].abs().max() for f in frames[-4:]]).max())})
            if qerr > .02 or rerr > .01 or not reset_match:
                result.update(status='trajectory_diverged', stopping_reason='pre-registered trajectory mismatch; no automatic retry')
                break
            if step % 25 == 0:
                _write_metrics(a.output / 'metrics.json', result)
                print(condition, step, 'q error', qerr, 'v error', verr, flush=True)
        else:
            result['status'] = 'completed'
        result['reset_schedule'] = schedule.records
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if frames:
            torch.save({'frames': frames, 'scope': result.get('torque_scope'),
                'joint_order': 'G1_JOINT_NAMES', 'sensor_body_names': result.get('sensor_body_names'),
                'robot_body_names': list(robot.body_names)}, a.output / 'substeps.pt')
        _write_metrics(a.output / 'metrics.json', result)
        if env is not None: env.close()
        if app is not None: app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original', 'fixture', 'tape', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    run(parser.parse_args())
