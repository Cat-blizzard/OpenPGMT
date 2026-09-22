"""Complete reference reset contract and floor support; CPU only."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pgmt.envs.actuators import validate_actuator_resume
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.reset_geometry import CollisionFloor

URDF = Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')


def test_floor_geometry_fixed_links_rotated_primitives_and_mesh(tmp_path):
    import trimesh
    trimesh.creation.box(extents=(.2, .4, .6)).export(tmp_path / 'box.stl')
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="test"><link name="root">
      <collision><origin xyz="0 0 1"/><geometry><mesh filename="box.stl" scale="1 1 2"/></geometry></collision>
      <collision><origin xyz="0 0 1" rpy="0 1.57079632679 0"/><geometry><cylinder radius=".2" length=".8"/></geometry></collision>
      </link><link name="fixed"><collision><geometry><sphere radius=".1"/></geometry></collision></link>
      <joint name="fixed_joint" type="fixed"><parent link="root"/><child link="fixed"/><origin xyz="0 0 .2"/></joint>
      </robot>''')
    geom = CollisionFloor(path, ['root'])
    p = torch.zeros(2, 1, 3)
    quat = torch.tensor([[[1., 0, 0, 0]], [[0., 1, 0, 0]]])
    # Upright sphere lowest=.1; upside-down scaled mesh lowest=-1.6.
    torch.testing.assert_close(geom.min_height(p, quat), torch.tensor([.1, -1.6]))


@pytest.fixture
def env():
    if not URDF.exists():
        pytest.skip('external licensed URDF absent')
    t = np.arange(12, dtype=np.float32)
    q = np.zeros((12, 29), np.float32)
    q[:, 3] = .2 + .01*t
    angle = .4 + .01*t
    # Moving, tilted root: preserve pitch, align only heading, rotate velocity.
    rot = np.stack([np.cos(angle/2), np.zeros_like(t), np.sin(angle/2), np.zeros_like(t)], -1)
    db = MotionDatabase.from_sequences([dict(qpos=q, qvel=np.gradient(q, .02, axis=0),
        root_pos=np.stack([t*.02, t*.01, .8+t*.002], -1), root_rot=rot, frame_time=.02)])
    return G1Env(G1EnvConfig(num_envs=3, reference_urdf_path=str(URDF), reset_mode='reference_state',
        default_root_pos=(0.,0.,.65), max_action_delay=2, enable_adaptive_sampling=False), reference_database=db)


def test_complete_state_height_velocity_and_first_command(env):
    obs = env.reset(seed=111)
    reference = env._reference_batch(env.reference_seq_idx, env.reference_frame)
    for name, key in [('qpos','joint_pos'), ('qvel','joint_vel'), ('root_pos','root_pos'),
                      ('root_quat','root_quat'), ('root_lin_vel','root_lin_vel'), ('root_ang_vel','root_ang_vel')]:
        torch.testing.assert_close(getattr(env, name), reference[key], atol=2e-5, rtol=1e-5)
    assert env._reset_height_lift.max() > 0
    assert env._reset_collision_floor.min_height(env.body_pos, env.body_quat).min() >= -1e-6
    assert env.qvel.abs().max() > .1 and env.root_ang_vel.abs().max() > .1
    torch.testing.assert_close(env._previous_body_lin_vel, env.body_lin_vel)
    torch.testing.assert_close(env._previous_root_lin_vel, env.root_lin_vel)
    for name in ('action', '_prev_action', 'target'):
        torch.testing.assert_close(getattr(env, name), env.qpos)
    assert obs['history'].eq(0).all()
    assert env.history[:, :-1].eq(0).all()
    assert env.contact_forces.eq(0).all() and env.foot_contact.eq(0).all()
    env._action_delay[:] = 2
    old = env.qpos.clone()
    supplied = old + .01
    env._apply_action(supplied)
    torch.testing.assert_close(env.action, supplied)
    torch.testing.assert_close(env.target, old)  # explicit delay, no nominal-zero command
    env._apply_action(supplied)
    env._apply_action(supplied)
    torch.testing.assert_close(env.target, supplied)


def test_subset_and_recovery_are_not_overwritten(env):
    before = {k: getattr(env, k).clone() for k in ('qpos','qvel','root_pos','root_lin_vel','history','_action_queue')}
    env._forced_reference[1] = (0, 2)
    env._sample_reference(torch.tensor([1]))
    for name, value in before.items():
        torch.testing.assert_close(getattr(env, name)[[0,2]], value[[0,2]])
    batch = env._reference_batch(env.reference_seq_idx, env.reference_frame)
    sentinel = env.qpos[2].clone().add(.1)
    env.qpos[2] = sentinel
    env._recovery_active[2] = 1
    env._initialize_reference_state(torch.tensor([0,1]), batch, torch.tensor([True,True,False]))
    torch.testing.assert_close(env.qpos[2], sentinel)
    calls = []
    env.articulation = SimpleNamespace(
        write_root_link_state_to_sim=lambda state, env_ids: calls.append(('link', env_ids.tolist(), state.clone())),
        write_root_state_to_sim=lambda state, env_ids: calls.append(('legacy', env_ids.tolist(), state.clone())))
    state = torch.cat([env.root_pos, env.root_quat, env.root_lin_vel, env.root_ang_vel], -1)
    env._write_reset_root_state(state)
    assert [(c[0], c[1]) for c in calls] == [('link',[0,1]), ('legacy',[2])]


def test_reset_mode_resume_guard_and_scope(env):
    saved = env.state_dict()
    env.load_state_dict(saved)
    saved['reset_mode'] = 'nominal'
    with pytest.raises(ValueError, match='reset_mode'):
        env.load_state_dict(saved)
    with pytest.raises(ValueError, match='reset_mode'):
        validate_actuator_resume({'environment_config': asdict(G1EnvConfig())}, env.cfg)
    for kwargs in ({'stage':2}, {'terrain_family':'stairs'}):
        with pytest.raises(ValueError, match='flat ground'):
            G1EnvConfig(reset_mode='reference_state', **kwargs)
    with pytest.raises(ValueError, match='requires reference'):
        G1Env(G1EnvConfig(reset_mode='reference_state'))


def test_reference_reset_preserves_real_pool_sample(env):
    from pgmt.envs.recovery import FallRecoveryPool
    pool = FallRecoveryPool(init_prob=1., prob_max=1.)
    state = dict(qpos=torch.full((29,), .12), qvel=torch.full((29,), .3),
        root_pos=torch.tensor([.1,.2,.25]), root_quat=torch.tensor([1.,0.,0.,0.]),
        root_lin_vel=torch.tensor([.2,.3,.4]), root_ang_vel=torch.tensor([.1,.2,.3]))
    pool.add(state, seq_idx=0, frame=2)
    env.recovery_pool = pool
    env.reset(seed=113)
    assert env._recovery_active.eq(1).all()
    for name, value in state.items():
        torch.testing.assert_close(getattr(env, name), value.expand_as(getattr(env, name)))
    assert env._reset_height_lift.eq(0).all()
    torch.testing.assert_close(env.target, env.default_q)
