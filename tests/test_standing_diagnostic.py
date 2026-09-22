"""Frozen target pairing, tamper guards and pre-reset rollout evidence."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pgmt.cfg.assumptions import get
from pgmt.envs.g1_env import G1Env,G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule,ObservedDiagnosticEnv,SubstepVelocityMonitor,sha256
from pgmt.train.standing_diagnostic import FrozenStandingFixture,StandingCapture,capture_storage
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO

URDF=Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')


def fixture(tmp_path):
    starts={'pose':torch.zeros(29),'control_target':torch.zeros(29),'candidate_target':torch.ones(29)*.01,
        'root_quat':torch.tensor([[1.,0,0,0],[.9999619,.008726535,0,0],[.9999619,-.008726535,0,0],[.9999619,0,.008726535,0]]),
        'root_height':torch.tensor([.793,.794,.794,.795])}
    torch.save(starts,tmp_path/'starts.pt')
    (tmp_path/'candidate.json').write_text(json.dumps({'pose':starts['pose'].tolist(),'target':starts['candidate_target'].tolist()}))
    m={'standing_training_contract':'frozen_static_load_v1','candidate_sha256':sha256(tmp_path/'candidate.json'),
       'starts_sha256':sha256(tmp_path/'starts.pt')}
    return m


def environment():
    n=600;q=np.zeros((n,29),np.float32)
    db=MotionDatabase.from_sequences([dict(name='static',qpos=q,qvel=q.copy(),root_pos=np.tile([0.,0.,.793],(n,1)).astype(np.float32),
        root_rot=np.tile([1.,0,0,0],(n,1)).astype(np.float32),frame_time=.02)])
    e=G1Env(G1EnvConfig(num_envs=8,episode_length_s=10,reference_urdf_path=str(URDF),enable_adaptive_sampling=False),reference_database=db)
    m={'sequence_names':['static'],'clips':[{'sequence':'static','clip_id':'static','start_frame':0,'horizon_s':10.,'split':'train'}]}
    return e,m


@pytest.mark.parametrize('name',['candidate.json','starts.pt'])
def test_frozen_standing_rejects_mutated_evidence(tmp_path,name):
    m=fixture(tmp_path);FrozenStandingFixture(tmp_path,m,'candidate')
    with (tmp_path/name).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):FrozenStandingFixture(tmp_path,m,'candidate')


@pytest.mark.skipif(not URDF.exists(),reason='external asset unavailable')
def test_four_start_replicas_and_only_actor_bias_changes(tmp_path):
    m=fixture(tmp_path);states=[];policies=[]
    for condition in ('control','candidate'):
        e,manifest=environment();f=FrozenStandingFixture(tmp_path,m,condition);target=f.apply(e)
        FixedClipSchedule(e,manifest);e.reset(seed=0)
        torch.testing.assert_close(e.root_quat[:4],e.root_quat[4:],atol=0,rtol=0)
        torch.testing.assert_close(e.qpos,torch.zeros_like(e.qpos),atol=0,rtol=0)
        states.append((e.root_pos.clone(),e.root_quat.clone()))
        torch.manual_seed(0);policies.append(Stage1Policy(initial_joint_targets=target).state_dict())
    assert all(torch.equal(a,b) for a,b in zip(*states))
    diff=[k for k in policies[0] if not torch.equal(policies[0][k],policies[1][k])]
    assert diff==['actor.mlp.net.6.bias']


@pytest.mark.skipif(not URDF.exists(),reason='external asset unavailable')
def test_capture_matches_collector_and_owns_snapshots(tmp_path):
    m=fixture(tmp_path);e,manifest=environment();target=FrozenStandingFixture(tmp_path,m,'candidate').apply(e)
    s=FixedClipSchedule(e,manifest);obs=e.reset(seed=1)
    capture=StandingCapture(ObservedDiagnosticEnv(e,s),e)
    ppo=PPO(Stage1Policy(initial_joint_targets=target),replace(get('A6').value,num_steps_per_env=4,num_mini_batches=1))
    _,metrics=ppo.collect_rollout(capture,obs)
    stored=capture_storage(ppo.storage);frames=capture.take()
    assert len(frames)==4 and not capture.frames and metrics['steps']==32
    torch.testing.assert_close(torch.stack([f['physics']['target'] for f in frames]),stored['actions'],atol=0,rtol=0)
    torch.testing.assert_close(torch.stack([f['reward'] for f in frames]),stored['rewards'],atol=0,rtol=0)
    original=stored['actions'].clone();ppo.storage.actions.zero_()
    assert torch.equal(stored['actions'],original)


def test_substep_capture_keeps_all_samples_without_aliasing():
    velocity=torch.ones(2,29)
    core=SimpleNamespace(num_envs=2,joint_ids=list(range(29)),articulation=SimpleNamespace(root_physx_view=SimpleNamespace(get_dof_velocities=lambda:velocity)))
    env=SimpleNamespace(env=SimpleNamespace(core=core,scene=SimpleNamespace(update=lambda dt:None)))
    monitor=SubstepVelocityMonitor(env,retain_raw=True)
    for i in range(4):velocity.fill_(i);env.env.scene.update(.005)
    raw=monitor.take_raw()
    assert raw.shape==(4,2,29) and raw[:,0,0].tolist()==[0,1,2,3] and not monitor.raw
    assert monitor.summary()['environment_substeps']==8
