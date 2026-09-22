"""Frozen clip scheduling and unmodified first-policy-action evidence."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pgmt.cfg.assumptions import get
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.fixed_clip_diagnostic import FixedClipSchedule, ObservedDiagnosticEnv, finite_tree, load_manifest, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from setup.build_tracking_diagnostic import screen

URDF=Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')


def make_env(horizon=.04, fk=False):
    frames=400
    seqs=[]
    for i in range(3):
        q=np.zeros((frames,29),np.float32);q[:,3]=.1*i
        seqs.append(dict(name=f'clip{i}', qpos=q,qvel=np.zeros_like(q),
            root_pos=np.tile([0.,0.,.8],(frames,1)).astype(np.float32),
            root_rot=np.tile([1.,0.,0.,0.],(frames,1)).astype(np.float32),frame_time=.02))
    db=MotionDatabase.from_sequences(seqs)
    cfg=G1EnvConfig(num_envs=2,episode_length_s=horizon,enable_adaptive_sampling=False,
        reference_urdf_path=str(URDF) if fk else None)
    env=G1Env(cfg,reference_database=db)
    manifest={'sequence_names':[s['name'] for s in db.seqs],
        'clips':[{'clip_id':f'id{i}','sequence':s['name'],'start_frame':20*i,
            'horizon_s':horizon,'split':'check' if i==2 else 'train'} for i,s in enumerate(db.seqs)]}
    return env,manifest


def test_per_environment_schedule_excludes_check_and_preserves_policy_rng():
    env,m=make_env();s=FixedClipSchedule(env,m)
    rng=torch.get_rng_state().clone()
    env.reset()
    assert torch.equal(rng,torch.get_rng_state())
    assert env.reference_seq_idx.tolist()==[0,1]
    s.sample(torch.tensor([1]));s.sample(torch.tensor([1]));s.sample(torch.tensor([0]))
    assert [(r['env_id'],r['episode'],r['clip_id']) for r in s.records]==[
        (0,0,'id0'),(1,0,'id1'),(1,1,'id0'),(1,2,'id1'),(0,1,'id1')]
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('change', ['horizon','padding','ordering','duplicate','randomization','terrain'])
def test_reject_invalid_or_unmatched_conditions(change):
    env,m=make_env()
    if change=='horizon':m['clips'][0]['horizon_s']+=.02
    if change=='padding':m['clips'][0]['start_frame']=399
    if change=='ordering':m['sequence_names'].reverse()
    if change=='duplicate':m['clips'][1]['clip_id']=m['clips'][0]['clip_id']
    if change=='randomization':env.cfg.corrupt_observations=True
    if change=='terrain':env.cfg.terrain_family='stairs'
    with pytest.raises(ValueError):FixedClipSchedule(env,m)


@pytest.mark.skipif(not URDF.exists(),reason='external URDF unavailable')
def test_real_first_action_and_timeout_bootstrap_survive_auto_reset():
    env,m=make_env(fk=True);s=FixedClipSchedule(env,m);obs=env.reset()
    wrapper=ObservedDiagnosticEnv(env,s)
    policy=Stage1Policy();ppo=PPO(policy,replace(get('A6').value,num_steps_per_env=2,num_mini_batches=1))
    obs,metrics=ppo.collect_rollout(wrapper,obs)
    assert metrics['timeouts']==2
    assert len(wrapper.ends)==2 and all(e['timeout'] for e in wrapper.ends)
    assert s.episodes.tolist()==[2,2]
    assert len(wrapper.first_steps)==2
    for row in wrapper.first_steps:
        assert torch.equal(row['requested_target'],row['executed_target'])
        assert not torch.equal(row['requested_target'],row['reference_qpos'])
        assert row['episode']==0
    # The real PPO collector rejects missing terminal observations on timeout.
    assert torch.isfinite(ppo.storage.returns).all()


def test_screen_is_deterministic_with_distinct_reserved_sequences():
    seqs=[]
    for prefix in ('aiming','walk'):
        for i in range(4):
            n=400;t=np.arange(n,dtype=np.float32)*.02
            q=np.zeros((n,29),np.float32)
            seqs.append(dict(name=f'{prefix}{i}',qpos=q,qvel=q.copy(),frame_time=.02,
                root_pos=np.stack([t*(.3 if prefix=='walk' else .02),t*0,t*0+.8],-1),
                root_rot=np.tile([1.,0.,0.,0.],(n,1)).astype(np.float32)))
    db=MotionDatabase.from_sequences(seqs)
    a,_=screen(db);b,_=screen(db)
    assert a==b and len(a)==8
    train={c['sequence'] for c in a if c['split']=='train'}
    check={c['sequence'] for c in a if c['split']=='check'}
    assert len(train)==len(check)==4 and not train&check
    assert {c['category'] for c in a}=={'low_motion','slow_walk'}


def test_finite_validation_catches_optimizer_and_numpy_state():
    assert finite_tree({'a':[torch.ones(2),np.ones(2),1.]})
    assert not finite_tree({'optimizer':{'moments':torch.tensor(float('nan'))}})
    assert not finite_tree({'a':np.array([float('inf')])})


def test_manifest_fingerprints_fail_after_data_mutation(tmp_path):
    import json
    data=tmp_path/'data';data.mkdir();npz=data/'clip.npz';npz.write_bytes(b'original')
    asset=tmp_path/'asset';asset.write_bytes(b'asset');urdf=tmp_path/'urdf';urdf.write_bytes(b'urdf')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema':'pgmt_fixed_clip_diagnostic_v1','control_dt':.02,'sim_dt':.005,
        'reference_files':{'clip.npz':sha256(npz)},'urdf_sha256':sha256(urdf),'asset_sha256':sha256(asset)}))
    load_manifest(manifest,data,urdf,asset)
    npz.write_bytes(b'changed')
    with pytest.raises(ValueError,match='differ'):load_manifest(manifest,data,urdf,asset)


def test_diagnostic_checkpoint_cannot_initialize_formal_stage2(tmp_path):
    from pgmt.train.train_stage2 import _load_stage1
    checkpoint=tmp_path/'diagnostic.pt'
    torch.save({'schema':'pgmt_fixed_clip_diagnostic_v1','diagnostic_ppo':PPO(Stage1Policy()).state_dict()},checkpoint)
    with pytest.raises(ValueError,match='physical Stage 1'):
        _load_stage1(checkpoint,torch.device('cpu'),require_physics=True)
