"""Budget/tamper guards, first-episode accounting and owned-process cleanup."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from setup.standing_multiseed_protocol import jobs,evaluation_jobs,prepare,validate_plan,TRAIN_TRANSITIONS,EVAL_TRANSITIONS
from setup.evaluate_standing_multiseed import first_episode_summary,select_checkpoint
from setup.run_standing_multiseed import stop_group

FIXTURE=Path('runs/standing_learning_20260922/fixture')
URDF=Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')
ASSET=URDF.parent.parent/'usd/g1.usd'


def test_six_training_and_24_unique_evaluation_jobs_cover_exact_budget():
    assert len(jobs())==6 and sum(16*24*100 for _ in jobs())==TRAIN_TRANSITIONS
    assert len({j['job'] for j in evaluation_jobs()})==24
    assert sum(4*500 for _ in evaluation_jobs())==EVAL_TRANSITIONS
    assert [j['seed'] for j in jobs()]==[0,0,1,1,2,2]


@pytest.mark.skipif(not (FIXTURE.exists() and URDF.exists() and ASSET.exists()),reason='frozen external fixture unavailable')
def test_prepared_batch_preserves_parent_and_rejects_mutated_seed_manifest(tmp_path):
    original=(FIXTURE/'manifest.json').read_bytes()
    a=SimpleNamespace(fixture=FIXTURE,urdf=URDF,asset=ASSET,output=tmp_path/'batch')
    plan=prepare(a)
    assert (FIXTURE/'manifest.json').read_bytes()==original
    assert (a.output/'fixture/manifest.json').read_bytes()==original
    assert all(p['only_parameter_difference']==['actor.mlp.net.6.bias'] for p in plan['cpu_paired_initialization'])
    m=a.output/'seed1.json';d=json.loads(m.read_text());d['learning_budget']['updates']=101;m.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='frozen batch file changed'):validate_plan(a.output/'plan.json')


def test_evaluation_accounts_for_inactive_environment_simulation():
    frames=[{'active':torch.tensor(mask),'tilt_deg':torch.tensor([1.,2.,3.,4.]),
             'joint_mse':torch.ones(4)*.01} for mask in
            ([True]*4,[False,True,True,True],[False,False,False,True])]
    ends=[{'timeout':False,'terminated':True}]*3+[{'timeout':True,'terminated':False}]
    r=first_episode_summary(frames,ends)
    assert r['executed_transitions']==12 and r['scored_transitions']==8
    assert r['active_steps_per_start']==[1,2,2,3] and r['completed_10s']==1
    assert r['mean_tilt_per_start_deg']==[1.,2.,3.,4.]


def test_evaluation_refuses_nonfinal_checkpoint(tmp_path,monkeypatch):
    import setup.evaluate_standing_multiseed as module
    monkeypatch.setattr(module,'validate_plan',lambda p:{})
    monkeypatch.setattr(module,'validate_runtime_source',lambda p:{})
    monkeypatch.setattr(module,'validate_checkpoint_file',lambda *args:None)
    plan=tmp_path/'plan.json';plan.write_text('{}')
    from pgmt.train.fixed_clip_diagnostic import sha256
    root=tmp_path/'train';root.mkdir()
    (root/'execution_ledger.json').write_text(json.dumps({'status':'completed','total_transitions':TRAIN_TRANSITIONS,'plan_sha256':sha256(plan)}))
    manifest=tmp_path/'seed0.json';manifest.write_text('{}')
    folder=root/'seed0_control';folder.mkdir()
    (folder/'metrics.json').write_text(json.dumps({'status':'completed','completed_updates':100,'environment_config':{}}))
    torch.save({'seed':0,'manifest_sha256':sha256(manifest),'actor_initialization':{'mode':'standing_control'},
                'environment_config':{},'diagnostic_ppo':{'update_count':20}},folder/'policy.pt')
    with pytest.raises(ValueError,match='incorrect initial/final'):select_checkpoint(plan,root,'seed0_control_final_stochastic')


def test_cleanup_terminates_owned_session():
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    try:
        stop_group(p)
        assert p.poll() is not None
    finally:
        if p.poll() is None:p.kill();p.wait()
