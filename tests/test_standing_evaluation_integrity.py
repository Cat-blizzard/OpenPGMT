import json
from pathlib import Path
import pytest

from pgmt.train.fixed_clip_diagnostic import sha256
from setup.standing_multiseed_protocol import jobs,TRAIN_TRANSITIONS
from setup.standing_evaluation_integrity import (validate_runtime_source,write_checkpoint_index,
                                                validate_checkpoint_file,validate_runtime_configuration,serialized_configuration)


def test_deferred_evaluation_rejects_runtime_drift(tmp_path,monkeypatch):
    import setup.standing_evaluation_integrity as module
    monkeypatch.setattr(module,'REPO',tmp_path)
    files={}
    for name in ('pgmt/train/policy.py','pgmt/envs/g1_env.py'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('frozen');files[name]=sha256(p)
    (tmp_path/'source_manifest.json').write_text(json.dumps(files))
    assert validate_runtime_source(tmp_path)==files
    (tmp_path/'pgmt/train/policy.py').write_text('changed')
    with pytest.raises(ValueError,match='runtime dependency changed'):validate_runtime_source(tmp_path)


def test_checkpoint_index_rejects_replacement(tmp_path):
    (tmp_path/'execution_ledger.json').write_text(json.dumps({'status':'completed','total_transitions':TRAIN_TRANSITIONS,'plan_sha256':'plan'}))
    for j in jobs():
        p=tmp_path/j['name'];p.mkdir()
        for n in ('initial.pt','policy.pt'):(p/n).write_bytes((j['name']+n).encode())
    write_checkpoint_index(tmp_path);p=tmp_path/'seed2_candidate/policy.pt'
    validate_checkpoint_file(tmp_path,p,'plan');p.write_bytes(b'replaced')
    with pytest.raises(ValueError,match='checkpoint differs'):validate_checkpoint_file(tmp_path,p,'plan')


def test_evaluation_allows_four_starts_but_rejects_physics_change():
    saved={'num_envs':16,'episode_length_s':10.,'control_dt':.02,'pd_kp':80,'device':'cuda:0'}
    actual={**saved,'num_envs':4}
    validate_runtime_configuration(saved,actual)
    with pytest.raises(ValueError,match='configuration differs'):
        validate_runtime_configuration(saved,{**actual,'pd_kp':81})


def test_checkpoint_tuple_and_path_config_matches_saved_json():
    saved={'num_envs':16,'episode_length_s':10.,'stiffness':(80.,)*29,'asset_path':Path('/asset.usd')}
    metrics={'num_envs':16,'episode_length_s':10.,'stiffness':[80.]*29,'asset_path':'/asset.usd'}
    assert serialized_configuration(saved)==metrics
    validate_runtime_configuration(saved,{**metrics,'num_envs':4})
