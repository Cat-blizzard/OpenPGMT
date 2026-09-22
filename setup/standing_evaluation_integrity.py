"""CPU provenance checks before deferred standing evaluation allocates CUDA."""
import json
from pathlib import Path

from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics, _json_default
from setup.standing_multiseed_protocol import jobs, TRAIN_TRANSITIONS

REPO=Path(__file__).resolve().parents[1]


def serialized_configuration(value):
    """Compare checkpoint Paths/tuples with their JSON metrics representation."""
    return json.loads(json.dumps(value,default=_json_default))


def validate_runtime_source(training_root):
    frozen=json.loads((Path(training_root)/'source_manifest.json').read_text())
    required={'pgmt/train/policy.py','pgmt/envs/g1_env.py'}
    if not required.issubset(frozen):raise ValueError('missing frozen runtime source evidence')
    selected={p:h for p,h in frozen.items() if p.startswith('pgmt/') or p in ('data/retarget_lafan1.py','data/bvh.py')}
    for path,digest in selected.items():
        if sha256(REPO/path)!=digest:raise ValueError(f'training runtime dependency changed: {path}')
    return selected


def write_checkpoint_index(training_root):
    root=Path(training_root);dest=root/'checkpoint_manifest.json'
    if dest.exists():raise FileExistsError(dest)
    ledger=json.loads((root/'execution_ledger.json').read_text())
    if ledger['status']!='completed' or ledger['total_transitions']!=TRAIN_TRANSITIONS:
        raise ValueError('checkpoint index requires completed training')
    files={str(Path(j['name'])/name):sha256(root/j['name']/name)
           for j in jobs() for name in ('initial.pt','policy.pt')}
    _write_metrics(dest,{'schema':'standing_checkpoint_index_v1','plan_sha256':ledger['plan_sha256'],'files':files})


def validate_checkpoint_file(training_root,path,plan_sha):
    root=Path(training_root);path=Path(path)
    index=json.loads((root/'checkpoint_manifest.json').read_text())
    rel=str(path.relative_to(root))
    expected={str(Path(j['name'])/n) for j in jobs() for n in ('initial.pt','policy.pt')}
    if (index['schema']!='standing_checkpoint_index_v1' or index['plan_sha256']!=plan_sha
            or set(index['files'])!=expected or index['files'].get(rel)!=sha256(path)):
        raise ValueError('checkpoint differs from the completed training index')


def validate_runtime_configuration(saved,actual):
    saved=serialized_configuration(saved);actual=serialized_configuration(actual)
    ignored={'num_envs','device','asset_path','reference_data_dir','reference_urdf_path'}
    if saved['num_envs']!=16 or actual['num_envs']!=4 or actual['episode_length_s']!=10.:
        raise ValueError('invalid standing evaluation layout')
    if {k:v for k,v in saved.items() if k not in ignored}!={k:v for k,v in actual.items() if k not in ignored}:
        raise ValueError('evaluation physics/configuration differs from training')
