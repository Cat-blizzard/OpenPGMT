"""CPU preparation and strict contracts for the six-run standing diagnostic."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.policy import Stage1Policy
from pgmt.train.standing_diagnostic import FrozenStandingFixture
from pgmt.train.train_stage1 import _write_metrics

SEEDS = (0, 1, 2)
CONDITIONS = ('control', 'candidate')
TRAIN_TRANSITIONS = 230400
EVAL_TRANSITIONS = 48000


def jobs():
    return [{'name':f'seed{s}_{c}', 'seed':s, 'condition':c} for s in SEEDS for c in CONDITIONS]


def evaluation_jobs():
    return [{**j, 'checkpoint':when, 'mode':mode,
             'job':f"{j['name']}_{when}_{mode}"}
            for j in jobs() for when in ('initial', 'final') for mode in ('deterministic', 'stochastic')]


def budget(seed):
    return dict(conditions=list(CONDITIONS), envs=16, steps_per_update=24,
                updates=100, total_transitions=76800, seed=seed, lr_schedule_updates=1000)


def validate_plan(path):
    path=Path(path); p=json.loads(path.read_text()); root=path.parent
    if (p.get('schema')!='standing_multiseed_v1' or p['runs']!=jobs()
            or p['training_transitions']!=TRAIN_TRANSITIONS or p['evaluation_transitions']!=EVAL_TRANSITIONS
            or p['evaluation_jobs']!=evaluation_jobs() or p['evaluation_chained'] is not False):
        raise ValueError('invalid standing batch or budget')
    for rel,digest in p['frozen_files'].items():
        if sha256(root/rel)!=digest: raise ValueError(f'frozen batch file changed: {rel}')
    fixture=root/'fixture'
    for seed in SEEDS:
        m=load_manifest(root/f'seed{seed}.json',fixture/'reference',p['urdf'],p['asset'])
        if m['learning_budget']!=budget(seed) or m['horizon_s']!=10.:
            raise ValueError('invalid per-seed standing budget')
        for condition in CONDITIONS: FrozenStandingFixture(fixture,m,condition)
    return p


def check_pair(root, seed):
    root=Path(root); paths=[root/f'seed{seed}_{c}' for c in CONDITIONS]
    initial=[torch.load(p/'initial.pt',map_location='cpu',weights_only=False) for p in paths]
    states=[s['diagnostic_ppo']['policy'] for s in initial]
    diff=[k for k in states[0] if not torch.equal(states[0][k],states[1][k])]
    if diff!=['actor.mlp.net.6.bias']: raise ValueError('initial policies are not paired')
    evidence=[torch.load(p/'initial_evidence.pt',map_location='cpu',weights_only=False) for p in paths]
    for key in evidence[0]['observations']:
        torch.testing.assert_close(evidence[0]['observations'][key],evidence[1]['observations'][key],atol=0,rtol=0)
    torch.testing.assert_close(evidence[0]['latent_std'],evidence[1]['latent_std'],atol=0,rtol=0)
    metrics=[json.loads((p/'metrics.json').read_text()) for p in paths]
    for key in ('environment_config','ppo_config'):
        if metrics[0][key]!=metrics[1][key]:raise ValueError(f'unpaired {key}')
    return {'seed':seed,'only_parameter_difference':diff,'initial_observations_equal':True,
            'initial_std_equal':True,'environment_and_ppo_equal':True}


def prepare(a):
    if a.output.exists():raise FileExistsError(a.output)
    source=load_manifest(a.fixture/'manifest.json',a.fixture/'reference',a.urdf,a.asset)
    a.output.mkdir(parents=True);shutil.copytree(a.fixture,a.output/'fixture')
    pairing=[]
    for seed in SEEDS:
        m={**source,'learning_budget':budget(seed),'parent_manifest_sha256':sha256(a.fixture/'manifest.json')}
        _write_metrics(a.output/f'seed{seed}.json',m)
        states=[]
        for c in CONDITIONS:
            f=FrozenStandingFixture(a.output/'fixture',m,c)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                states.append(Stage1Policy(initial_joint_targets=f.starts[c+'_target']).state_dict())
        diff=[k for k in states[0] if not torch.equal(states[0][k],states[1][k])]
        if diff!=['actor.mlp.net.6.bias']:raise ValueError('unpaired CPU initialization')
        pairing.append({'seed':seed,'only_parameter_difference':diff})
    files={str(p.relative_to(a.output)):sha256(p) for p in a.output.rglob('*') if p.is_file()}
    plan={'schema':'standing_multiseed_v1','runs':jobs(),'evaluation_jobs':evaluation_jobs(),
          'training_transitions':TRAIN_TRANSITIONS,'evaluation_transitions':EVAL_TRANSITIONS,
          'evaluation_chained':False,'urdf':str(a.urdf.resolve()),'asset':str(a.asset.resolve()),
          'frozen_files':files,'cpu_paired_initialization':pairing,
          'evaluation':{'envs':4,'steps':500,'horizon_s':10.,'first_episode_only':True,
                        'transition_accounting':'all simulated environments including inactive episodes',
                        'noise_seed_by_training_seed':{str(s):10000+s for s in SEEDS}},
          'source_sha256':sha256(__file__)}
    _write_metrics(a.output/'plan.json',plan);validate_plan(a.output/'plan.json')
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('fixture','urdf','asset','output'):p.add_argument('--'+k,type=Path,required=True)
    prepare(p.parse_args())
