"""CPU-only policy-head gradient diagnostics at preregistered updates 20/100.

No optimizer or simulator steps. Head advantages use the same per-head
standardization as PPO. Cosines describe these rollout states, not causation.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from pgmt.train.policy import Stage1Policy
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.standing_multiseed_protocol import jobs


def analyze(folder,update):
    before=folder/'captures'/f'update_{update:02d}_before.pt'
    b=torch.load(before,map_location='cpu',weights_only=False)
    after=torch.load(folder/'captures'/f'update_{update:02d}_after.pt',map_location='cpu',weights_only=False)
    s=b['storage'];p=Stage1Policy().eval();p.load_state_dict(b['policy_before'])
    obs={k:v.flatten(0,1) for k,v in s['observations'].items()}
    output=p.evaluate_actions(obs,s['actions'].flatten(0,1),latent_actions=s['latent_actions'].flatten(0,1))
    adv=s['advantages'].flatten(0,1);normed=(adv-adv.mean(0))/(adv.std(0,unbiased=False)+1e-8)
    torch.testing.assert_close(normed.mean(-1),s['policy_advantages'].flatten(),atol=2e-5,rtol=2e-5)
    params=[(n,v) for n,v in p.named_parameters() if not n.startswith('critic.')]
    grads=[]
    ratio=(output.log_probs-output.log_probs.detach()).exp()
    for head in range(3):
        loss=-(ratio*normed[:,head]).mean()
        g=torch.autograd.grad(loss,[v for _,v in params],retain_graph=head<2,allow_unused=True)
        grads.append({n:torch.zeros_like(v) if value is None else value for (n,v),value in zip(params,g)})
    groups={}
    for group,predicate in [('actor',lambda n:n.startswith('actor.') or n=='log_std'),
                            ('shared',lambda n:n.startswith(('ifm.','history_encoder.')))]:
        keys=[n for n,_ in params if predicate(n)]
        vec=[torch.cat([g[k].flatten() for k in keys]) for g in grads]
        cosine=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-12))
        delta=torch.cat([(after['policy_after'][k]-b['policy_before'][k]).flatten() for k in keys])
        base=torch.cat([b['policy_before'][k].flatten() for k in keys])
        groups[group]={'gradient_norm_upper_lower_aux':[float(v.norm()) for v in vec],
            'cosine_upper_lower':cosine(vec[0],vec[1]),'cosine_upper_aux':cosine(vec[0],vec[2]),
            'cosine_lower_aux':cosine(vec[1],vec[2]),'accepted_parameter_step_l2':float(delta.norm()),
            'relative_parameter_step_l2':float(delta.norm()/base.norm().clamp_min(1e-12))}
    return {'update':update,'groups':groups,'advantage_std_upper_lower_aux':adv.std(0,unbiased=False).tolist(),
            'accepted_actor_steps':after['metrics']['optimizer_steps'],'rollout_kl':after['metrics']['exact_kl'],
            'capture_sha256':sha256(before)}


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    rows=[];done=set()
    while True:
        ledger=json.loads((a.source/'execution_ledger.json').read_text())
        for job in jobs():
            row=next((r for r in ledger['runs'] if r['name']==job['name']),None)
            if job['name'] in done or row is None or row.get('status')!='completed':continue
            rows.append({**job,'snapshots':[analyze(a.source/job['name'],u) for u in (20,100)]})
            done.add(job['name']);print('CPU gradient audit',job['name'],flush=True)
            _write_metrics(a.output,{'status':'running','runs':rows,'gpu_used':False,'optimizer_steps':0})
        if ledger['status']!='running' or not a.watch:break
        time.sleep(10)
    _write_metrics(a.output,{'status':'completed' if len(rows)==6 else 'incomplete','runs':rows,'gpu_used':False,
        'optimizer_steps':0,'scope':'behavior-policy gradient geometry on training rollouts, no causal or policy-evaluation claim'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--watch',action='store_true');run(p.parse_args())
