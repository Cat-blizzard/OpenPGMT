"""Compare the same eight source clips and test height/waist explanations."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.bvh import load_bvh
from data.g1_kinematics import G1Kinematics
from data.ik_refine import FULL_KEYPOINTS,FULL_WEIGHTS
from data.retarget_lafan1 import G1_JOINT_NAMES,W
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    causes=json.loads((a.causes/'summary.json').read_text());model=G1Kinematics(a.urdf);rows=[]
    for cause in causes['clips']:
        name=cause['clip_id'];n=cause['scored_frames'];seqs=[]
        for directory in (a.v1,a.v2):
            with np.load(directory/(name+'.npz')) as z:seqs.append({k:z[k].copy() for k in z.files})
        with np.load(a.causes/(name+'_stages.npz')) as z:stages={k:z[k].copy() for k in z.files}
        b=load_bvh(a.bvh/(cause['source']+'.bvh'));gp,_=b.fk(unit_scale=1.)
        gp=gp@W.T*float(seqs[0]['scale']);gp[:,:,2]-=gp[:,:,2].min()
        start=cause['source_start_frame'];sl=slice(start,start+n)
        toe=gp[sl][:,[b.joint_index('LeftToe'),b.joint_index('RightToe')],2]
        def cost(q):
            p,_=model.forward(q,stages['analytic_root'],stages['root_rot'])
            error=np.stack([p[name] for _,name in FULL_KEYPOINTS],1)-stages['source_targets']
            return (error*FULL_WEIGHTS[None,:,None]).reshape(n,-1).__pow__(2).sum(-1)+.05*((q-stages['analytic_q'])**2).sum(-1)
        old_q,new_q=[s['qpos'][:n].astype(np.float64) for s in seqs]
        c1,c2=cost(old_q),cost(new_q)
        inward=old_q.copy();inward[:,14]-=.001;dc=cost(inward)-c1;bound=abs(old_q[:,14]-.52)<1e-4
        velocity=[float(abs(s['qvel'][:n]).max()) for s in seqs]
        delta=[float(abs(np.diff(q,axis=0)).max()) for q in (old_q,new_q)]
        pair=cause['peak_joint_step']['source_frames'];local=pair[0]-start;j=G1_JOINT_NAMES.index(cause['peak_joint_step']['joint'])
        # Existing full-sequence collision minimum is already ~0. A global
        # downward shift sufficient for contact in this clip would penetrate
        # another frame in the same frozen trajectory.
        minimum=float(stages['full_collision_min'].min());clip_floor=cause['scored_minimum_all_collision_height_m']
        lower=-minimum;upper=.02-clip_floor
        row={'clip_id':name,'max_joint_step_v1_v2_rad':delta,'max_joint_speed_v1_v2_rad_s':velocity,
            'original_peak_pair':{'joint':G1_JOINT_NAMES[j],'source_frames':pair,
                'step_v1_v2_rad':[float(q[local+1,j]-q[local,j]) for q in (old_q,new_q)]},
            'ik_objective_mean_v1_v2':[float(c1.mean()),float(c2.mean())],
            'ik_objective_max_increase_v2':float((c2-c1).max()),
            'waist_bound_inward_cost_increase_fraction':float((dc[bound]>0).mean()) if bound.any() else None,
            'waist_bound_inward_cost_delta_range':([float(dc[bound].min()),float(dc[bound].max())] if bound.any() else None),
            'source_toe_height_proxy_m':{'min':toe.min(0).tolist(),'median':np.median(toe,axis=0).tolist(),
                'either_toe_below_2cm_fraction':float((toe.min(-1)<=.02).mean()),
                'scope':'source joint-point geometry proxy; not replacement mesh contact labels'},
            'constant_shift_feasibility':{'nonpenetration_lower_bound_m':lower,'contact_upper_bound_m':upper,
                'feasible':lower<=upper,'scope':'unchanged full-sequence trajectory and one flat source plane'},
            'sha256_v1_v2':[sha256(d/(name+'.npz')) for d in (a.v1,a.v2)]}
        if (c2-c1).max()>2e-6:raise ValueError('longer IK increased its declared optimization objective')
        rows.append(row);print(name,row,flush=True)
    _write_metrics(a.output,{'status':'completed','clips':rows,'v2_change':'30 LM iterations instead of 10; same objective, limits, source clips and grounding rule',
        'training_approved':False,'reason':'Height/support and waist objective mismatch remain; v2 only addresses demonstrated under-iteration.',
        'source_sha256':sha256(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('causes','v1','v2','bvh','urdf','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
