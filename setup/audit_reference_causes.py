"""Locate height anchors, waist saturation and IK jumps without rewriting data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.bvh import load_bvh,quat_to_mat
from data.g1_kinematics import G1Kinematics
from data.retarget_lafan1 import retarget,G1_JOINT_NAMES,M_RIG,W,_decompose_chain
from data.ik_refine import FULL_KEYPOINTS,FULL_WEIGHTS,_refine
from pgmt.envs.reset_geometry import CollisionFloor
from pgmt.envs.reference_motion import _qmat
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def collision_anchor(model,floor,data):
    best=None;frame_min=[]
    for start in range(0,len(data['qpos']),128):
        sl=slice(start,start+128);p,q=model.forward(data['qpos'][sl],data['root_pos'][sl],data['root_rot'][sl])
        bp=torch.tensor(np.stack([p[n] for n in model.order],1),dtype=torch.float32)
        bq=torch.tensor(np.stack([q[n] for n in model.order],1),dtype=torch.float32);R=_qmat(bq)
        z=bp[:,floor.indices,2]+(R[:,floor.indices,2]*floor.points).sum(-1)-floor.radii
        values,idx=z.min(-1);body=floor.indices[idx];point=idx.clone()
        for ci,(bi,center,axis,r,h) in enumerate(floor.cylinders):
            cz=bp[:,bi,2]+(R[:,bi,2]*center).sum(-1);cos=(R[:,bi,2]*axis).sum(-1).clamp(-1,1)
            cz-=h*cos.abs()+r*(1-cos.square()).clamp_min(0).sqrt()
            lower=cz<values;values=torch.where(lower,cz,values);body=torch.where(lower,bi,body);point=torch.where(lower,-ci-1,point)
        frame_min.extend(values.tolist());i=int(values.argmin())
        if best is None or float(values[i])<best['height_m']:
            best={'frame':start+i,'height_m':float(values[i]),'body':model.order[int(body[i])],
                  'geometry_point_index':int(point[i]),'root_height_m':float(data['root_pos'][start+i,2])}
    return best,np.array(frame_min)


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model=G1Kinematics(a.urdf);floor=CollisionFloor(a.urdf,model.order)
    manifest=json.loads(a.manifest.read_text());rows=[]
    for clip in manifest['clips']:
        name=clip['source_sequence'];b=load_bvh(a.bvh/(name+'.bvh'))
        print('START',clip['clip_id'],flush=True)
        with np.load(a.full/(name+'_full.npz')) as z:full={k:z[k].copy() for k in z.files}
        analytic=retarget(b,kinematics=model)
        start=clip['source_start_frame'];n=round(clip['horizon_s']/float(full['frame_time']));sl=slice(start,start+n)
        anchor,heights=collision_anchor(model,floor,full)
        gp,gq=b.fk(unit_scale=1.);gpos=gp@W.T*float(analytic['scale'])
        sf,sj=np.unravel_index(np.argmin(gpos[:,:,2]),gpos[:,:,2].shape)
        hips=b.joint_index('Hips');chest=b.joint_index('Spine2')
        raw=np.stack(_decompose_chain(M_RIG@(quat_to_mat(gq[:,hips]).transpose(0,2,1)@quat_to_mat(gq[:,chest]))@M_RIG.T,'waist'),-1)
        targets=np.stack([gpos[:,b.joint_index(src)] for src,_ in FULL_KEYPOINTS],1)-(gpos[:,hips]-analytic['root_pos'])[:,None]
        small={k:v[sl].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==b.num_frames else v for k,v in analytic.items()}
        refined=_refine(small,targets[sl],list(range(29)),[name for _,name in FULL_KEYPOINTS],FULL_WEIGHTS,max_iter=10,kinematics=model)
        mismatch=float(abs(refined-full['qpos'][sl]).max())
        if mismatch>2e-6:raise ValueError('saved IK cannot be reproduced')
        diff=np.diff(full['qpos'][sl],axis=0);local,j=map(int,np.unravel_index(abs(diff).argmax(),diff.shape));pair=start+local
        window=slice(max(0,pair-2),min(b.num_frames,pair+4));window_ids=list(range(window.start,window.stop))
        initial={k:v[window].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==b.num_frames else v for k,v in analytic.items()}
        probes={str(it):_refine(initial,targets[window],list(range(29)),[name for _,name in FULL_KEYPOINTS],FULL_WEIGHTS,
                             max_iter=it,kinematics=model) for it in (1,5,10,30)}
        pos_a,_=model.forward(analytic['qpos'][sl],analytic['root_pos'][sl],analytic['root_rot'][sl])
        # Root translation is irrelevant to position residuals if targets move with it.
        pos_i,_=model.forward(refined,analytic['root_pos'][sl],analytic['root_rot'][sl])
        body_errors=[float(np.linalg.norm(np.stack([p[name] for _,name in FULL_KEYPOINTS],1)-targets[sl],axis=-1).mean()) for p in (pos_a,pos_i)]
        at=lambda x:float((abs(x-.52)<1e-4).mean())
        row={'clip_id':clip['clip_id'],'source':name,'source_start_frame':start,'scored_frames':n,
             'source_floor_anchor':{'frame':int(sf),'body':b.names[sj],'height_before_shift_m':float(gpos[sf,sj,2])},
             'collision_anchor':anchor,'anchor_inside_scored_clip':start<=anchor['frame']<start+n,
             'full_sequence_additional_height_shift_m':float(full['full_sequence_height_shift_m']),
             'scored_minimum_all_collision_height_m':float(heights[sl].min()),
             'whole_sequence_clearance_quantiles_m':np.quantile(heights,[0,.01,.1,.5,.9,1]).tolist(),
             'waist':{'source_pitch_min_max_rad':[float(raw[sl,2].min()),float(raw[sl,2].max())],
                'source_above_limit_fraction':float((raw[sl,2]>.52).mean()),
                'analytic_limit_fraction':at(analytic['qpos'][sl,14]),'ik_limit_fraction':at(refined[:,14]),
                'analytic_to_ik_mean_pitch_delta_rad':float((refined[:,14]-analytic['qpos'][sl,14]).mean())},
             'keypoint_mean_error_analytic_ik_m':body_errors,'saved_ik_max_error_rad':mismatch,
             'peak_joint_step':{'joint':G1_JOINT_NAMES[j],'source_frames':[pair,pair+1],
                'saved_step_rad':float(diff[local,j]),'analytic_step_rad':float(analytic['qpos'][pair+1,j]-analytic['qpos'][pair,j]),
                'window_frames':window_ids,'analytic_values_rad':analytic['qpos'][window,j].tolist(),
                'ik_values_by_iteration':{k:v[:,j].tolist() for k,v in probes.items()},
                'source_max_local_rotation_change_rad':float((2*np.arccos(np.clip(abs((gq[pair+1]*gq[pair]).sum(-1)),0,1))).max())},
             'source_sha256':sha256(a.bvh/(name+'.bvh')),'full_npz_sha256':sha256(a.full/(name+'_full.npz'))}
        np.savez_compressed(a.output/(clip['clip_id']+'_stages.npz'),analytic_q=analytic['qpos'][sl],ik_q=refined,
            source_waist=raw[sl],source_targets=targets[sl],analytic_root=analytic['root_pos'][sl],
            root_rot=analytic['root_rot'][sl],full_collision_min=heights,
            **{'jump_window_ik_'+k:v for k,v in probes.items()})
        rows.append(row);_write_metrics(a.output/'progress.json',{'status':'running','clips':rows})
        print('DONE',row,flush=True)
    _write_metrics(a.output/'summary.json',{'status':'completed','clips':rows,'source_sha256':sha256(__file__),
        'urdf_sha256':model.sha256,'inputs_preserved':True,'ik_regularizer':'per-frame attraction to analytic pose, not temporal smoothing'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bvh','full','manifest','urdf','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
