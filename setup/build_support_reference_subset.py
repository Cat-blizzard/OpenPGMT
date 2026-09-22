"""Opt-in support/anatomy v3 for the original eight windows; old files stay intact."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.bvh import load_bvh
from data.build_mesh_contacts import CONTACT_SOURCE, ReferenceMesh, sphere_geometry, contact_labels
from data.g1_kinematics import G1Kinematics
from data.retarget_lafan1 import retarget, bounded_joint_trajectory, G1_JOINT_NAMES, G1_JOINT_LIMITS, W
from data.ik_refine import FULL_KEYPOINTS
from data.support_retarget import source_support, anatomical_targets, refine_limbs, solve_support_height, contiguous_runs
from pgmt.envs.reset_geometry import CollisionFloor
from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry


def floor_min(model, floor, data):
    chunks = []
    for start in range(0, len(data['qpos']), 128):
        sl = slice(start, start+128)
        p, q = model.forward(data['qpos'][sl], data['root_pos'][sl], data['root_rot'][sl])
        bp = torch.tensor(np.stack([p[n] for n in model.order],1),dtype=torch.float32)
        bq = torch.tensor(np.stack([q[n] for n in model.order],1),dtype=torch.float32)
        chunks.append(floor.min_height(bp,bq).numpy())
    return np.concatenate(chunks)


def run(a):
    if a.evidence.exists() or (not a.audit_only and a.output.exists()): raise FileExistsError('fresh output required')
    manifest = load_manifest(a.manifest,a.v2,a.urdf,a.asset)
    a.evidence.mkdir(parents=True)
    if not a.audit_only: a.output.mkdir(parents=True)
    model=G1Kinematics(a.urdf);floor=CollisionFloor(a.urdf,model.order);geometry=SupportGeometry(a.urdf)
    mesh=ReferenceMesh();feet_geometry=sphere_geometry(a.urdf);rows=[]
    lo,hi=np.array([G1_JOINT_LIMITS[n] for n in G1_JOINT_NAMES]).T
    for clip in manifest['clips']:
        name=clip['clip_id'];source=clip['source_sequence'];start=clip['source_start_frame']
        print('START',name,'audit' if a.audit_only else 'full export',flush=True)
        b=load_bvh(a.bvh/(source+'.bvh'));analytic=retarget(b,kinematics=model);gp,gq=b.fk()
        plane,support,toes,speed=source_support(b,gp,float(analytic['scale']))
        # Use the independently estimated source foot plane; no robot minimum is used here.
        analytic['root_pos']=analytic['root_pos'].copy()
        analytic['root_pos'][:,2]=(gp[:,b.joint_index('Hips')]@W.T)[:,2]*float(analytic['scale'])-plane
        initial,targets=anatomical_targets(b,analytic,model,gp,gq)
        n=round(clip['horizon_s']/b.frame_time);count=round((clip['horizon_s']+.7)/b.frame_time)+1
        sl=slice(start,start+count)
        with np.load(a.v2/(name+'.npz')) as z:old={k:z[k].copy() for k in z.files}
        if a.audit_only:
            data={k:v[sl].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==b.num_frames else v for k,v in initial.items()}
            fit_targets=targets[sl];active=support[sl];score=slice(0,n)
        else:
            data=initial;fit_targets=targets;active=support;score=slice(start,start+n)
        q=refine_limbs(data,fit_targets,model)
        q,qvel=bounded_joint_trajectory(q,b.frame_time)
        if ((q<lo-1e-6)|(q>hi+1e-6)).any():raise ValueError('invalid v3 joint limits')
        if not np.array_equal(q[:,12:15],data['qpos'][:,12:15]):raise ValueError('IK changed bounded source waist')
        data={**data,'qpos':q,'qvel':qvel}
        minimum=floor_min(model,floor,data)
        g=geometry.compute(q,data['root_pos'],data['root_rot']);clearance=g['clearance'].min(-1)
        delta,fit=solve_support_height(clearance,minimum,active,b.frame_time)
        data['root_pos']=data['root_pos'].copy();data['root_pos'][:,2]+=delta
        after_floor=floor_min(model,floor,data)
        if after_floor.min() < -2e-5:raise ValueError('root trajectory penetrates URDF collision geometry')
        labels=[]
        for s in range(0,len(q),256):
            seg=slice(s,s+256);p,r=model.forward(q[seg],data['root_pos'][seg],data['root_rot'][seg])
            bp=torch.tensor(np.stack([p[k] for k in model.order],1),dtype=torch.float32)
            bq=torch.tensor(np.stack([r[k] for k in model.order],1),dtype=torch.float32)
            labels.append(contact_labels(bp,bq,model.order,feet_geometry,mesh).numpy())
        contacts=np.concatenate(labels)
        shifted=clearance+delta[:,None]
        if not np.array_equal(contacts,shifted<=.02):raise ValueError('independent mesh query differs')
        # Compare v2 and candidate against identical, new anatomical targets at identical root poses.
        old_positions,_=model.forward(old['qpos'][:n],initial['root_pos'][start:start+n],initial['root_rot'][start:start+n])
        new_positions,_=model.forward(q[score],initial['root_pos'][start:start+n],initial['root_rot'][start:start+n])
        errors=[float(np.linalg.norm(np.stack([p[body] for _,body in FULL_KEYPOINTS],1)-targets[start:start+n],axis=-1).mean())
                for p in (old_positions,new_positions)]
        score_support=active[score];score_clearance=shifted[score]
        row={'clip_id':name,'source_sequence':source,'source_start_frame':start,
            'source_plane_m':plane,'source_support_fraction':support[start:start+n].mean(0).tolist(),
            'source_bouts_intersecting_clip':[[l,h] for l,h in contiguous_runs(support.any(-1)) if l<start+count and h>start],
            'waist_limit_v2_v3':[float((abs(v[:,14]-.52)<1e-4).mean()) for v in (old['qpos'][:n],q[score])],
            'anatomical_target_error_v2_v3_m':errors,
            'max_joint_step_v2_v3_rad':[float(abs(np.diff(v,axis=0)).max()) for v in (old['qpos'][:n],q[score])],
            'max_joint_speed_v2_v3_rad_s':[float(abs(v).max()) for v in (old['qvel'][:n],qvel[score])],
            'foot_clearance_quantiles_m':np.quantile(score_clearance.min(-1),[0,.5,1]).tolist(),
            'mesh_contact_fraction':contacts[score].mean(0).tolist(),
            'source_support_above_2cm_fraction':float((score_clearance[score_support]>.02).mean()) if score_support.any() else None,
            'minimum_all_collision_height_m':float(after_floor.min()),
            'root_z_correction_range_m':[float(delta.min()),float(delta.max())],
            'max_root_step_m':float(np.linalg.norm(np.diff(data['root_pos'][score],axis=0),axis=-1).max()),
            'max_root_vertical_acceleration_m_s2':float(abs(np.diff(data['root_pos'][score,2],n=2)/b.frame_time**2).max()),
            'root_solve':fit,'old_npz_sha256':sha256(a.v2/(name+'.npz')),
            'source_bvh_sha256':sha256(a.bvh/(source+'.bvh'))}
        np.savez_compressed(a.evidence/(name+'_support.npz'),source_toe_height=toes[...,2],source_toe_speed=speed,
            source_support=support,root_delta=delta,collision_min_before=minimum,collision_min_after=after_floor,
            foot_clearance=shifted,anatomical_targets=fit_targets)
        if not a.audit_only:
            data.update(contacts=contacts,contact_source=np.array(CONTACT_SOURCE),reference_frame_contract=np.array('flat_ground_v1'),
                reference_ground_z=np.array(0.,np.float32),kinematic_urdf_sha256=np.array(model.sha256),
                retarget_contract=np.array('anatomical_support_v3'),source_ground_plane_m=np.array(plane),
                contact_provenance=np.array(json.dumps({'source':CONTACT_SOURCE,'mesh':mesh.provenance,'tolerance_m':.02,
                    'urdf_sha256':model.sha256,'root_height':'convex full-sequence support trajectory; no final rigid grounding',
                    'source_support_is_not_contact_labels':True})))
            np.savez_compressed(a.evidence/(source+'_full.npz'),**data)
            sub={k:v[sl].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==b.num_frames else v for k,v in data.items()}
            sub.update(source_sequence=np.array(source),source_start_frame=np.array(start),source_bvh_sha256=np.array(row['source_bvh_sha256']))
            np.savez_compressed(a.output/(name+'.npz'),**sub)
        rows.append(row);_write_metrics(a.evidence/'progress.json',{'status':'running','clips':rows})
        print('DONE',row,flush=True)
    summary={'status':'completed','scope':'scored-window feasibility audit' if a.audit_only else 'full-sequence v3 candidate',
        'clips':rows,'training_approved':False,'source_sha256':sha256(__file__),
        'algorithm_sha256':sha256('data/support_retarget.py'),'urdf_sha256':model.sha256,
        'note':'new offline retarget conventions, not changed RL rewards; no claim of physical feasibility'}
    _write_metrics(a.evidence/'summary.json',summary)
    if not a.audit_only:
        _write_metrics(a.output/'contact_manifest.json',{'source':CONTACT_SOURCE,'sequence_names':manifest['sequence_names'],
            'urdf_sha256':model.sha256,'reference_frame_contract':'flat_ground_v1','retarget_contract':'anatomical_support_v3'})
        _write_metrics(a.output/'quality_status.json',{'status':'diagnostic_only','training_approved':False,
            'reason':'source-support violations and temporal quality require review; no physical reference rollout was run'})
        _write_metrics(a.evidence/'manifest.json',{**manifest,'reference_files':{p.name:sha256(p) for p in sorted(a.output.glob('*.npz'))},
            'parent_manifest_sha256':sha256(a.manifest),'retarget_contract':'anatomical_support_v3',
            'selection':{'basis':'same original eight source windows, full sequence optimized before cropping',
                         'height_anchor':'source stationary-foot plane and constrained smooth root trajectory'},
            'source_script_sha256':sha256(__file__),'algorithm_sha256':sha256('data/support_retarget.py')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bvh','v2','manifest','urdf','asset','output','evidence'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true');run(p.parse_args())
