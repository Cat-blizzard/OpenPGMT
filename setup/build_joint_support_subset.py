"""CPU-only v4 candidate on the original eight windows, with context and 50 Hz export."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.build_mesh_contacts import CONTACT_SOURCE, ReferenceMesh, sphere_geometry, contact_labels
from data.g1_kinematics import G1Kinematics
from data.joint_support_retarget import fit_joint_support
from data.support_retarget import solve_support_height
from data.retarget_lafan1 import bounded_joint_trajectory, G1_JOINT_NAMES, G1_JOINT_LIMITS
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.reset_geometry import CollisionFloor
from pgmt.train.stage1 import TorchMotionDatabase
from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry
from setup.build_support_reference_subset import floor_min


def source_audit(source, start, count, plane):
    h=source['source_toe_height'];v=source['source_toe_speed'];mask=source['source_support']
    sl=slice(start,start+count);both=mask[sl].all(-1);diff=abs(h[sl,0]-h[sl,1])
    stationary=v[sl]<=.15
    return {'plane_m':float(plane),'scored_frames':count,'support_samples':int(mask[sl].sum()),
        'double_support_frames':int(both.sum()),
        'double_support_height_difference_p50_p95_max_m':np.quantile(diff[both],[.5,.95,1]).tolist() if both.any() else None,
        'double_support_difference_over_1cm_fraction':float((diff[both]>.01).mean()) if both.any() else None,
        'stationary_but_outside_source_plane_band_samples':int((stationary & (abs(h[sl]-plane)>.02)).sum()),
        'support_within_1cm_of_plane_fraction':float((abs(h[sl]-plane)[mask[sl]]<=.01).mean()),
        'support_near_speed_threshold_fraction':float((v[sl][mask[sl]]>.12).mean()),
        'interpretation':'toe-pivot proxy, not measured source contact; original mask retained without relabeling ambiguous frames'}


def run(a):
    if a.output.exists() or a.evidence.exists():raise FileExistsError('fresh v4 output required')
    manifest=load_manifest(a.manifest,a.v3,a.urdf,a.asset)
    a.output.mkdir(parents=True);a.evidence.mkdir(parents=True)
    model=G1Kinematics(a.urdf);floor=CollisionFloor(a.urdf,model.order);geom=SupportGeometry(a.urdf)
    mesh=ReferenceMesh();footgeom=sphere_geometry(a.urdf);rows=[];audits=[]
    # Complete the source-label review for all windows before any optimization.
    for clip in manifest['clips']:
        with np.load(a.v3/(clip['clip_id']+'.npz')) as z:plane=float(z['source_ground_plane_m']);dt=float(z['frame_time'])
        with np.load(a.v3_evidence/(clip['clip_id']+'_support.npz')) as z:
            audit=source_audit(z,clip['source_start_frame'],round(clip['horizon_s']/dt),plane)
        audits.append({'clip_id':clip['clip_id'],**audit})
    _write_metrics(a.evidence/'source_support_audit.json',{'status':'completed','clips':audits})
    for clip in manifest['clips']:
        name=clip['clip_id'];source=clip['source_sequence'];start=clip['source_start_frame']
        print('START',name,flush=True)
        with np.load(a.v3/(name+'.npz')) as z:source_bvh_sha=str(z['source_bvh_sha256'])
        with np.load(a.v3_evidence/(source+'_full.npz')) as z:full={k:z[k].copy() for k in z.files}
        with np.load(a.v3_evidence/(name+'_support.npz')) as z:source_support=z['source_support'].copy()
        dt=float(full['frame_time']);n=round(clip['horizon_s']/.02);count=round((clip['horizon_s']+.7)/.02)+1
        # Half a second of real source context on each side; never optimize an
        # isolated scoring frame or claim that the entire source was refitted.
        before=min(25,int(start*dt/.02));after=min(25,int(((len(full['qpos'])-1-start)*dt)/.02)-(count-1))
        if after<0:raise ValueError('insufficient source padding')
        frames=start+torch.arange(-before,count+after,dtype=torch.float32)*(.02/dt)
        db=TorchMotionDatabase(MotionDatabase.from_sequences([full]))
        q,_,rp,rr=db.ref_at(torch.zeros(len(frames),dtype=torch.long),frames)
        base={'qpos':q.numpy(),'root_pos':rp.numpy(),'root_rot':rr.numpy(),'frame_time':np.array(.02,np.float32)}
        active=source_support[frames.round().long().numpy()];score=slice(before,before+n);export=slice(before,before+count)
        qfit,rfit,fit=fit_joint_support(base,active,model)
        if not np.array_equal(qfit[:,12:],base['qpos'][:,12:]):raise ValueError('upper-body/waist changed')
        fitted={**base,'qpos':qfit,'root_pos':rfit}
        clear=geom.compute(qfit,rfit,base['root_rot'])['clearance'].min(-1)
        minima=floor_min(model,floor,fitted)
        # Whole-body nonpenetration and root temporal regularity use the v3
        # constrained height solve AFTER the joint fit, not a per-frame clamp.
        delta,height_fit=solve_support_height(clear,minima,active,.02)
        fitted['root_pos']=rfit.copy();fitted['root_pos'][:,2]+=delta
        minimum=floor_min(model,floor,fitted)
        uniform_margin=max(0.,.0005-float(minimum.min()))
        fitted['root_pos'][:,2]+=uniform_margin
        qfit,qvel=bounded_joint_trajectory(qfit,.02)
        fitted.update(qpos=qfit,qvel=qvel)
        minimum=floor_min(model,floor,fitted)
        if minimum.min()<-.00001:raise ValueError('v4 collision geometry penetrates the floor')
        p,r=model.forward(qfit,fitted['root_pos'],fitted['root_rot'])
        bp=torch.tensor(np.stack([p[k] for k in model.order],1),dtype=torch.float32)
        bq=torch.tensor(np.stack([r[k] for k in model.order],1),dtype=torch.float32)
        contacts=contact_labels(bp,bq,model.order,footgeom,mesh).numpy()
        clears=[geom.compute(x['qpos'],x['root_pos'],x['root_rot'])['clearance'].min(-1) for x in (base,fitted)]
        if not np.array_equal(contacts,clears[1]<=.02):raise ValueError('independent mesh labels disagree')
        pbase,_=model.forward(base['qpos'],base['root_pos'],base['root_rot'])
        displacement=np.linalg.norm(np.stack([p[k]-pbase[k] for k in model.order],1),axis=-1)
        aq=active[score];lo,hi=np.array([G1_JOINT_LIMITS[k] for k in G1_JOINT_NAMES]).T
        row={'clip_id':name,'source_start_frame':start,'frame_time':.02,'context_frames_before_after':[before,after],
            'source_support_above_2cm_v3_v4':[float((cl[score][aq]>.02).mean()) for cl in clears],
            'source_support_clearance_p50_p95_max_v3_v4_m':[np.quantile(cl[score][aq],[.5,.95,1]).tolist() for cl in clears],
            'mesh_contact_fraction_v3_v4':[(cl[score]<=.02).mean(0).tolist() for cl in clears],
            'swing_clearance_p50_p95_v3_v4_m':[np.quantile(cl[score][~aq],[.5,.95]).tolist() if (~aq).any() else None for cl in clears],
            'max_joint_speed_v3_v4_rad_s':[float(abs(np.diff(x['qpos'][score],axis=0)/.02).max()) for x in (base,fitted)],
            'max_joint_acceleration_v3_v4_rad_s2':[float(abs(np.diff(x['qpos'][score],n=2,axis=0)/.02**2).max()) for x in (base,fitted)],
            'max_root_acceleration_v3_v4_m_s2':[float(np.linalg.norm(np.diff(x['root_pos'][score],n=2,axis=0)/.02**2,axis=-1).max()) for x in (base,fitted)],
            'root_z_change_p50_max_m':np.quantile(abs(fitted['root_pos'][score,2]-base['root_pos'][score,2]),[.5,1]).tolist(),
            'body_displacement_from_v3_mean_max_m':[float(displacement[score].mean()),float(displacement[score].max())],
            'waist_at_upper_limit_v3_v4':[float((abs(x['qpos'][score,14]-.52)<1e-4).mean()) for x in (base,fitted)],
            'joint_limit_violations':int(((qfit<lo-1e-6)|(qfit>hi+1e-6)).sum()),
            'minimum_all_collision_height_m':float(minimum.min()),'uniform_margin_added_m':uniform_margin,
            'joint_fit':fit,'height_fit':height_fit,'v3_full_sha256':sha256(a.v3_evidence/(source+'_full.npz'))}
        # Preserve nontrajectory metadata; write the new sampling interval and
        # fresh derivatives/contacts explicitly. Old NPZ files stay untouched.
        data={k:v.copy() for k,v in full.items() if not (v.ndim and len(v)==len(full['qpos']))}
        data.update({k:v[export].astype(np.float32) for k,v in fitted.items() if isinstance(v,np.ndarray) and v.ndim})
        data.update(contacts=contacts[export],frame_time=np.array(.02,np.float32),source_sequence=np.array(source),
            source_start_frame=np.array(start),source_frame_time=np.array(dt),source_bvh_sha256=np.array(source_bvh_sha),
            retarget_contract=np.array('anatomical_joint_support_v4_50hz'),contact_source=np.array(CONTACT_SOURCE),
            contact_provenance=np.array(json.dumps({'source':CONTACT_SOURCE,'mesh':mesh.provenance,'tolerance_m':.02,
                'urdf_sha256':model.sha256,'source_support_is_not_contact_labels':True,'native_query_hz':50})))
        # Derivative at the cropped first frame retains its actual context.
        np.savez_compressed(a.output/(name+'.npz'),**data)
        np.savez_compressed(a.evidence/(name+'_evidence.npz'),source_support=active[export],
            v3_qpos=base['qpos'][export],v3_root_pos=base['root_pos'][export],v3_root_rot=base['root_rot'][export],
            v3_clearance=clears[0][export],v4_clearance=clears[1][export],source_query_frames=frames.numpy()[export])
        rows.append(row);_write_metrics(a.evidence/'progress.json',{'status':'running','clips':rows});print('DONE',name,row,flush=True)
    _write_metrics(a.evidence/'summary.json',{'status':'completed','training_approved':False,'clips':rows,
        'scope':'joint root/leg correction of original eight windows with real context; CPU geometric candidate',
        'source_sha256':sha256(__file__),'algorithm_sha256':sha256('data/joint_support_retarget.py')})
    _write_metrics(a.output/'quality_status.json',{'status':'diagnostic_only','training_approved':False,
        'reason':'requires independent temporal/runtime checks and physical tracking; source contact remains a proxy'})
    _write_metrics(a.output/'contact_manifest.json',{'source':CONTACT_SOURCE,'sequence_names':manifest['sequence_names'],
        'urdf_sha256':model.sha256,'reference_frame_contract':'flat_ground_v1','retarget_contract':'anatomical_joint_support_v4_50hz'})
    _write_metrics(a.evidence/'manifest.json',{**manifest,'reference_files':{p.name:sha256(p) for p in sorted(a.output.glob('*.npz'))},
        'parent_manifest_sha256':sha256(a.manifest),'retarget_contract':'anatomical_joint_support_v4_50hz',
        'selection':{'basis':'same eight source time windows; 0.5 s context where available; 50 Hz export'},
        'source_script_sha256':sha256(__file__),'algorithm_sha256':sha256('data/joint_support_retarget.py')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v3','v3-evidence','manifest','urdf','asset','output','evidence'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
