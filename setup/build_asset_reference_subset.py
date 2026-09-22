"""Regenerate the frozen eight clips with explicit URDF/full-sequence grounding."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.bvh import load_bvh
from data.g1_kinematics import G1Kinematics
from data.ik_refine import refine_full
from data.retarget_lafan1 import retarget, G1_JOINT_NAMES, G1_JOINT_LIMITS
from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry


def run(a):
    if a.output.exists() or a.evidence.exists():
        raise FileExistsError('new data and evidence directories required')
    iterations=getattr(a,'ik_iterations',10)
    if iterations<=0:raise ValueError('IK iterations must be positive')
    old = load_manifest(a.manifest, a.old_reference, a.urdf, a.asset)
    a.output.mkdir(parents=True);a.evidence.mkdir(parents=True)
    model, geometry = G1Kinematics(a.urdf), SupportGeometry(a.urdf)
    rows, clips = [], []
    lo, hi = np.array([G1_JOINT_LIMITS[n] for n in G1_JOINT_NAMES]).T
    for clip in old['clips']:
        name = clip['sequence'];bvh = load_bvh(a.bvh/(name+'.bvh'))
        print('START', clip['clip_id'], 'full frames', bvh.num_frames, flush=True)
        data = refine_full(retarget(bvh, kinematics=model), bvh, max_iter=iterations, kinematics=model)
        if ((data['qpos'] < lo-1e-6) | (data['qpos'] > hi+1e-6)).any():
            raise ValueError('new IK exceeds legal targets')
        np.savez_compressed(a.evidence/(name+'_full.npz'), **data)
        start = clip['start_frame'];count = round((clip['horizon_s']+.7)/bvh.frame_time)+1
        if start+count >= bvh.num_frames:
            raise ValueError('insufficient original future padding')
        sub = {k: v[start:start+count].copy() if isinstance(v, np.ndarray) and v.ndim and len(v)==bvh.num_frames
               else v for k, v in data.items()}
        sub.update(source_sequence=np.array(name), source_start_frame=np.array(start),
                   source_bvh_sha256=np.array(sha256(a.bvh/(name+'.bvh'))))
        np.savez_compressed(a.output/(clip['clip_id']+'.npz'), **sub)
        n = round(clip['horizon_s']/bvh.frame_time)
        with np.load(a.old_reference/(name+'.npz')) as z:
            prior = {k:z[k][start:start+n].copy() for k in ('qpos','qvel','root_pos','root_rot','contacts')}
        g = geometry.compute(sub['qpos'][:n], sub['root_pos'][:n], sub['root_rot'][:n])
        recomputed = (g['clearance'] <= .02).any(-1)
        if not np.array_equal(recomputed, sub['contacts'][:n]):
            raise ValueError('saved and independently queried contacts differ')
        if g['clearance'].min() < -2e-5:
            raise ValueError('new reference foot penetrates source ground')
        row = {'clip_id':clip['clip_id'], 'source_sequence':name, 'source_start_frame':start,
            'full_frames':bvh.num_frames,'exported_frames':count,'scored_frames':n,'ik_iterations':iterations,
            'full_sequence_height_shift_m':float(data['full_sequence_height_shift_m']),
            'q_delta_rms_rad':float(np.sqrt(((sub['qpos'][:n]-prior['qpos'])**2).mean())),
            'waist_limit_old_new':[float((abs(q[:,14]-.52)<1e-4).mean()) for q in (prior['qpos'],sub['qpos'][:n])],
            'max_frame_joint_step_old_new_rad':[float(abs(np.diff(q,axis=0)).max()) for q in (prior['qpos'],sub['qpos'][:n])],
            'max_joint_speed_old_new_rad_s':[float(abs(v).max()) for v in (prior['qvel'],sub['qvel'][:n])],
            'foot_min_clearance_m':float(g['clearance'].min()),
            'contacts_old_new':[prior['contacts'].mean(0).tolist(),sub['contacts'][:n].mean(0).tolist()],
            'offline_contact_requery_passed':True,'all_joint_limits_passed':True,
            'full_source_sha256':sha256(a.evidence/(name+'_full.npz'))}
        rows.append(row)
        clips.append(dict(clip, sequence=clip['clip_id'], start_frame=0,
                          source_sequence=name, source_start_frame=start))
        _write_metrics(a.evidence/'progress.json', {'status':'running', 'clips':rows})
        print('DONE', row, flush=True)
    provenance={'source':'offline_reference_mesh_v2','sequence_names':sorted(c['sequence'] for c in clips),
                'urdf_sha256':model.sha256,'reference_frame_contract':'flat_ground_v1'}
    _write_metrics(a.output/'contact_manifest.json',provenance)
    manifest={**old,'clips':clips,'sequence_names':provenance['sequence_names'],
        'reference_files':{p.name:sha256(p) for p in sorted(a.output.glob('*.npz'))},
        'parent_manifest_sha256':sha256(a.manifest),'selection':{
            'basis':'identical source sequences/start frames to original frozen eight clips',
            'height_anchor':'whole original sequence, all URDF collision geometry; crop after finalization'},
        'source_script_sha256':sha256(__file__),'ik_iterations':iterations}
    _write_metrics(a.evidence/'manifest.json',manifest)
    _write_metrics(a.evidence/'summary.json',{'status':'completed','clips':rows,
        'legacy_reference_preserved':True,'data_path':str(a.output.resolve()),'urdf_sha256':model.sha256})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bvh','old-reference','manifest','urdf','asset','output','evidence'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ik-iterations',type=int,default=10)
    run(p.parse_args())
