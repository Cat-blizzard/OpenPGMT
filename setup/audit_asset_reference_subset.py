"""CPU validation of frozen asset-bound clips in both runtime reset modes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from data.bvh import load_bvh
from data.g1_kinematics import G1Kinematics
from data.ik_refine import FULL_KEYPOINTS
from data.retarget_lafan1 import W
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    manifest=load_manifest(a.manifest,a.reference,a.urdf,a.asset)
    old=load_manifest(a.old_manifest,a.old_reference,a.urdf,a.asset)
    model=G1Kinematics(a.urdf);geometry=SupportGeometry(a.urdf)
    rows=[]
    for clip in manifest['clips']:
        with np.load(a.reference/(clip['sequence']+'.npz')) as z:
            seq={k:z[k].copy() for k in z.files}
        start=clip['source_start_frame'];n=round(clip['horizon_s']/float(seq['frame_time']))
        with np.load(a.old_reference/(clip['source_sequence']+'.npz')) as z:
            prior={k:z[k].copy() for k in z.files}
        b=load_bvh(a.bvh/(clip['source_sequence']+'.bvh'))
        small=replace(b,root_pos=b.root_pos[start:start+n],rotations=b.rotations[start:start+n],local_trans=b.local_trans[start:start+n])
        gp,_=small.fk(unit_scale=1.0);relative=(gp-gp[:,small.joint_index('Hips'),None])@W.T
        errors=[]
        for data,sl in ((prior,slice(start,start+n)),(seq,slice(0,n))):
            bp,_=model.forward(data['qpos'][sl],data['root_pos'][sl],data['root_rot'][sl])
            target=np.stack([relative[:,small.joint_index(src)] for src,_ in FULL_KEYPOINTS],1)*float(data['scale'])
            actual=np.stack([bp[name]-data['root_pos'][sl] for _,name in FULL_KEYPOINTS],1)
            errors.append(float(np.linalg.norm(actual-target,axis=-1).mean()))
        row={'clip_id':clip['clip_id'],'body_point_mean_error_old_new_m':errors,
             'max_root_translation_step_m':float(np.linalg.norm(np.diff(seq['root_pos'][:n],axis=0),axis=-1).max()),
             'max_root_rotation_step_rad':float((2*np.arccos(np.clip(abs((seq['root_rot'][1:n]*seq['root_rot'][:n-1]).sum(-1)),0,1))).max()),
             'exported_qvel_difference_error_rad_s':float(abs(seq['qvel'][1:] - np.diff(seq['qpos'],axis=0)/float(seq['frame_time'])).max()),
             'runtime':{}}
        db=MotionDatabase.from_sequences([seq])
        for mode in ('nominal','reference_state'):
            env=G1Env(G1EnvConfig(num_envs=n,reference_urdf_path=str(a.urdf),reset_mode=mode,
                default_root_pos=(2.,3.,1.2),enable_adaptive_sampling=False),reference_database=db)
            env._forced_reference.update({i:(0,i) for i in range(n)})
            env.reset(seed=0)
            ref=env.reference_body
            g=geometry.compute(ref['joint_pos'].numpy(),ref['root_pos'].numpy(),ref['root_quat'].numpy())
            contact=(g['clearance']<=.02).any(-1)
            mismatch=float((contact!=seq['contacts'][:n]).mean())
            z_error=float(abs(ref['root_pos'][:,2].numpy()-seq['root_pos'][:n,2]).max())
            if mismatch or z_error>2e-6:raise ValueError('source/runtime ground or contact mismatch')
            row['runtime'][mode]={'contact_mismatch_fraction':mismatch,'root_height_error_m':z_error,
                'minimum_foot_clearance_m':float(g['clearance'].min()),'maximum_reset_lift_m':float(env._reset_height_lift.max())}
        rows.append(row)
        print(clip['clip_id'],row,flush=True)
    _write_metrics(a.output,{'status':'completed','clips':rows,'source_manifest_sha256':sha256(a.manifest),
        'old_reference_hashes_still_match':True,'old_manifest_sha256':sha256(a.old_manifest),
        'source_script_sha256':sha256(__file__),'training_approved':False,
        'reason':'Geometry/placement consistency does not remove floating feet, source morphology mismatch or IK continuity regressions.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','manifest','old-reference','old-manifest','bvh','urdf','asset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
