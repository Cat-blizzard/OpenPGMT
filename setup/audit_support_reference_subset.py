"""Independent runtime/source-support checks for v3, including 50 Hz interpolation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.g1_kinematics import G1Kinematics
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.reference_motion import ReferenceMotion
from pgmt.envs.reset_geometry import CollisionFloor
from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry
from setup.build_support_reference_subset import floor_min


def run(a):
    if a.output.exists(): raise FileExistsError(a.output)
    manifest=load_manifest(a.manifest,a.reference,a.urdf,a.asset)
    model=G1Kinematics(a.urdf);floor=CollisionFloor(a.urdf,model.order);geometry=SupportGeometry(a.urdf);rows=[]
    for clip in manifest['clips']:
        name=clip['clip_id'];n=150;start=clip['source_start_frame']
        with np.load(a.reference/(name+'.npz')) as z:seq={k:z[k].copy() for k in z.files}
        with np.load(a.v2/(name+'.npz')) as z:old={k:z[k].copy() for k in z.files}
        with np.load(a.evidence/(name+'_support.npz')) as z:
            support=z['source_support'][start:start+n]
            clear=z['foot_clearance'][start:start+n]
            all_min=z['collision_min_after'][start:start+n]
        active=support.any(-1)
        # Any root-only translation preserves this bound conflict.
        highest_active=np.where(support,clear,-np.inf).max(-1)
        infeasible=active & (highest_active-all_min>.02)
        row={'clip_id':name,'runtime':{},'root_only_infeasible_support_frame_fraction':float(infeasible[active].mean()),
             'root_vertical_acceleration_v2_v3_m_s2':[float(abs(np.diff(s['root_pos'][:n,2],n=2)/(float(s['frame_time'])**2)).max()) for s in (old,seq)],
             'root_step_v2_v3_m':[float(np.linalg.norm(np.diff(s['root_pos'][:n],axis=0),axis=-1).max()) for s in (old,seq)],
             'max_qvel_difference_error_rad_s':float(abs(seq['qvel'][1:]-np.diff(seq['qpos'],axis=0)/float(seq['frame_time'])).max()),
             'max_root_rotation_step_rad':float((2*np.arccos(np.clip(abs((seq['root_rot'][1:n]*seq['root_rot'][:n-1]).sum(-1)),0,1))).max()),
             'foot_clearance_per_side_min_median_max_m':[np.quantile(clear[:,side],[0,.5,1]).tolist() for side in (0,1)]}
        db=MotionDatabase.from_sequences([seq])
        for mode in ('nominal','reference_state'):
            env=G1Env(G1EnvConfig(num_envs=n,reference_urdf_path=str(a.urdf),reset_mode=mode,
                default_root_pos=(2.,3.,1.2),enable_adaptive_sampling=False),reference_database=db)
            env._forced_reference.update({i:(0,i) for i in range(n)});env.reset(seed=0)
            ref=env.reference_body;g=geometry.compute(ref['joint_pos'].numpy(),ref['root_pos'].numpy(),ref['root_quat'].numpy())
            contact=(g['clearance']<=.02).any(-1);mismatch=float((contact!=seq['contacts'][:n]).mean())
            zerr=float(abs(ref['root_pos'][:,2].numpy()-seq['root_pos'][:n,2]).max())
            if mismatch or zerr>2e-6:raise ValueError('source-frame placement/contact mismatch')
            row['runtime'][mode]={'contact_mismatch_fraction':mismatch,'root_height_error_m':zerr,
                                 'reset_lift_max_m':float(env._reset_height_lift.max())}
        motion=ReferenceMotion(db,a.urdf)
        frames=torch.arange(250)*.02/float(seq['frame_time']);ref=motion.sample(torch.zeros(250,dtype=torch.long),frames)
        interp={'qpos':ref['joint_pos'].numpy(),'root_pos':ref['root_pos'].numpy(),'root_rot':ref['root_quat'].numpy()}
        contact=(geometry.compute(interp['qpos'],interp['root_pos'],interp['root_rot'])['clearance']<=.02).any(-1)
        minima=floor_min(model,floor,interp)
        row['interpolated_50hz']={'minimum_collision_height_m':float(minima.min()),
            'penetrating_more_than_0_1mm_fraction':float((minima<-.0001).mean()),
            'mesh_vs_interpolated_label_mismatch_fraction':float((contact!=ref['foot_contact'].numpy()).mean()),
            'scope':'continuous pose interpolation and majority-interpolated binary labels; no automatic repair'}
        rows.append(row);print(name,row,flush=True)
    _write_metrics(a.output,{'status':'completed','clips':rows,'training_approved':False,
        'source_sha256':sha256(__file__),'manifest_sha256':sha256(a.manifest),
        'scope':'CPU runtime geometry, not dynamic reference tracking'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','v2','manifest','evidence','urdf','asset','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
