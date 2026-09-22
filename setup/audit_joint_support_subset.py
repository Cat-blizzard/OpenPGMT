"""Independent CPU runtime, temporal and common-target checks of the v4 candidate."""
import argparse
from pathlib import Path
import numpy as np
import torch

from data.g1_kinematics import G1Kinematics
from data.ik_refine import FULL_KEYPOINTS
from pgmt.envs.g1_env import G1Env,G1EnvConfig
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.reference_motion import ReferenceMotion
from pgmt.envs.reset_geometry import CollisionFloor
from pgmt.train.fixed_clip_diagnostic import load_manifest,sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry
from setup.build_support_reference_subset import floor_min


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    manifest=load_manifest(a.manifest,a.reference,a.urdf,a.asset)
    model=G1Kinematics(a.urdf);floor=CollisionFloor(a.urdf,model.order);geometry=SupportGeometry(a.urdf);rows=[]
    for clip in manifest['clips']:
        name=clip['clip_id'];n=round(clip['horizon_s']/.02)
        with np.load(a.reference/(name+'.npz')) as z:seq={k:z[k].copy() for k in z.files}
        with np.load(a.evidence/(name+'_evidence.npz')) as z:e={k:z[k].copy() for k in z.files}
        with np.load(a.v3_evidence/(name+'_support.npz')) as z:targets=z['anatomical_targets'].copy()
        times=e['source_query_frames'][:n];idx=np.floor(times).astype(int);weight=(times-idx)[:,None,None]
        targets=targets[idx]*(1-weight)+targets[idx+1]*weight
        target_relative=targets-targets[:,:1]
        comparable=[];directions=[]
        for q,rp,rr in ((e['v3_qpos'][:n],e['v3_root_pos'][:n],e['v3_root_rot'][:n]),
                        (seq['qpos'][:n],seq['root_pos'][:n],seq['root_rot'][:n])):
            p,_=model.forward(q,rp,rr);points=np.stack([p[b] for _,b in FULL_KEYPOINTS],1)
            comparable.append(float(np.linalg.norm((points-points[:,:1])-target_relative,axis=-1).mean()))
            angles=[]
            for knee,ankle in ((9,10),(11,12)):
                actual=points[:,ankle]-points[:,knee];desired=targets[:,ankle]-targets[:,knee]
                cosine=(actual*desired).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(desired,axis=-1))
                angles.append(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
            directions.append(float(np.stack(angles).mean()))
        db=MotionDatabase.from_sequences([seq]);motion=ReferenceMotion(db,a.urdf)
        row={'clip_id':name,'runtime':{},'same_anatomical_target_torso_relative_error_v3_v4_m':comparable,
             'same_anatomical_shank_direction_error_v3_v4_deg':directions,
             'qvel_export_difference_error_rad_s':float(abs(seq['qvel'][1:]-np.diff(seq['qpos'].astype(float),axis=0)/.02).max()),
             'maximum_stored_joint_step_rad':float(abs(np.diff(seq['qpos'],axis=0)).max())}
        for mode in ('nominal','reference_state'):
            env=G1Env(G1EnvConfig(num_envs=n,reference_urdf_path=str(a.urdf),reset_mode=mode,
                default_root_pos=(2.,3.,1.2),enable_adaptive_sampling=False),reference_database=db)
            env._forced_reference.update({i:(0,i) for i in range(n)});env.reset(seed=0)
            ref=env.reference_body;g=geometry.compute(ref['joint_pos'].numpy(),ref['root_pos'].numpy(),ref['root_quat'].numpy())
            mismatch=float(((g['clearance']<=.02).any(-1)!=seq['contacts'][:n]).mean())
            zerr=float(abs(ref['root_pos'][:,2].numpy()-seq['root_pos'][:n,2]).max())
            if mismatch or zerr>2e-6:raise ValueError('source frame placement/contact mismatch')
            row['runtime'][mode]={'contact_mismatch_fraction':mismatch,'root_height_error_m':zerr,
                                  'reset_lift_max_m':float(env._reset_height_lift.max())}
        for hz in (50,100):
            count=round(clip['horizon_s']*hz);frames=torch.arange(count)*(1/hz)/float(seq['frame_time'])
            ref=motion.sample(torch.zeros(count,dtype=torch.long),frames)
            interp={'qpos':ref['joint_pos'].numpy(),'root_pos':ref['root_pos'].numpy(),'root_rot':ref['root_quat'].numpy()}
            clear=geometry.compute(interp['qpos'],interp['root_pos'],interp['root_rot'])['clearance'].min(-1)
            minima=floor_min(model,floor,interp)
            row[f'query_{hz}hz']={'minimum_collision_height_m':float(minima.min()),
                'penetrating_more_than_0_1mm_fraction':float((minima<-.0001).mean()),
                'mesh_vs_stored_label_mismatch_fraction':float(((clear<=.02)!=ref['foot_contact'].numpy()).mean())}
        # Future padding is part of the exported trajectory, not just scored frames.
        row['all_exported_frames_minimum_collision_height_m']=float(floor_min(model,floor,seq).min())
        rows.append(row);print(name,row,flush=True)
    _write_metrics(a.output,{'status':'completed','clips':rows,'training_approved':False,
        'source_sha256':sha256(__file__),'manifest_sha256':sha256(a.manifest),
        'scope':'CPU geometric consistency; 100 Hz is an extra interpolation probe, no physical tracking'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','reference','evidence','v3-evidence','urdf','asset','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
