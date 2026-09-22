"""CPU evidence for reference provenance and support geometry; no data rewrite."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from data.bvh import load_bvh, quat_to_mat
from data.retarget_lafan1 import G1_JOINT_NAMES, M_RIG, W, _decompose_chain, g1_forward_kinematics, retarget
from data.ik_refine import refine_full
from data.build_mesh_contacts import FEET, sphere_geometry
from pgmt.envs.reference_motion import ReferenceMotion, _qmat
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.fixed_clip_diagnostic import sha256


def stats(x):
    x=np.asarray(x)
    return {'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max())}


class SupportGeometry:
    """URDF mass centers and foot spheres; geometry is not a balance controller."""
    def __init__(self, urdf):
        seq={'name':'geometry','qpos':np.zeros((2,29),np.float32),'qvel':np.zeros((2,29),np.float32),
             'root_pos':np.zeros((2,3),np.float32),'root_rot':np.tile([1,0,0,0],(2,1)).astype(np.float32),'frame_time':.02}
        self.motion=ReferenceMotion(MotionDatabase.from_sequences([seq]),urdf)
        # A dedicated audit instance also includes fixed inertial links.
        self.motion._body_order=tuple(sorted(self.motion._links))
        self.names=list(self.motion.body_names);self.feet=sphere_geometry(urdf)
        tree=ET.parse(urdf).getroot();mass=[];com=[]
        for name in self.names:
            inertial=tree.find(f"link[@name='{name}']/inertial")
            mass.append(float(inertial.find('mass').get('value')) if inertial is not None else 0.)
            origin=None if inertial is None else inertial.find('origin')
            com.append([float(v) for v in origin.get('xyz','0 0 0').split()] if origin is not None else [0,0,0])
        self.mass=torch.tensor(mass);self.com=torch.tensor(com)

    def compute(self,q,root_pos,root_quat):
        q,rp,rq=[torch.as_tensor(v,dtype=torch.float32) for v in (q,root_pos,root_quat)]
        b=self.motion.kinematics(q,torch.zeros_like(q),rp,rq,torch.zeros_like(rp),torch.zeros_like(rp))
        R=_qmat(b['body_quat']);centers=b['body_pos']+(R@self.com[...,None]).squeeze(-1)
        com=(centers*self.mass[None,:,None]).sum(1)/self.mass.sum()
        points=[];clearance=[]
        for name,(local,radii) in zip(FEET,self.feet):
            i=self.names.index(name)
            p=b['body_pos'][:,i,None]+torch.einsum('bij,kj->bki',R[:,i],local)
            points.append(p);clearance.append(p[...,2]-radii)
        return {'com':com.numpy(),'foot_points':torch.stack(points,1).numpy(),
                'clearance':torch.stack(clearance,1).numpy(),'body_pos':b['body_pos'].numpy(),
                'mass_kg':float(self.mass.sum())}


def audit_clip(clip,args,geometry):
    name=clip['sequence'];bvh_path=args.bvh/(name+'.bvh');npz_path=args.reference_data/(name+'.npz')
    b=load_bvh(bvh_path);analytic=retarget(b)
    start=clip['start_frame'];end=start+round(clip['horizon_s']/b.frame_time)
    small=replace(b,root_pos=b.root_pos[start:end],rotations=b.rotations[start:end],local_trans=b.local_trans[start:end])
    sliced={k:v[start:end].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==b.num_frames else v
            for k,v in analytic.items()}
    refined=refine_full(sliced,small)
    with np.load(npz_path,allow_pickle=False) as archive:
        saved={k:archive[k][start:end].copy() for k in ('qpos','qvel','root_pos','root_rot','contacts')}
    # q refinement is independent across frames. A sliced finalize has a
    # different constant root-z anchor and zero qvel at its new first frame.
    q_error=float(abs(refined['qpos']-saved['qpos']).max())
    if q_error>2e-6:raise ValueError(f'cannot reproduce saved IK coordinates: {name}: {q_error}')
    gp,gq=small.fk();hips=small.joint_index('Hips');chest=small.joint_index('Spine2')
    Rp=quat_to_mat(gq[:,hips]);Rc=quat_to_mat(gq[:,chest])
    raw=np.stack(_decompose_chain(M_RIG@(Rp.transpose(0,2,1)@Rc)@M_RIG.T,'waist'),-1)
    neck=gp[:,small.joint_index('Neck')]-gp[:,hips]
    trunk=np.einsum('ij,tjk,tk->ti',M_RIG,Rp.transpose(0,2,1),neck)
    source_pitch=np.arctan2(trunk[:,0],trunk[:,2])
    geom=geometry.compute(saved['qpos'],saved['root_pos'],saved['root_rot'])
    xml=g1_forward_kinematics(saved['qpos'],saved['root_pos'],saved['root_rot'])
    errors={n:float(abs(xml[n]-geom['body_pos'][:,geometry.names.index(n)]).max()) for n in xml if n in geometry.names}
    contact=(geom['clearance']<=.02).any(-1)
    if not np.array_equal(contact,saved['contacts']):raise ValueError('saved mesh contacts do not match source-frame URDF query')
    zshift=saved['root_pos'][:,2]-refined['root_pos'][:,2]
    row={'clip_id':clip['clip_id'],'sequence':name,'start':start,'stop_exclusive':end,'frames':end-start,
         'source_bvh_sha256':sha256(bvh_path),'saved_npz_sha256':sha256(npz_path),
         'mapped_source_waist_pitch_rad':stats(raw[:,2]),'source_hip_to_neck_pitch_rad':stats(source_pitch),
         'analytic_waist_pitch_rad':stats(sliced['qpos'][:,14]),'ik_waist_pitch_rad':stats(saved['qpos'][:,14]),
         'source_pitch_above_limit_fraction':float((raw[:,2]>.52).mean()),
         'analytic_pitch_at_limit_fraction':float((abs(sliced['qpos'][:,14]-.52)<1e-4).mean()),
         'ik_pitch_at_limit_fraction':float((abs(saved['qpos'][:,14]-.52)<1e-4).mean()),
         'recomputed_saved_q_max_error_rad':q_error,'local_finalize_vs_saved_root_z_shift_m':stats(zshift),
         'xml_urdf_body_max_abs_difference_m':errors,
         'source_foot_min_clearance_m':[stats(geom['clearance'][:,i].min(-1)) for i in (0,1)],
         'source_mesh_contact_fraction':contact.mean(0).tolist(),'source_mesh_contact_labels_reproduced':True}
    np.savez_compressed(args.output/(clip['clip_id']+'_stages.npz'),source_waist=raw,
        analytic_q=sliced['qpos'],ik_q=saved['qpos'],source_trunk_pitch=source_pitch,
        com=geom['com'],foot_points=geom['foot_points'],foot_clearance=geom['clearance'])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bvh','reference-data','urdf','asset','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh evidence directory')
    a.output.mkdir(parents=True);geometry=SupportGeometry(a.urdf)
    manifest=json.loads(a.manifest.read_text());rows=[]
    for clip in manifest['clips']:
        row=audit_clip(clip,a,geometry);rows.append(row)
        print(clip['clip_id'],row['mapped_source_waist_pitch_rad'],row['ik_pitch_at_limit_fraction'],flush=True)
    neutral=geometry.compute(np.zeros((1,29)),np.array([[0,0,.793]]),np.array([[1,0,0,0]]))
    summary={'status':'completed','scope':'CPU kinematic/data audit, not dynamic feasibility',
        'asset_sha256':sha256(a.asset),'urdf_sha256':sha256(a.urdf),'manifest_sha256':sha256(a.manifest),
        'source_sha256':sha256(__file__),'clips':rows,
        'neutral_geometry':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in neutral.items() if k!='body_pos'},
        'notes':['source contact labels use raw source placement, not reset placement',
                 'source joint-frame rotation and hip-to-neck vector are different quantities',
                 'saved q is reproduced; sliced IK finalize cannot reproduce the whole-sequence z anchor',
                 'XML and URDF are distinct robot geometries; neither file is modified by this audit']}
    with (a.output/'summary.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')


if __name__=='__main__':main()
