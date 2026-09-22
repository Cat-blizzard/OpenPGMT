"""CPU URDF/USD consistency audit and one statically balanced PD target.

Requires pxr, but never starts SimulationApp. Static feasibility is not stability.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from data.bvh import quat_rot_vec
from data.g1_kinematics import G1Kinematics
from data.retarget_lafan1 import G1_JOINT_NAMES, G1_JOINT_LIMITS
from data.build_mesh_contacts import sphere_geometry, FEET
from pgmt.envs.actuators import G1_EFFORT_LIMITS
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def wxyz(value):
    return np.array([value.GetReal(), *value.GetImaginary()])


def audit_asset(model, asset):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(asset))
    if stage.GetMetadata('metersPerUnit') != 1. or stage.GetMetadata('kilogramsPerUnit') != 1.:
        raise ValueError('asset must be in SI units')
    p, q = model.forward(np.zeros((1,29)))
    cache = UsdGeom.XformCache()
    rigid_names = {p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints, bodies, collisions = [], [], []
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'PhysicsRevoluteJoint':
            name = prim.GetName().removesuffix('_joint');child=model.actuated[name]
            parent, _, _, xyz, quat, axis=model.joints[child]
            pos0=np.array(prim.GetAttribute('physics:localPos0').Get());pos1=np.array(prim.GetAttribute('physics:localPos1').Get())
            r0=Rotation.from_quat(wxyz(prim.GetAttribute('physics:localRot0').Get())[[1,2,3,0]]).as_matrix()
            r1=Rotation.from_quat(wxyz(prim.GetAttribute('physics:localRot1').Get())[[1,2,3,0]]).as_matrix()
            restR=r0@r1.T;restp=pos0-restR@pos1
            R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            usd_axis=np.array([float(k==prim.GetAttribute('physics:axis').Get()) for k in 'XYZ'])
            link_names=[prim.GetRelationship('physics:body'+str(i)).GetTargets()[0].name for i in (0,1)]
            joints.append({'name':name,'origin_error_m':float(abs(restp-xyz).max()),
                'rotation_matrix_error':float(abs(restR-R).max()),'axis_error':float(abs(r1@usd_axis-axis).max()),
                'links_match':link_names==[parent,child]})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name=prim.GetName();mass=float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
            com=np.array(UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get())
            matrix=cache.GetLocalToWorldTransform(prim)
            usd_pos=np.array(matrix.ExtractTranslation())
            # Gf's near-identity rotation extraction can lose small angles;
            # compare the actual transform matrix with a stable conversion.
            usd_quat=Rotation.from_matrix(np.array(matrix)[:3,:3].T).as_quat()[[3,0,1,2]]
            merged_mass=0.;merged_com=np.zeros(3)
            for link,node in model.links.items():
                ancestor=link
                while ancestor!=name and ancestor!='pelvis' and ancestor not in rigid_names and model.joints[ancestor][2]=='fixed':
                    ancestor=model.joints[ancestor][0]
                if ancestor!=name:continue
                inertia=node.find('inertial')
                if inertia is None:continue
                m=float(inertia.find('mass').get('value'));origin=inertia.find('origin')
                xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                world=p[link][0]+quat_rot_vec(q[link][0],xyz)
                local=quat_rot_vec(q[name][0]*np.array([1,-1,-1,-1]),world-p[name][0])
                merged_mass+=m;merged_com+=m*local
            if merged_mass == 0 and mass == 0:
                # The fixed camera/head placeholder has no mass or collision;
                # USD uses (-inf,-inf,-inf) for its unspecified COM.
                if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
                    raise ValueError('zero authored mass with collision would require runtime mass inference')
                com = np.zeros(3)
            else:
                merged_com/=merged_mass
            bodies.append({'name':name,'mass_kg':mass,'com_local':com.tolist(),
                'merged_urdf_mass_error_kg':abs(mass-merged_mass),'merged_urdf_com_error_m':float(abs(com-merged_com).max()),
                'rest_position_error_m':float(abs(usd_pos-p[name][0]).max()),
                'rest_rotation_error':float(min(abs(usd_quat-q[name][0]).max(),abs(usd_quat+q[name][0]).max()))})
    for name,(centers,radii) in zip(FEET,sphere_geometry(model.path)):
        found=[]
        body=stage.GetPrimAtPath('/g1_29dof/'+name)
        inv=cache.GetLocalToWorldTransform(body).GetInverse()
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()).startswith(str(body.GetPath())+'/'):
                if prim.GetTypeName()!='Sphere':raise ValueError('unexpected USD foot collision')
                matrix=cache.GetLocalToWorldTransform(prim)*inv
                found.append([*matrix.ExtractTranslation(),float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())])
        expected=np.concatenate((centers.numpy(),radii.numpy()[:,None]),axis=1)
        error=float(abs(np.array(sorted(found))-np.array(sorted(expected.tolist()))).max())
        collisions.append({'body':name,'sphere_count':len(found),'max_geometry_error_m':error})
    ok=(len(joints)==29 and all(r['links_match'] and max(r['origin_error_m'],r['rotation_matrix_error'],r['axis_error'])<2e-6 for r in joints)
        and all(r['merged_urdf_mass_error_kg']<1e-5 and r['merged_urdf_com_error_m']<2e-6
                and r['rest_position_error_m']<2e-6 for r in bodies)
        and all(r['sphere_count']==4 and r['max_geometry_error_m']<2e-6 for r in collisions))
    return {'passed':ok,'joints':joints,'bodies':bodies,'foot_collisions':collisions,
            'authored_rest_rotation_max_quaternion_error':max(r['rest_rotation_error'] for r in bodies),
            'rotation_contract':'Joint frame transforms define FK. Authored flattened rest-body poses have small rotation rounding differences; verify runtime poses separately.',
            'collision_scope':'exact sphere-foot geometry; whole-body URDF collision support is used offline'}


def static_target(model, bodies, kp=80.):
    pose=np.zeros((1,29));root=np.array([[0,0,model.neutral_height]])
    p,q=model.forward(pose,root)
    names=[b['name'] for b in bodies];local=np.array([b['com_local'] for b in bodies]);m=np.array([b['mass_kg'] for b in bodies])
    world=np.array([p[n][0]+quat_rot_vec(q[n][0],v) for n,v in zip(names,local)])
    com=(world*m[:,None]).sum(0)/m.sum();weight=m.sum()*9.81
    foot_bodies=[];foot_local=[];foot_world=[]
    for name,(centers,radii) in zip(FEET,sphere_geometry(model.path)):
        for center in centers.numpy():
            foot_bodies.append(name);foot_local.append(center);foot_world.append(p[name][0]+quat_rot_vec(q[name][0],center))
    foot_world=np.array(foot_world)
    A=np.stack((np.ones(8),foot_world[:,0],foot_world[:,1]))
    desired=weight*np.array([1.,com[0],com[1]])
    forces=A.T@np.linalg.solve(A@A.T,desired) # minimum-norm vertical support forces
    if (forces<=0).any():raise ValueError('neutral support cannot use all eight contacts with positive normal force')
    Jcom=model.jacobian(p,q,names,local_points=local)[0].reshape(-1,3,29)
    Jfeet=model.jacobian(p,q,foot_bodies,local_points=foot_local)[0].reshape(-1,3,29)
    gravity=np.einsum('k,kj->j',m*9.81,Jcom[:,2,:])
    contact=np.einsum('k,kj->j',forces,Jfeet[:,2,:])
    tau=gravity-contact;target=pose[0]+tau/kp
    lo,hi=np.array([G1_JOINT_LIMITS[n] for n in G1_JOINT_NAMES]).T
    if (target<=lo).any() or (target>=hi).any() or (abs(tau)>=G1_EFFORT_LIMITS).any():
        raise ValueError('static target violates actuator constraints')
    return {'pose':pose[0].tolist(),'root_height_m':model.neutral_height,'target':target.tolist(),
        'gravity_generalized_nm':gravity.tolist(),'contact_generalized_nm':contact.tolist(),'required_pd_torque_nm':tau.tolist(),
        'normal_forces_n':forces.tolist(),'mass_kg':float(m.sum()),'com_world_m':com.tolist(),
        'force_balance_error_n':float(abs(A[0]@forces-desired[0])),
        'horizontal_moment_error_nm':float(abs(A[1:]@forces-desired[1:]).max()),
        'pd_balance_residual_nm':float(abs(kp*(target-pose[0])+contact-gravity).max()),
        'max_effort_fraction':float((abs(tau)/np.array(G1_EFFORT_LIMITS)).max()),
        'max_target_offset_rad':float(abs(target-pose[0]).max()),
        'friction_assumption':'vertical static gravity load: tangential forces zero; positive normal forces',
        'scope':'static equilibrium at q=0; no assertion of passive/dynamic stability'}


def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);model=G1Kinematics(a.urdf)
    asset=audit_asset(model,a.asset);_write_metrics(a.output/'asset_audit.json',asset)
    if not asset['passed']:raise RuntimeError('URDF/USD asset mismatch')
    candidate=static_target(model,asset['bodies'])
    _write_metrics(a.output/'candidate.json',candidate)
    # Same q=0 physical/reference posture in all conditions. Only the target
    # differs. An initial 1 mm gap avoids geometric penetration at contact start.
    quat=np.tile([1.,0,0,0],(4,1));angle=np.deg2rad(1.)/2
    for row,axis,sign in ((1,1,1),(2,1,-1),(3,2,1)):
        quat[row,0]=np.cos(angle);quat[row,axis]=sign*np.sin(angle)
    from setup.audit_reference_control import SupportGeometry
    geom=SupportGeometry(a.urdf);root=np.tile([0,0,model.neutral_height+.001],(4,1))
    g=geom.compute(np.zeros((4,29)),root,quat)
    root[:,2]+=np.maximum(.001-g['clearance'].min((1,2)),0)
    torch.save({'root_quat':torch.tensor(quat,dtype=torch.float32),'root_height':torch.tensor(root[:,2],dtype=torch.float32),
        'pose':torch.zeros(29),'control_target':torch.zeros(29),'candidate_target':torch.tensor(candidate['target'],dtype=torch.float32)},a.output/'starts.pt')
    data=a.output/'reference';data.mkdir();n=600
    np.savez_compressed(data/'static_loaded.npz',qpos=np.zeros((n,29),np.float32),qvel=np.zeros((n,29),np.float32),
        root_pos=np.tile([0,0,model.neutral_height],(n,1)).astype(np.float32),root_rot=np.tile([1,0,0,0],(n,1)).astype(np.float32),
        contacts=np.ones((n,2),bool),frame_time=np.array(.02),contact_source=np.array('offline_reference_mesh_v2'),
        reference_frame_contract=np.array('flat_ground_v1'),reference_ground_z=np.array(0.),kinematic_urdf_sha256=np.array(model.sha256))
    _write_metrics(data/'contact_manifest.json',{'sequence_names':['static_loaded']})
    _write_metrics(a.output/'manifest.json',{'schema':'pgmt_fixed_clip_diagnostic_v1','control_dt':.02,'sim_dt':.005,'horizon_s':10.,
        'clips':[{'clip_id':'static_loaded','sequence':'static_loaded','start_frame':0,'horizon_s':10.,'split':'train'}],
        'sequence_names':['static_loaded'],'reference_files':{p.name:sha256(p) for p in data.glob('*.npz')},
        'asset_sha256':sha256(a.asset),'urdf_sha256':model.sha256,'candidate_sha256':sha256(a.output/'candidate.json'),
        'budget':{'conditions':['control','candidate'],'modes':['fixed','deterministic'],'envs':4,'steps_per_run':500,'total_max_transitions':8000}})
    print('PREPARED',candidate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('urdf','asset','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
