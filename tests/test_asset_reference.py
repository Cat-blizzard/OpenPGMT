"""Asset-bound FK, grounding and reward derivative contracts."""
from pathlib import Path

import numpy as np
import pytest
import torch

from data.g1_kinematics import G1Kinematics
from data.retarget_lafan1 import G1_JOINT_NAMES
from pgmt.envs.reference_motion import ReferenceMotion
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.g1_env import G1Env, G1EnvConfig

URDF=Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')
pytestmark=pytest.mark.skipif(not URDF.exists(),reason='external robot asset absent')


def sequence(n=20):
    return dict(qpos=np.zeros((n,29),np.float32),qvel=np.zeros((n,29),np.float32),
        root_pos=np.tile([0,0,.9],(n,1)).astype(np.float32),
        root_rot=np.tile([1,0,0,0],(n,1)).astype(np.float32),frame_time=.02)


def test_asset_fk_and_point_jacobian_against_independent_runtime():
    model=G1Kinematics(URDF);rng=np.random.default_rng(14)
    q=rng.uniform(-.4,.4,(5,29));rp=rng.normal(size=(5,3));rq=rng.normal(size=(5,4));rq/=np.linalg.norm(rq,axis=1,keepdims=True)
    p,r=model.forward(q,rp,rq)
    ref=ReferenceMotion(MotionDatabase.from_sequences([sequence()]),URDF)
    args=[torch.tensor(v,dtype=torch.float32) for v in (q,np.zeros_like(q),rp,rq,np.zeros_like(rp),np.zeros_like(rp))]
    actual=ref.kinematics(*args)
    for i,name in enumerate(ref.body_names):
        np.testing.assert_allclose(p[name],actual['body_pos'][:,i],atol=1e-6)
    names=['left_wrist_yaw_link','torso_link','right_ankle_roll_link']
    J=model.jacobian(p,r,names)
    for j in range(29):
        dq=np.zeros_like(q);dq[:,j]=1e-6
        plus,_=model.forward(q+dq,rp,rq);minus,_=model.forward(q-dq,rp,rq)
        for i,name in enumerate(names):
            np.testing.assert_allclose(J[:,3*i:3*i+3,j],(plus[name]-minus[name])/2e-6,atol=2e-8,rtol=1e-5)


@pytest.mark.parametrize('mode',['nominal','reference_state'])
def test_ground_plane_survives_runtime_root_placement(mode):
    model=G1Kinematics(URDF);seq=model.finalize_reference(sequence())
    env=G1Env(G1EnvConfig(num_envs=2,reference_urdf_path=str(URDF),reset_mode=mode,
        default_root_pos=(2.,3.,1.2),enable_adaptive_sampling=False),reference_database=MotionDatabase.from_sequences([seq]))
    env.reset(seed=7)
    torch.testing.assert_close(env.reference_root_pos[:,2],torch.full((2,),model.neutral_height),atol=2e-6,rtol=0)
    assert env.reference_body['foot_contact'].all()
    assert env._reference_placement_pos[:,2].eq(0).all()
    if mode=='nominal':assert env.root_pos[:,2].eq(1.2).all()
    else:
        assert env._reset_height_lift.max()<2e-6
        torch.testing.assert_close(env.root_pos,env.reference_root_pos,atol=2e-6,rtol=0)


def test_grounded_references_reject_wrong_assets_or_mixed_contracts():
    seq=G1Kinematics(URDF).finalize_reference(sequence())
    seq['kinematic_urdf_sha256']=np.array('0'*64)
    cfg=G1EnvConfig(reference_urdf_path=str(URDF))
    with pytest.raises(ValueError,match='geometry differs'):
        G1Env(cfg,reference_database=MotionDatabase.from_sequences([seq]))
    with pytest.raises(ValueError,match='cannot mix'):
        G1Env(cfg,reference_database=MotionDatabase.from_sequences([seq,sequence()]))


@pytest.mark.parametrize('acceleration',[0.,2.])
@pytest.mark.parametrize('source_dt',[.02,1/30])
def test_reference_acceleration_matches_backward_physical_velocity_difference(acceleration,source_dt):
    seq=sequence();seq['frame_time']=source_dt;t=np.arange(20,dtype=np.float32)*source_dt
    seq['root_pos'][:,0]=t+.5*acceleration*t*t
    motion=ReferenceMotion(MotionDatabase.from_sequences([seq]),URDF)
    frames=torch.tensor([0.,.6,5.2,18.,19.]);sample=motion.sample(torch.zeros(5,dtype=torch.long),frames)
    expected=torch.tensor(1+acceleration*frames.numpy()*source_dt)
    torch.testing.assert_close(sample['root_lin_vel'][:,0],expected,atol=2e-5,rtol=0)
    torch.testing.assert_close(sample['root_accel'][1:,0],torch.full((4,),acceleration),atol=5e-4,rtol=0)
    assert sample['body_accel'][0].abs().max()==0  # no prior interval at first source frame
    torch.testing.assert_close(sample['body_accel'][1:,:,0],torch.full_like(sample['body_accel'][1:,:,0],acceleration),atol=2e-3,rtol=0)
    yaw=torch.tensor([[2**-.5,0,0,2**-.5]]).expand(5,-1)
    rotated=motion.sample(torch.zeros(5,dtype=torch.long),frames,placement_quat=yaw)
    torch.testing.assert_close(rotated['body_accel'][1:,:,1],sample['body_accel'][1:,:,0],atol=2e-5,rtol=0)


def test_full_sequence_grounding_is_one_constant_height_shift():
    model=G1Kinematics(URDF);seq=sequence();seq['root_pos'][:,2]+=np.arange(20)*.003
    out=model.finalize_reference(seq)
    shift=out['root_pos'][:,2]-seq['root_pos'][:,2]
    np.testing.assert_allclose(shift,shift[0],atol=1e-7)
    np.testing.assert_array_equal(out['qpos'],seq['qpos'])
    np.testing.assert_allclose(np.diff(out['root_pos'][:,2]),np.diff(seq['root_pos'][:,2]),atol=1e-7)


def test_resume_and_recovery_fingerprints_detect_changed_reference_contracts():
    seq=G1Kinematics(URDF).finalize_reference(sequence())
    db=MotionDatabase.from_sequences([seq])
    env=G1Env(G1EnvConfig(num_envs=1,reference_urdf_path=str(URDF)),reference_database=db)
    env.reset(seed=0)
    state=env.state_dict()
    env.load_state_dict(state)
    old=dict(state);old.pop('reference_kinematics_contract')
    with pytest.raises(ValueError,match='kinematics contract'):
        env.load_state_dict(old)
    old=dict(state);old.pop('reference_placement_contract')
    with pytest.raises(ValueError,match='placement contract'):
        env.load_state_dict(old)
    legacy={k:v for k,v in seq.items() if k not in ('reference_frame_contract','reference_ground_z','kinematic_urdf_sha256')}
    assert MotionDatabase.from_sequences([legacy]).pose_fingerprint()!=db.pose_fingerprint()


def test_static_load_target_satisfies_virtual_work_and_limits():
    from setup.prepare_static_load_candidate import static_target
    from data.bvh import quat_rot_vec
    from data.build_mesh_contacts import sphere_geometry, FEET
    model=G1Kinematics(URDF);bodies=[]
    for name,link in model.links.items():
        inertia=link.find('inertial')
        if inertia is None:continue
        origin=inertia.find('origin')
        bodies.append({'name':name,'mass_kg':float(inertia.find('mass').get('value')),
            'com_local':np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)})
    result=static_target(model,bodies)
    assert result['max_effort_fraction']<.11
    assert result['force_balance_error_n']<1e-9 and result['horizontal_moment_error_nm']<1e-9
    forces=np.array(result['normal_forces_n'])
    def potential(q):
        p,r=model.forward(q)
        gravity=sum(b['mass_kg']*9.81*(p[b['name']][0]+quat_rot_vec(r[b['name']][0],b['com_local']))[2] for b in bodies)
        heights=[]
        for name,(centers,_) in zip(FEET,sphere_geometry(URDF)):
            heights.extend((p[name][0]+quat_rot_vec(r[name][0],centers.numpy().astype(np.float64)))[:,2])
        return gravity-forces@np.array(heights)
    numeric=[]
    for j in range(29):
        dq=np.zeros((1,29));dq[0,j]=1e-6
        numeric.append((potential(dq)-potential(-dq))/2e-6)
    np.testing.assert_allclose(numeric,result['required_pd_torque_nm'],atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(np.array(result['target'])*80,result['required_pd_torque_nm'],atol=1e-12)
