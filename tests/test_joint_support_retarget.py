"""Physical geometry checks for root/leg fitting, independent of RL training."""
from pathlib import Path
import numpy as np
import pytest

from data.g1_kinematics import G1Kinematics
from data.joint_support_retarget import fit_joint_support
from data.retarget_lafan1 import G1_JOINT_LIMITS
from setup.audit_reference_control import SupportGeometry

URDF=Path('/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf')
pytestmark=pytest.mark.skipif(not URDF.exists(),reason='external geometry unavailable')


def test_asymmetric_support_requires_leg_changes_and_preserves_upper_body():
    model=G1Kinematics(URDF);geometry=SupportGeometry(URDF)
    q=np.zeros((12,29));q[:,2]=.15;q[:,3]=.25
    rp=np.tile([0.,0.,model.neutral_height+.02],(12,1));rr=np.tile([1.,0,0,0],(12,1))
    base=geometry.compute(q,rp,rr)['clearance'].min(-1)
    fitted,root,info=fit_joint_support(dict(qpos=q,root_pos=rp,root_rot=rr),np.ones((12,2),bool),model,max_nfev=50)
    clear=geometry.compute(fitted,root,rr)['clearance'].min(-1)
    assert np.max(clear)<.01 and np.min(clear)>0
    assert abs(fitted[:,:12]-q[:,:12]).max()>.02
    assert np.array_equal(fitted[:,12:],q[:,12:]) and np.array_equal(root[:,:2],rp[:,:2])
    assert np.max(abs(np.diff(fitted,axis=0)))<1e-5
    assert info['final_cost']<info['initial_cost'] and np.ptp(clear[0])<np.ptp(base[0])


def test_airborne_motion_is_not_forced_to_stance():
    model=G1Kinematics(URDF);q=np.zeros((8,29))
    rp=np.tile([0.,0.,model.neutral_height+.1],(8,1));rr=np.tile([1.,0,0,0],(8,1))
    fitted,root,_=fit_joint_support(dict(qpos=q,root_pos=rp,root_rot=rr),np.zeros((8,2),bool),model)
    np.testing.assert_allclose(fitted,q,atol=1e-7)
    np.testing.assert_allclose(root,rp,atol=1e-7)


def test_joint_on_bound_does_not_freeze_the_entire_optimizer():
    model=G1Kinematics(URDF);q=np.zeros((12,29));q[:,2]=.15;q[:,3]=.25
    q[:,8]=G1_JOINT_LIMITS['right_hip_yaw'][0]
    rp=np.tile([0.,0.,model.neutral_height+.02],(12,1));rr=np.tile([1.,0,0,0],(12,1))
    fitted,root,info=fit_joint_support(dict(qpos=q,root_pos=rp,root_rot=rr),np.ones((12,2),bool),model,max_nfev=50)
    assert info['function_evaluations']>2
    assert info['final_cost']<info['initial_cost']*.9
    assert abs(fitted[:,:12]-q[:,:12]).max()>.01
