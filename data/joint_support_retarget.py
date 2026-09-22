"""Opt-in v4: temporally coupled root/leg fitting on the 50 Hz query grid.

Source stance is an uncertain geometric proxy. Its labels never replace the
independent robot-foot mesh query. No online policy/control code is changed.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags, eye, kron, vstack

from data.bvh import quat_rot_vec
from data.build_mesh_contacts import FEET, sphere_geometry
from data.retarget_lafan1 import G1_JOINT_NAMES, G1_JOINT_LIMITS


def fit_joint_support(data, support, model, dt=.02, max_nfev=200):
    """Fit 12 leg angles and root z jointly, retaining upper body and root XY.

    Residuals are in meters (pose/leg smoothness use 0.1 m/rad).
    Nonpenetration of all body geometry is checked independently after this
    foot-constrained solve; this function makes no whole-body safety claim.
    """
    q0=np.asarray(data['qpos'],float);rp0=np.asarray(data['root_pos'],float)
    n=len(q0);support=np.asarray(support,bool)
    if n<3 or support.shape!=(n,2) or not np.isfinite(q0).all():raise ValueError('invalid support trajectory')
    lo,hi=np.array([G1_JOINT_LIMITS[k] for k in G1_JOINT_NAMES[:12]]).T
    leg_names=G1_JOINT_NAMES[:12]
    geometry=sphere_geometry(model.path)
    bodies=[];locals_=[];radii=[];sides=[]
    for side,(name,(centers,rs)) in enumerate(zip(FEET,geometry)):
        for center,r in zip(centers.numpy(),rs.numpy()):
            bodies.append(name);locals_.append(center);radii.append(r);sides.append(side)
    sides=np.array(sides);radii=np.array(radii)
    key_bodies=['left_knee_link','left_ankle_pitch_link','right_knee_link','right_ankle_pitch_link']
    p0,r0=model.forward(q0,rp0,data['root_rot'])
    original_points=np.stack([p0[b]+quat_rot_vec(r0[b],l) for b,l in zip(bodies,locals_)],1)
    original_keys=np.stack([p0[b] for b in key_bodies],1)
    scale=np.r_[np.full(12,.1),1.]
    d1=diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))/dt
    d2=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))/dt**2
    temporal=vstack([.1*kron(d1,diags(scale)),.01*kron(d2,diags(scale))],format='csr')
    prior=np.r_[np.full(12,np.sqrt(.02)),np.sqrt(.05)]
    origin=np.c_[q0[:,:12],rp0[:,2]]
    cache={}

    def evaluate(flat):
        if 'x' in cache and np.array_equal(flat,cache['x']):return cache['r'],cache['j']
        x=flat.reshape(n,13)-origin;q=q0.copy();q[:,:12]+=x[:,:12];rp=rp0.copy();rp[:,2]+=x[:,12]
        p,r=model.forward(q,rp,data['root_rot'])
        points=np.stack([p[b]+quat_rot_vec(r[b],l) for b,l in zip(bodies,locals_)],1)
        jp=model.jacobian(p,r,bodies,leg_names,locals_).reshape(n,len(bodies),3,12)
        jp=np.concatenate([jp,np.broadcast_to([0.,0.,1.],(n,len(bodies),3))[...,None]],-1)
        clear=points[...,2]-radii
        residual=[];jac=[]
        # Only the lowest sphere on a stance foot must touch; pitching/toe-off
        # is retained, rather than forcing every sole sphere to the plane.
        for side in (0,1):
            ids=np.flatnonzero(sides==side);idx=ids[clear[:,ids].argmin(-1)]
            weight=np.sqrt(10.)*support[:,side]
            residual.append(((clear[np.arange(n),idx]-.003)*weight)[:,None])
            jac.append((jp[np.arange(n),idx,2]*weight[:,None])[:,None])
        # Preserve swinging foot geometry and leg keypoints under morphology.
        swing=(~support[:,sides]).astype(float)
        residual.append(((points-original_points)*swing[...,None]).reshape(n,-1))
        jac.append((jp*swing[...,None,None]).reshape(n,-1,13))
        keys=np.stack([p[b] for b in key_bodies],1)
        jk=model.jacobian(p,r,key_bodies,leg_names).reshape(n,4,3,12)
        jk=np.concatenate([jk,np.broadcast_to([0.,0.,1.],(n,4,3))[...,None]],-1)
        residual.append((.5*(keys-original_keys)).reshape(n,-1));jac.append((.5*jk).reshape(n,-1,13))
        negative=np.minimum(clear-.001,0)
        residual.append(10.*negative);jac.append(10.*jp[:,:,2]*(clear<.001)[...,None])
        residual.append(x*prior);jac.append(np.broadcast_to(np.diag(prior),(n,13,13)))
        local=np.concatenate(residual,1);jlocal=np.concatenate(jac,1);nr=local.shape[1]
        rows=np.broadcast_to(np.arange(n*nr).reshape(n,nr,1),jlocal.shape).ravel()
        cols=np.broadcast_to((np.arange(n)[:,None,None]*13+np.arange(13)),jlocal.shape).ravel()
        js=coo_matrix((jlocal.ravel(),(rows,cols)),shape=(n*nr,n*13)).tocsr()
        result=np.r_[local.ravel(),temporal@x.ravel()];derivative=vstack([js,temporal],format='csr')
        cache.update(x=flat.copy(),r=result,j=derivative)
        return result,derivative

    lower=np.c_[np.broadcast_to(lo,(n,12)),np.full(n,-np.inf)].ravel()
    upper=np.c_[np.broadcast_to(hi,(n,12)),np.full(n,np.inf)].ravel()
    # Use absolute coordinates for the optimizer's trust radius. A nearly-zero
    # correction vector nudged off a joint bound yields a ~1e-8 initial radius
    # and misleading ftol success after two negligible steps in SciPy TRF.
    initial=np.clip(origin.ravel(),lower+1e-8,upper-1e-8)
    initial_cost=float(np.square(evaluate(initial)[0]).sum()/2)
    fit=least_squares(lambda x:evaluate(x)[0],initial,jac=lambda x:evaluate(x)[1],bounds=(lower,upper),
                      method='trf',tr_solver='lsmr',max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-5)
    x=fit.x.reshape(n,13)-origin;q=q0.copy();q[:,:12]+=x[:,:12];rp=rp0.copy();rp[:,2]+=x[:,12]
    if not np.isfinite(x).all() or fit.cost>initial_cost+1e-8:raise RuntimeError('joint support fit invalid')
    return q,rp,{'converged':bool(fit.success),'message':fit.message,'function_evaluations':fit.nfev,
                 'initial_cost':initial_cost,'final_cost':float(fit.cost),'optimality':float(fit.optimality),
                 'maximum_joint_change_rad':float(abs(x[:,:12]).max()),'maximum_root_change_m':float(abs(x[:,12]).max()),
                 'stance_target_m':.003,'foot_margin_m':.001,'pose_weight_m2_per_rad2':.02,
                 'root_weight':.05,'stance_weight':10.,'foot_penetration_weight':100.,
                 'temporal_velocity_weight_s2':.01,'temporal_acceleration_weight_s4':.0001,
                 'leg_temporal_conversion_m_per_rad':.1}
