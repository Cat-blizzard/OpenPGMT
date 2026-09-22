"""Opt-in anatomical targets and support-constrained root height for diagnostic v3.

This does not change the legacy exporter or any RL reward. Source stance is a
geometric/velocity proxy used by offline retargeting; exported contact labels
are still independent collision-sphere/ground-mesh queries.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

from data.bvh import quat_to_mat
from data.retarget_lafan1 import (G1_JOINT_NAMES, M_RIG, W, _decompose_chain,
                                  _chain_joints, _mat_mean, bounded_joint_trajectory)
from data.ik_refine import FULL_KEYPOINTS, FULL_WEIGHTS, _refine


def contiguous_runs(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edge == 1).tolist(), np.flatnonzero(edge == -1).tolist()))


def source_support(bvh, positions, scale):
    """One robust flat source plane per sequence, then contiguous stance bouts."""
    feet = positions[:, [bvh.joint_index(n) for n in ('LeftToe', 'RightToe')]] @ W.T * scale
    speed = np.linalg.norm(np.gradient(feet, bvh.frame_time, axis=0), axis=-1)
    z = feet[..., 2]
    low = z <= z.min(-1, keepdims=True) + .02
    candidates = (speed <= .15) & low
    if candidates.sum() < 30:
        raise ValueError('insufficient stationary source feet to estimate a flat plane')
    plane = float(np.median(z[candidates]))
    support = candidates & (abs(z-plane) <= .02)
    for side in range(2):
        for start, end in contiguous_runs(support[:, side]):
            if end-start < 3:
                support[start:end, side] = False
    return plane, support, feet, speed


def anatomical_targets(bvh, analytic, model, positions, rotations):
    """Map actual shoulder/elbow/wrist pivots and adapt bone lengths to G1.

    A bounded source waist orientation owns the torso target. Position IK
    adjusts limbs, not that orientation, avoiding morphology-induced waist
    saturation. A genuinely out-of-range source waist remains projected.
    """
    out = dict(analytic)
    q = analytic['qpos'].astype(np.float64).copy()
    rotations = quat_to_mat(rotations)
    for side, prefix in [('left', 'Left'), ('right', 'Right')]:
        for suffix, child, parent, base in [
            ('shoulder', prefix+'Arm', 'Spine2', side+'_shoulder_pitch_link'),
            ('elbow', prefix+'ForeArm', prefix+'Arm', side+'_shoulder_yaw_link'),
            ('wrist', prefix+'Hand', prefix+'ForeArm', side+'_elbow_link')]:
            align = quat_to_mat(model.rest[base][1]) @ M_RIG
            relative = rotations[:, bvh.joint_index(parent)].transpose(0,2,1) @ rotations[:, bvh.joint_index(child)]
            aligned = align @ relative @ align.T
            centered = aligned @ _mat_mean(aligned).T
            chain = side+'_'+suffix
            for name, angle in zip(_chain_joints(chain), _decompose_chain(centered, chain)):
                q[:, G1_JOINT_NAMES.index(name)] = angle
    q, _ = bounded_joint_trajectory(q, float(analytic['frame_time']))
    out['qpos'] = q
    body, _ = model.forward(q, out['root_pos'], out['root_rot'])
    source = positions @ W.T
    targets = {'torso_link': body['torso_link']}

    def direction(a, b):
        d = source[:, bvh.joint_index(b)] - source[:, bvh.joint_index(a)]
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)

    def length(a, b):
        return float(np.linalg.norm(model.rest[a][0]-model.rest[b][0]))

    for side, prefix in [('left', 'Left'), ('right', 'Right')]:
        shoulder, elbow, roll, wrist = [side+'_'+name+'_link' for name in
                                      ('shoulder_pitch', 'elbow', 'wrist_roll', 'wrist_yaw')]
        targets[shoulder] = body[shoulder]
        targets[elbow] = targets[shoulder] + direction(prefix+'Arm', prefix+'ForeArm')*length(shoulder, elbow)
        forearm = direction(prefix+'ForeArm', prefix+'Hand')
        targets[roll] = targets[elbow] + forearm*length(elbow, roll)
        targets[wrist] = targets[elbow] + forearm*length(elbow, wrist)
        hip, knee, ankle = [side+'_'+name+'_link' for name in ('hip_pitch', 'knee', 'ankle_pitch')]
        targets[knee] = body[hip] + direction(prefix+'UpLeg', prefix+'Leg')*length(hip, knee)
        targets[ankle] = targets[knee] + direction(prefix+'Leg', prefix+'Foot')*length(knee, ankle)
    return out, np.stack([targets[body] for _, body in FULL_KEYPOINTS], 1)


def refine_limbs(data, targets, model, iterations=30):
    return _refine(data, targets, list(range(12))+list(range(15,29)),
                   [body for _, body in FULL_KEYPOINTS], FULL_WEIGHTS,
                   max_iter=iterations, kinematics=model)


def solve_support_height(foot_clearance, all_collision_min, support, dt):
    """One convex trajectory solve: stance residual + source motion prior.

    delta has meters; velocity/acceleration penalties act on delta, retaining
    the source root trajectory when no correction is required. Collision
    nonpenetration is a hard lower bound, not a later per-frame floor clamp.
    """
    n = len(foot_clearance)
    if n < 3 or np.shape(support) != (n, 2) or np.shape(foot_clearance) != (n, 2):
        raise ValueError('expected at least three frames of two-foot constraints')
    if not all(np.isfinite(x).all() for x in (foot_clearance, all_collision_min)):
        raise ValueError('non-finite support geometry')
    d1 = diags([-np.ones(n-1), np.ones(n-1)], [0,1], shape=(n-1,n)) / dt
    d2 = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0,1,2], shape=(n-2,n)) / dt**2
    prior, velocity, acceleration = .01, .01, .0001
    h = diags(support.sum(-1)+prior) + velocity*(d1.T@d1) + acceleration*(d2.T@d2)
    b = (foot_clearance*support).sum(-1)
    lower = -np.asarray(all_collision_min) + 1e-5
    def objective(delta):
        hd = h @ delta
        return .5*float(delta@hd)+float(b@delta), hd+b
    initial = np.maximum(-b/(support.sum(-1)+prior), lower)
    fit = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=list(zip(lower, np.full(n,np.inf))),
                   options={'maxiter':2000, 'ftol':1e-12, 'gtol':1e-7, 'maxls':40})
    if not fit.success or not np.isfinite(fit.x).all() or (fit.x<lower-1e-8).any():
        raise RuntimeError(f'support trajectory solve failed: {fit.message}')
    return fit.x, {'status':str(fit.message), 'iterations':int(fit.nit),
        'root_prior_weight':prior, 'velocity_weight_s2':velocity,
        'acceleration_weight_s4':acceleration, 'nonpenetration_margin_m':1e-5}
