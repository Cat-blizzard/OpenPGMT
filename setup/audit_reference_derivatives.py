"""Analytic SI/time checks and frozen static-fixture equivalence, CPU only."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from data.g1_kinematics import G1Kinematics
from pgmt.envs.reference_motion import ReferenceMotion
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def run(root, urdf, output):
    if output.exists():raise FileExistsError(output)
    spec=importlib.util.spec_from_file_location('frozen_reference_motion',root/'source_executed/pgmt/envs/reference_motion.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    with np.load(root/'fixture_v4/reference/static_loaded.npz') as z:seq={k:z[k] for k in z.files}
    db=MotionDatabase.from_sequences([seq]);before=frozen.ReferenceMotion(db,urdf);after=ReferenceMotion(db,urdf)
    frames=torch.arange(501,dtype=torch.float32);ids=torch.zeros(501,dtype=torch.long)
    a=before.sample(ids,frames);b=after.sample(ids,frames)
    errors={k:float((a[k].float()-b[k].float()).abs().max()) for k in a if isinstance(a[k],torch.Tensor)}
    if any(errors.values()):raise ValueError('post-physics derivative change altered the frozen static reference')
    model=G1Kinematics(urdf);rows=[]
    frames=torch.tensor([0.,.6,5.2,18.,38.,39.]);ids=torch.zeros(len(frames),dtype=torch.long)
    t=np.arange(40,dtype=np.float64)/30;sample_t=frames.numpy().astype(np.float64)/30;prev_t=np.maximum(sample_t-.02,0)
    for case in ('static','translation','accelerating_translation','yaw','knee_sine'):
        q=np.zeros((40,29));rp=np.tile([0.,0.,.9],(40,1));rq=np.tile([1.,0.,0.,0.],(40,1))
        if case in ('translation','accelerating_translation'):
            acc=2. if case=='accelerating_translation' else 0.
            rp[:,0]=t+.5*acc*t*t
        if case=='yaw':rq[:,0]=np.cos(.2*t/2);rq[:,3]=np.sin(.2*t/2)
        if case=='knee_sine':q[:,3]=.1*np.sin(.8*t)
        seq=dict(qpos=q.astype(np.float32),qvel=np.zeros_like(q,dtype=np.float32),root_pos=rp.astype(np.float32),
                 root_rot=rq.astype(np.float32),frame_time=1/30)
        motion=ReferenceMotion(MotionDatabase.from_sequences([seq]),urdf);actual=motion.sample(ids,frames)
        def exact_velocity(times):
            q=np.zeros((len(times),29));qd=np.zeros_like(q);rp=np.zeros((len(times),3));rq=np.tile([1.,0.,0.,0.],(len(times),1))
            if case=='yaw':rq[:,0]=np.cos(.2*times/2);rq[:,3]=np.sin(.2*times/2)
            if case=='knee_sine':q[:,3]=.1*np.sin(.8*times);qd[:,3]=.08*np.cos(.8*times)
            p,r=model.forward(q,rp,rq);names=motion.body_names
            if case=='yaw':return np.cross(np.array([0.,0.,.2]),np.stack([p[n] for n in names],1))
            if case in ('translation','accelerating_translation'):
                v=np.zeros((len(times),len(names),3));v[:,:,0]=(1+acc*times)[:,None];return v
            return np.einsum('nbj,nj->nb',model.jacobian(p,r,names),qd).reshape(len(times),len(names),3)
        expected_v=exact_velocity(sample_t);previous=exact_velocity(prev_t)
        expected_a=(expected_v-previous)/np.maximum(sample_t-prev_t,1e-8)[:,None,None]
        ev=float(abs(actual['body_lin_vel'].numpy()-expected_v).max());ea=float(abs(actual['body_accel'].numpy()-expected_a).max())
        if ev>2e-4 or ea>2e-3:raise ValueError(f'analytic derivative check failed: {case}: {ev}, {ea}')
        rows.append({'case':case,'source_hz':30,'control_hz':50,'velocity_max_error_m_s':ev,'acceleration_max_error_m_s2':ea})
    _write_metrics(output,{'status':'completed','static_fixture_before_after_max_errors':errors,'analytic_checks':rows,
        'reference_kinematics_contract':ReferenceMotion.kinematics_contract,
        'physics_source_sha256':sha256(root/'source_executed/pgmt/envs/reference_motion.py'),
        'final_source_sha256':sha256('pgmt/envs/reference_motion.py'),'script_sha256':sha256(__file__),
        'conventions':{'position':'world-frame body/link origins, metres','velocity':'m/s at t',
            'acceleration':'(v(t)-v(max(t-control_dt,0)))/actual elapsed seconds; zero at source start',
            'physical_acceleration':'pre/post control-step world velocities, dt=0.02 s',
            'reward':'-0.001 * sum over four EE and xyz of squared acceleration difference; unchanged'}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','urdf','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.urdf,a.output)
