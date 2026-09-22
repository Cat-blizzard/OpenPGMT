"""CPU verification and plots for the bounded standing-action replay."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pgmt.envs.g1_env import G1_JOINT_NAMES
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.train_stage1 import _write_metrics


def run(source, output):
    if output.with_suffix('.json').exists(): raise FileExistsError(output)
    rows=[];fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ri,(condition,step,eid) in enumerate([('control',54,11),('candidate',274,5)]):
        folder=source/condition;m=json.loads((folder/'metrics.json').read_text())
        assert m['status']=='completed' and m['ppo_updates']==0 and m['transitions']==4800
        assert all(r['max_q_error_rad']==r['max_root_error_m']==r['max_substep_velocity_error_rad_s']==0 and r['reset_match'] for r in m['steps'])
        data=torch.load(folder/'substeps.pt',map_location='cpu',weights_only=False);fs=data['frames']
        assert len(fs)==1200
        j=G1_JOINT_NAMES.index('left_ankle_roll');fidx=data['sensor_body_names'].index('left_ankle_roll_link')
        torque_limit=torch.tensor(m['environment_config']['torque_limit'])
        kp=torch.tensor(m['environment_config']['stiffness']);kd=torch.tensor(m['environment_config']['damping'])
        max_torque_error=0.
        for f in fs:
            predicted=(kp*(f['target']-f['qpos_before'])-kd*f['qvel_before']).clamp(-torque_limit,torque_limit)
            torch.testing.assert_close(predicted,f['estimated_drive_torque'],atol=2e-4,rtol=2e-5)
            max_torque_error=max(max_torque_error,float((predicted-f['estimated_drive_torque']).abs().max()))
        context=[]
        for f in fs[(step-3)*4:(step+2)*4]:
            norm=f['contact_forces'][eid].norm(dim=-1)
            context.append({'control_step':f['step'],'substep':f['substep'],
                'time_relative_to_control_start_s':((f['step']-step)*4+f['substep'])*.005,
                'q_rad':float(f['qpos'][eid,j]),'q_before_rad':float(f['qpos_before'][eid,j]),
                'v_rad_s':float(f['qvel'][eid,j]),'v_before_rad_s':float(f['qvel_before'][eid,j]),
                'position_difference_velocity_rad_s':float(f['qpos_fd'][eid,j]),
                'target_rad':float(f['target'][eid,j]),'pd_estimate_nm':float(f['estimated_drive_torque'][eid,j]),
                'left_foot_force_norm_n':float(norm[fidx]),
                'left_foot_force_w_n':f['contact_forces'][eid,fidx].tolist(),
                'root_height_m':float(f['root_pos'][eid,2]),
                'root_vertical_velocity_m_s':float(f['root_lin_vel'][eid,2]),
                'contact_bodies_over_1n':{name:float(norm[k]) for k,name in enumerate(data['sensor_body_names']) if norm[k]>1}})
        peak=max(context,key=lambda r:abs(r['v_rad_s']))
        rows.append({'condition':condition,'transitions':4800,'ppo_updates':0,'matched_control_steps':300,
            'exact_substep_velocity_match':True,'raw_cache_contact_timestamps_checked_in_replay':True,
            'pd_estimate_max_formula_error_nm':max_torque_error,'peak':peak,'context':context,
            'substeps_sha256':sha256(folder/'substeps.pt'),'metrics_sha256':sha256(folder/'metrics.json')})
        x=[r['time_relative_to_control_start_s']*1000 for r in context]
        axes[ri,0].plot(x,[r['v_rad_s'] for r in context],label='raw velocity')
        axes[ri,0].plot(x,[r['position_difference_velocity_rad_s'] for r in context],label='position difference / dt')
        axes[ri,1].plot(x,[r['left_foot_force_norm_n'] for r in context],label='left foot contact')
        axes[ri,2].plot(x,[r['pd_estimate_nm'] for r in context],label='pre-substep PD estimate')
        for ax,title in zip(axes[ri],('Velocity (rad/s)','Contact norm (N)','Estimated drive torque (Nm)')):
            ax.set_title(condition+' | '+title);ax.set_xlabel('Time from event control-step start (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    release=json.loads((source/'gpu_release.json').read_text());assert not release['task_pids_alive']
    result={'status':'completed','physical_transitions':9600,'ppo_updates':0,'environment_substeps':38400,
        'conditions':rows,'gpu_release':release,'source_sha256':sha256(__file__),
        'interpretation':'Exact replay and raw/cache/contact consistency rule out an action/readback mismatch in these runs. '
            'Peaks coincide with falling foot contact; large PD estimates follow the peaks. '
            'End-of-substep velocities differ from position increments, including a sign reversal; '
            'the observations do not establish exact physical impulse fidelity or measured solver drive torque.',
        'training_decision':'No demonstrated interface bug requiring a reward/PD/termination change. '
            'Retain transient monitoring for a separately authorized bounded learning comparison; no training started here.'}
    _write_metrics(output.with_suffix('.json'),result)
    fig.suptitle('Identical saved-action replay | 9600 transitions, no learning')
    fig.savefig(output.with_suffix('.png'),dpi=150);plt.close(fig)
    print('All 600 control steps / 2400 substeps match; PD estimates verified; GPU released.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
