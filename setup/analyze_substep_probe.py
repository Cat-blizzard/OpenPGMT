"""Compare raw 5-ms PhysX traces without hiding overspeed in observations."""
import argparse
import json
from pathlib import Path

import torch

from pgmt.train.fixed_clip_diagnostic import sha256


def read(path):return json.loads(Path(path).read_text())


def summarize(path):
    m=read(path/'metrics.json')
    if m['status']!='completed' or len(m['steps'])!=m['arguments']['steps']:raise ValueError('incomplete probe')
    t=torch.load(path/'substeps.pt',map_location='cpu',weights_only=False)
    frames=t['frames'];q=torch.stack([f['qpos'] for f in frames]);v=torch.stack([f['qvel'] for f in frames])
    fd=torch.stack([f['qpos_fd'] for f in frames]);contact=torch.stack([f['contact_forces'] for f in frames])
    names=t['joint_names'];per_joint=v.abs().amax((0,1));order=per_joint.argsort(descending=True)
    first_ends=m['first_episode_ends'];duration=[x['duration_s'] if x else m['arguments']['steps']*.02 for x in first_ends]
    end_all=[e for s in m['steps'] for e in s['ended']]
    result={'path':str(path),'physics':m['physics_config'],'substeps':len(frames),
        'max_cache_q_error':m['max_cache_q_error'],'max_cache_v_error':m['max_cache_v_error'],
        'max_abs_velocity':float(v.abs().max()),'max_abs_position_fd':float(fd.abs().max()),
        'p99_abs_velocity':float(torch.quantile(v.abs().flatten(),.99)),
        'substep_envs_over45':int((v.abs().amax(-1)>45).sum()),
        'end_control_envs_over45':int((v[3::4].abs().amax(-1)>45).sum()),
        'velocity_minus_fd_rms':float((v-fd).square().mean().sqrt()),
        'contact_force_peak':float(contact.norm(dim=-1).max()),
        'joint_peaks':{names[j]:float(per_joint[j]) for j in order.tolist()},
        'mean_observed_first_episode_s':sum(duration)/len(duration),
        'first_episode_durations':duration,'first_episode_censored':sum(x is None for x in first_ends),
        'overspeed_terminations':sum('joint_speed' in x['reasons'] for x in end_all),
        'actions_sha256':sha256(path/'actions.pt'),'trace_sha256':sha256(path/'substeps.pt')}
    i,j,k=[int(x) for x in torch.unravel_index(v.abs().argmax(),v.shape)]
    f=frames[i]
    result['peak_event']={'trace_index':i,'step':f['step'],'substep':f['substep'],'env_id':j,'joint':names[k],
        'qvel':float(v[i,j,k]),'qpos_fd':float(fd[i,j,k]),'qpos':float(q[i,j,k]),
        'target':float(f['target'][j,k]),'estimated_drive_torque':float(f['estimated_drive_torque'][j,k]),
        'episode_age_s':float((f['episode_step_before'][j]+f['substep']/4)*.02),
        'nearby_qvel':v[max(0,i-4):i+5,j,k].tolist(),'nearby_qpos_fd':fd[max(0,i-4):i+5,j,k].tolist()}
    return result,m,q,v


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[];base=None
    for path in a.paths:
        row,m,q,v=summarize(path)
        tape=torch.load(path/'actions.pt',map_location='cpu',weights_only=False)['actions']
        if base is None:base=(m,q,v,tape)
        else:
            if not torch.equal(tape,base[3]):raise ValueError('action tapes differ')
            # Solver fields were made explicit in G1EnvConfig after the first
            # probe series. They are the experimental variables, not controls.
            solver_fields={'physics_external_forces_every_iteration','physics_min_velocity_iterations'}
            controls=lambda cfg:{k:v for k,v in cfg.items() if k not in solver_fields}
            if controls(m['environment_config'])!=controls(base[0]['environment_config']):raise ValueError('environment contracts differ')
            if m['initial']!=base[0]['initial']:raise ValueError('initial states differ')
            row.update(action_tape_identical=True,initial_state_identical=True,
                qpos_trace_equal_to_baseline=torch.equal(q,base[1]),qvel_trace_equal_to_baseline=torch.equal(v,base[2]))
        rows.append(row)
    with a.output.open('x') as f:json.dump({'schema':'pgmt_substep_comparison_v1','runs':rows},f,indent=2);f.write('\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
