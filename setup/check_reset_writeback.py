"""Read the 12 tested reset states directly from PhysX, without stepping."""
import json
from pathlib import Path
import sys
import traceback

import torch

from pgmt.train.train_stage1 import _seed_everything, _write_metrics, build_env
from pgmt.train.diagnostics import cpu_copy


def main():
    root = Path(sys.argv[1])
    output = root / 'reset_writeback_verified' / 'writeback.json'
    if output.exists():
        raise ValueError('refusing to overwrite writeback audit')
    records = [json.loads((root / f'reference_state_seed{seed}' / 'response.json').read_text()) for seed in (17,18,19)]
    args = records[0]['arguments']
    _seed_everything(17)
    from pgmt.envs.isaac_app import launch_isaac_app
    from pgmt.envs.g1_env import _quat_to_mat
    app, env = launch_isaac_app('cuda:0'), None
    result = {'status':'running', 'physics_steps':0, 'source':'direct root_physx_view getters, bypassing ArticulationData', 'batches':[]}
    try:
        env = build_env(4, torch.device('cuda:0'), backend='isaaclab', app=app,
            reference_data=args['reference_data'], asset_path=args['asset'], reference_urdf=args['urdf'],
            env_options={'seed':17, 'enable_adaptive_sampling':False, 'randomize_dynamics':False,
                         'corrupt_observations':False, 'max_action_delay':0, 'reset_mode':'reference_state'})
        core = env.env.core
        for record in records:
            initial = record['initial']
            core._forced_reference.update({i:(int(s),float(f)) for i,(s,f) in enumerate(
                zip(initial['reference_seq_idx'], initial['reference_frame']))})
            env.reset()
            view = core.articulation.root_physx_view
            pose = view.get_root_transforms().clone()
            q = view.get_dof_positions()[:, core.joint_ids].clone()
            qd = view.get_dof_velocities()[:, core.joint_ids].clone()
            vel_com = view.get_root_velocities().clone()
            quat = pose[:, [6,3,4,5]]  # PhysX xyzw -> repository wxyz
            com_offset = view.get_coms().to(core.device)[:,0,:3]
            offset_world = torch.bmm(_quat_to_mat(quat), com_offset.unsqueeze(-1)).squeeze(-1)
            linear = vel_com[:,:3] - torch.linalg.cross(vel_com[:,3:], offset_world)
            observed = {'qpos':q, 'qvel':qd, 'root_pos':pose[:,:3], 'root_quat':quat,
                        'root_lin_vel':linear, 'root_ang_vel':vel_com[:,3:]}
            expected = {k:getattr(core,k) for k in observed}
            # Align quaternion signs before comparing identical rotations.
            observed['root_quat'] = torch.where((quat*expected['root_quat']).sum(-1,keepdim=True)<0,-quat,quat)
            errors = {k:(v-expected[k]).abs().max().item() for k,v in observed.items()}
            result['batches'].append(cpu_copy({'seed':record['arguments']['seed'], 'reference_seq_idx':core.reference_seq_idx,
                'reference_frame':core.reference_frame, 'errors':errors, 'physx_state':observed,
                'expected_state':expected, 'physx_com_velocity':vel_com, 'physx_com_offset_link':com_offset}))
            _write_metrics(output,result)
            if max(errors.values())>2e-5:
                raise RuntimeError(f'direct PhysX reset mismatch: {errors}')
        result['status']='completed'
        _write_metrics(output,result)
    except BaseException:
        result['status']='failed'
        result['error']=traceback.format_exc()
        traceback.print_exc()
        _write_metrics(output,result)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
