"""Frozen static-load fixtures and passive evidence capture for short PPO runs."""
import json
from pathlib import Path

import torch

from pgmt.train.diagnostics import cpu_copy
from pgmt.train.fixed_clip_diagnostic import sha256


class FrozenStandingFixture:
    def __init__(self, path, manifest, condition):
        self.path=Path(path);self.condition=condition
        if condition not in ('control','candidate') or manifest.get('standing_training_contract')!='frozen_static_load_v1':
            raise ValueError('invalid frozen standing condition/contract')
        for name,key in (('candidate.json','candidate_sha256'),('starts.pt','starts_sha256')):
            if sha256(self.path/name)!=manifest.get(key):raise ValueError(f'frozen standing {name} fingerprint differs')
        self.starts=torch.load(self.path/'starts.pt',map_location='cpu',weights_only=True)
        self.candidate=json.loads((self.path/'candidate.json').read_text())
        expected={'pose':(29,),'control_target':(29,),'candidate_target':(29,),'root_quat':(4,4),'root_height':(4,)}
        for name,shape in expected.items():
            value=self.starts[name]
            if tuple(value.shape)!=shape or not torch.isfinite(value).all():raise ValueError(f'invalid standing {name}')
        torch.testing.assert_close(self.starts['candidate_target'],torch.tensor(self.candidate['target']),atol=1e-7,rtol=0)
        torch.testing.assert_close(self.starts['pose'],torch.tensor(self.candidate['pose']),atol=0,rtol=0)
        torch.testing.assert_close(self.starts['root_quat'].norm(dim=-1),torch.ones(4),atol=1e-6,rtol=0)
        self.evidence={'condition':condition,'candidate_sha256':manifest['candidate_sha256'],
                       'starts_sha256':manifest['starts_sha256'],'contract':manifest['standing_training_contract']}

    def apply(self, core):
        if core.cfg.reset_mode!='nominal' or core.num_envs%4 or core.cfg.episode_length_s!=10.:
            raise ValueError('standing fixture requires nominal reset, four-start replicas and 10 s horizon')
        target=self.starts[self.condition+'_target'].to(core.device)
        if ((target<=core.joint_low)|(target>=core.joint_high)).any():raise ValueError('standing target outside legal bounds')
        if not (core.pd.kp.eq(80).all() and core.pd.kd.eq(2).all()):raise ValueError('standing PD differs from 80/2')
        mapping=torch.arange(core.num_envs,device=core.device)%4
        core.default_q.copy_(self.starts['pose'].to(core.device).expand_as(core.default_q))
        core.default_root_quat.copy_(self.starts['root_quat'].to(core.device)[mapping])
        core.default_root_pos[:,2]=self.starts['root_height'].to(core.device)[mapping]
        self.evidence['start_indices']=mapping.cpu().tolist()
        return target


class StandingCapture:
    """Observe pre-reset physics and collector outputs without modifying actions."""
    def __init__(self, env, core):
        self.env=env;self.core=core;self.num_envs=core.num_envs;self.device=core.device
        self.frames=[];self.total_steps=0;self.last_physics=None
        self.original_reward=core._reward
        core._reward=self.reward

    def reward(self):
        c=self.core
        state={key:getattr(c,attr) for key,attr in (
            ('joint_pos','qpos'),('joint_vel','qvel'),('root_pos','root_pos'),('root_quat','root_quat'),
            ('root_lin_vel','root_lin_vel'),('root_ang_vel','root_ang_vel'),('body_pos','body_pos'),
            ('body_quat','body_quat'),('body_lin_vel','body_lin_vel'),('body_ang_vel','body_ang_vel'),
            ('contact_forces','contact_forces'),('joint_low','joint_low'),('joint_high','joint_high'))}
        state['body_accel']=(c.body_lin_vel-c._previous_body_lin_vel)/c.cfg.control_dt
        self.last_physics=cpu_copy({'state':state,'reference':c.reference_body,
            'previous_state':{'root_lin_vel':c._previous_root_lin_vel,'body_lin_vel':c._previous_body_lin_vel,
                              'contact_forces':c._previous_contact_forces},
            'action':c.action,'previous_action':c._prev_action,'target':c.target,
            'corrected_velocity':c.reference_root_lin_vel,'recovery_mask':c._recovery_active,
            'reference_frame':c.reference_frame,'episode_steps':c.episode_length_buf,
            'foot_contact':c.foot_contact,'estimated_torque':(torch.zeros_like(c.qpos) if c.articulation is None
                else c.articulation.data.applied_torque[:,c.joint_ids])})
        return self.original_reward()

    def step(self, actions):
        old_frame=self.core.reference_frame.detach().cpu().clone()
        result=self.env.step(actions);self.total_steps+=1
        obs,reward,terminated,timeout,info=result
        p=self.last_physics
        torch.testing.assert_close(p['target'],actions.detach().cpu(),atol=0,rtol=0)
        dt=self.core._torch_reference.frame_time[self.core.reference_seq_idx].detach().cpu()
        torch.testing.assert_close(p['reference_frame'],old_frame+self.core.cfg.control_dt/dt,atol=1e-4,rtol=0)
        self.frames.append(cpu_copy({'physics':p,'next_observations':obs,'reward':reward,
            'terminated':terminated,'timeouts':timeout,'terminal_observation':info.get('terminal_observation'),
            'weighted_aux_terms':info.get('weighted_aux_terms',{}),'tracking':info['tracking']}))
        return result

    def take(self):
        frames=self.frames;self.frames=[]
        return frames


def capture_storage(storage):
    return cpu_copy({key:getattr(storage,key) for key in ('observations','actions','latent_actions','log_probs',
        'values','rewards','next_values','terminated','timeouts','advantages','returns','policy_advantages')})
