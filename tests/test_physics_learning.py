"""Solver compatibility and explicit bounded initialization contracts."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.actuators import validate_actuator_resume
from pgmt.train.fixed_clip_diagnostic import diagnostic_initial_targets
from pgmt.train.policy import Stage1Policy


def test_physics_change_cannot_silently_resume_legacy_checkpoint():
    legacy=G1Env(G1EnvConfig(physics_external_forces_every_iteration=False,physics_min_velocity_iterations=0))
    saved=legacy.state_dict();saved.pop('physics_settings')
    legacy.load_state_dict(saved)
    candidate=G1Env()
    with pytest.raises(ValueError,match='physics'):candidate.load_state_dict(saved)
    cfg=asdict(legacy.cfg);cfg.pop('physics_external_forces_every_iteration');cfg.pop('physics_min_velocity_iterations')
    with pytest.raises(ValueError,match='physics'):validate_actuator_resume({'environment_config':cfg},candidate.cfg)
    validate_actuator_resume({'environment_config':cfg},legacy.cfg)
    candidate.load_state_dict(candidate.state_dict())


@pytest.mark.parametrize('kwargs',[
    {'physics_min_velocity_iterations':-1},{'physics_min_velocity_iterations':256},
    {'physics_min_velocity_iterations':1.5},{'physics_min_velocity_iterations':True},
    {'physics_external_forces_every_iteration':1}])
def test_invalid_solver_settings(kwargs):
    with pytest.raises(ValueError,match='physics'):G1EnvConfig(**kwargs)


def test_clip_initialization_is_fixed_legal_bias_with_no_new_action_transform():
    core=G1Env();core.cfg.reset_mode='reference_state'
    core.reference_qpos[:]=core.joint_high
    schedule=SimpleNamespace(clips=[{}])
    targets=diagnostic_initial_targets(core,schedule,'clip_start')
    assert ((targets>core.joint_low)&(targets<core.joint_high)).all()
    policy=Stage1Policy(initial_joint_targets=targets)
    torch.testing.assert_close(policy._target(policy.actor.mlp.net[-1].bias),targets,atol=2e-6,rtol=1e-6)
    assert policy.action_contract=='joint_targets_tanh_v2'
    assert diagnostic_initial_targets(core,schedule,'zero') is None
    with pytest.raises(ValueError,match='one fixed clip'):
        diagnostic_initial_targets(core,SimpleNamespace(clips=[{},{}]),'clip_start')
    core.cfg.reset_mode='nominal'
    with pytest.raises(ValueError,match='reference_state'):diagnostic_initial_targets(core,schedule,'clip_start')
