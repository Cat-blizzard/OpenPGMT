"""v4 curriculum, termination coordinates, value completion and LR resume."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from pgmt.cfg.assumptions import get
from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import MockStage1Env, _lr_schedule_horizon


def test_local_tracking_termination_separates_world_drift_from_body_deformation():
    env = G1Env()
    env._body_state_available = True
    local = torch.zeros_like(env.body_pos)
    local[..., 0] = torch.linspace(-.4, .4, local.shape[1])
    env.body_pos.copy_(local + env.root_pos[:, None])
    env.reference_body = {"body_pos": env.body_pos.clone(), "root_pos": env.root_pos.clone(),
                          "root_quat": env.root_quat.clone()}
    # Robot is translated and yaw-rotated as one rigid configuration.
    angle = .7
    rot = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                        [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    env.root_pos[:, :2] += torch.tensor([3., 2.])
    env.root_quat[:] = torch.tensor([math.cos(angle/2), 0., 0., math.sin(angle/2)])
    env.body_pos.copy_(local @ rot.T + env.root_pos[:, None])
    for _ in range(30):
        assert not env._dones()[0].any()
    assert not env._last_termination_reasons["ref_deviation"].any()
    # Deforming every body relative to its root must still terminate.
    env.body_pos[..., 2] += .7
    for _ in range(30):
        terminated, _ = env._dones()
    assert terminated.all()
    assert env._last_termination_reasons["ref_deviation"].all()


def test_simultaneous_failure_reasons_are_captured_before_auto_reset():
    env = G1Env(G1EnvConfig(num_envs=2))
    env.root_pos[:, 2] = .1
    env.root_quat[:] = torch.tensor([0., 1., 0., 0.])
    env.qvel[:] = 100.
    terminated, _ = env._dones()
    info = env.episode_diagnostics()
    assert terminated.all()
    for name in ("root_low", "tilted", "joint_speed"):
        assert info["termination_reasons"][name].all()
    env.reset()
    assert info["termination_reasons"]["root_low"].all()


def test_lr_stop_point_does_not_shorten_schedule_and_resume_cannot_change_it():
    assert _lr_schedule_horizon(20) == 1000
    saved = {"ppo": {"total_updates": 4000}}
    assert _lr_schedule_horizon(200, checkpoint=saved) == 4000
    with pytest.raises(ValueError, match="preserve"):
        _lr_schedule_horizon(200, 1000, saved)
    with pytest.raises(ValueError, match="cover"):
        _lr_schedule_horizon(200, 20)


def fitted_policy(complete):
    torch.manual_seed(23)
    policy = Stage1Policy()
    cfg = replace(get("A6").value, num_steps_per_env=4, num_learning_epochs=3,
                  num_mini_batches=2, learning_rate=.001, target_kl=.0005,
                  complete_critic_epochs=complete)
    ppo = PPO(policy, cfg, total_updates=1000)
    env = MockStage1Env(4, torch.device("cpu"), horizon=32)
    obs, _ = ppo.collect_rollout(env, env.reset())
    metrics = ppo.update()
    return ppo, env, obs, metrics


def test_critic_completion_preserves_actor_and_shared_encoder_exactly():
    joint, _, _, baseline = fitted_policy(False)
    completed, _, _, result = fitted_policy(True)
    assert result["critic_only_steps"] > 0
    assert result["optimizer_steps"] == baseline["optimizer_steps"]
    assert result["value_optimizer_steps"] == 6
    assert result["exact_kl"] == baseline["exact_kl"]
    for key, value in joint.policy.state_dict().items():
        if not key.startswith("critic."):
            assert torch.equal(value, completed.policy.state_dict()[key]), key
    assert sum(result["value_mse_raw_post_"+h] for h in joint.policy.head_names) < sum(
        result["value_mse_raw_joint_"+h] for h in joint.policy.head_names)
    for name, parameter in completed.policy.named_parameters():
        if parameter in completed.optimizer.state:
            expected = result["value_optimizer_steps"] if name.startswith("critic.") else result["optimizer_steps"]
            assert completed.optimizer.state[parameter]["step"].item() == expected


def test_critic_completion_checkpoint_replays_next_update():
    ppo, env, obs, _ = fitted_policy(True)
    saved = deepcopy(ppo.state_dict())
    ppo.collect_rollout(env, obs)
    expected = ppo.update()
    restored = PPO(Stage1Policy(), ppo.config, total_updates=1000)
    restored.load_state_dict(saved)
    replay_env = MockStage1Env(4, torch.device("cpu"), horizon=32)
    restored.collect_rollout(replay_env, replay_env.reset())
    actual = restored.update()
    assert actual == expected
    for key, value in ppo.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[key])
