"""Regression coverage for production v3 scaling and transactional KL updates."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.distributions import Normal

from pgmt.cfg.assumptions import get
from pgmt.train.policy import PolicyOutput, Stage1Policy, Stage2Policy
from pgmt.train.ppo import PPO, scaled_value_errors
from pgmt.train.train_stage1 import _zeros_observations


class SmallNormalPolicy(torch.nn.Module):
    head_names = ("upper", "lower", "aux")

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(0.))
        self.critic = torch.nn.Parameter(torch.zeros(3))

    def latent_distribution(self, obs):
        mean = (self.mean * obs["obs"]).expand(-1, 29)
        return Normal(mean, torch.ones_like(mean))

    def value(self, obs):
        return self.critic.expand(obs["obs"].shape[0], -1)

    def evaluate_actions(self, obs, actions):
        dist = self.latent_distribution(obs)
        return PolicyOutput(actions, dist.log_prob(actions).sum(-1),
                            self.value(obs), dist.entropy().sum(-1))

    def act(self, obs):
        return self.evaluate_actions(obs, self.latent_distribution(obs).sample())


class SmallEnv:
    def __init__(self):
        self.obs = {k: torch.ones(8, 1) for k in ("obs", "history", "future", "privileged")}

    def step(self, actions):
        done = torch.zeros(8, dtype=torch.bool)
        return self.obs, actions.mean(-1, keepdim=True).expand(-1, 3), done, done, {}


def prepared_ppo(**overrides):
    torch.manual_seed(12)
    cfg = replace(get("A6").value, num_steps_per_env=4, num_mini_batches=2,
                  num_learning_epochs=3, learning_rate=1., **overrides)
    ppo = PPO(SmallNormalPolicy(), cfg)
    env = SmallEnv()
    ppo.collect_rollout(env, env.obs)
    return ppo


@pytest.mark.parametrize("policy_cls", [Stage1Policy, Stage2Policy])
def test_fresh_actor_is_near_default_posture_with_small_noise(policy_cls):
    torch.manual_seed(0)
    policy = policy_cls()
    obs = _zeros_observations(2, torch.device("cpu"))
    obs["elevation"] = torch.zeros(2, get("A8").value.map_size, get("A8").value.map_size)
    out = policy.act(obs, deterministic=True)
    assert out.actions.abs().max() < .005
    assert torch.allclose(policy.log_std.exp(), torch.full((29,), .1))


def test_value_loss_is_invariant_to_per_head_units_and_clips_in_scaled_units():
    values = torch.tensor([[4., 8.], [8., 4.]])
    old = torch.zeros_like(values)
    returns = torch.full_like(values, 10.)
    scales = torch.tensor([2., 4.])
    actual = scaled_value_errors(values, old, returns, scales, .2)
    assert torch.allclose(actual, torch.tensor([4.8**2, 2.3**2]))
    units = torch.tensor([10., 1000.])
    assert torch.allclose(actual, scaled_value_errors(values*units, old*units,
                                                      returns*units, scales*units, .2))


def test_kl_guard_rejects_large_steps_without_advancing_adam_state():
    ppo = prepared_ppo()
    before = ppo.policy.mean.detach().clone()
    result = ppo.update()
    assert result["kl_backtracks"] > 0
    assert result["optimizer_steps"] > 0
    assert not torch.equal(ppo.policy.mean.detach(), before)
    assert 0 < result["exact_kl"] <= .02
    assert result["max_accepted_kl"] <= .02
    for state in ppo.optimizer.state.values():
        assert state["step"].item() == result["optimizer_steps"]


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_rejected_update_restores_populated_adam_moments_and_weights():
    ppo = prepared_ppo()
    ppo.update()  # Establish nonzero Adam moments first.
    ppo.config = replace(ppo.config, target_kl=1e-12, max_kl_backtracks=0)
    env = SmallEnv()
    ppo.collect_rollout(env, env.obs)
    # update() reapplies this scheduled LR before proposing any step.
    for group in ppo.optimizer.param_groups:
        group["lr"] = ppo.config.learning_rate
    weights = deepcopy(ppo.policy.state_dict())
    adam = deepcopy(ppo.optimizer.state_dict())
    result = ppo.update()
    assert result["optimizer_steps"] == 0
    assert result["kl_backtracks"] == 1
    assert result["early_stopped"]
    assert result["exact_kl"] == 0
    assert_tree_equal(weights, ppo.policy.state_dict())
    assert_tree_equal(adam, ppo.optimizer.state_dict())


def test_full_kl_weights_short_final_chunk_and_includes_outlying_state():
    ppo = prepared_ppo(kl_chunk_size=7)
    ppo.storage.observations["obs"][-1, -1] = 10.
    old = ppo._snapshot_behavior()
    with torch.no_grad():
        ppo.policy.mean.fill_(1.)
    mean, maximum = ppo._exact_kl(old)
    # 31 states with a mean shift of 1, one with a shift of 10, in 29 dims.
    assert mean == pytest.approx(29 * .5 * (31 + 100) / 32)
    assert maximum == pytest.approx(29 * .5 * 100)


def test_legacy_reward_checkpoint_is_rejected_before_mutating_policy():
    ppo = PPO(SmallNormalPolicy())
    checkpoint = deepcopy(ppo.state_dict())
    del checkpoint["training_contract"]
    checkpoint["policy"]["mean"].fill_(123.)
    with pytest.raises(ValueError, match="training/reward contract"):
        ppo.load_state_dict(checkpoint)
    assert ppo.policy.mean.item() == 0.


def test_grouped_rollout_metrics_use_pre_reset_recovery_and_duration():
    class GroupEnv(SmallEnv):
        def step(self, actions):
            obs, reward, terminated, timeout, _ = super().step(actions)
            reward = torch.arange(8.).view(8, 1).expand(-1, 3)
            recovering = torch.arange(8) >= 4
            terminated[0] = True
            # Do not alias timeouts to terminations in this test double.
            timeout = torch.zeros_like(terminated)
            info = {"recovery_active": recovering, "episode_elapsed_s": torch.full((8,), .4),
                    "physics": {"contact_force_max_n": torch.arange(8.) * 10}}
            return obs, reward, terminated, timeout, info
    env = GroupEnv()
    ppo = PPO(SmallNormalPolicy())
    _, metrics = ppo.collect_rollout(env, env.obs, num_steps=1)
    ordinary, recovery = (metrics["episode_groups"][x] for x in ("ordinary", "recovery"))
    assert ordinary["steps"] == recovery["steps"] == 4
    assert ordinary["terminations"] == 1 and recovery["terminations"] == 0
    assert ordinary["ended_episode_duration_mean_s"] == pytest.approx(.4)
    assert ordinary["mean"]["reward_aux"] == 1.5
    assert recovery["mean"]["reward_aux"] == 5.5
    assert recovery["peak_abs"]["physics/contact_force_max_n"] == 70.
