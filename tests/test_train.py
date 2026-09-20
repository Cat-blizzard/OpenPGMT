"""Regression tests for the simulator-independent Stage 1 training pieces."""

from dataclasses import replace

import torch

from pgmt.cfg.assumptions import get
from pgmt.train.ppo import PPO
from pgmt.train.storage import RolloutStorage
from pgmt.train.policy import Stage1Policy


def _small_observations(batch=1):
    return {
        "obs": torch.zeros(batch, 1),
        "history": torch.zeros(batch, 1),
        "future": torch.zeros(batch, 1),
        "privileged": torch.zeros(batch, 1),
    }


def test_rollout_timeout_bootstraps_but_does_not_carry_gae():
    storage = RolloutStorage(
        num_steps=2,
        num_envs=1,
        observation_shapes={key: tuple(value.shape[1:])
                            for key, value in _small_observations().items()},
        action_dim=1,
        num_heads=3,
    )
    observations = _small_observations()
    zeros = torch.zeros(1, 3)
    storage.add(observations, torch.zeros(1, 1), torch.zeros(1), zeros,
                zeros, torch.zeros(1, dtype=torch.bool),
                torch.ones(1, dtype=torch.bool), torch.full((1, 3), 2.0))
    storage.add(observations, torch.zeros(1, 1), torch.zeros(1), zeros,
                torch.ones(1, 3), torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool), zeros)
    storage.compute_returns(gamma=1.0, lam=1.0)

    # The first transition uses its reset-before timeout value (2), while the
    # following episode's reward (1) cannot leak backward through the timeout.
    assert torch.equal(storage.advantages[:, 0], torch.tensor([[2., 2., 2.], [1., 1., 1.]]))
    assert torch.equal(storage.returns[:, 0], storage.advantages[:, 0])


def test_stage1_policy_wrapper_preserves_contract_shapes():
    policy = Stage1Policy()
    observations = {
        "obs": torch.zeros(1, 96),
        "history": torch.zeros(1, 10, 96),
        "future": torch.zeros(1, 6, 61),
        "privileged": torch.zeros(1, 48),
    }
    output = policy.act(observations, deterministic=True)
    assert output.actions.shape == (1, 29)
    assert output.log_probs.shape == output.entropy.shape == (1,)
    assert output.values.shape == (1, 3)
    assert torch.isfinite(torch.cat((output.actions, output.values), dim=-1)).all()


class _TinyPolicy(torch.nn.Module):
    """Small policy used to exercise PPO's environment protocol."""

    head_names = ("upper", "lower", "aux")

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def _output(self, observations, actions=None):
        n = observations["obs"].shape[0]
        if actions is None:
            actions = self.bias.expand(n, 29)
        values = self.bias.expand(n, 3)
        return type("Output", (), {
            "actions": actions,
            "log_probs": torch.zeros(n),
            "values": values,
            "entropy": torch.ones(n),
        })()

    def act(self, observations, deterministic=False):
        return self._output(observations)

    def value(self, observations):
        return self.bias.expand(observations["obs"].shape[0], 3)

    def evaluate_actions(self, observations, actions):
        return self._output(observations, actions)


class _TinyEnv:
    def __init__(self, observations):
        self.observations = observations

    def step(self, actions):
        n = actions.shape[0]
        # Deliberately return CPU rewards; PPO must normalize them to its
        # policy device before accumulating the rollout metrics.
        rewards = torch.ones(n, 3)
        terminated = torch.zeros(n, dtype=torch.bool)
        truncated = torch.zeros(n, dtype=torch.bool)
        return self.observations, rewards, terminated, truncated, {}


class _GaussianTinyPolicy(torch.nn.Module):
    """log_prob 依赖可训练参数的最小策略，用于观测 PPO 更新幅度。"""

    head_names = ("upper", "lower", "aux")

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(1, 29))
        self.log_std = torch.nn.Parameter(torch.zeros(1, 29))

    def _dist(self, n):
        return torch.distributions.Normal(self.mean.expand(n, 29),
                                          self.log_std.exp().expand(n, 29))

    def act(self, observations, deterministic=False):
        n = observations["obs"].shape[0]
        dist = self._dist(n)
        actions = dist.mean if deterministic else dist.sample()
        return _policy_output(actions, dist.log_prob(actions).sum(-1),
                              torch.zeros(n, 3), dist.entropy().sum(-1))

    def value(self, observations):
        return torch.zeros(observations["obs"].shape[0], 3)

    def evaluate_actions(self, observations, actions):
        n = actions.shape[0]
        dist = self._dist(n)
        return _policy_output(actions, dist.log_prob(actions).sum(-1),
                              torch.zeros(n, 3), dist.entropy().sum(-1))


def _policy_output(actions, log_probs, values, entropy):
    from types import SimpleNamespace
    return SimpleNamespace(actions=actions, log_probs=log_probs,
                           values=values, entropy=entropy)


def test_single_minibatch_update_reports_post_update_kl():
    """单一巨批（1 epoch × 1 minibatch）下 pre-step 的 KL 恒为 0，必须靠
    更新后重估的 ``approx_kl_post`` 观测策略是否真的移动（2026-09-20 GPU8
    256env 验证曾因此被误读为 actor 冻结）。"""
    torch.manual_seed(0)
    policy = _GaussianTinyPolicy()
    config = replace(get("A6").value, num_steps_per_env=4,
                     num_learning_epochs=1, num_mini_batches=1)

    class _RandomRewardEnv:
        def __init__(self, n):
            self.n = n
            self.observations = _small_observations(n)

        def step(self, actions):
            n = actions.shape[0]
            rewards = torch.randn(n, 3)
            terminated = torch.zeros(n, dtype=torch.bool)
            truncated = torch.zeros_like(terminated)
            return self.observations, rewards, terminated, truncated, {}

    observations = _small_observations(3)
    ppo = PPO(policy, config=config, total_updates=1)
    observations, _ = ppo.collect_rollout(_RandomRewardEnv(3), observations)
    metrics = ppo.update()
    assert metrics["approx_kl"] == 0.0
    assert metrics["approx_kl_post"] > 0.0


def test_ppo_collect_and_update_with_environment_protocol():
    observations = _small_observations(batch=2)
    config = replace(get("A6").value, num_steps_per_env=2,
                     num_learning_epochs=1, num_mini_batches=1)
    ppo = PPO(_TinyPolicy(), config=config)
    next_observations, collected = ppo.collect_rollout(
        _TinyEnv(observations), observations)
    assert next_observations is observations
    assert collected["steps"] == 4
    updated = ppo.update()
    assert updated["num_updates"] == 1
    assert torch.isfinite(torch.tensor(list(updated.values()), dtype=torch.float32)).all()
