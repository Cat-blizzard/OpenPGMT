"""CPU replay of captured on-policy samples; no simulator or behavior claims."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pgmt.cfg.assumptions import get
from pgmt.train.diagnostics import cpu_copy
from pgmt.train.fixed_clip_diagnostic import sha256
from pgmt.train.policy import PolicyOutput, Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import _write_metrics


def independent_gae(rewards, values, next_values, terminated, timeouts, gamma, lam):
    """Float64 NumPy reference with explicit per-environment episode breaks."""
    result = np.zeros_like(rewards, dtype=np.float64)
    for env in range(rewards.shape[1]):
        carry = np.zeros(rewards.shape[-1], dtype=np.float64)
        for t in range(len(rewards) - 1, -1, -1):
            bootstrap = np.zeros_like(carry) if terminated[t, env] else next_values[t, env]
            if terminated[t, env] or timeouts[t, env]:
                carry = np.zeros_like(carry)
            carry = rewards[t, env] + gamma * bootstrap - values[t, env] + gamma * lam * carry
            result[t, env] = carry
    return result


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    saved = torch.load(source / 'rollout.pt', weights_only=False, map_location='cpu')
    frames = saved['frames']
    if saved['schema'] != 'pgmt_static_rollout_v1':
        raise ValueError('unknown capture')
    policy = Stage1Policy().eval()
    policy.load_state_dict(torch.load(source / 'initial_policy.pt', weights_only=True, map_location='cpu'))
    ppo = PPO(policy)
    errors = {'action': 0., 'log_prob': 0., 'value': 0., 'next_value': 0.}
    with torch.no_grad():
        for frame in frames:
            actual = policy.evaluate_actions(frame['observations'], frame['actions'], latent_actions=frame['latent_actions'])
            errors['action'] = max(errors['action'], float((policy._target(frame['latent_actions']) - frame['actions']).abs().max()))
            errors['log_prob'] = max(errors['log_prob'], float((actual.log_probs - frame['log_probs']).abs().max()))
            errors['value'] = max(errors['value'], float((actual.values - frame['values']).abs().max()))

    class Replay:
        index = 0

        def act(self, observations):
            frame = frames[self.index]
            for key, value in observations.items():
                torch.testing.assert_close(value, frame['observations'][key], atol=0, rtol=0)
            return PolicyOutput(frame['actions'], frame['log_probs'], frame['values'],
                                torch.zeros_like(frame['log_probs']), frame['latent_actions'])

        def step(self, actions):
            frame = frames[self.index]
            torch.testing.assert_close(actions, frame['actions'], atol=0, rtol=0)
            self.index += 1
            info = {}
            if frame['terminal_observation'] is not None:
                info['terminal_observation'] = frame['terminal_observation']
            return frame['next_observations'], frame['rewards'], frame['terminated'], frame['timeouts'], info

    replay = Replay()
    original_act = policy.act
    policy.act = replay.act  # frozen samples; production collector still computes bootstrap/GAE
    try:
        _, metrics = ppo.collect_rollout(replay, frames[0]['observations'], num_steps=len(frames))
    finally:
        policy.act = original_act
    s = ppo.storage
    next_saved = torch.stack([f['next_values'] for f in frames])
    errors['next_value'] = float((s.next_values - next_saved).abs().max())
    gae = independent_gae(*(getattr(s, k).numpy().astype(np.float64) for k in
        ('rewards', 'values', 'next_values')), s.terminated.numpy(), s.timeouts.numpy(),
        ppo.config.gamma, ppo.config.lam)
    reference = torch.tensor(gae, dtype=torch.float32)
    torch.testing.assert_close(s.advantages, reference, atol=2e-4, rtol=2e-5)
    flat = gae.reshape(-1, 3)
    normalized = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    torch.testing.assert_close(s.policy_advantages.flatten(),
        torch.tensor(normalized.mean(-1), dtype=torch.float32), atol=2e-5, rtol=2e-5)
    if errors['action'] > 2e-6 or errors['log_prob'] > 2e-4 or max(errors['value'], errors['next_value']) > 2e-4:
        raise ValueError(f'capture/CPU replay mismatch: {errors}')

    obs = {k: v.flatten(0, 1) for k, v in s.observations.items()}
    actions, latent = s.actions.flatten(0, 1), s.latent_actions.flatten(0, 1)
    old_logp = s.log_probs.flatten()
    evaluated = policy.evaluate_actions(obs, actions, latent_actions=latent)
    ratio = (evaluated.log_probs - old_logp).exp()
    parameters = [(n, p) for n, p in policy.named_parameters() if not n.startswith('critic.')]
    gradients = []
    per_module = []
    for h in range(3):
        loss = -(ratio * torch.tensor(normalized[:, h], dtype=torch.float32)).mean()
        grad = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        pieces = [torch.zeros_like(p).flatten() if g is None else g.flatten() for (_, p), g in zip(parameters, grad)]
        gradients.append(torch.cat(pieces).detach())
        per_module.append({prefix: float(torch.stack([g.square().sum() for (n, _), g in zip(parameters, pieces)
            if n.startswith(prefix)]).sum().sqrt()) for prefix in ('actor.', 'history_encoder.', 'ifm.', 'log_std')})
    vectors = torch.stack(gradients)
    norms = vectors.norm(dim=1)
    cosine = (vectors @ vectors.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
    if not torch.isfinite(vectors).all() or (norms <= 0).any():
        raise RuntimeError('missing or non-finite actor learning signal')
    del evaluated, ratio, gradients, vectors
    # One isolated CPU optimizer replay on a copy of the captured initial
    # policy. This policy is never deployed in the simulator or resumed.
    torch.manual_seed(12345)
    before = cpu_copy(policy.state_dict())
    advantages = s.policy_advantages.flatten().clone()
    update = ppo.update()
    with torch.no_grad():
        after_logp = policy.evaluate_actions(obs, actions, latent_actions=latent).log_probs
        delta = after_logp - old_logp
        surrogate_before = float(advantages.mean())
        surrogate_after = float(((delta.exp()).clamp(1-ppo.config.clip_param, 1+ppo.config.clip_param) * advantages
                                  ).minimum(delta.exp() * advantages).mean())
    torch.save({'scope': 'isolated CPU replay, not a training checkpoint', 'before': before,
                'after': cpu_copy(policy.state_dict()), 'optimizer': cpu_copy(ppo.optimizer.state_dict()),
                'update_metrics': update}, output / 'cpu_optimizer_replay.pt')
    result = {'status': 'completed', 'scope': 'real static stochastic samples; one isolated CPU PPO update, no behavior evaluation',
        'source_rollout_sha256': sha256(source / 'rollout.pt'), 'source_script_sha256': sha256(__file__),
        'transitions': len(frames) * 4, 'head_names': list(policy.head_names), 'collector_metrics': metrics,
        'replay_max_abs_error': errors, 'gae_max_abs_error': float((s.advantages-reference).abs().max()),
        'gae_and_normalized_advantages_passed': True,
        'advantage_mean': flat.mean(0).tolist(), 'advantage_std': flat.std(0).tolist(),
        'policy_gradient_norm_by_head': norms.tolist(), 'gradient_norm_by_module': per_module,
        'gradient_cosine': cosine.tolist(), 'update_metrics': update,
        'clipped_surrogate_before': surrogate_before, 'clipped_surrogate_after': surrogate_after,
        'positive_advantage_mean_logp_change': float(delta[advantages > 0].mean()),
        'negative_advantage_mean_logp_change': float(delta[advantages < 0].mean()),
        'notes': ['No historical training rollout is reconstructed.',
                  'Gradient and surrogate checks do not establish better physical control.',
                  'Timeout behavior is physically exercised only if collector_metrics.timeouts > 0.']}
    _write_metrics(output / 'summary.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
