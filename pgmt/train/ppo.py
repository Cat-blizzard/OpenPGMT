"""独立于仿真器的 Stage 1 多头 PPO，支持自动 reset 和断点恢复。"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import math
from typing import Optional

import torch
from torch.distributions import Normal, kl_divergence

from pgmt.cfg.assumptions import get
from pgmt.train.policy import Stage1Policy
from pgmt.train.storage import RolloutStorage


def scaled_value_errors(values, old_values, returns, scales, clip_param):
    """Per-head loss in standardized units; inputs/GAE remain in raw units."""
    v, old, target = values / scales, old_values / scales, returns / scales
    clipped = old + (v - old).clamp(-clip_param, clip_param)
    return torch.maximum((v - target).square(), (clipped - target).square()).mean(0)


class PPO:
    """奖励与价值的列序固定为 ``policy.head_names``。

    环境协议：``step(actions) -> obs, rewards, terminated, truncated, info``。
    自动 reset 时，``info['terminal_observation']`` 必须携带纯 timeout 的
    reset 前状态（全 batch，或按 timeout mask 顺序排列的子 batch）。

    ``total_updates`` 是完整学习率日程长度，与 runner 的本次停止点独立。
    恢复时日程必须与 checkpoint 一致。保存点应在 update 之后。
    """

    training_contract = "pgmt_curriculum_critic_v4"

    def __init__(self, policy: Stage1Policy, config=None, value_loss_coef: float = 1.0,
                 total_updates: Optional[int] = None, diagnostics=None):
        self.policy = policy
        self.config = get("A6").value if config is None else config
        if value_loss_coef < 0 or (total_updates is not None and total_updates <= 0):
            raise ValueError("invalid value loss coefficient or update target")
        self.value_loss_coef = float(value_loss_coef)
        self.total_updates = total_updates
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        self.update_count = 0
        self.diagnostics = diagnostics
        self.storage = None
        self._rollout_recovery = None
        # Extra value minibatches should not consume the actor/environment's
        # sampling stream in paired completion-on/off experiments.
        self._critic_generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed() + 100003)
        budget = getattr(self.config, "target_kl", None)
        if budget is not None:
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("target_kl must be positive and finite, or None")
            if not callable(getattr(policy, "latent_distribution", None)):
                raise TypeError("KL-constrained PPO requires policy.latent_distribution")
            if not 0 < self.config.kl_stop_fraction <= 1 or self.config.max_kl_backtracks < 0:
                raise ValueError("invalid KL stopping/backtracking configuration")
            if self.config.kl_chunk_size <= 0:
                raise ValueError("kl_chunk_size must be positive")
        if getattr(self.config, "normalize_value_loss", False) and self.config.value_scale_floor <= 0:
            raise ValueError("value_scale_floor must be positive")
        if getattr(self.config, "complete_critic_epochs", False) and not callable(getattr(policy, "detached_critic_inputs", None)):
            raise TypeError("critic completion requires detached_critic_inputs")

    @property
    def device(self):
        return next(self.policy.parameters()).device

    @torch.no_grad()
    def collect_rollout(self, env, observations, num_steps: Optional[int] = None):
        """收集完整 rollout、计算分头 GAE，返回 ``(next_obs, metrics)``。"""
        if self.storage is not None and self.storage.step:
            raise RuntimeError("update the pending rollout before collecting another")
        steps = self.config.num_steps_per_env if num_steps is None else num_steps
        count = observations["obs"].shape[0]
        shapes = {key: tuple(value.shape[1:]) for key, value in observations.items()}
        if (self.storage is None or self.storage.num_steps != steps
                or self.storage.num_envs != count):
            self.storage = RolloutStorage(steps, count, observation_shapes=shapes,
                                          num_heads=len(self.policy.head_names), device=self.device)
        self.policy.eval()
        self._rollout_recovery = None
        recovery_labels = []
        reward_sum = torch.zeros(len(self.policy.head_names), device=self.device)
        terminations, timeouts = 0, 0
        recovery_steps, recovery_terminations, recovery_timeouts = 0, 0, 0
        aux_sums, aux_peaks = {}, {}
        groups = {name: {"steps": 0, "terminations": 0, "timeouts": 0,
                          "ended_duration_sum": 0., "ended_duration_count": 0,
                          "sums": {}, "peaks": {}, "reasons": {},
                          "reset_count": 0, "reset_joint_rmse_sum": 0., "reset_joint_rmse_max": 0.}
                  for name in ("ordinary", "recovery")}
        latent_saturated, latent_count = 0, 0
        for _ in range(steps):
            # 环境可能原位改写观测缓冲；先保存本步策略输入，确保不是 reset 后的状态。
            current_obs = {key: value.detach().clone() for key, value in observations.items()}
            output = self.policy.act(current_obs)
            next_obs, rewards, terminated, truncated, info = env.step(output.actions)
            # Keep rollout accounting on the policy device.  Isaac Gym envs normally
            # return GPU tensors, but a CPU reward tensor is easy to produce in a
            # Python/NumPy reward adapter; ``reward_sum +=`` below otherwise fails
            # with a cross-device error before storage can copy the sample.
            rewards = rewards.to(device=self.device, dtype=torch.float32)
            terminated = terminated.to(device=self.device, dtype=torch.bool)
            truncated = truncated.to(device=self.device, dtype=torch.bool)
            next_values = self.policy.value(next_obs)
            timeout_mask = truncated & ~terminated
            if timeout_mask.any():
                terminal = info.get("terminal_observation")
                if terminal is None:
                    raise ValueError("timeout requires reset-before terminal_observation for bootstrap")
                num_timeouts = int(timeout_mask.sum().item())
                terminal_count = terminal["obs"].shape[0]
                if terminal_count == count:
                    terminal = {key: value[timeout_mask] for key, value in terminal.items()}
                elif terminal_count != num_timeouts:
                    raise ValueError("terminal_observation must contain full batch or pure timeouts")
                next_values[timeout_mask] = self.policy.value(terminal)
            self.storage.add(current_obs, output.actions, output.log_probs, output.values,
                             rewards, terminated, truncated, next_values,
                             latent_actions=getattr(output, "latent_actions", None))
            reward_sum += rewards.sum(0)
            terminations += int(terminated.sum().item())
            timeouts += int(timeout_mask.sum().item())
            if getattr(output, "latent_actions", None) is not None:
                latent_saturated += int((output.latent_actions.abs() > 3).sum())
                latent_count += output.latent_actions.numel()
            for name, contribution in info.get("weighted_aux_terms", {}).items():
                aux_sums[name] = aux_sums.get(name, 0.) + contribution.sum().item()
                aux_peaks[name] = max(aux_peaks.get(name, 0.), contribution.abs().max().item())
            if "recovery_active" in info:
                recovering = info["recovery_active"].to(self.device, torch.bool)
                recovery_labels.append(recovering.clone())
                recovery_steps += int(recovering.sum().item())
                recovery_terminations += int((recovering & terminated).sum().item())
                recovery_timeouts += int((recovering & timeout_mask).sum().item())
                measurements = {"reward_" + head: rewards[:, j]
                                for j, head in enumerate(self.policy.head_names)}
                for category in ("weighted_aux_terms", "tracking", "physics"):
                    measurements.update({category + "/" + k: v
                                         for k, v in info.get(category, {}).items()})
                for name, mask in (("ordinary", ~recovering), ("recovery", recovering)):
                    group = groups[name]
                    n = int(mask.sum())
                    if not n:
                        continue
                    group["steps"] += n
                    group["terminations"] += int((mask & terminated).sum())
                    group["timeouts"] += int((mask & timeout_mask).sum())
                    for reason, fired in info.get("termination_reasons", {}).items():
                        group["reasons"][reason] = group["reasons"].get(reason, 0) + int((mask & fired.to(self.device)).sum())
                    if "reset_joint_rmse" in info:
                        starts = mask & (info["episode_steps"] == 1)
                        if starts.any():
                            initial_error = info["reset_joint_rmse"][starts]
                            group["reset_count"] += int(starts.sum())
                            group["reset_joint_rmse_sum"] += initial_error.sum().item()
                            group["reset_joint_rmse_max"] = max(group["reset_joint_rmse_max"], initial_error.max().item())
                    ended = mask & (terminated | truncated)
                    if "episode_elapsed_s" in info:
                        group["ended_duration_sum"] += info["episode_elapsed_s"][ended].sum().item()
                        group["ended_duration_count"] += int(ended.sum())
                    for key, value in measurements.items():
                        selected = value.to(self.device)[mask]
                        group["sums"][key] = group["sums"].get(key, 0.) + selected.sum().item()
                        group["peaks"][key] = max(group["peaks"].get(key, 0.), selected.abs().max().item())
            observations = next_obs
        self.storage.compute_returns(self.config.gamma, self.config.lam)
        if len(recovery_labels) == steps:
            self._rollout_recovery = torch.stack(recovery_labels).flatten()
        means = (reward_sum / (steps * count)).tolist()
        metrics = {"reward_" + name: value for name, value in zip(self.policy.head_names, means)}
        metrics.update(reward_mean=sum(means), terminations=terminations, timeouts=timeouts,
                       episode_count=terminations + timeouts, steps=steps * count,
                       recovery_steps=recovery_steps, recovery_terminations=recovery_terminations,
                       recovery_timeouts=recovery_timeouts)
        if aux_sums:
            metrics["weighted_aux_terms_mean"] = {k: v / (steps * count) for k, v in aux_sums.items()}
            metrics["weighted_aux_terms_peak_abs"] = aux_peaks
        if latent_count:
            metrics["action_tanh_saturated_fraction"] = latent_saturated / latent_count
        if any(group["steps"] for group in groups.values()):
            metrics["episode_groups"] = {}
            for name, group in groups.items():
                result = {k: group[k] for k in ("steps", "terminations", "timeouts")}
                if group["steps"]:
                    result["mean"] = {k: v / group["steps"] for k, v in group["sums"].items()}
                    result["peak_abs"] = group["peaks"]
                if group["ended_duration_count"]:
                    result["ended_episode_duration_mean_s"] = group["ended_duration_sum"] / group["ended_duration_count"]
                result["termination_reasons"] = group["reasons"]
                if group["reset_count"]:
                    result["resets_observed"] = group["reset_count"]
                    result["reset_joint_rmse_mean"] = group["reset_joint_rmse_sum"] / group["reset_count"]
                    result["reset_joint_rmse_max"] = group["reset_joint_rmse_max"]
                metrics["episode_groups"][name] = result
        if "recovery_curriculum" in info:
            metrics["recovery_curriculum"] = info["recovery_curriculum"]
        return observations, metrics

    def update(self):
        if self.storage is None or not self.storage.ready:
            raise RuntimeError("collect_rollout or prepared storage is required before update")
        cfg = self.config
        self.policy.train()
        # 第 0 次更新用初始 LR；最后一次计划更新仍有正步长。
        fraction = (1.0 if self.total_updates is None else
                    max(0.0, 1.0 - self.update_count / self.total_updates))
        learning_rate = cfg.learning_rate * fraction
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        budget = getattr(cfg, "target_kl", None)
        behavior = self._snapshot_behavior() if budget is not None else None
        diagnostic_before = None
        if behavior is not None and self.diagnostics is not None and self.diagnostics.ready("kl", self.update_count + 1):
            from pgmt.train.diagnostics import cpu_copy
            diagnostic_before = cpu_copy(self.policy.state_dict())
        returns = self.storage.returns.flatten(0, 1)
        scales = (returns.std(0, unbiased=False).clamp_min(cfg.value_scale_floor)
                  if getattr(cfg, "normalize_value_loss", False) else torch.ones_like(returns[0]))
        raw_mse_pre = (self.storage.values.flatten(0, 1) - returns).square().mean(0)
        totals = {key: 0.0 for key in (
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
        )}
        totals.update({"value_loss_" + name: 0.0 for name in self.policy.head_names})
        samples, optimizer_steps, backtracks = 0, 0, 0
        early_stopped, max_accepted_kl = False, 0.
        last_batch = None
        for batch in self.storage.minibatches(cfg.num_mini_batches, cfg.num_learning_epochs):
            last_batch = batch
            output = self._evaluate_batch(batch)
            log_ratio = output.log_probs - batch["log_probs"]
            ratio = log_ratio.exp()
            advantage = batch["policy_advantages"]
            surrogate = ratio * advantage
            clipped_surrogate = ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantage
            policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
            value_per_head = scaled_value_errors(output.values, batch["values"],
                                                  batch["returns"], scales, cfg.clip_param)
            value_loss = value_per_head.mean()
            entropy = output.entropy.mean()
            loss = policy_loss + self.value_loss_coef * value_loss - cfg.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite PPO gradient")
            if behavior is None:
                self.optimizer.step()
            else:
                # A rejected proposal must not leave Adam's moments/step
                # advanced. Reuse the same gradient while reducing its LR.
                weights_before = deepcopy(self.policy.state_dict())
                optimizer_before = deepcopy(self.optimizer.state_dict())
                proposal_lr = self.optimizer.param_groups[0]["lr"]
                accepted = False
                for retry in range(cfg.max_kl_backtracks + 1):
                    for group in self.optimizer.param_groups:
                        group["lr"] = proposal_lr * .5 ** retry
                    self.optimizer.step()
                    kl, _ = self._exact_kl(behavior)
                    if math.isfinite(kl) and kl <= budget:
                        accepted = True
                        max_accepted_kl = max(max_accepted_kl, kl)
                        break
                    self.policy.load_state_dict(weights_before)
                    # Optimizer loading may alias same-device state tensors.
                    # Each retry needs its own copy or step() would mutate
                    # the supposedly frozen rollback snapshot.
                    self.optimizer.load_state_dict(deepcopy(optimizer_before))
                    backtracks += 1
                if not accepted:
                    early_stopped = True
                    break
            optimizer_steps += 1
            size = batch["actions"].shape[0]
            with torch.no_grad():
                metrics = {
                    "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                    "entropy": entropy.item(), "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                    "clip_fraction": ((ratio - 1).abs() > cfg.clip_param).float().mean().item(),
                    "grad_norm": grad_norm.item(),
                }
                metrics.update({"value_loss_" + name: value for name, value in
                                zip(self.policy.head_names, value_per_head.tolist())})
            for key, value in metrics.items():
                totals[key] += value * size
            samples += size
            if behavior is not None and kl >= budget * cfg.kl_stop_fraction:
                early_stopped = True
                break
        # Completing only the private critic preserves the accepted actor
        # and shared encoder. Total value steps stay within the epoch budget.
        critic_only_steps = 0
        if getattr(cfg, "complete_critic_epochs", False):
            remaining = cfg.num_learning_epochs * cfg.num_mini_batches - optimizer_steps
            values_joint = self._rollout_values()
            raw_mse_joint = (values_joint - returns).square().mean(0)
            critic_only_steps = self._complete_critic(remaining, scales, learning_rate)
        self.update_count += 1
        metrics = {key: value / max(samples, 1) for key, value in totals.items()}
        # 上面的 ratio 指标在每个 minibatch 的 optimizer.step() 之前求值；当
        # epochs × minibatches = 1（单一巨批）时它们恒为 ratio=1 处的恒等值，
        # 对策略是否移动完全失明。这里在更新后的参数下重估同一批数据，
        # 给出真实的更新幅度。
        with torch.no_grad():
            output = self._evaluate_batch(last_batch)
            log_ratio = output.log_probs - last_batch["log_probs"]
            metrics["approx_kl_post"] = (((log_ratio.exp() - 1) - log_ratio).mean().item())
        if behavior is not None:
            metrics["exact_kl"], metrics["exact_kl_max_state"] = self._exact_kl(behavior)
            metrics["max_accepted_kl"] = max_accepted_kl
            if not math.isfinite(metrics["exact_kl"]) or metrics["exact_kl"] > budget:
                raise RuntimeError("accepted PPO update exceeded the rollout KL budget")
            metrics.update(self._kl_statistics(behavior))
            if diagnostic_before is not None:
                self.diagnostics.capture_kl(self, behavior, diagnostic_before)
        # Raw errors expose value underfitting even when normalized loss is
        # small. Compute on the entire rollout, not just its last minibatch.
        values_post = self._rollout_values()
        raw_mse_post = (values_post - returns).square().mean(0)
        variance = returns.var(0, unbiased=False)
        explained = torch.where(variance > 1e-8,
                                1 - (returns-values_post).var(0, unbiased=False) / variance.clamp_min(1e-8),
                                torch.zeros_like(variance))
        for j, head in enumerate(self.policy.head_names):
            metrics["value_scale_" + head] = scales[j].item()
            metrics["value_mse_raw_pre_" + head] = raw_mse_pre[j].item()
            metrics["value_mse_raw_post_" + head] = raw_mse_post[j].item()
            if getattr(cfg, "complete_critic_epochs", False):
                metrics["value_mse_raw_joint_" + head] = raw_mse_joint[j].item()
            metrics["explained_variance_" + head] = explained[j].item()
        self.storage.clear()
        metrics.update(learning_rate=self.optimizer.param_groups[0]["lr"],
                       scheduled_learning_rate=learning_rate, num_updates=self.update_count,
                       optimizer_steps=optimizer_steps, kl_backtracks=backtracks,
                       critic_only_steps=critic_only_steps,
                       value_optimizer_steps=optimizer_steps + critic_only_steps,
                       early_stopped=early_stopped)
        return metrics

    def _complete_critic(self, remaining, scales, learning_rate):
        if remaining <= 0 or self.value_loss_coef == 0:
            return 0
        features = [self.policy.detached_critic_inputs(obs) for obs in self._observation_chunks()]
        privileged, intent = (torch.cat([item[j] for item in features]).detach() for j in (0, 1))
        returns = self.storage.returns.flatten(0, 1)
        old_values = self.storage.values.flatten(0, 1)
        previous_lr = [g["lr"] for g in self.optimizer.param_groups]
        steps = 0
        try:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            while steps < remaining:
                for ix in torch.tensor_split(torch.randperm(len(returns), device=self.device,
                                                            generator=self._critic_generator), self.config.num_mini_batches):
                    values = self.policy.critic(privileged[ix], intent[ix])
                    loss = self.value_loss_coef * scaled_value_errors(
                        values, old_values[ix], returns[ix], scales, self.config.clip_param).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite private critic loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.max_grad_norm)
                    if not torch.isfinite(norm):
                        raise RuntimeError("non-finite private critic gradient")
                    self.optimizer.step()
                    steps += 1
                    if steps == remaining:
                        break
        finally:
            for group, lr in zip(self.optimizer.param_groups, previous_lr):
                group["lr"] = lr
        return steps

    def _observation_chunks(self):
        flat = {k: v.flatten(0, 1) for k, v in self.storage.observations.items()}
        count = self.storage.num_steps * self.storage.num_envs
        chunk_size = getattr(self.config, "kl_chunk_size", 256)
        for start in range(0, count, chunk_size):
            yield {k: v[start:start + chunk_size] for k, v in flat.items()}

    @torch.no_grad()
    def _snapshot_behavior(self):
        # The policy is unchanged throughout collection. Cache its Normal
        # parameters once; all subsequent proposals use this fixed reference.
        batches = []
        for obs in self._observation_chunks():
            distribution = self.policy.latent_distribution(obs)
            if not isinstance(distribution, Normal):
                raise TypeError("analytic KL guard requires a diagonal Normal")
            batches.append((obs, Normal(distribution.loc.clone(), distribution.scale.clone())))
        return batches

    @torch.no_grad()
    def _exact_kl(self, behavior):
        total, count, maximum = 0., 0, 0.
        for obs, old in behavior:
            new = self.policy.latent_distribution(obs)
            per_state = kl_divergence(old, new).sum(-1)
            if not torch.isfinite(per_state).all():
                return float("inf"), float("inf")
            total += per_state.sum().item()
            count += per_state.numel()
            maximum = max(maximum, per_state.max().item())
        return max(0., total / count), maximum

    @torch.no_grad()
    def _kl_statistics(self, behavior):
        per_state = torch.cat([kl_divergence(old, self.policy.latent_distribution(obs)).sum(-1)
                               for obs, old in behavior])
        metrics = {}
        groups = {"all": torch.ones_like(per_state, dtype=torch.bool)}
        if self._rollout_recovery is not None:
            groups.update(ordinary=~self._rollout_recovery, recovery=self._rollout_recovery)
        for name, mask in groups.items():
            if not mask.any():
                continue
            values = per_state[mask]
            metrics["kl_" + name + "_mean"] = values.mean().item()
            metrics["kl_" + name + "_max"] = values.max().item()
            for q in (.5, .95, .99):
                metrics[f"kl_{name}_p{round(100*q)}"] = torch.quantile(values, q).item()
        return metrics

    @torch.no_grad()
    def _rollout_values(self):
        return torch.cat([self.policy.value(obs) for obs in self._observation_chunks()])

    def _evaluate_batch(self, batch):
        kwargs = ({"latent_actions": batch["latent_actions"]} if "latent_actions" in batch else {})
        return self.policy.evaluate_actions(batch["observations"], batch["actions"], **kwargs)

    def state_dict(self):
        if self.storage is not None and self.storage.step:
            raise RuntimeError("checkpoint after update; pending rollout is not serialized")
        return {"action_contract": getattr(self.policy, "action_contract", "test_policy"),
                "training_contract": self.training_contract,
                "critic_rng": self._critic_generator.get_state(),
                "rng_cpu": torch.get_rng_state(),
                "rng_device": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count, "config": asdict(self.config),
                "value_loss_coef": self.value_loss_coef, "total_updates": self.total_updates}

    def load_state_dict(self, state, strict: bool = True):
        if self.storage is not None and self.storage.step:
            raise RuntimeError("cannot restore with a pending rollout")
        if state.get("action_contract") != getattr(self.policy, "action_contract", "test_policy"):
            raise ValueError("checkpoint action contract is obsolete; start a new training run")
        if state.get("training_contract") != self.training_contract:
            raise ValueError("checkpoint training/reward contract is obsolete; start a new training run")
        if state["config"] != asdict(self.config) or state["value_loss_coef"] != self.value_loss_coef:
            raise ValueError("PPO checkpoint configuration differs from current configuration")
        if state.get("total_updates") != self.total_updates:
            raise ValueError("PPO checkpoint learning-rate schedule differs; keep its original horizon")
        self.policy.load_state_dict(state["policy"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state["update_count"])
        self._critic_generator.set_state(state["critic_rng"].cpu())
        if state.get("rng_cpu") is not None:
            torch.set_rng_state(state["rng_cpu"].cpu())
        if self.device.type == "cuda" and state.get("rng_device") is not None:
            torch.cuda.set_rng_state(state["rng_device"].cpu(), self.device)
