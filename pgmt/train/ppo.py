"""独立于仿真器的 Stage 1 多头 PPO，支持自动 reset 和断点恢复。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch

from pgmt.cfg.assumptions import get
from pgmt.train.policy import Stage1Policy
from pgmt.train.storage import RolloutStorage


class PPO:
    """奖励与价值的列序固定为 ``policy.head_names``。

    环境协议：``step(actions) -> obs, rewards, terminated, truncated, info``。
    自动 reset 时，``info['terminal_observation']`` 必须携带纯 timeout 的
    reset 前状态（全 batch，或按 timeout mask 顺序排列的子 batch）。

    ``total_updates`` 是绝对目标迭代数，用于线性学习率递减；恢复时保留新构造
    实例的目标，继续使用 checkpoint 的 update_count。保存点应在 update 之后。
    """

    def __init__(self, policy: Stage1Policy, config=None, value_loss_coef: float = 1.0,
                 total_updates: Optional[int] = None):
        self.policy = policy
        self.config = get("A6").value if config is None else config
        if value_loss_coef < 0 or (total_updates is not None and total_updates <= 0):
            raise ValueError("invalid value loss coefficient or update target")
        self.value_loss_coef = float(value_loss_coef)
        self.total_updates = total_updates
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        self.update_count = 0
        self.storage = None

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
        reward_sum = torch.zeros(len(self.policy.head_names), device=self.device)
        terminations, timeouts = 0, 0
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
                             rewards, terminated, truncated, next_values)
            reward_sum += rewards.sum(0)
            terminations += int(terminated.sum().item())
            timeouts += int(timeout_mask.sum().item())
            observations = next_obs
        self.storage.compute_returns(self.config.gamma, self.config.lam)
        means = (reward_sum / (steps * count)).tolist()
        metrics = {"reward_" + name: value for name, value in zip(self.policy.head_names, means)}
        metrics.update(reward_mean=sum(means), terminations=terminations, timeouts=timeouts,
                       episode_count=terminations + timeouts, steps=steps * count)
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
        totals = {key: 0.0 for key in (
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
        )}
        totals.update({"value_loss_" + name: 0.0 for name in self.policy.head_names})
        samples = 0
        for batch in self.storage.minibatches(cfg.num_mini_batches, cfg.num_learning_epochs):
            output = self.policy.evaluate_actions(batch["observations"], batch["actions"])
            log_ratio = output.log_probs - batch["log_probs"]
            ratio = log_ratio.exp()
            advantage = batch["policy_advantages"]
            surrogate = ratio * advantage
            clipped_surrogate = ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantage
            policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
            clipped_values = batch["values"] + (output.values - batch["values"]).clamp(
                -cfg.clip_param, cfg.clip_param)
            # 先按样本求每头误差，再对头等权平均；每个价值头拟合自己的回报。
            value_per_head = torch.maximum((output.values - batch["returns"]).square(),
                                           (clipped_values - batch["returns"]).square()).mean(0)
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
            self.optimizer.step()
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
        self.update_count += 1
        self.storage.clear()
        metrics = {key: value / samples for key, value in totals.items()}
        metrics.update(learning_rate=learning_rate, num_updates=self.update_count)
        return metrics

    def state_dict(self):
        if self.storage is not None and self.storage.step:
            raise RuntimeError("checkpoint after update; pending rollout is not serialized")
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count, "config": asdict(self.config),
                "value_loss_coef": self.value_loss_coef, "total_updates": self.total_updates}

    def load_state_dict(self, state, strict: bool = True):
        if self.storage is not None and self.storage.step:
            raise RuntimeError("cannot restore with a pending rollout")
        if state["config"] != asdict(self.config) or state["value_loss_coef"] != self.value_loss_coef:
            raise ValueError("PPO checkpoint configuration differs from current configuration")
        self.policy.load_state_dict(state["policy"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state["update_count"])
