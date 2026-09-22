#!/usr/bin/env python
"""Capture a balanced physical diagnostic batch, then compare PPO updates on CPU.

All candidates start from identical policy weights, transitions and minibatch
orders. Contact-count and value-scaling choices are explicit ablations, not
paper claims. Captures explicitly reconstruct the historical force-squared
contact reward for the baseline; this is NOT the production v3 reward.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.distributions import Normal, kl_divergence

from pgmt.cfg.assumptions import get
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import _seed_everything, _write_metrics, build_env


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    return x


def capture(a):
    from pgmt.envs.isaac_app import launch_isaac_app
    from pgmt.envs.recovery import load_fall_pool
    _seed_everything(7)
    app, env = launch_isaac_app(a.device), None
    try:
        pool = load_fall_pool(a.fall_pool)
        env = build_env(16, torch.device(a.device), backend="isaaclab", app=app,
                        reference_data=a.reference_data, asset_path=a.asset,
                        reference_urdf=a.urdf, fall_pool=pool,
                        env_options={"seed": 7, "randomize_dynamics": True,
                                     "corrupt_observations": True, "max_action_delay": 2})
        core = env.env.core
        pool.init_prob = pool.prob_max = 0.
        env.reset()
        pool.init_prob = pool.prob_max = 1.
        env.env._reset_idx(torch.arange(8, 16, device=core.device))
        pool.init_prob, pool.prob_max = .1, .5
        env.env._get_observations()
        policy = Stage1Policy(init_noise_std=a.noise_std, neutral_init=a.neutral_init,
                              initial_joint_targets=core.default_q[0].cpu()).to(core.device)
        ppo = PPO(policy, total_updates=1000)
        diagnostics = []
        compute = core._reward_computer.compute

        def traced(state, reference, **kwargs):
            reward, metrics = compute(state, reference, **kwargs)
            feet = [core._reward_computer.bi[x] for x in
                    ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
            allowed = torch.zeros(state["contact_forces"].shape[1], device=core.device, dtype=torch.bool)
            allowed[feet[:2]] = True
            force = state["contact_forces"].norm(dim=-1)
            count = ((force > get("A21").value.contact_force_threshold) & ~allowed[None]).sum(-1)
            # Keep this ablation's historical baseline explicit after v3
            # changed the production reward to contact counts.
            from pgmt.rewards.spec import AUX
            weight = dict(AUX.terms)["undesired_contact"]
            old_cost = ((force - get("A21").value.contact_force_threshold).clamp_min(0).square()
                        * ~allowed[None]).sum(-1)
            reward[:, 2] += weight * (old_cost - metrics["aux"]["undesired_contact"])
            metrics["aux"]["undesired_contact"] = old_cost
            diagnostics.append(cpu_tree({
                "recovering": core._recovery_active.bool(), "episode_step": core.episode_length_buf,
                "seq": core.reference_seq_idx, "frame": core.reference_frame,
                "state_accel": state["body_accel"][:, feet],
                "reference_accel": reference["body_accel"][:, feet],
                "contact_count": count, "terms": metrics["aux"],
            }))
            return reward, metrics

        core._reward_computer.compute = traced
        _, rollout = ppo.collect_rollout(env, env.get_observations(), 24)
        storage = cpu_tree(vars(ppo.storage))
        storage["device"] = torch.device("cpu")
        torch.save({"policy": cpu_tree(policy.state_dict()), "storage": storage,
                    "reward_contract": "legacy_force_square_ablation",
                    "diagnostics": diagnostics, "rollout": rollout,
                    "sequence_names": [s["name"] for s in core.reference_database.seqs],
                    "initialization": {"noise_std":a.noise_std,"neutral_init":a.neutral_init}}, a.output)
        print(json.dumps(rollout), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def compare(a):
    from pgmt.train.storage import RolloutStorage
    payload = torch.load(a.capture, map_location="cpu", weights_only=False)
    base = payload["storage"]
    records = payload["diagnostics"]
    state = torch.stack([d["state_accel"] for d in records])
    reference = torch.stack([d["reference_accel"] for d in records])
    recovering = torch.stack([d["recovering"] for d in records])
    contact_count = torch.stack([d["contact_count"] for d in records])
    old_contact = torch.stack([d["terms"]["undesired_contact"] for d in records])
    def stats(x):
        x = x.flatten().float()
        return dict(zip(("p50", "p95", "p99", "max"), torch.quantile(x, torch.tensor([.5, .95, .99, 1.])).tolist()))
    report = {"batch": payload["rollout"], "diagnostics": {}, "candidates": {}}
    for name, mask in (("ordinary", ~recovering), ("recovery", recovering)):
        if mask.any():
            report["diagnostics"][name] = {
                "transitions": int(mask.sum()), "state_accel_norm_m_s2": stats(state[mask].norm(dim=-1)),
                "reference_accel_norm_m_s2": stats(reference[mask].norm(dim=-1)),
                "weighted_state_accel_energy": float(-.001*state[mask].square().sum((-1,-2)).mean()),
                "weighted_reference_accel_energy": float(-.001*reference[mask].square().sum((-1,-2)).mean()),
                "weighted_mismatch": float(-.001*(state-reference)[mask].square().sum((-1,-2)).mean()),
                "weighted_contact_square": float(-.1*old_contact[mask].mean()),
                "weighted_contact_count": float(-.1*contact_count[mask].float().mean()),
            }
    candidates = [
        ("baseline", 1e-3, False, False, None, "both"),
        ("lr_1e4", 1e-4, False, False, None, "both"),
        ("lr_1e5", 1e-5, False, False, None, "both"),
        ("actor_only", 1e-3, False, False, None, "actor"),
        ("critic_only", 1e-3, False, False, None, "critic"),
        ("scaled_value_1e4", 1e-4, True, False, None, "both"),
        ("count_contact_1e4", 1e-4, False, True, None, "both"),
        ("count_scaled_1e4", 1e-4, True, True, None, "both"),
        ("count_scaled_guard_1e4", 1e-4, True, True, .02, "both"),
    ]
    for name, lr, normalized, count_contact, budget, objective in candidates:
        _seed_everything(123)
        policy = Stage1Policy()
        policy.load_state_dict(payload["policy"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        storage = object.__new__(RolloutStorage)
        storage.__dict__.update(deepcopy(base))
        if count_contact:
            storage.rewards[..., 2] += .1*(old_contact-contact_count)
            storage.compute_returns(.99, .95)
        obs = {k: v.flatten(0, 1) for k,v in storage.observations.items()}
        with torch.no_grad():
            behavior, _ = policy._distribution_and_values(obs)
            behavior = Normal(behavior.loc.clone(), behavior.scale.clone())
        def full_kl():
            with torch.no_grad():
                now, _ = policy._distribution_and_values(obs)
                return float(kl_divergence(behavior, now).sum(-1).mean())
        scale = storage.returns.flatten(0, 1).std(0, unbiased=False).clamp_min(1.) if normalized else torch.ones(3)
        losses, kls, backtracks, accepted = [], [], 0, 0
        first_grad = None
        for batch in storage.minibatches(4, 5):
            out = policy.evaluate_actions(batch["observations"], batch["actions"], latent_actions=batch["latent_actions"])
            ratio = (out.log_probs-batch["log_probs"]).exp()
            adv = batch["policy_advantages"]
            policy_loss = -torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv).mean() - .01*out.entropy.mean()
            v, old, target = out.values/scale, batch["values"]/scale, batch["returns"]/scale
            clipped = old+(v-old).clamp(-.2,.2)
            value_loss = torch.maximum((v-target).square(), (clipped-target).square()).mean()
            if first_grad is None:
                shared = list(policy.history_encoder.parameters()) + list(policy.ifm.parameters())
                def norm(loss):
                    g=torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
                    return float(torch.stack([x.square().sum() for x in g if x is not None]).sum().sqrt())
                first_grad = {"actor_shared":norm(policy_loss), "critic_shared":norm(value_loss)}
            loss = policy_loss if objective == "actor" else value_loss if objective == "critic" else policy_loss+value_loss
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            old_weights, old_opt = (deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())) if budget else (None,None)
            taken = False
            for retry in range(9 if budget else 1):
                optimizer.step()
                kl = full_kl()
                if budget is None or kl <= budget:
                    taken = True
                    break
                policy.load_state_dict(old_weights)
                optimizer.load_state_dict(deepcopy(old_opt))
                for group in optimizer.param_groups:
                    group["lr"] = lr * .5 ** (retry+1)
                backtracks += 1
            if not taken:
                break
            lr=optimizer.param_groups[0]["lr"]
            accepted += 1
            losses.append(float(loss.detach()))
            kls.append(kl)
            # Once the rollout's KL budget is nearly spent, finish this
            # update; do not keep shrinking the LR into ineffective steps.
            if budget is not None and kl >= .9 * budget:
                break
        report["candidates"][name] = {"first_shared_gradient_norm": first_grad,
            "value_scale": scale.tolist(), "first_step_exact_kl":kls[0] if kls else 0.,
            "final_exact_kl":full_kl(), "max_accepted_kl":max(kls,default=0.),
            "accepted_steps":accepted,"backtracks":backtracks,"final_lr":lr,
            "normalized_value_loss":normalized,"contact_count":count_contact,
            "final_objective":losses[-1] if losses else None}
        _write_metrics(a.output, report)
        print(name, json.dumps(report["candidates"][name]), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture","compare"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--capture",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--fall-pool")
    p.add_argument("--asset")
    p.add_argument("--urdf")
    p.add_argument("--reference-data")
    p.add_argument("--noise-std",type=float,default=.3)
    p.add_argument("--neutral-init",action="store_true")
    a=p.parse_args()
    if a.output.exists(): p.error("output exists")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (capture if a.mode=="capture" else compare)(a)


if __name__=="__main__":
    main()
