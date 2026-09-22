"""Replay an accepted-update KL event on CPU, without any simulator.

Use ``CUDA_VISIBLE_DEVICES='' python -m setup.replay_kl_diagnostic event.pt``.
Load only trusted local snapshots. This verifies saved observations/parameters;
it does not reconstruct the minibatch/Adam trajectory that produced the update.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.distributions import kl_divergence

from pgmt.train.policy import Stage1Policy, Stage2Policy


@torch.no_grad()
def replay(path):
    event = torch.load(path, map_location="cpu", weights_only=False)
    if event.get("schema") != "pgmt.diagnostic.v1" or event.get("kind") != "kl":
        raise ValueError("expected a pgmt KL diagnostic snapshot")
    policy = Stage1Policy() if len(event["heads"]) == 3 else Stage2Policy()
    if list(policy.head_names) != event["heads"]:
        raise ValueError("unsupported policy head contract")
    policy.eval()
    distributions, errors = [], {}
    for when, prefix in (("before", "old"), ("after", "new")):
        policy.load_state_dict(event["policy_" + when])
        dist = policy.latent_distribution(event["observations"])
        distributions.append(dist)
        for parameter in ("loc", "scale"):
            actual, expected = getattr(dist, parameter), event[prefix + "_" + parameter]
            errors[prefix + "_" + parameter] = (actual - expected).abs().max().item()
            # GPU-to-CPU GEMM can differ slightly; errors are also reported.
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    per_joint = kl_divergence(*distributions)
    torch.testing.assert_close(per_joint, event["kl_per_joint"], atol=2e-5, rtol=2e-4)
    observations = {k: {"abs_max": v.abs().max().item(), "rms": v.square().mean().sqrt().item(),
                         "finite": bool(v.isfinite().all())} for k, v in event["observations"].items()}
    return {"verified": True, "step": event["step"], "kl": per_joint.sum(-1).tolist(),
            "recovery": None if event["recovery"] is None else event["recovery"].tolist(),
            "parameter_max_abs_error": errors, "observations": observations,
            "scope": event["snapshot_scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event")
    args = parser.parse_args()
    print(json.dumps(replay(args.event), indent=2))


if __name__ == "__main__":
    main()
