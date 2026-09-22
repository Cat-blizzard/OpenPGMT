"""Run a frozen matched manifest in IsaacLab (no learning, no mock results).

One environment is used deliberately so per-episode random streams are
independent of early failures and batch scheduling. JSONL is flushed after
every episode and can be resumed. Failed episodes remain in the denominator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pgmt.eval.manifest import file_hash
from pgmt.envs.terrain.generators import FAMILIES
from pgmt.envs.randomization import RANDOMIZATION_SPEC
from pgmt.train.policy import Stage1Policy, Stage2Policy
from pgmt.train.train_stage1 import build_env, _write_metrics


def summarize(rows):
    def group(items):
        n = len(items)
        if not n:
            return {"episodes": 0}
        successes = sum(r["success"] for r in items)
        rate, z = successes/n, 1.959963984540054
        center = (rate + z*z/(2*n)) / (1+z*z/n)
        margin = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        steps = sum(r["steps"] for r in items)
        result = {"episodes": n, "completion": rate, "completion_wilson95": [center-margin, center+margin],
                  "mean_duration_s": sum(r["duration_s"] for r in items)/n}
        for key in ("joint", "body", "root"):
            result[key+"_rmse"] = math.sqrt(sum(r[key+"_mse_sum"] for r in items)/steps)
        tp, fp, fn = (sum(r["contact_"+k] for r in items) for k in ("tp", "fp", "fn"))
        result["contact_f1"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        result["terrain_boundary_failures"] = sum(r["terrain_out_of_bounds"] for r in items)
        return result
    return {"overall": group(rows), "L9": group([r for r in rows if r["level"] == 9]),
            "cells": {f"{f}/L{level}": group([r for r in rows if r["family"] == f and r["level"] == level])
                      for f in FAMILIES for level in range(10)}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "reference-data", "asset", "urdf", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cuda:8")
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--limit", type=int, help="explicit diagnostic subset, never reported as the full benchmark")
    from pgmt.envs.actuators import ACTUATOR_PROFILES, actuator_options
    p.add_argument("--actuator-profile", choices=ACTUATOR_PROFILES, default="asset_effort_v1")
    a = p.parse_args(argv)
    if a.limit is not None and a.limit <= 0:
        p.error("limit must be positive")
    manifest = json.loads(a.manifest.read_text())
    if manifest["schema"] != "pgmt_matched_eval_v2" or manifest["randomization"] != RANDOMIZATION_SPEC:
        raise ValueError("evaluation configuration differs from frozen manifest")
    if manifest["control_dt"] != .02 or manifest["sim_dt"] != .005:
        raise ValueError("manifest clocks differ from the evaluated environment")
    hashes = {f.name: file_hash(f) for f in a.reference_data.glob("*.npz")}
    if hashes != manifest["reference_files"] or file_hash(a.asset) != manifest["asset_sha256"] or file_hash(a.urdf) != manifest["urdf_sha256"]:
        raise ValueError("evaluation assets/references differ from frozen manifest")
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["ppo"]
    policy = Stage1Policy() if a.stage == 1 else Stage2Policy()
    if payload.get("action_contract") != policy.action_contract:
        raise ValueError("obsolete checkpoint action contract")
    policy.load_state_dict(payload["policy"])
    policy.to(a.device).eval()
    # Termination semantics changed in v4. Never append new-rule episodes to
    # older rows merely because their policy and motion manifest match.
    import pgmt.envs.g1_env as env_module
    import pgmt.cfg.assumptions as cfg_module
    import pgmt.rewards.batched as reward_module
    import pgmt.envs.actuators as actuator_module
    identity = {"manifest_sha256": file_hash(a.manifest), "checkpoint_sha256": file_hash(a.checkpoint), "stage": a.stage,
                "evaluation_protocol": "pgmt_root_relative_v4",
                "environment_sha256": file_hash(Path(env_module.__file__)),
                "actuator_profile": a.actuator_profile,
                "actuator_implementation_sha256": file_hash(Path(actuator_module.__file__)),
                "assumptions_sha256": file_hash(Path(cfg_module.__file__)),
                "reward_implementation_sha256": file_hash(Path(reward_module.__file__))}
    rows = []
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            row = json.loads(line)
            if any(row.get(k) != v for k, v in identity.items()):
                raise ValueError("output belongs to a different checkpoint/manifest/evaluation protocol")
            rows.append(row)
    seen = {r["episode_id"] for r in rows}
    if len(seen) != len(rows):
        raise ValueError("duplicate episode results")
    expected = {e["episode_id"]: e for e in manifest["episodes"]}
    for row in rows:
        episode = expected.get(row["episode_id"])
        if episode is None or any(row.get(k) != v for k, v in episode.items()):
            raise ValueError("saved episode differs from manifest")
    from pgmt.envs.isaac_app import launch_isaac_app
    app = launch_isaac_app(a.device)
    env = None
    try:
        env = build_env(1, torch.device(a.device), backend="isaaclab", app=app,
                        reference_data=str(a.reference_data), asset_path=str(a.asset), reference_urdf=str(a.urdf),
                        env_options={"stage": 2, "terrain_curriculum": False, "enable_adaptive_sampling": False,
                                     "randomize_dynamics": True, "corrupt_observations": True, "max_action_delay": 2,
                                     **actuator_options(a.actuator_profile),
                                     **manifest["terrain"]})
        core = env.env.core
        names = [s["name"] for s in core.reference_database.seqs]
        episodes = manifest["episodes"] if a.limit is None else manifest["episodes"][:a.limit]
        a.output.parent.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            if episode["episode_id"] in seen:
                continue
            core._rng = np.random.default_rng(episode["seed"])
            torch.manual_seed(episode["seed"])
            core._terrain_families[0] = FAMILIES.index(episode["family"])
            core._terrain_levels[0] = episode["level"]
            core._forced_reference[0] = (names.index(episode["sequence"]), episode["start_frame"])
            obs = env.reset()
            totals = {k: 0. for k in ("joint_mse", "body_mse", "root_mse", "contact_tp", "contact_fp", "contact_fn")}
            horizon = round(episode["horizon_s"] / manifest["control_dt"])
            for step in range(1, horizon+1):
                with torch.no_grad():
                    action = policy(obs)
                obs, _, terminated, truncated, info = env.step(action)
                for key in totals:
                    totals[key] += float(info["tracking"][key][0])
                if bool(terminated[0] | truncated[0]):
                    break
            row = {**episode, **identity, "steps": step, "duration_s": step*manifest["control_dt"],
                   "success": step == horizon and bool(truncated[0]) and not bool(terminated[0]),
                   "terrain_out_of_bounds": bool(info.get("terrain_out_of_bounds", [False])[0])}
            row.update({(k+"_sum" if k.endswith("mse") else k): v for k, v in totals.items()})
            with a.output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            rows.append(row)
            report = {**identity, "complete": len(rows) == len(manifest["episodes"]),
                      "expected_episodes": len(manifest["episodes"]), **summarize(rows)}
            _write_metrics(a.output.with_suffix(".summary.json"), report)
            print(f"evaluation {len(rows)}/{len(manifest['episodes'])}", flush=True)
    finally:
        if env is not None:
            env.env.close()
        app.close()


if __name__ == "__main__":
    main()
