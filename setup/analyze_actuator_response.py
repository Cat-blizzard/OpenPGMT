"""CPU analysis of paired, short actuator probes; no policy/evaluation claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def arrays(run):
    return {key: np.asarray([s["pre_reset"][key] for s in run["steps"]]) for key in run["steps"][0]["pre_reset"]}


def prefix_steps(run):
    return np.array([len(run["steps"]) if end is None else end["step"] for end in run["first_episode_end"]])


def summarize(run, ends=None):
    data = arrays(run)
    stops = prefix_steps(run) if ends is None else ends
    mask = np.arange(len(run["steps"]))[:, None] < stops[None]
    qvel = np.abs(data["qvel"])
    state_accel = np.linalg.norm(data["state_ee_accel"], axis=-1)
    reference_accel = np.linalg.norm(data["reference_ee_accel"], axis=-1)
    error = np.linalg.norm(data["state_ee_accel"]-data["reference_ee_accel"], axis=-1)
    torque = np.abs(data["estimated_torque_nm"])
    limit = np.asarray(run["environment_config"]["torque_limit"])
    error_masked = np.where(mask[..., None], error, -1)
    step, eid, body = np.unravel_index(error_masked.argmax(), error.shape)
    tracking = {k: np.asarray([s["per_env_tracking"][k] for s in run["steps"]])
                for k in ("joint_mse", "body_mse", "root_mse")}
    per_joint_peak = np.where(mask[..., None], qvel, -1).reshape(-1, 29).max(0)
    top_joints = per_joint_peak.argsort()[::-1][:5]
    return {
        "observed_first_episode_s": (stops * run["environment_config"]["control_dt"]).tolist(),
        "failures": sum(end is not None and end["terminated"] for end in run["first_episode_end"]),
        "censored_at_budget": sum(end is None for end in run["first_episode_end"]),
        "first_end_reasons": [None if end is None else end["reasons"] for end in run["first_episode_end"]],
        "max_joint_speed_rad_s": float(qvel[mask].max()),
        "max_joint_name": run["initial"]["joint_names"][int(np.where(mask[..., None], qvel, -1).reshape(-1, 29).max(0).argmax())],
        "fastest_joints": {run["initial"]["joint_names"][int(i)]: float(per_joint_peak[i]) for i in top_joints},
        "overspeed_transition_count": int((qvel[mask].max(-1) > 45).sum()),
        "min_root_height_m": float(data["root_pos"][..., 2][mask].min()),
        "max_state_ee_accel_m_s2": float(state_accel[mask].max()),
        "max_reference_ee_accel_m_s2": float(reference_accel[mask].max()),
        "max_ee_error_m_s2": float(error[mask].max()),
        "p95_ee_error_m_s2": float(np.quantile(error[mask], .95)),
        "estimated_saturation_fraction": float((torque[mask] >= .99*limit).mean()),
        "first_step": {"max_joint_speed_rad_s": float(qvel[0].max()),
                       "max_state_ee_accel_m_s2": float(state_accel[0].max()),
                       "max_reference_ee_accel_m_s2": float(reference_accel[0].max()),
                       "max_target_change_from_reset_rad": float(np.abs(data["target"][0] - np.asarray(run["initial"]["qpos"])).max())}
                       if "target" in data and "qpos" in run["initial"] else None,
        **{k.replace("mse", "rmse"): float(np.sqrt(v[mask].mean())) for k, v in tracking.items()},
        "worst_ee_event": {"step": int(step+1), "env_id": int(eid), "body": run["ee_names"][body],
            "episode_step": int(data["episode_steps"][step, eid]),
            "reference_seq_idx": int(data["reference_seq_idx"][step, eid]),
            "reference_frame": float(data["reference_frame"][step, eid]),
            "state_accel": data["state_ee_accel"][step, eid, body].tolist(),
            "reference_accel": data["reference_ee_accel"][step, eid, body].tolist()}}


def analyze(root):
    runs, failed = {}, []
    for path in sorted(root.glob("*/response.json")):
        run = json.loads(path.read_text())
        if run["status"] != "completed":
            failed.append({"path": str(path), "status": run["status"], "error": run.get("error")})
            continue
        if len(run["steps"]) != run["arguments"]["steps"] or (path.parent/"exit_code").read_text().strip() != "0":
            raise ValueError(f"incomplete result: {path}")
        key = (run["arguments"]["target_mode"], run["arguments"]["seed"], run["arguments"]["actuator_profile"])
        if key in runs:
            raise ValueError(f"duplicate completed condition: {key}")
        readback = torch.load(path.with_suffix(".diagnostics")/"actuators_000.pt", map_location="cpu", weights_only=False)
        expected = torch.tensor(run["environment_config"]["torque_limit"])
        torch.testing.assert_close(readback["effort_nm"], expected.expand_as(readback["effort_nm"]))
        torch.testing.assert_close(readback["stiffness_nm_rad"], torch.full_like(readback["stiffness_nm_rad"], 80.))
        torch.testing.assert_close(readback["damping_nm_s_rad"], torch.full_like(readback["damping_nm_s_rad"], 2.))
        runs[key] = (run, path, readback)
    pairs = []
    for mode, seed in sorted({(k[0], k[1]) for k in runs}):
        keys = [(mode, seed, p) for p in ("legacy_uniform120", "asset_effort_v1")]
        if not all(k in runs for k in keys):
            continue
        (old, old_path, old_rb), (new, new_path, new_rb) = [runs[k] for k in keys]
        old_source = json.loads((old_path.parent/"manifest.json").read_text())["source_sha256"]
        new_source = json.loads((new_path.parent/"manifest.json").read_text())["source_sha256"]
        for key in old_source:
            if key.startswith("pgmt/") or key in ("setup/check_actuator_response.py", "setup/run_actuator_check.sh"):
                if old_source[key] != new_source.get(key):
                    raise ValueError(f"unmatched experiment source: {key}")
        for key in old["environment_config"]:
            if key != "torque_limit" and old["environment_config"][key] != new["environment_config"][key]:
                raise ValueError(f"unmatched configuration: {mode}/{seed}/{key}")
        if old["initial"] != new["initial"]:
            raise ValueError(f"unmatched initial state: {mode}/{seed}")
        torch.testing.assert_close(old_rb["velocity_rad_s"], new_rb["velocity_rad_s"])
        common = np.minimum(prefix_steps(old), prefix_steps(new))
        old_data, new_data = arrays(old), arrays(new)
        mask = np.arange(len(old["steps"]))[:, None] < common[None]
        for key in ("target", "reference_seq_idx", "reference_frame"):
            if not np.array_equal(old_data[key][mask], new_data[key][mask]):
                raise ValueError(f"unmatched input before first end: {mode}/{seed}/{key}")
        pairs.append({"mode": mode, "seed": seed, "initial_and_inputs_match": True, "experiment_source_matches": True,
                      "legacy_path": str(old_path), "candidate_path": str(new_path),
                      "legacy": summarize(old), "candidate": summarize(new),
                      "common_prefix_steps": common.tolist(),
                      "common_prefix_legacy": summarize(old, common),
                      "common_prefix_candidate": summarize(new, common)})
    groups = {}
    for mode in sorted({p["mode"] for p in pairs}):
        subset = [p for p in pairs if p["mode"] == mode]
        groups[mode] = {"seeds": [p["seed"] for p in subset]}
        for profile in ("legacy", "candidate"):
            metrics = {
                "mean_first_episode_s": [np.mean(p[profile]["observed_first_episode_s"]) for p in subset],
                "common_joint_rmse_rad": [p["common_prefix_"+profile]["joint_rmse"] for p in subset],
                "common_body_rmse_m": [p["common_prefix_"+profile]["body_rmse"] for p in subset],
                "first_step_peak_state_ee_accel_m_s2": [p[profile]["first_step"]["max_state_ee_accel_m_s2"] for p in subset],
                "common_p95_ee_error_m_s2": [p["common_prefix_"+profile]["p95_ee_error_m_s2"] for p in subset],
            }
            groups[mode][profile] = {
                "seed_summary": {k: {"mean": float(np.mean(v)), "sample_sd": float(np.std(v, ddof=1)) if len(v)>1 else None,
                                     "values": v} for k, v in metrics.items()},
                "first_episode_failures": sum(p[profile]["failures"] for p in subset),
                "first_episode_count": sum(len(p[profile]["observed_first_episode_s"]) for p in subset),
                "overspeed_transitions": sum(p[profile]["overspeed_transition_count"] for p in subset),
                "max_joint_speed_rad_s": max(p[profile]["max_joint_speed_rad_s"] for p in subset),
            }
    return {"scope": "first episodes only, 3-second diagnostic budget; censored durations are not 30-second success",
            "groups": groups,
            "pairs": pairs, "failed_attempts": failed, "completed_runs": len(runs),
            "unpaired_completed_runs": len(runs)-2*len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"pairs": len(result["pairs"]), "completed_runs": result["completed_runs"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
