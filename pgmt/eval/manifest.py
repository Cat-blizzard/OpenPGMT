"""Freeze identical episodes and corruption seeds across compared policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.terrain.generators import FAMILIES
from pgmt.envs.terrain.curriculum import is_compatible, motion_category
from pgmt.envs.randomization import RANDOMIZATION_SPEC


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(database, seed=0, per_cell=192):
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    rng = np.random.default_rng(seed)
    episodes = []
    for family in FAMILIES:
        for level in range(10):
            usable = []
            for seq in database.seqs:
                starts = len(seq["qpos"]) - math.ceil(30. / seq["frame_time"])
                if starts > 0 and is_compatible(motion_category(seq["name"]), family, level):
                    usable.append((seq["name"], starts))
            if not usable:
                raise ValueError(f"no compatible full 30-second reference for {family}/L{level}")
            weights = np.array([x[1] for x in usable], dtype=float)
            weights /= weights.sum()
            for _ in range(per_cell):
                name, starts = usable[int(rng.choice(len(usable), p=weights))]
                episodes.append({"episode_id": len(episodes), "family": family, "level": level,
                                 "sequence": name, "start_frame": int(rng.integers(starts)),
                                 "seed": int(rng.integers(2**31)), "horizon_s": 30.})
    return {"schema": "pgmt_matched_eval_v2", "seed": seed, "per_cell": per_cell,
            "control_dt": .02, "sim_dt": .005, "action_contract": "joint_targets_tanh_v2",
            "terrain": {"terrain_repeats": 8, "terrain_resolution": .1},
            "randomization": RANDOMIZATION_SPEC,
            "selection": "frame-uniform within compatible motions with >=30s remaining; coarse rules are project assumptions",
            "episodes": episodes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-data", type=Path, required=True)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-cell", type=int, default=192)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("manifest already exists; use it for every policy or choose a new path")
    manifest = make_manifest(MotionDatabase(str(a.reference_data)), a.seed, a.per_cell)
    manifest["reference_files"] = {f.name: file_hash(f) for f in sorted(a.reference_data.glob("*.npz"))}
    manifest["asset_sha256"], manifest["urdf_sha256"] = file_hash(a.asset), file_hash(a.urdf)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"froze {len(manifest['episodes'])} episodes in {a.output}")


if __name__ == "__main__":
    main()
