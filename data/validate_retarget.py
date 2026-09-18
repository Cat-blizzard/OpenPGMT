"""Validate stored retargeting artifacts and record their exact provenance.

This checks exported data integrity, not dynamic feasibility. Velocity spikes,
source contact agreement, and IK errors remain diagnostics for physical review.
No motion or quality-report files are regenerated or modified by this command.

Examples::

    # Audit exported NPZs without touching them or requiring the source BVHs.
    python -m data.validate_retarget --npz-dir data/processed/lafan1_g1 \
        --out /tmp/lafan1_g1_validation.json --allow-violations

    python -m data.validate_retarget --bvh-dir data/raw/lafan1 \
        --npz-dir data/processed/lafan1_g1_fixed \
        --quality-report data/processed/quality_report_fixed.csv \
        --generation-record data/processed/generation_fixed.json \
        --expected-sequences 77 --out data/processed/acceptance_fixed.json \
        --provenance-out data/processed/provenance_fixed.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Dict, Optional
import zipfile

import numpy as np

from data.retarget_lafan1 import (
    G1_JOINT_LIMITS, G1_JOINT_NAMES, contact_labels, g1_forward_kinematics,
)
from pgmt.cfg.assumptions import get


# These checks deliberately operate on the exported artifact.  They do not
# regenerate motion or rewrite any source/report files.  The default directory
# is the one used by ``data.retarget_lafan1`` for training references.
DEFAULT_NPZ_DIR = Path("data/processed/lafan1_g1")
_JOINT_LIMIT_TOLERANCE = 1e-6
_QVEL_ERROR_TOLERANCE = 1e-4
_ROOT_QUATERNION_TOLERANCE = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    path = path.resolve()
    try:
        name = str(path.relative_to(root))
    except ValueError:
        name = str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def source_metadata(path: Path):
    """Read source frame metadata without doing an unnecessary second BVH FK."""
    with path.open(encoding="utf-8") as source:
        for line in source:
            match = re.fullmatch(r"\s*Frames:\s*(\d+)\s*", line)
            if match:
                frames = int(match.group(1))
                break
        else:
            raise ValueError("source BVH has no Frames declaration")
        for line in source:
            match = re.fullmatch(r"\s*Frame Time:\s*([^\s]+)\s*", line)
            if match:
                dt = float(match.group(1))
                if frames <= 0 or not np.isfinite(dt) or dt <= 0:
                    raise ValueError("source BVH has invalid frame metadata")
                return frames, dt
    raise ValueError("source BVH has no Frame Time declaration")


def validate_sequence(data: Dict, source_frames: Optional[int] = None,
                      source_dt: Optional[float] = None) -> dict:
    """Check one stored NPZ; large speeds are reported, not silently repaired.

    ``source_frames`` and ``source_dt`` are optional so the checker can be used
    as a standalone artifact audit when the original BVH directory is not
    available.  If provided, they are still checked against the artifact.
    Structural problems raise ``ValueError``; quality violations are returned
    in ``failed_checks`` for the caller to apply its chosen policy.
    """
    if source_frames is None:
        if "qpos" not in data:
            raise ValueError("qpos is missing")
        qpos_shape = np.asarray(data["qpos"]).shape
        if len(qpos_shape) != 2:
            raise ValueError("qpos must have shape (frames, 29)")
        source_frames = int(qpos_shape[0])
    source_frames = int(source_frames)
    if source_frames <= 0:
        raise ValueError("qpos must contain at least one frame")
    if list(data.get("joint_names", [])) != G1_JOINT_NAMES:
        raise ValueError("joint_names do not match canonical G1 order")
    shapes = {"qpos": (source_frames, 29), "qvel": (source_frames, 29),
              "root_pos": (source_frames, 3), "root_rot": (source_frames, 4),
              "contacts": (source_frames, 2)}
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in ("scale", "frame_time"):
        if (key not in data or data[key].shape != ()
                or not np.isfinite(data[key]) or data[key] <= 0):
            raise ValueError(f"{key} must be a positive finite scalar")
    if not np.isin(data["contacts"], (0, 1)).all():
        raise ValueError("contacts must contain boolean labels")
    dt = float(data["frame_time"])
    if source_dt is not None and not np.isclose(dt, source_dt, rtol=1e-5, atol=0):
        raise ValueError("frame_time differs from source BVH")

    qpos = data["qpos"].astype(np.float64)
    qvel = data["qvel"].astype(np.float64)
    lo = np.array([G1_JOINT_LIMITS[name][0] for name in G1_JOINT_NAMES])
    hi = np.array([G1_JOINT_LIMITS[name][1] for name in G1_JOINT_NAMES])
    limit_excess = np.maximum(qpos - hi, lo - qpos)
    limit_count = int((limit_excess > _JOINT_LIMIT_TOLERANCE).sum())
    expected_qvel = np.zeros_like(qpos)
    expected_qvel[1:] = np.diff(qpos, axis=0) / dt
    velocity_error = np.abs(expected_qvel - qvel)
    velocity_bad = int((velocity_error > _QVEL_ERROR_TOLERANCE).sum())
    rotation_error = np.abs(np.linalg.norm(data["root_rot"], axis=-1) - 1)

    pos = g1_forward_kinematics(data["qpos"], data["root_pos"], data["root_rot"])
    feet = np.stack([pos["left_ankle_roll_link"], pos["right_ankle_roll_link"]], axis=1)
    labels = contact_labels(feet, dt, get("A14").value.foot_vel_thresh)
    contact_bad = int((labels != data["contacts"]).sum())
    failures = []
    if limit_count:
        failures.append("joint_limits")
    if velocity_bad:
        failures.append("qvel_direct_difference")
    if rotation_error.max() > _ROOT_QUATERNION_TOLERANCE:
        failures.append("root_quaternion_norm")
    if contact_bad:
        failures.append("stored_contact_consistency")
    return {
        "passed": not failures, "failed_checks": failures, "frames": source_frames,
        "frame_time_s": dt, "duration_s": source_frames * dt,
        "joint_limit_violation_samples": limit_count,
        "joint_limit_max_excess_rad": max(0.0, float(limit_excess.max())),
        "qvel_mismatch_samples": velocity_bad,
        "qvel_max_abs_error_rad_s": float(velocity_error.max()),
        "root_quaternion_max_norm_error": float(rotation_error.max()),
        "contact_mismatch_samples": contact_bad,
        "contact_true_samples": int(data["contacts"].sum()),
        "velocity_spike_samples_gt_30_rad_s": int((np.abs(qvel) > 30).sum()),
        "max_abs_qvel_rad_s": float(np.abs(qvel).max()),
        "min_ankle_origin_z_m": float(feet[..., 2].min()),
    }


def validate_quality_report(path: Path, sequences: Dict[str, dict]) -> dict:
    """Ensure diagnostics describe this full set of successfully checked NPZs."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row.get("seq") for row in rows]
    if len(names) != len(set(names)) or set(names) != set(sequences):
        raise ValueError("quality report sequence set differs from NPZ set")
    metrics = ("fitted_err_cm", "holdout_err_cm", "min_foot_z", "spike_pct",
               "contact_agree", "contact_label_agree", "limit_over_pct")
    for row in rows:
        if row.get("reference_source") != "stored_npz":
            raise ValueError("quality report must evaluate stored_npz")
        seq = sequences[row["seq"]]
        if int(row["frames"]) != seq["frames"]:
            raise ValueError(f"quality report frame count differs: {row['seq']}")
        for key in metrics:
            if not np.isfinite(float(row[key])):
                raise ValueError(f"quality report {key} is not finite: {row['seq']}")
        n = seq["frames"]
        measured = {
            "limit_over_pct": 100 * seq["joint_limit_violation_samples"] / (n * 29),
            "spike_pct": 100 * seq["velocity_spike_samples_gt_30_rad_s"] / (n * 29),
            "contact_label_agree": 100 * (1 - seq["contact_mismatch_samples"] / (n * 2)),
            "min_foot_z": seq["min_ankle_origin_z_m"],
        }
        for key, expected in measured.items():
            if not np.isclose(float(row[key]), expected, rtol=1e-7, atol=1e-8):
                raise ValueError(f"quality report {key} differs from NPZ: {row['seq']}")
    groups = {"all": rows, "ground": [row for row in rows if row.get("type") == "ground"]}
    return {
        group: {key: {"min": min(float(row[key]) for row in values),
                      "mean_per_sequence": float(np.mean([float(row[key]) for row in values])),
                      "max": max(float(row[key]) for row in values)} for key in metrics}
        for group, values in groups.items() if values
    }


def _load_npz(path: Path) -> dict:
    """Load one artifact without allowing object-array code execution."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _ground_filter_stats(artifacts: Dict[str, Path], sequences: Dict[str, dict]) -> dict:
    """Report what excluding ``ground*`` files would remove from the corpus."""
    ground_names = sorted(name for name in artifacts if name.lower().startswith("ground"))
    retained_names = sorted(name for name in artifacts if name not in ground_names)

    def frame_total(names):
        return int(sum(sequences[name]["frames"] for name in names if name in sequences))

    total_frames = frame_total(sorted(artifacts))
    ground_frames = frame_total(ground_names)
    total_duration = float(sum(sequences[name]["duration_s"] for name in artifacts
                               if name in sequences))
    ground_duration = float(sum(sequences[name]["duration_s"] for name in ground_names
                                if name in sequences))
    return {
        "rule": "filename starts with 'ground' (case-insensitive)",
        "total_sequences": len(artifacts),
        "ground_sequences": len(ground_names),
        "retained_sequences": len(retained_names),
        "total_frames_checked": total_frames,
        "ground_frames_checked": ground_frames,
        "retained_frames_checked": max(0, total_frames - ground_frames),
        "total_duration_s_checked": total_duration,
        "ground_duration_s_checked": ground_duration,
        "retained_duration_s_checked": max(0.0, total_duration - ground_duration),
        "ground_sequence_fraction": (len(ground_names) / len(artifacts) if artifacts else 0.0),
        "ground_frame_fraction": (ground_frames / total_frames if total_frames else 0.0),
    }


def validate_dataset(bvh_dir: Optional[Path], npz_dir: Path,
                     expected_sequences: Optional[int] = None,
                     quality_report: Optional[Path] = None,
                     allow_violations: bool = False) -> dict:
    """Validate a collection of stored NPZs.

    When ``bvh_dir`` is ``None`` this is an artifact-only audit: frame counts
    and ``frame_time`` are taken from each NPZ and no source files are needed.
    Supplying a BVH directory additionally checks source/artifact membership and
    frame metadata.  ``allow_violations`` permits quality diagnostics (joint
    limits, finite-difference velocity, quaternion/contact consistency) while
    still failing malformed or unreadable artifacts.
    """
    npz_dir = Path(npz_dir)
    bvh_dir = Path(bvh_dir) if bvh_dir is not None else None
    sources = ({path.stem: path for path in bvh_dir.glob("*.bvh")}
               if bvh_dir is not None else {})
    artifacts = {path.stem: path for path in npz_dir.glob("*.npz")}
    failures = []
    if bvh_dir is not None and not sources:
        failures.append("no source BVH files")
    if not artifacts:
        failures.append("no NPZ files")
    expected_count = len(sources) if bvh_dir is not None else len(artifacts)
    if expected_sequences is not None and expected_count != expected_sequences:
        kind = "source" if bvh_dir is not None else "artifact"
        failures.append(f"expected {expected_sequences} {kind} sequences, found {expected_count}")
    missing = sorted(set(sources) - set(artifacts)) if bvh_dir is not None else []
    extra = sorted(set(artifacts) - set(sources)) if bvh_dir is not None else []
    if missing or extra:
        failures.append("source and NPZ sequence sets differ")
    sequences = {}
    for name in sorted(artifacts):
        try:
            frames = dt = None
            if bvh_dir is not None and name in sources:
                frames, dt = source_metadata(sources[name])
            result = validate_sequence(_load_npz(artifacts[name]), frames, dt)
            result["is_ground"] = name.lower().startswith("ground")
            sequences[name] = result
            if result["failed_checks"] and not allow_violations:
                failures.append(name + ": " + ", ".join(result["failed_checks"]))
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            failures.append(f"{name}: {exc}")
    quality = None
    if quality_report is not None:
        try:
            quality = validate_quality_report(quality_report, sequences)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(f"quality report: {exc}")
    summary = {key: sum(seq[key] for seq in sequences.values()) for key in (
        "frames", "duration_s", "joint_limit_violation_samples", "qvel_mismatch_samples",
        "contact_mismatch_samples", "contact_true_samples", "velocity_spike_samples_gt_30_rad_s")}
    for key in ("qvel_max_abs_error_rad_s", "joint_limit_max_excess_rad",
                "root_quaternion_max_norm_error", "max_abs_qvel_rad_s"):
        summary[key] = max((seq[key] for seq in sequences.values()), default=None)
    return {
        "schema_version": 1, "passed": not failures, "failures": failures,
        "allow_violations": bool(allow_violations),
        "scope": "stored artifact integrity; dynamic feasibility is not certified",
        "thresholds": {"joint_limit_tolerance_rad": _JOINT_LIMIT_TOLERANCE,
                       "qvel_error_tolerance_rad_s": _QVEL_ERROR_TOLERANCE,
                       "root_quaternion_norm_tolerance": _ROOT_QUATERNION_TOLERANCE,
                       "diagnostic_velocity_spike_threshold_rad_s": 30},
        "source_sequences": len(sources), "artifact_sequences": len(artifacts),
        "checked_sequences": len(sequences), "missing_sequences": missing,
        "extra_sequences": extra, "summary": summary, "quality_diagnostics": quality,
        "ground_filter": _ground_filter_stats(artifacts, sequences),
        "sequences": sequences,
    }


def validate_artifacts(npz_dir: Path, allow_violations: bool = False) -> dict:
    """Convenience wrapper for the standalone, BVH-free artifact audit."""
    return validate_dataset(None, npz_dir, allow_violations=allow_violations)


def verify_generation_record(record: dict, root: Path, bvh_dir: Path) -> list:
    """Refuse to attribute products to code/source files that changed mid-run."""
    errors = []
    expected_sources = {str((root / item["path"]).resolve()) for item in record["source_files"]}
    actual_sources = {str(path.resolve()) for path in bvh_dir.glob("*.bvh")}
    if expected_sources != actual_sources:
        errors.append("generation record source file set differs")
    for kind in ("algorithm_files", "source_files", "preserved_files"):
        for item in record.get(kind, []):
            path = root / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                errors.append(f"generation record changed {kind}: {item['path']}")
    return errors


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                     suffix=".partial.json", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", type=Path,
                        help="optional source BVH directory; omit for an NPZ-only audit")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--expected-sequences", type=int)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--generation-record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path,
                        help="optional separate provenance JSON (never an input file)")
    parser.add_argument("--allow-violations", action="store_true",
                        help="report quality violations but return success; malformed files still fail")
    args = parser.parse_args()
    protected = {path.resolve() for directory, pattern in ((args.bvh_dir, "*.bvh"),
                                                           (args.npz_dir, "*.npz"))
                 if directory for path in directory.glob(pattern)}
    protected.update(path.resolve() for path in (args.quality_report, args.generation_record) if path)
    if args.provenance_out and args.out.resolve() == args.provenance_out.resolve():
        parser.error("--out and --provenance-out must be distinct")
    if args.out.resolve() in protected or (args.provenance_out and
                                           args.provenance_out.resolve() in protected):
        parser.error("output paths must be distinct from each other and all input files")

    root = Path(__file__).resolve().parents[1]
    result = validate_dataset(args.bvh_dir, args.npz_dir, args.expected_sequences,
                              args.quality_report, allow_violations=args.allow_violations)
    record = None
    if args.generation_record and args.bvh_dir is not None:
        try:
            record = json.loads(args.generation_record.read_text(encoding="utf-8"))
            result["failures"].extend(verify_generation_record(record, root, args.bvh_dir))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["failures"].append(f"generation record: {exc}")
    result["passed"] = not result["failures"]
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out, result)
    if args.provenance_out:
        provenance = {
            "schema_version": 1, "checked_at_utc": result["checked_at_utc"],
            "generation": record,
            "validation": {"command": [sys.executable, "-m", "data.validate_retarget", *sys.argv[1:]],
                           "python_version": platform.python_version(), "numpy_version": np.__version__,
                           "tool": file_record(Path(__file__), root), "passed": result["passed"],
                           "git_commit_at_validation": subprocess.check_output(
                               ["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()},
            "source_files": ([file_record(path, root) for path in sorted(args.bvh_dir.glob("*.bvh"))]
                             if args.bvh_dir is not None else []),
            "artifact_files": [file_record(path, root) for path in sorted(args.npz_dir.glob("*.npz"))],
            "acceptance_report": file_record(args.out, root),
            "quality_report": file_record(args.quality_report, root) if args.quality_report else None,
        }
        write_json(args.provenance_out, provenance)
    print(json.dumps({"passed": result["passed"], "checked_sequences": result["checked_sequences"],
                      "summary": result["summary"], "ground_filter": result["ground_filter"],
                      "failures": result["failures"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
