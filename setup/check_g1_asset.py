#!/usr/bin/env python3
"""Inspect a local G1 asset without copying or modifying third-party files.

The project deliberately keeps the Unitree/ProtoMotions asset outside git.  Run
this checker with paths into that checkout to produce an auditable manifest::

    python setup/check_g1_asset.py \
      --urdf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf \
      --mjcf /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/mjcf/g1.xml \
      --usd /data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/usd/g1.usd \
      --json /tmp/g1_asset_manifest.json

USD inspection is intentionally conservative.  A USD file is reported as a
candidate and its file hash is recorded; joint-level inspection needs an
Isaac-Sim process (the normal environment does not expose ``pxr`` directly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

EXPECTED = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
    "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
    "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
    "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "right_wrist_pitch", "right_wrist_yaw",
]
EXPECTED_XML = [f"{name}_joint" for name in EXPECTED]
EXPECTED_BODIES = (
    "pelvis", "torso_link",
    *tuple(f"{name}_link" for name in EXPECTED if name not in {"waist_yaw", "waist_roll", "waist_pitch"}),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_info(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    p = Path(path).expanduser().resolve()
    out: dict[str, Any] = {"path": str(p), "exists": p.is_file()}
    if p.is_file():
        out.update({"bytes": p.stat().st_size, "sha256": _sha256(p)})
    return out


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _axis(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        return [float(x) for x in value.split()]
    except ValueError:
        return None


def _license_candidates(path: Path) -> list[str]:
    """List nearby license files; this does not infer permission to redistribute."""
    found: list[str] = []
    for parent in [path.parent, *path.parents]:
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
            candidate = parent / name
            if candidate.is_file() and str(candidate) not in found:
                found.append(str(candidate))
    # The ProtoMotions layout keeps the G1 mesh notice beside (rather than
    # above) the URDF/MJCF/USD.  Include it when present so a manifest records
    # both the project and model notices.
    for parent in [path.parent, *path.parents]:
        candidate = parent / "mesh" / "G1" / "LICENSE"
        if candidate.is_file() and str(candidate) not in found:
            found.append(str(candidate))
    return found


def analyze_urdf(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"format": "urdf", **_path_info(p), "licenses": _license_candidates(p)}
    if not p.is_file():
        result["errors"] = ["URDF file does not exist"]
        return result
    root = ET.parse(p).getroot()
    joints: list[dict[str, Any]] = []
    meshes: list[str] = []
    for joint in root.findall("joint"):
        typ = joint.attrib.get("type", "")
        if typ not in ("revolute", "continuous", "prismatic"):
            continue
        limit = joint.find("limit")
        parent = joint.find("parent")
        child = joint.find("child")
        axis_elem = joint.find("axis")
        joints.append({
            "name": joint.attrib.get("name"),
            "type": typ,
            "parent": parent.attrib.get("link") if parent is not None else None,
            "child": child.attrib.get("link") if child is not None else None,
            "axis": _axis(axis_elem.attrib.get("xyz") if axis_elem is not None else None),
            "lower": _float(limit.attrib.get("lower")) if limit is not None else None,
            "upper": _float(limit.attrib.get("upper")) if limit is not None else None,
        })
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            meshes.append(filename)
    result["joints"] = joints
    result["joint_names"] = [j["name"] for j in joints]
    result["joint_count"] = len(joints)
    result["body_names"] = sorted({j["child"] for j in joints if j["child"]} | {"pelvis", "torso_link"})
    result["actuators"] = []  # URDF has no actuator declarations.
    result["mesh_refs"] = meshes
    resolved = [(p.parent / ref).resolve() for ref in meshes]
    result["missing_meshes"] = [ref for ref, target in zip(meshes, resolved) if not target.is_file()]
    result["checks"] = {
        "29_revolute_joints": len(joints) == 29 and all(j["type"] == "revolute" for j in joints),
        "expected_joint_order": result["joint_names"] == EXPECTED_XML,
        "mesh_files_resolve": not result["missing_meshes"],
        "actuators_declared": False,
        "required_body_names_present": set(EXPECTED_BODIES).issubset(result["body_names"]),
    }
    return result


def analyze_mjcf(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"format": "mjcf", **_path_info(p), "licenses": _license_candidates(p)}
    if not p.is_file():
        result["errors"] = ["MJCF file does not exist"]
        return result
    root = ET.parse(p).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_refs = [m.attrib.get("file") for m in root.findall(".//asset/mesh") if m.attrib.get("file")]
    resolved = [(p.parent / meshdir / ref).resolve() for ref in mesh_refs]
    joint_by_name: dict[str, dict[str, Any]] = {}
    def walk(body: ET.Element, body_name: str | None = None) -> None:
        current = body.attrib.get("name", body_name)
        for joint in body.findall("joint"):
            name = joint.attrib.get("name")
            if name and joint.attrib.get("type", "hinge") != "free":
                rng = (joint.attrib.get("range") or "").split()
                joint_by_name[name] = {
                    "name": name,
                    "body": current,
                    "type": joint.attrib.get("type", "hinge"),
                    "axis": _axis(joint.attrib.get("axis")),
                    "lower": _float(rng[0]) if len(rng) > 0 else None,
                    "upper": _float(rng[1]) if len(rng) > 1 else None,
                    "effort_lower": _float((joint.attrib.get("actuatorfrcrange") or "").split()[0]) if (joint.attrib.get("actuatorfrcrange") or "").split() else None,
                    "effort_upper": _float((joint.attrib.get("actuatorfrcrange") or "").split()[-1]) if (joint.attrib.get("actuatorfrcrange") or "").split() else None,
                }
        for child in body.findall("body"):
            walk(child, current)
    for body in root.findall(".//worldbody/body"):
        walk(body)
    actuators = []
    for motor in root.findall(".//actuator/motor"):
        actuators.append({"name": motor.attrib.get("name"), "joint": motor.attrib.get("joint"), "gear": motor.attrib.get("gear")})
    joints = [joint_by_name[name] for name in joint_by_name]
    result.update({
        "joint_count": len(joints),
        "joint_names": [j["name"] for j in joints],
        "joints": joints,
        "body_names": sorted({j["body"] for j in joints if j["body"]} | {"pelvis", "torso_link"}),
        "actuators": actuators,
        "mesh_refs": mesh_refs,
        "missing_meshes": [ref for ref, target in zip(mesh_refs, resolved) if not target.is_file()],
    })
    result["checks"] = {
        "29_hinge_joints": len(joints) == 29 and all(j["type"] == "hinge" for j in joints),
        "expected_joint_order": result["joint_names"] == EXPECTED_XML,
        "30_one_to_one_actuators": len(actuators) == 29 and [a["joint"] for a in actuators] == result["joint_names"],
        "mesh_files_resolve": not result["missing_meshes"],
        "required_body_names_present": set(EXPECTED_BODIES).issubset(result["body_names"]),
    }
    return result


def analyze_usd(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"format": "usd", **_path_info(p), "licenses": _license_candidates(p)}
    if p.is_file():
        with p.open("rb") as f:
            result["magic"] = f.read(8).decode("ascii", errors="replace")
        result["checks"] = {"usd_file_present": True, "joint_mapping_inspected": False}
        result["note"] = "Binary USD joint mapping requires Isaac Sim/pxr; use the runtime probe before training."
    else:
        result["checks"] = {"usd_file_present": False, "joint_mapping_inspected": False}
    return result


def build_manifest(urdf: str | None, mjcf: str | None, usd: str | None) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    if urdf:
        assets["urdf"] = analyze_urdf(urdf)
    if mjcf:
        assets["mjcf"] = analyze_mjcf(mjcf)
    if usd:
        assets["usd"] = analyze_usd(usd)
    # URDF intentionally has no actuator section; its ``actuators_declared``
    # field is informational and must not make an otherwise complete URDF fail.
    checks = [check for info in assets.values() for name, check in info.get("checks", {}).items()
              if name != "actuators_declared"]
    return {"schema": "pgmt.g1_asset_manifest.v1", "assets": assets, "all_static_checks_pass": bool(checks) and all(checks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf")
    parser.add_argument("--mjcf")
    parser.add_argument("--usd")
    parser.add_argument("--json", dest="json_path", help="write manifest JSON to this path")
    args = parser.parse_args(argv)
    if not any((args.urdf, args.mjcf, args.usd)):
        parser.error("provide at least one of --urdf, --mjcf, --usd")
    manifest = build_manifest(args.urdf, args.mjcf, args.usd)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_path:
        Path(args.json_path).expanduser().write_text(payload + "\n", encoding="utf-8")
    # Missing meshes and absent MJCF actuators are expected for URDF-only probes,
    # so the checker is informative by default. CI can inspect ``checks``.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
