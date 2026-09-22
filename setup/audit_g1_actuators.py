"""CPU-only URDF/USD actuator audit. Requires pxr, never starts SimulationApp.

Run as ``python -m setup.audit_g1_actuators --urdf ... --usd ... --output ...``.
The JSON preserves raw USD attributes as well as SI conversions. Runtime
PhysX readback is a separate GPU check; this report does not claim it passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from data.retarget_lafan1 import G1_JOINT_NAMES, G1_JOINT_LIMITS
from pgmt.envs.g1_env import G1EnvConfig


def audit(urdf, usd):
    from pxr import Usd
    stage = Usd.Stage.Open(str(usd))
    if stage is None:
        raise ValueError("cannot open USD")
    # Fail rather than apply SI conversions to an unfamiliar stage scale.
    if stage.GetMetadata("metersPerUnit") != 1. or stage.GetMetadata("kilogramsPerUnit") != 1.:
        raise ValueError("audit requires a USD stage in meters and kilograms")
    xml = {j.attrib["name"].removesuffix("_joint"): j for j in ET.parse(urdf).getroot().findall("joint")
           if j.attrib.get("type") != "fixed"}
    prims = {p.GetName().removesuffix("_joint"): p for p in stage.Traverse()
             if p.GetTypeName() == "PhysicsRevoluteJoint"}
    if set(xml) != set(G1_JOINT_NAMES) or set(prims) != set(G1_JOINT_NAMES):
        raise ValueError("URDF and USD must each have exactly the canonical 29 G1 joints")
    cfg = G1EnvConfig()
    rows = []
    for i, name in enumerate(G1_JOINT_NAMES):
        j, p = xml[name], prims[name]
        limit = {k: float(v) for k, v in j.find("limit").attrib.items()}
        raw = {a.GetName(): a.Get() for a in p.GetAttributes()
               if a.GetName().startswith(("physics:", "drive:", "physxJoint:"))
               and isinstance(a.Get(), (int, float, str, bool))}
        drive = "drive:angular:physics:"
        if raw.get(drive + "type") != "force":
            raise ValueError("expected force-based USD joint drives")
        lower, upper = [math.radians(raw["physics:" + k + "Limit"]) for k in ("lower", "upper")]
        effort = raw[drive + "maxForce"]
        velocity = math.radians(raw["physxJoint:maxJointVelocity"])
        bodies = [p.GetRelationship("physics:body" + str(k)).GetTargets()[0].name for k in (0, 1)]
        axis = [float(x) for x in j.find("axis").attrib["xyz"].split()]
        usd_axis = [float(k == raw["physics:axis"]) for k in "XYZ"]
        checks = {
            "links": bodies == [j.find("parent").attrib["link"], j.find("child").attrib["link"]],
            "axis": axis == usd_axis,
            "position": all(math.isclose(x, y, abs_tol=2e-6) for x, y in zip((lower, upper), (limit["lower"], limit["upper"]))),
            "effort": math.isclose(effort, limit["effort"], abs_tol=1e-6),
            "velocity": math.isclose(velocity, limit["velocity"], abs_tol=1e-5),
            # Retargeting constants are rounded to four decimal places.
            "policy_position_within_5e_5_rad": all(math.isclose(x, y, abs_tol=5e-5) for x, y in zip(G1_JOINT_LIMITS[name], (limit["lower"], limit["upper"]))),
            "candidate_effort": cfg.torque_limit[i] == effort,
        }
        rows.append(dict(name=name, urdf=limit, usd_raw=raw, usd_bodies=bodies, checks=checks,
                         policy_position_rad=list(G1_JOINT_LIMITS[name]),
                         usd_effort_nm=effort, usd_velocity_rad_s=velocity,
                         usd_kp_nm_rad=raw[drive+"stiffness"] * 180 / math.pi,
                         usd_kd_nm_s_rad=raw[drive+"damping"] * 180 / math.pi,
                         usd_target_rad=math.radians(raw[drive+"targetPosition"]),
                         legacy_effort_nm=120., candidate_effort_nm=cfg.torque_limit[i],
                         configured_kp_nm_rad=cfg.stiffness[i], configured_kd_nm_s_rad=cfg.damping[i],
                         configured_reset_rad=cfg.default_joint_pos[i]))
    paths = {"urdf": Path(urdf).resolve(), "usd": Path(usd).resolve()}
    return {"schema": "pgmt.actuator_audit.v1", "runtime_verified": False,
            "assets": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
            "all_checks_pass": all(all(r["checks"].values()) for r in rows), "joints": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.urdf, args.usd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Audit evidence should not silently replace an earlier report.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    columns = ["name", "usd_effort_nm", "usd_velocity_rad_s", "usd_kp_nm_rad", "usd_kd_nm_s_rad",
               "usd_target_rad", "legacy_effort_nm", "candidate_effort_nm", "configured_kp_nm_rad",
               "configured_kd_nm_s_rad", "configured_reset_rad"]
    with args.output.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["joints"])
    print(json.dumps({"output": str(args.output), "joints": len(result["joints"]),
                      "all_checks_pass": result["all_checks_pass"], "runtime_verified": False}))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
