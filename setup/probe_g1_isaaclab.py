#!/usr/bin/env python
"""Load a licensed G1 USD in Isaac Lab and verify runtime names.

This is intentionally separate from the static XML checker: flattened USD has
no reliable text representation of its articulation.  The probe instantiates
the asset, reads Isaac Lab's joint/body names, and applies the same strict
29-DoF mapping used by :mod:`pgmt.envs.g1_env`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.asset.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        parser.error("--asset must be a USD/USDA/USDC file")
    if not args.asset.is_file():
        parser.error(f"asset does not exist: {args.asset}")

    simulation_app = None
    sim = None
    report = {"asset": str(args.asset.resolve()), "ok": False}

    def save_report() -> None:
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=True).app
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.actuators import ImplicitActuatorCfg
        from pgmt.envs.g1_env import REQUIRED_BODY_NAMES, validate_g1_mapping

        log_dir = os.environ.get(
            "PGMT_ISAACLAB_LOG_DIR",
            os.path.join(os.getcwd(), "data", "processed", "isaaclab_logs"),
        )
        os.makedirs(log_dir, exist_ok=True)
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0", log_dir=log_dir)
        sim = SimulationContext(sim_cfg)
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
        robot_cfg = ArticulationCfg(
            prim_path="/World/G1",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args.asset.resolve())),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.793)),
            # Use a broad expression only for loading.  The strict project
            # mapping below is the acceptance gate for the actual 29 drives.
            actuators={"g1_probe": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=80.0, damping=2.0,
                effort_limit_sim=120.0,
            )},
        )
        robot = Articulation(robot_cfg)
        sim.reset()
        robot.update(sim_cfg.dt)
        joint_names = list(robot.joint_names)
        body_names = list(robot.body_names)
        maps = validate_g1_mapping(joint_names, body_names, required_body_names=REQUIRED_BODY_NAMES)
        report.update({
            "joint_names": joint_names,
            "body_names": body_names,
            "joint_count": len(joint_names),
            "body_count": len(body_names),
            "joint_ids_canonical": maps["joint_ids"].tolist(),
            "body_ids_required": maps["body_ids"].tolist(),
            "ok": True,
        })
        save_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    finally:
        try:
            if sim is not None:
                sim.close()
        except Exception:
            pass
        try:
            if simulation_app is not None:
                try:
                    simulation_app.close(skip_cleanup=True)
                except TypeError:
                    simulation_app.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
