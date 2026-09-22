#!/usr/bin/env python
"""Bounded real-PhysX diagnostics; does not train or measure policy success.

Stage 1 forces recovery sampling to exercise the reset/grace path. Stage 2
raycasts all 50 collision tiles and deliberately triggers timeout/failure to
check curriculum plumbing. Those triggers are not learned completions.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--asset", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--reference-data", required=True)
    p.add_argument("--fall-pool")
    from pgmt.envs.actuators import ACTUATOR_PROFILES, actuator_options
    p.add_argument("--actuator-profile", choices=ACTUATOR_PROFILES, default="asset_effort_v1")
    p.add_argument("--diagnostics-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output already exists")
    from pgmt.train.train_stage1 import _seed_everything, _write_metrics
    from pgmt.envs.isaac_app import launch_isaac_app
    from pgmt.envs.recovery import load_fall_pool
    from pgmt.train.policy import Stage1Policy, Stage2Policy

    _seed_everything(7)
    report = {"status": "running", "stage": a.stage, "checks": {},
              "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
              "vulkan_index": os.getenv("PGMT_RENDER_GPU")}

    def record(name, value):
        report["checks"][name] = value
        _write_metrics(a.output, report)
        print(f"PASS {name}: {json.dumps(value)}", flush=True)

    app, env = launch_isaac_app(a.device), None
    try:
        import pgmt.envs.g1_env as g1
        g1 = importlib.reload(g1)
        pool = load_fall_pool(a.fall_pool) if a.stage == 1 else None
        if pool is not None:
            pool.init_prob = pool.prob_max = 1.
            pool._probability = 1.
        n = 8 if a.stage == 1 else 5
        c = g1.G1EnvConfig(num_envs=n, device=a.device, stage=a.stage, seed=7,
                          asset_path=a.asset, reference_urdf_path=a.urdf,
                          reference_data_dir=a.reference_data,
                          terrain_curriculum=a.stage == 2,
                          randomize_dynamics=True, corrupt_observations=True,
                          max_action_delay=2, **actuator_options(a.actuator_profile))
        cfg = g1.G1DirectRLEnvCfg(pgmt_cfg=c, robot_cfg=g1.make_g1_articulation_cfg(a.asset, c))
        cfg.scene.num_envs, cfg.scene.env_spacing = n, 2.5
        cfg.sim.enable_scene_query_support = a.stage == 2
        physical = g1.IsaacLabG1Env(cfg, recovery_pool=pool)
        env = g1.IsaacLabPPOAdapter(physical)
        from pgmt.train.diagnostics import attach_diagnostics
        attach_diagnostics(env, a.diagnostics_dir)
        obs = env.reset()
        core, robot = physical.core, physical.robot
        assert all(torch.isfinite(x).all() for x in obs.values())
        assert core._contact_sensor_available
        record("environment", {"envs": n, "dt": core.cfg.control_dt,
                               "reference_sequences": len(core.reference_database.seqs),
                               "contact_sensor": True})

        r, view = physical.randomizer, robot.root_physx_view
        mass = view.get_masses().cpu()[:, r.pelvis] / r.masses[:, r.pelvis]
        com = view.get_coms().cpu()[:, r.pelvis, :3] - r.coms[:, r.pelvis, :3]
        mat = view.get_material_properties().cpu()[:, :, :2]
        assert torch.allclose(mass, core.priv_mass[:, 0].cpu(), atol=1e-6)
        assert torch.allclose(com, core.priv_com.cpu(), atol=1e-6)
        assert torch.allclose(mat, core.priv_friction.cpu()[:, None, :].expand_as(mat), atol=1e-6)
        gains = view.get_dof_stiffnesses().cpu()[:, core.joint_ids.cpu()]
        assert torch.allclose(gains, (core.pd.kp * core.priv_motor_strength).cpu(), atol=1e-5)
        limits = view.get_dof_max_forces().cpu()[:, core.joint_ids.cpu()]
        assert torch.allclose(limits, core.pd.torque_limit.cpu().expand_as(limits), atol=1e-5)
        record("physx_effort_readback", {"profile": a.actuator_profile, "joint_names": list(g1.G1_JOINT_NAMES),
                                       "effort_nm": limits[0].tolist(),
                                       "velocity_rad_s": view.get_dof_max_velocities().cpu()[0, core.joint_ids.cpu()].tolist()})
        assert float((mass - 1).abs().max()) > 0
        record("physx_randomization_readback", {"mass_scale": mass.tolist(),
                                                "friction": mat[:, 0, 0].tolist(),
                                                "com_max_m": float(com.abs().max()),
                                                "pd_strength_range": [float(core.priv_motor_strength.min()), float(core.priv_motor_strength.max())]})
        policy = (Stage1Policy() if a.stage == 1 else Stage2Policy()).to(core.device)
        with torch.no_grad():
            out = policy.act(obs)
            again = policy.evaluate_actions(obs, out.actions, latent_actions=out.latent_actions)
        assert torch.isfinite(again.log_probs).all()
        assert torch.allclose(out.log_probs, again.log_probs, atol=1e-5)
        assert ((out.actions >= core.joint_low) & (out.actions <= core.joint_high)).all()
        record("bounded_action_density", {"max_log_prob_error": float((out.log_probs-again.log_probs).abs().max())})

        if a.stage == 1:
            assert core._recovery_active.bool().all()
            assert (core.root_pos[:, 2] < .5).all()
            assert (core.reference_root_pos[:, 2] > .6).all()
            record("physical_recovery_reset", {"fallen_root_z": core.root_pos[:, 2].tolist(),
                                                "target_root_z": core.reference_root_pos[:, 2].tolist()})
        core._action_delay[:] = 2
        first = core.default_q + .03
        for target in (first, core.default_q + .06, core.default_q + .09):
            obs, rewards, terminated, timeout, _ = env.step(target)
            assert not terminated.any() and not timeout.any()
            assert torch.isfinite(rewards).all()
        assert torch.equal(core.target, first)
        assert torch.equal(robot.data.joint_pos_target[:, core.joint_ids], first)
        record("physical_delayed_target", {"delay_control_steps": 2, "exact_target_readback": True})

        if a.stage == 1:
            first_end = None
            ends_after_grace = 0
            max_contact = 0.
            for step in range(3, 180):
                obs, rewards, terminated, timeout, _ = env.step(core.default_q)
                assert all(torch.isfinite(x).all() for x in obs.values())
                assert torch.isfinite(rewards).all()
                ended = int((terminated | timeout).sum())
                if step + 1 < 150:
                    assert ended == 0, "recovery terminated inside 3-second grace"
                if ended:
                    first_end = first_end or step + 1
                    ends_after_grace += ended
                max_contact = max(max_contact, float(core.contact_forces.norm(dim=-1).max()))
            assert max_contact > 1
            record("recovery_grace_rollout", {"control_steps": 180, "first_episode_end_step": first_end,
                                              "ends_after_grace": ends_after_grace, "max_contact_force_N": max_contact})
        else:
            import omni.physx
            from pgmt.envs.terrain.generators import FAMILIES
            from pgmt.envs.terrain.curriculum import is_compatible, motion_category
            query = omni.physx.get_physx_scene_query_interface()
            assert not physical.sim.stage.GetPrimAtPath("/World/ground").IsValid()
            rng = np.random.default_rng(17)
            errors, z_values = [], []
            for level in range(10):
                for family in range(5):
                    # Offset to an interior repeated tile, away from robots at
                    # the origin. Include positive and negative rough heights.
                    xy = rng.uniform(.6, 3.4, (12, 2)) + np.array([family, level]) * core._terrain.size
                    expected = core._terrain.query(torch.tensor(xy, device=core.device, dtype=torch.float32)).cpu().numpy()
                    for point, z in zip(xy, expected):
                        hit = query.raycast_closest((float(point[0]), float(point[1]), 5.), (0., 0., -1.), 10.)
                        assert hit["hit"], (family, level, point)
                        assert str(hit["collision"]).startswith("/World/terrain"), hit
                        errors.append(abs(float(hit["position"][2]) - float(z)))
                        z_values.append(float(z))
            assert max(errors) < .001, max(errors)
            assert min(z_values) < -.005 and max(z_values) > .3
            record("collision_mesh_raycast", {"tiles": 50, "rays": len(errors),
                                               "max_height_error_m": max(errors),
                                               "terrain_z_range_m": [min(z_values), max(z_values)],
                                               "ground_plane_absent": True})

            env.reset()
            physical.episode_length_buf[:] = physical.max_episode_length - 1
            _, _, terminated, timeout, _ = env.step(core.default_q)
            assert timeout.all() and not terminated.any()
            assert core._terrain_levels.tolist() == [1] * n
            promoted_xy = core.root_pos[:, :2].clone()
            expected_xy = core._terrain.origins(core._terrain_families, core._terrain_levels)[:, :2]
            assert torch.allclose(promoted_xy, expected_xy)
            # Trigger a physical out-of-field failure, then verify auto-reset
            # writes the lower-level pose back into the articulation.
            root = robot.data.root_state_w.clone()
            root[:, 0] += core._terrain.size
            robot.write_root_state_to_sim(root)
            _, _, terminated, timeout, _ = env.step(core.default_q)
            assert terminated.all() and not timeout.any()
            assert core._terrain_levels.tolist() == [0] * n
            expected_xy = core._terrain.origins(core._terrain_families, core._terrain_levels)[:, :2]
            assert torch.allclose(robot.data.root_pos_w[:, :2], expected_xy)
            record("physical_curriculum_reset", {"forced_timeout_levels": [1] * n,
                                                 "forced_failure_levels": [0] * n,
                                                 "promoted_world_xy": promoted_xy.tolist()})

            ids = torch.arange(n, device=core.device)
            core._terrain_levels[:] = 9
            for state in core.terrain_curriculum.states:
                state.level = 9
            physical._reset_idx(ids)
            for i, seq in enumerate(core.reference_seq_idx.tolist()):
                assert is_compatible(motion_category(core.reference_database.seqs[seq]["name"]), FAMILIES[i], 9)
            root = robot.data.root_state_w.clone()
            root[:, :2] += 2.
            root[:, 2] = core._terrain.query(root[:, :2]) + .85
            root[:, 7:] = 0
            robot.write_root_state_to_sim(root)
            robot.set_joint_position_target(core.default_q, joint_ids=core.joint_ids)
            peak_force = torch.zeros(n, device=core.device)
            # Raw stepping intentionally bypasses tracking termination while
            # testing whether each physical family supports the falling robot.
            for _ in range(160):
                physical.scene.write_data_to_sim()
                physical.sim.step(render=False)
                physical.scene.update(core.cfg.sim_dt)
                core._read_articulation(update_previous=False)
                physical._sync_contact_forces()
                peak_force = torch.maximum(peak_force, core.contact_forces.norm(dim=-1).max(-1).values)
                assert torch.isfinite(core.root_pos).all()
            assert (peak_force > 1.).all(), peak_force
            record("five_family_physical_contacts", {"level": 9, "physics_steps": 160,
                                                      "peak_force_N": peak_force.tolist(),
                                                      "final_root_z": core.root_pos[:, 2].tolist(),
                                                      "compatible_references": True})
        report["status"] = "passed"
        _write_metrics(a.output, report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_metrics(a.output, report)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
