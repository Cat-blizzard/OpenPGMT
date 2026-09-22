"""Collect offline post-fall states in IsaacLab, without policy training.

Robots receive randomized reset velocity, then settle under gravity with zero
stiffness and weak damping. Only physically fallen, finite states are retained.
The initial reference frame remains the recovery target; pool XY is local.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pgmt.envs.recovery import FallRecoveryPool
from pgmt.train.train_stage1 import build_env, _atomic_output, _seed_everything


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:8")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--states", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--asset", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--reference-data", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if min(a.num_envs, a.states, a.max_batches) <= 0:
        p.error("counts must be positive")
    if a.output.exists():
        p.error("output already exists")
    _seed_everything(a.seed)
    from pgmt.envs.isaac_app import launch_isaac_app
    app = launch_isaac_app(a.device)
    env = None
    try:
        env = build_env(a.num_envs, torch.device(a.device), backend="isaaclab", app=app,
                        reference_data=a.reference_data, asset_path=a.asset, reference_urdf=a.urdf,
                        env_options={"seed": a.seed, "fall_collection": True, "episode_length_s": 10.})
        physical, core = env.env, env.env.core
        pool = FallRecoveryPool(capacity=a.states)
        pool.provenance = {
            "backend": "isaaclab", "seed": a.seed, "method": "gravity_settle_3s_zero_kp_v1",
            "asset": str(Path(a.asset).resolve()), "urdf": str(Path(a.urdf).resolve()),
            "sequence_names": [s["name"] for s in core.reference_database.seqs],
            "pose_fingerprint": core.reference_database.pose_fingerprint(),
            "reference_data": str(Path(a.reference_data).resolve()),
        }
        ids = torch.arange(a.num_envs, device=core.device)
        for _ in range(a.max_batches):
            env.reset()
            initial_frames = core.reference_frame.clone()
            initial_seqs = core.reference_seq_idx.clone()
            root = torch.cat((core.root_pos, core.root_quat, core.root_lin_vel, core.root_ang_vel), -1).clone()
            root[:, 7:9] = torch.as_tensor(core._rng.uniform(-1.5, 1.5, (a.num_envs, 2)), device=core.device)
            root[:, 10:12] = torch.as_tensor(core._rng.uniform(-2., 2., (a.num_envs, 2)), device=core.device)
            physical.robot.write_root_state_to_sim(root, ids)
            physical.robot.write_joint_stiffness_to_sim(torch.zeros_like(core.qpos), joint_ids=core.joint_ids, env_ids=ids)
            physical.robot.write_joint_damping_to_sim(torch.full_like(core.qpos, .5), joint_ids=core.joint_ids, env_ids=ids)
            for actuator in physical.robot.actuators.values():
                actuator.stiffness.zero_()
                actuator.damping.fill_(.5)
            for _ in range(150):
                env.step(core.default_q)
            q = core.root_quat
            up = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
            fallen = (core.root_pos[:, 2] < .5) & (up < .5)
            fallen &= core.root_lin_vel.norm(dim=-1) < 2.
            for i in fallen.nonzero().flatten().tolist():
                state = {k: getattr(core, k)[i].clone() for k in
                         ("qpos", "qvel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel")}
                state["root_pos"][:2] -= core.default_root_pos[i, :2]
                pool.add(state, seq_idx=int(initial_seqs[i]), frame=float(initial_frames[i]))
                if len(pool) == a.states:
                    break
            print(f"fall states {len(pool)}/{a.states}", flush=True)
            if len(pool) == a.states:
                break
        if len(pool) < a.states:
            raise RuntimeError(f"only {len(pool)}/{a.states} qualified physical falls; no final pool published")
        with _atomic_output(a.output) as stream:
            torch.save({"schema": "pgmt_fall_pool_v2", "pool": pool.state_dict()}, stream)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.env.close()
        app.close()


if __name__ == "__main__":
    main()
