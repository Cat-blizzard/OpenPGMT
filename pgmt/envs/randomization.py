"""Explicit reproduction settings, applied to PhysX rather than critic-only noise.

Ranges are engineering assumptions, not hyperparameters disclosed by the paper.
Mass/inertia/COM perturb the pelvis; material friction affects all robot shapes;
motor strength scales the implicit PD gains. Reset velocity is an impulse in m/s.
"""
from __future__ import annotations

import numpy as np
import torch

RANDOMIZATION_SPEC = {
    "version": 1, "friction": [0.6, 1.2], "friction_buckets": 32, "pelvis_mass_scale": [0.9, 1.1],
    "pelvis_com_m": [-0.02, 0.02], "pd_strength": [0.85, 1.15],
    "reset_velocity_xy_m_s": [-0.15, 0.15], "max_action_delay_steps": 2,
    "proprio_noise_std": {"rotation6d": .01, "omega": .02, "q": .01, "qd": .1},
}


class IsaacRandomizer:
    def __init__(self, robot, core):
        self.robot, self.core = robot, core
        view = robot.root_physx_view
        self.masses = view.get_masses().clone().cpu()
        self.inertias = view.get_inertias().clone().cpu()
        self.coms = view.get_coms().clone().cpu()
        self.materials = view.get_material_properties().clone().cpu()
        self.pelvis = robot.body_names.index("pelvis")

    def apply(self, ids):
        c, robot = self.core, self.robot
        if not c.cfg.randomize_dynamics or not len(ids):
            return
        n, cpu_ids = len(ids), ids.cpu()
        # Flat L0 is nominal; higher flat levels alter dynamics. Stage 1 uses
        # the full range; other terrain families retain a minimum perturbation.
        scale = np.ones(n)
        if c.cfg.stage == 2:
            levels = c._terrain_levels[ids].cpu().numpy() / 9
            scale = np.where(c._terrain_families[ids].cpu().numpy() == 0, levels, .2 + .8 * levels)

        def sample(lo, hi, neutral, width=1):
            draws = c._rng.uniform(lo, hi, (n, width))
            return torch.as_tensor(neutral + (draws - neutral) * scale[:, None], dtype=torch.float32)

        mass = sample(.9, 1.1, 1)
        com = sample(-.02, .02, 0, 3)
        # Bound the number of distinct PhysX materials across repeated resets.
        buckets = c._rng.integers(0, 32, size=(n, 1))
        friction = torch.as_tensor(1 + (.6 + .6 * buckets / 31 - 1) * scale[:, None], dtype=torch.float32)
        strength = sample(.85, 1.15, 1, 29).to(c.device)
        impulse = sample(-.15, .15, 0, 2).to(c.device)
        view = robot.root_physx_view
        masses, inertias = view.get_masses().clone().cpu(), view.get_inertias().clone().cpu()
        coms, materials = view.get_coms().clone().cpu(), view.get_material_properties().clone().cpu()
        masses[cpu_ids, self.pelvis] = self.masses[cpu_ids, self.pelvis] * mass[:, 0]
        inertias[cpu_ids, self.pelvis] = self.inertias[cpu_ids, self.pelvis] * mass
        coms[cpu_ids, self.pelvis, :3] = self.coms[cpu_ids, self.pelvis, :3] + com
        materials[cpu_ids, :, :2] = friction[:, None, :]
        view.set_masses(masses, cpu_ids)
        view.set_inertias(inertias, cpu_ids)
        view.set_coms(coms, cpu_ids)
        view.set_material_properties(materials, cpu_ids)
        kp, kd = c.pd.kp * strength, c.pd.kd * strength
        robot.write_joint_stiffness_to_sim(kp, joint_ids=c.joint_ids, env_ids=ids)
        robot.write_joint_damping_to_sim(kd, joint_ids=c.joint_ids, env_ids=ids)
        # Keep implicit actuator torque estimates consistent with PhysX gains.
        canonical_to_runtime = c.joint_ids
        kp_runtime, kd_runtime = torch.zeros_like(kp), torch.zeros_like(kd)
        kp_runtime[:, canonical_to_runtime], kd_runtime[:, canonical_to_runtime] = kp, kd
        for actuator in robot.actuators.values():
            actuator.stiffness[ids] = kp_runtime[:, actuator.joint_indices]
            actuator.damping[ids] = kd_runtime[:, actuator.joint_indices]
        c.priv_mass[ids] = mass.to(c.device)
        c.priv_com[ids] = com.to(c.device)
        c.priv_friction[ids] = friction.to(c.device).expand(-1, 2)
        c.priv_motor_strength[ids] = strength
        c.priv_push[ids] = 0
        c.priv_push[ids, :2] = impulse
        c.root_lin_vel[ids, :2] += impulse
