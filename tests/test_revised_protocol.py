"""Behavioral checks at the action, curriculum, physics-data and eval boundaries."""
from collections import Counter

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform, AffineTransform

from pgmt.envs.g1_env import G1Env, G1EnvConfig
from pgmt.envs.recovery import FallRecoveryPool, load_fall_pool
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.terrain.batched import TerrainAtlas
from pgmt.train.policy import Stage1Policy
from pgmt.train.ppo import PPO
from pgmt.eval.manifest import make_manifest
from pgmt.eval.run import summarize
from data.build_mesh_contacts import ReferenceMesh, contact_labels, FEET


def database():
    n = 1600
    return MotionDatabase.from_sequences([{
        "name": "walk_test", "qpos": np.zeros((n, 29)), "qvel": np.zeros((n, 29)),
        "root_pos": np.tile([0., 0., .793], (n, 1)),
        "root_rot": np.tile([1., 0., 0., 0.], (n, 1)), "frame_time": .02,
    }])


def test_action_density_matches_change_of_variables_and_env_executes_targets():
    policy = Stage1Policy()
    latent = torch.linspace(-1., 1., 29).unsqueeze(0)
    normal = Normal(torch.zeros_like(latent), torch.ones_like(latent))
    target = policy._target(latent)
    transformed = TransformedDistribution(normal, [TanhTransform(), AffineTransform(policy.action_mid, policy.action_half_range)])
    assert torch.allclose(policy._log_prob(normal, latent), transformed.log_prob(target).sum(-1), atol=2e-5)
    env = G1Env()
    env._apply_action(target)
    assert torch.equal(env.target, target)
    with pytest.raises(ValueError, match="joint position targets"):
        env._apply_action(target + 100.)


def test_saturated_targets_keep_finite_probabilities_and_unit_ppo_ratio(monkeypatch):
    policy = Stage1Policy()
    mean = torch.full((2, 29), 30., requires_grad=True)
    normal = Normal(mean, torch.ones_like(mean))
    monkeypatch.setattr(policy, "_distribution_and_values", lambda obs: (normal, torch.zeros(2, 3)))
    out = policy.act({})
    evaluated = policy.evaluate_actions({}, out.actions, latent_actions=out.latent_actions)
    assert torch.equal(out.log_probs, evaluated.log_probs)
    assert torch.isfinite(evaluated.log_probs).all()
    (-evaluated.log_probs.mean() - evaluated.entropy.mean()).backward()
    assert torch.isfinite(mean.grad).all()


def test_legacy_checkpoint_rejected_before_loading_weights():
    ppo = PPO(Stage1Policy())
    old = ppo.state_dict()
    old.pop("action_contract")
    with pytest.raises(ValueError, match="obsolete"):
        ppo.load_state_dict(old)


def test_action_delay_applies_old_targets_without_rewriting_policy_action():
    env = G1Env(G1EnvConfig(max_action_delay=2))
    env._action_delay[:] = 2
    first, second, third = (torch.full((1, 29), v) for v in (.05, .1, .15))
    env._apply_action(first)
    assert torch.equal(env.target, env.default_q)
    env._apply_action(second)
    env._apply_action(third)
    assert torch.equal(env.target, first)
    assert torch.equal(env.action, third)
    env.reset()
    assert torch.equal(env._action_queue[:, 0], env.default_q)


def test_repeated_terrain_mesh_and_query_use_identical_triangles():
    atlas = TerrainAtlas(resolution=.25, repeats=2)
    vertices, faces = atlas.mesh()
    rng = np.random.default_rng(5)
    triangles = vertices[faces[rng.integers(len(faces), size=1000)]]
    points = .2 * triangles[:, 0] + .3 * triangles[:, 1] + .5 * triangles[:, 2]
    queried = atlas.query(torch.from_numpy(points[:, :2]))
    assert np.allclose(queried.numpy(), points[:, 2], atol=3e-5)


def test_stage2_assigns_five_families_and_updates_placement_and_level():
    env = G1Env(G1EnvConfig(num_envs=5, stage=2, terrain_curriculum=True, terrain_resolution=.25, terrain_repeats=2))
    ids = torch.arange(5)
    assert env._terrain_families.tolist() == [0, 1, 2, 3, 4]
    env.record_episode_outcomes(torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool))
    env._sample_reference(ids)
    assert env._terrain_levels.tolist() == [1] * 5
    expected = env._terrain.origins(env._terrain_families, env._terrain_levels)
    assert torch.equal(env.root_pos[:, :2], expected[:, :2])
    assert torch.allclose(env.root_pos[:, 2] - env._terrain.query(expected[:, :2]), torch.full((5,), .793))
    env.record_episode_outcomes(torch.ones(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool))
    assert env._terrain_levels.tolist() == [0] * 5


def test_recovery_targets_upright_reference_and_allows_time_to_stand():
    pool = FallRecoveryPool(init_prob=1., prob_max=1.)
    pool.add({"qpos": torch.zeros(29), "qvel": torch.zeros(29),
              "root_pos": torch.tensor([0., 0., .3]), "root_quat": torch.tensor([.7071068, .7071068, 0., 0.]),
              "root_lin_vel": torch.zeros(3), "root_ang_vel": torch.zeros(3)}, seq_idx=0, frame=0.)
    env = G1Env(G1EnvConfig(), reference_database=database(), recovery_pool=pool)
    assert env.root_pos[0, 2] == pytest.approx(.3)
    assert env.reference_root_pos[0, 2] == pytest.approx(.793)
    assert env.reference_root_quat[0, 0] == pytest.approx(1.)
    for _ in range(100):
        assert not env._dones()[0].any()
    env.episode_length_buf[:] = 200
    for _ in range(100):
        terminated, _ = env._dones()
    assert terminated.all()


def test_formal_pool_rejects_missing_physics_provenance(tmp_path):
    path = tmp_path / "pool.pt"
    torch.save(FallRecoveryPool(), path)
    with pytest.raises(ValueError, match="empty"):
        load_fall_pool(path)


def test_mesh_contacts_use_collision_geometry_not_foot_velocity():
    geometry = [(torch.tensor([[0., 0., -.03]]), torch.tensor([.005])) for _ in FEET]
    positions = torch.tensor([[[0., 0., .05], [0., 0., .2]]])
    quats = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    labels = contact_labels(positions, quats, list(FEET), geometry, ReferenceMesh())
    assert labels.tolist() == [[True, False]]


def test_observation_corruption_is_cached_between_reads_and_seeded():
    cfg = G1EnvConfig(stage=2, corrupt_observations=True, terrain_resolution=.25, terrain_repeats=1)
    env = G1Env(cfg)
    a = env.reset(seed=11)
    b = env.get_observations()
    assert torch.equal(a["obs"], b["obs"])
    assert torch.equal(a["elevation"], b["elevation"])
    other = G1Env(cfg)
    again = other.reset(seed=11)
    assert torch.equal(a["obs"], again["obs"])


def test_matched_manifest_has_exact_cells_and_reproducible_seeds():
    a, b = make_manifest(database()), make_manifest(database())
    assert a == b
    counts = Counter((e["family"], e["level"]) for e in a["episodes"])
    assert len(a["episodes"]) == 9600 and len(counts) == 50
    assert set(counts.values()) == {192}
    assert all(e["start_frame"] + 1500 < 1600 for e in a["episodes"])


def test_failed_episodes_stay_in_evaluation_denominator():
    row = {"family": "flat", "level": 9, "steps": 10, "duration_s": .2,
           "joint_mse_sum": 40., "body_mse_sum": 10., "root_mse_sum": 10.,
           "contact_tp": 10., "contact_fp": 5., "contact_fn": 0., "terrain_out_of_bounds": False}
    report = summarize([{**row, "success": True}, {**row, "success": False}])
    assert report["overall"]["completion"] == .5
    assert report["overall"]["joint_rmse"] == 2.
    assert report["L9"]["contact_f1"] == .8


def test_stage2_requires_pretraining_before_constructing_a_physical_env():
    from pgmt.train.train_stage2 import run
    with pytest.raises(ValueError, match="requires --stage1-checkpoint"):
        run(backend="isaaclab")


def test_randomization_writes_physics_properties_without_compounding_baselines():
    from types import SimpleNamespace
    from pgmt.envs.randomization import IsaacRandomizer

    class View:
        def __init__(self):
            self.values = {"masses": torch.ones(2, 1), "inertias": torch.ones(2, 1, 9),
                           "coms": torch.zeros(2, 1, 7), "material_properties": torch.ones(2, 3, 3)}

        def __getattr__(self, name):
            op, key = name.split("_", 1)
            if op == "get":
                return lambda: self.values[key]
            return lambda value, ids: self.values[key].__setitem__(ids, value[ids])

    class Robot:
        body_names = ["pelvis"]

        def __init__(self):
            self.root_physx_view = View()
            self.actuators = {"pd": SimpleNamespace(stiffness=torch.zeros(2, 29), damping=torch.zeros(2, 29), joint_indices=slice(None))}

        def write_joint_stiffness_to_sim(self, value, **kw):
            self.kp = value.clone()

        def write_joint_damping_to_sim(self, value, **kw):
            self.kd = value.clone()

    core = G1Env(G1EnvConfig(num_envs=2, randomize_dynamics=True))
    robot = Robot()
    randomizer = IsaacRandomizer(robot, core)
    ids = torch.arange(2)
    core._rng = np.random.default_rng(100)
    randomizer.apply(ids)
    first = {k: v.clone() for k, v in robot.root_physx_view.values.items()}
    assert not torch.equal(first["masses"], torch.ones(2, 1))
    assert torch.equal(robot.kp, core.pd.kp * core.priv_motor_strength)
    assert torch.equal(first["coms"][:, 0, :3], core.priv_com)
    core._rng = np.random.default_rng(100)
    randomizer.apply(ids)
    assert all(torch.equal(value, robot.root_physx_view.values[k]) for k, value in first.items())


def test_revised_environment_checkpoint_preserves_delay_noise_and_curriculum():
    cfg = G1EnvConfig(num_envs=5, stage=2, terrain_curriculum=True, max_action_delay=2,
                      corrupt_observations=True, terrain_resolution=.25, terrain_repeats=1)
    env = G1Env(cfg)
    env.record_episode_outcomes(torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool))
    env._sample_reference(torch.arange(5))
    target = Stage1Policy()._target(torch.ones(5, 29) * .1)
    env._apply_action(target)
    saved = env.state_dict()
    restored = G1Env(cfg)
    restored.load_state_dict(saved)
    assert torch.equal(restored._action_queue, env._action_queue)
    assert torch.equal(restored._terrain_levels, env._terrain_levels)
    assert restored.terrain_curriculum.level_of(1) == 1
    assert torch.equal(restored.get_observations()["elevation"], env.get_observations()["elevation"])
