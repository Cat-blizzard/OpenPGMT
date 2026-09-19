import numpy as np
import torch

from pgmt.contracts import ACT_DIM
from pgmt.envs.g1_env import (
    G1Env,
    G1EnvConfig,
    REQUIRED_BODY_NAMES,
    G1_JOINT_NAMES,
    preflight_asset,
    resolve_name_indices,
)
from pgmt.envs import g1_env as g1_env_module
from pgmt.envs.reference_sampler import MotionDatabase


def test_runtime_mapping_accepts_unitree_joint_suffix_without_permuting_order():
    actual = [name + "_joint" for name in reversed(G1_JOINT_NAMES)]
    ids = resolve_name_indices(actual, G1_JOINT_NAMES, kind="joint")
    assert ids.tolist() == list(reversed(range(ACT_DIM)))


def test_external_urdf_preflight_normalizes_joint_suffix_when_available():
    path = "/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf"
    report = preflight_asset(path)
    if not report.exists:
        return
    assert report.ok
    assert report.parsed_joint_names[:3] == ("floating_base", "pelvis_contour", "left_hip_pitch")


def test_g1_env_batch_contract_and_timeout_terminal_observation():
    env = G1Env(G1EnvConfig(num_envs=3, episode_length_s=0.02))
    obs = env.reset()
    assert {key: tuple(value.shape) for key, value in obs.items()} == {
        "obs": (3, 96), "history": (3, 10, 96), "future": (3, 6, 61), "privileged": (3, 48)
    }
    next_obs, rewards, terminated, truncated, info = env.step(torch.zeros(3, ACT_DIM))
    assert rewards.shape == (3, 3)
    assert not terminated.any()
    assert truncated.all()
    assert info["terminal_observation"]["obs"].shape == (3, 96)
    assert torch.isfinite(rewards).all()


def test_g1_env_observation_uses_reference_relative_orientation():
    env = G1Env(G1EnvConfig(num_envs=1))
    env.reference_root_quat[:] = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])
    env.root_quat[:] = env.reference_root_quat
    e6 = g1_env_module._relative_rot6d(env.root_quat, env.reference_root_quat)
    assert torch.allclose(e6, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]), atol=1e-5)


def test_g1_env_uses_batched_reference_motion():
    t = np.arange(5, dtype=np.float32)
    sequence = {
        "qpos": np.repeat(t[:, None], ACT_DIM, axis=1),
        "qvel": np.ones((5, ACT_DIM), dtype=np.float32),
        "root_pos": np.stack((t, np.zeros_like(t), np.zeros_like(t)), axis=1),
        "root_rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (5, 1)),
        "frame_time": 1.0,
    }
    env = G1Env(G1EnvConfig(num_envs=2), reference_database=MotionDatabase.from_sequences([sequence]))
    before = env.reference_frame.clone()
    assert torch.all((env.reference_qpos >= 0) & (env.reference_qpos <= 4))
    env.step(torch.zeros(2, ACT_DIM))
    assert torch.all(env.reference_frame > before)
    assert env.reference_future.shape == (2, 6, 61)
