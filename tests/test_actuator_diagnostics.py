"""Actuator mapping and passive evidence capture; no simulator/GPU required."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from data.retarget_lafan1 import G1_JOINT_NAMES
from pgmt.cfg.assumptions import get
from pgmt.envs.actuators import actuator_options, validate_actuator_resume
from pgmt.envs.g1_env import G1Env, G1EnvConfig, REQUIRED_BODY_NAMES
from pgmt.train.diagnostics import BoundedDiagnostics, DiagnosticConfig, attach_diagnostics
from pgmt.train.policy import Stage1Policy, Stage2Policy
from pgmt.train.ppo import PPO
from pgmt.train.train_stage1 import MockStage1Env
from pgmt.train.train_stage2 import MockStage2Env
from setup.replay_kl_diagnostic import replay


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_efforts_match_external_urdf_by_name_and_keep_pd_reset():
    path = Path("/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf")
    if not path.exists():
        pytest.skip("external licensed asset absent")
    joints = {j.attrib["name"].removesuffix("_joint"): j for j in ET.parse(path).getroot().findall("joint")}
    candidate = G1EnvConfig()
    legacy = G1EnvConfig(**actuator_options("legacy_uniform120"))
    for i, name in enumerate(G1_JOINT_NAMES):
        assert candidate.torque_limit[i] == float(joints[name].find("limit").attrib["effort"])
    assert candidate.stiffness == legacy.stiffness
    assert candidate.damping == legacy.damping
    assert candidate.default_joint_pos == legacy.default_joint_pos
    for key in asdict(candidate):
        if key != "torque_limit":
            assert getattr(candidate, key) == getattr(legacy, key)


def test_resume_rejects_silent_effort_change():
    old = {"environment_config": asdict(G1EnvConfig(**actuator_options("legacy_uniform120")))}
    with pytest.raises(ValueError, match="torque_limit differs"):
        validate_actuator_resume(old, G1EnvConfig())
    validate_actuator_resume(old, G1EnvConfig(**actuator_options("legacy_uniform120")))


def test_runtime_readback_uses_canonical_mapping_and_rejects_wrong_limit(tmp_path):
    core = G1Env(G1EnvConfig(num_envs=2))
    core.joint_ids = torch.arange(28, -1, -1)
    efforts = core.pd.torque_limit.flip(0).repeat(2, 1)
    view = SimpleNamespace(get_dof_max_forces=lambda: efforts,
                           get_dof_max_velocities=lambda: torch.ones(2, 29),
                           get_dof_stiffnesses=lambda: torch.ones(2, 29)*80,
                           get_dof_dampings=lambda: torch.ones(2, 29)*2)
    core.articulation = SimpleNamespace(root_physx_view=view)
    attach_diagnostics(core, tmp_path)
    event = load(tmp_path / "actuators_000.pt")
    assert torch.equal(event["effort_nm"], core.pd.torque_limit.repeat(2, 1))
    efforts[0, 0] = 120
    with pytest.raises(RuntimeError, match="PhysX effort limits differ"):
        attach_diagnostics(core, tmp_path / "bad")


def test_recorder_rate_count_and_restart_limits_preserve_evidence(tmp_path):
    cfg = DiagnosticConfig(max_events=2, min_interval=10)
    recorder = BoundedDiagnostics(tmp_path, cfg)
    value = torch.tensor([2.])
    for step in range(25):
        recorder.emit("joint_speed", step, {"value": value})
    paths = sorted(tmp_path.glob("*.pt"))
    assert len(paths) == 2
    assert [load(p)["step"] for p in paths] == [0, 10]
    value.zero_()
    assert load(paths[0])["value"].item() == 2
    BoundedDiagnostics(tmp_path, cfg).emit("joint_speed", 100, {})
    assert sorted(tmp_path.glob("*.pt")) == paths


def test_first_action_records_delay_history_without_changing_transition(tmp_path):
    cfg = G1EnvConfig(num_envs=2, max_action_delay=2, seed=19)
    control, observed = G1Env(cfg), G1Env(cfg)
    observed.diagnostics = BoundedDiagnostics(tmp_path)
    for env in (control, observed):
        env.reset(seed=19)
        env._action_delay[:] = 2
    action = torch.full((2, 29), .1)
    expected, actual = control.step(action), observed.step(action)
    for k in expected[0]:
        assert torch.equal(expected[0][k], actual[0][k])
    assert torch.equal(expected[1], actual[1])
    event = load(tmp_path / "first_action_000.pt")
    assert event["before"]["episode_length_buf"].eq(0).all()
    assert event["before"]["qvel"].eq(0).all()
    assert event["before"]["_action_queue"].eq(0).all()
    assert event["after"]["action"].eq(.1).all()
    assert event["after"]["target"].eq(0).all()  # delayed nominal target
    assert torch.equal(event["before"]["history"], event["after"]["history"])


def test_physics_events_keep_pre_reset_state_and_separate_accelerations(tmp_path):
    env = G1Env(G1EnvConfig(num_envs=2))
    env.diagnostics = BoundedDiagnostics(tmp_path, DiagnosticConfig(max_samples=1))
    env._diagnostic_step = 12
    env.qvel[1, G1_JOINT_NAMES.index("left_wrist_yaw")] = 60
    env.episode_length_buf[1] = 149
    env._recovery_active[1] = 1
    state_accel = torch.zeros(2, len(REQUIRED_BODY_NAMES), 3)
    reference_accel = torch.zeros_like(state_accel)
    body = REQUIRED_BODY_NAMES.index("left_wrist_yaw_link")
    state_accel[1, body, 0] = 200
    reference_accel[1, body, 0] = 30
    env.diagnostics.physics(env, {"body_accel": state_accel}, {"body_accel": reference_accel})
    env.reset()
    speed = load(tmp_path / "joint_speed_000.pt")
    accel = load(tmp_path / "ee_acceleration_000.pt")
    assert speed["env_ids"].tolist() == [1]
    assert speed["qvel"].max() == 60
    assert speed["episode_length_buf"].item() == 149
    assert speed["_recovery_active"].item() == 1
    assert speed["overspeed_mask"].sum() == 1
    assert "estimate" in speed["torque_source"]
    assert accel["state_accel_m_s2"].max() == 200
    assert accel["reference_accel_m_s2"].max() == 30
    assert accel["error_m_s2"].max() == 170


@pytest.mark.parametrize("stage", [1, 2])
def test_diagnostic_on_off_bitwise_training_and_cpu_replay(tmp_path, stage):
    cfg = replace(get("A6").value, num_steps_per_env=2, num_learning_epochs=1,
                  num_mini_batches=1, learning_rate=.001, complete_critic_epochs=True, kl_chunk_size=2)
    def train(recording):
        torch.manual_seed(219)
        policy = Stage1Policy() if stage == 1 else Stage2Policy()
        env = MockStage1Env(2, torch.device("cpu")) if stage == 1 else MockStage2Env(2, torch.device("cpu"))
        recorder = BoundedDiagnostics(tmp_path, DiagnosticConfig(kl_threshold=1e-10, min_interval=0, max_samples=2)) if recording else None
        ppo = PPO(policy, cfg, total_updates=1000, diagnostics=recorder)
        obs = env.reset()
        result = []
        for _ in range(2):
            obs, _ = ppo.collect_rollout(env, obs)
            # Supply mixed labels to verify time-major selected-index alignment.
            ppo._rollout_recovery = torch.tensor([False, True, True, False])
            result.append(ppo.update())
        return ppo, result, torch.get_rng_state()
    control, expected, rng = train(False)
    observed, actual, other_rng = train(True)
    assert actual == expected
    assert torch.equal(rng, other_rng)
    for k, value in control.policy.state_dict().items():
        assert torch.equal(value, observed.policy.state_dict()[k]), k
    paths = sorted(tmp_path.glob("kl_*.pt"))
    assert len(paths) == 2
    for path in paths:
        event = load(path)
        assert len(event["rollout_flat_indices"]) <= 2
        assert torch.equal(event["recovery"], torch.tensor([False, True, True, False])[event["rollout_flat_indices"]])
        assert replay(path)["verified"]
    bad = load(paths[0])
    bad["old_loc"].add_(1.)
    torch.save(bad, tmp_path / "corrupted.pt")
    with pytest.raises(AssertionError):
        replay(tmp_path / "corrupted.pt")
