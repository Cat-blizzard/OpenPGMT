import math

import torch

from pgmt.rewards.batched import BatchedRewardComputer
from pgmt.rewards.tracking import default_partitions
from pgmt.rewards.tracking import RobotState, ReferenceFrame, compute_residuals
from pgmt.rewards.terrain_contact import TerrainContactState, compute_terrain_values


def _fixture(n=2):
    p = default_partitions()
    bodies = list(dict.fromkeys(p.upper_bodies + p.lower_bodies))
    joints = list(p.lower_joints + p.upper_joints)
    b = len(bodies)
    state = {
        "joint_pos": torch.zeros(n, 29), "joint_vel": torch.zeros(n, 29),
        "root_pos": torch.zeros(n, 3), "root_quat": torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
        "root_lin_vel": torch.zeros(n, 3), "root_ang_vel": torch.zeros(n, 3),
        "body_pos": torch.zeros(n, b, 3), "body_quat": torch.tensor([[1., 0, 0, 0]]).repeat(n, b, 1),
        "body_lin_vel": torch.zeros(n, b, 3), "body_ang_vel": torch.zeros(n, b, 3),
        "contact_forces": torch.zeros(n, b, 3),
    }
    ref = {k: v.clone() for k, v in state.items()}
    return bodies, joints, state, ref


def test_exact_match_has_max_tracking_rewards():
    bodies, joints, state, ref = _fixture()
    c = BatchedRewardComputer(bodies, joints)
    r, m = c.compute(state, ref, action=torch.zeros(2, 29))
    assert r.shape == (2, 3)
    assert torch.allclose(r[:, :2], torch.tensor(4.5))
    assert torch.all(r[:, 2] > 0)
    assert torch.allclose(m["upper"]["link_pos"], torch.ones(2))


def test_body_and_joint_errors_are_batch_sensitive_and_partitioned():
    bodies, joints, state, ref = _fixture(1)
    c = BatchedRewardComputer(bodies, joints)
    state["body_pos"][0, bodies.index("torso_link"), 0] = 0.5
    state["joint_pos"][0, joints.index("left_hip_pitch")] = 0.5
    r, m = c.compute(state, ref, action=torch.zeros(1, 29))
    assert m["upper"]["link_pos"].item() < 1
    assert m["lower"]["ta_joint_pos"].item() < 1
    assert r[0, 0] < 4.5 and r[0, 1] < 4.5


def test_common_translation_and_yaw_do_not_change_tracking():
    bodies, joints, state, ref = _fixture(1)
    state["body_pos"][0, :, 0] = torch.arange(len(bodies), dtype=torch.float32)
    ref = {k: v.clone() for k, v in state.items()}
    c = BatchedRewardComputer(bodies, joints)
    base, _ = c.compute(state, ref, action=torch.zeros(1, 29))
    angle = math.pi / 3
    q = torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
    for d in (state, ref):
        d["root_pos"] += torch.tensor([2., -1., .4])
        d["root_quat"] = q.expand(1, -1)
        x, y, z = d["body_pos"][..., 0], d["body_pos"][..., 1], d["body_pos"][..., 2]
        d["body_pos"] = torch.stack((
            torch.cos(torch.tensor(angle)) * x - torch.sin(torch.tensor(angle)) * y + 2,
            torch.sin(torch.tensor(angle)) * x + torch.cos(torch.tensor(angle)) * y - 1,
            z + .4), -1)
    transformed, _ = c.compute(state, ref, action=torch.zeros(1, 29))
    assert torch.allclose(base, transformed, atol=1e-5)


def test_stage2_requires_real_contact_and_height_inputs():
    bodies, joints, state, ref = _fixture(1)
    c = BatchedRewardComputer(bodies, joints)
    state["contact_forces"] = torch.zeros(1, len(bodies), 3)
    ref["foot_contact"] = torch.zeros(1, 2, dtype=torch.bool)
    try:
        c.compute(state, ref, action=torch.zeros(1, 29), stage2=True, terrain={})
    except KeyError as e:
        assert "required" in str(e)
    else:
        raise AssertionError("missing terrain measurements must not become placeholder reward")


def test_tracking_terms_match_numpy_oracle_after_residual_reduction():
    bodies, joints, state, ref = _fixture(1)
    state["body_pos"][0, :, 0] = torch.linspace(0, 1, len(bodies))
    state["joint_pos"][0] = torch.linspace(0, .4, 29)
    c = BatchedRewardComputer(bodies, joints)
    _, metrics = c.compute(state, ref, action=torch.zeros(1, 29))
    s = RobotState(
        base_pos=state["root_pos"][0].numpy(), base_quat=state["root_quat"][0].numpy(),
        base_lin_vel=state["root_lin_vel"][0].numpy(), base_ang_vel=state["root_ang_vel"][0].numpy(),
        gravity_z=-1., joint_pos=state["joint_pos"][0].numpy(),
        joint_vel=state["joint_vel"][0].numpy(), joint_acc=torch.zeros(29).numpy(),
        link_pos={n: state["body_pos"][0, i].numpy() for i, n in enumerate(bodies)},
        link_quat={n: state["body_quat"][0, i].numpy() for i, n in enumerate(bodies)},
        link_lin_vel={n: state["body_lin_vel"][0, i].numpy() for i, n in enumerate(bodies)},
        link_ang_vel={n: state["body_ang_vel"][0, i].numpy() for i, n in enumerate(bodies)})
    rr = ReferenceFrame(
        joint_pos=ref["joint_pos"][0].numpy(), joint_vel=ref["joint_vel"][0].numpy(),
        anchor_pos=ref["root_pos"][0].numpy(), anchor_quat=ref["root_quat"][0].numpy(),
        link_pos={n: ref["body_pos"][0, i].numpy() for i, n in enumerate(bodies)},
        link_quat={n: ref["body_quat"][0, i].numpy() for i, n in enumerate(bodies)},
        link_lin_vel={n: ref["body_lin_vel"][0, i].numpy() for i, n in enumerate(bodies)},
        link_ang_vel={n: ref["body_ang_vel"][0, i].numpy() for i, n in enumerate(bodies)})
    p = default_partitions()
    ur = compute_residuals(s, rr, p.upper_bodies, [joints.index(x) for x in p.upper_joints])
    # The batched implementation applies exp(mean residual / sigma), the
    # repository oracle's group contract, rather than mean(exp(element)).
    expected = math.exp(-(ur["link_pos"] ** 2) / .06)
    assert math.isclose(metrics["upper"]["link_pos"].item(), expected, rel_tol=1e-5)
