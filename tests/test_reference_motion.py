import numpy as np
import torch
from pathlib import Path
import pytest

from pgmt.envs.reference_motion import ReferenceMotion
from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.g1_env import G1_JOINT_NAMES, REQUIRED_BODY_NAMES


_URDF_CANDIDATES = (
    Path("/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/urdf/g1.urdf"),
    Path(__file__).parents[1] / "data/raw/g1/external/protomotions/g1.urdf",
)
URDF = next((p for p in _URDF_CANDIDATES if p.is_file()), None)
pytestmark = pytest.mark.skipif(URDF is None, reason="licensed G1 URDF is not available")


def _db(T=4):
    q = np.zeros((T, 29), np.float32)
    q[:, G1_JOINT_NAMES.index("left_knee")] = np.linspace(0, .2, T)
    root = np.zeros((T, 3), np.float32)
    root[:, 0] = np.arange(T, dtype=np.float32) * .02
    return MotionDatabase.from_sequences([{
        "qpos": q, "qvel": np.gradient(q, .02, axis=0),
        "root_pos": root, "root_rot": np.tile([1, 0, 0, 0], (T, 1)),
        "frame_time": .02,
    }])


def test_reference_motion_has_complete_body_mapping_and_batch_fk():
    motion = ReferenceMotion(_db(), URDF, device="cpu")
    assert motion.body_names == REQUIRED_BODY_NAMES
    assert len(motion.body_names) == 28
    seq, frame = motion.sample_segments(3)
    out = motion.sample(seq, frame)
    assert out["body_pos"].shape == (3, 28, 3)
    assert out["body_quat"].shape == (3, 28, 4)
    assert out["future"].shape == (3, 6, 61)
    assert out["foot_contact"].shape == (3, 2)
    assert torch.isfinite(out["body_pos"]).all()


def test_reference_motion_root_placement_preserves_relative_body_pose():
    motion = ReferenceMotion(_db(), URDF, device="cpu")
    seq = torch.tensor([0]); frame = torch.tensor([0.])
    raw = motion.sample(seq, frame)
    placed = motion.sample(seq, frame, robot_root_pos=torch.tensor([[4., -2., 9.]]))
    assert torch.allclose(placed["body_pos"], raw["body_pos"], atol=1e-5)
    assert torch.allclose(placed["position_error"], torch.tensor([[4., -2.]]), atol=1e-5)


def test_reference_motion_root_velocity_uses_source_frame_time():
    motion = ReferenceMotion(_db(), URDF, device="cpu")
    out = motion.sample(torch.tensor([0]), torch.tensor([0.]))
    assert torch.allclose(out["root_lin_vel"][0], torch.tensor([1., 0., 0.]), atol=1e-5)


def test_pi_rotation_conversion_and_kinematics_interface():
    motion = ReferenceMotion(_db(), URDF, device="cpu")
    qd = torch.zeros(1, 29)
    # A pi yaw root rotation must round-trip with a valid unit quaternion.
    root_q = torch.tensor([[0., 0., 0., 1.]])
    out = motion.kinematics(qd, qd, torch.zeros(1, 3), root_q,
                            torch.zeros(1, 3), torch.zeros(1, 3))
    assert torch.allclose(out["body_quat"].norm(dim=-1), torch.ones(1, 28), atol=1e-5)
    assert torch.isfinite(out["body_ang_vel"]).all()
