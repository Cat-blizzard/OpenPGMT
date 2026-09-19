import numpy as np
import torch

from pgmt.contracts import ACT_DIM, REF_FRAME_DIM
from pgmt.train.stage1 import BatchRewardAdapter, PDConfig, PDBatchController, TorchMotionDatabase


class _DB:
    def __init__(self):
        t = np.arange(4, dtype=np.float32)
        self.seqs = [{
            "qpos": np.repeat(t[:, None], ACT_DIM, axis=1),
            "qvel": np.ones((4, ACT_DIM), np.float32),
            "root_pos": np.stack((t, t * 0, t * 0), axis=1),
            "root_rot": np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
            "frame_time": 1.0,
        }]


def test_torch_reference_database_is_batched_and_device_resident():
    db = TorchMotionDatabase(_DB())
    seq = torch.tensor([0, 0])
    frame = torch.tensor([0.5, 2.0])
    batch = db.batch(seq, frame)
    assert batch.qpos.shape == (2, ACT_DIM)
    assert batch.qvel.shape == (2, ACT_DIM)
    assert batch.future.shape == (2, 6, REF_FRAME_DIM)
    assert torch.allclose(batch.qpos[0], torch.full((ACT_DIM,), 0.5))


def test_pd_controller_maps_and_limits_targets_and_torque():
    c = PDBatchController(PDConfig(
        action_scale=torch.ones(ACT_DIM), kp=torch.full((ACT_DIM,), 2.),
        kd=torch.ones(ACT_DIM), default_q=torch.zeros(ACT_DIM),
        torque_limits=torch.full((ACT_DIM,), 0.5)))
    action = torch.full((2, ACT_DIM), 2.)
    target = c.target(action)
    tau = c.torque(action, torch.zeros_like(action), torch.zeros_like(action))
    assert torch.allclose(target, torch.ones_like(target))
    assert torch.allclose(tau, torch.full_like(tau, 0.5))


def test_batch_reward_returns_upper_lower_aux_heads():
    names = (
        "link_pos", "link_ori", "link_lin_vel", "link_ang_vel", "joint_pos", "joint_vel",
        "ta_link_pos", "ta_link_ori", "ta_joint_pos", "root_ori", "corrected_root_vel",
        "floating_anchor_pos", "recovery_upward_vel", "pelvis_vert_accel", "ee_accel_mismatch",
        "action_rate", "joint_limit", "undesired_contact", "head_torso_impact",
    )
    values = {name: torch.zeros(3) for name in names}
    values.update({name: torch.ones(3) for name in ("root_ori", "corrected_root_vel", "floating_anchor_pos", "recovery_upward_vel")})
    rewards = BatchRewardAdapter()(values)
    assert rewards.shape == (3, 3)
    assert torch.isfinite(rewards).all()
    assert torch.all(rewards[:, :2] > 0)
