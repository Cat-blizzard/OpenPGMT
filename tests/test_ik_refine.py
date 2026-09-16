"""data/ik_refine.py：几何雅可比数值差分回归 + 精修效果冒烟。

回归防护：雅可比曾有三个实现 bug（上游判定含自身、轴缺少子 body
静止 quat、旋转支点用父原点而非子原点），差分测试逐一锁定。
"""

import os

import numpy as np
import pytest

from data.bvh import load_bvh
from data.ik_refine import LEG_KEYPOINTS, _batch_fk, _batch_jacobian, refine_legs
from data.retarget_lafan1 import W, g1_forward_kinematics, retarget

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lafan1")
HAS_DATA = os.path.isdir(DATA_DIR) and any(f.endswith(".bvh") for f in os.listdir(DATA_DIR))

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="需要 data/raw/lafan1 真实数据")


@pytest.fixture(scope="module")
def walk():
    bvh = load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh"))
    d = retarget(bvh)
    return bvh, d


def test_jacobian_matches_finite_difference(walk):
    from data.retarget_lafan1 import G1_JOINT_NAMES
    bvh, d = walk
    rng = np.random.default_rng(0)
    joint_names = G1_JOINT_NAMES[:12]
    bodies = [b for _, b in LEG_KEYPOINTS]
    for t in rng.choice(bvh.num_frames, 5, replace=False):
        q0 = d["qpos"][t].copy()
        pos, quat = _batch_fk(q0[None], d["root_pos"][t:t + 1], d["root_rot"][t:t + 1])
        J = _batch_jacobian(pos, quat, bodies, joint_names)[0]
        eps = 0.02
        for j in range(12):
            q1 = q0.copy()
            q1[j] += eps
            pos1, _ = _batch_fk(q1[None], d["root_pos"][t:t + 1], d["root_rot"][t:t + 1])
            for k, (src, g1b) in enumerate(LEG_KEYPOINTS):
                fd = (pos1[g1b][0] - pos[g1b][0]) / eps
                an = J[3 * k:3 * k + 3, j]
                assert np.allclose(fd, an, atol=0.01), \
                    f"帧{t} 关节{j} {src}: 差分 {fd} vs 解析 {an}"


def test_refine_reduces_leg_error(walk):
    bvh, d = walk
    d2 = refine_legs(d, bvh)
    scale = float(d["scale"])
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale

    def leg_err(data):
        g1 = g1_forward_kinematics(data["qpos"], data["root_pos"], data["root_rot"])
        delta = gpos[:, bvh.joint_index("Hips")] - g1["pelvis"]
        errs = []
        for src, g1b in [("LeftLeg", "left_knee_link"), ("LeftFoot", "left_ankle_pitch_link"),
                         ("RightLeg", "right_knee_link"), ("RightFoot", "right_ankle_pitch_link")]:
            errs.append(np.linalg.norm(gpos[:, bvh.joint_index(src)] - delta - g1[g1b], axis=-1).mean())
        return np.mean(errs)

    assert leg_err(d2) < leg_err(d) * 0.7, "IK 精修应显著降低腿部误差"
    assert leg_err(d2) * 100 < 8.0, f"精修后腿部误差 {leg_err(d2)*100:.1f}cm 应 ≤8cm（M1.5a 验收线）"


def test_refine_no_velocity_spikes(walk):
    bvh, d = walk
    d2 = refine_legs(d, bvh)
    assert (np.abs(d2["qvel"]) > 30).mean() < 1e-4, "精修不应引入速度尖峰"


def test_refine_respects_joint_limits(walk):
    from data.retarget_lafan1 import G1_JOINT_LIMITS, G1_JOINT_NAMES
    bvh, d = walk
    d2 = refine_legs(d, bvh)
    for i, jname in enumerate(G1_JOINT_NAMES[:12]):
        lo, hi = G1_JOINT_LIMITS[jname]
        assert (d2["qpos"][:, i] >= lo - 1e-6).all() and (d2["qpos"][:, i] <= hi + 1e-6).all(), \
            f"{jname} 超出限位"
