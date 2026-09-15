"""data/bvh.py：合成 BVH 的解析与 FK 数值校验。"""

import numpy as np
import pytest

from data.bvh import euler_to_quat, parse_bvh, quat_mul, quat_rot_vec

# 两关节合成 BVH：root(6 通道) + child(3 通道)，3 帧
SYNTHETIC = """HIERARCHY
ROOT Hips
{
	OFFSET 0.0 90.0 0.0
	CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
	JOINT Spine
	{
		OFFSET 0.0 10.0 0.0
		CHANNELS 3 Zrotation Yrotation Xrotation
		End Site
		{
			OFFSET 0.0 5.0 0.0
		}
	}
}
MOTION
Frames: 3
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0  0.0 0.0 0.0
10.0 0.0 0.0 0.0 0.0 0.0  0.0 0.0 0.0
20.0 0.0 0.0 0.0 0.0 0.0  90.0 0.0 0.0
"""


def test_parse_structure():
    bvh = parse_bvh(SYNTHETIC)
    assert bvh.names == ["Hips", "Spine"]
    assert list(bvh.parents) == [-1, 0]
    assert bvh.euler_order == ["zyx", "zyx"]
    assert bvh.frame_time == pytest.approx(1 / 30, rel=1e-4)
    assert bvh.num_frames == 3
    assert bvh.end_sites["Spine"] == pytest.approx([0.0, 5.0, 0.0])


def test_root_position_and_rotations():
    bvh = parse_bvh(SYNTHETIC)
    assert bvh.root_pos[:, 0] == pytest.approx([0.0, 10.0, 20.0])
    # 第 3 帧 Spine 绕 z 转 90°
    assert bvh.rotations[2, 1, 0] == pytest.approx(np.radians(90.0))
    assert bvh.rotations[2, 1, 1:].sum() == pytest.approx(0.0)


def test_fk_positions():
    bvh = parse_bvh(SYNTHETIC)
    gpos, grot = bvh.fk(unit_scale=1.0)
    # 第 1 帧：根位置 = 通道值 (0,0,0)（根 OFFSET 是绑定偏移，FK 不用），
    # Spine = root + offset(0,10,0)
    assert gpos[0, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert gpos[0, 1] == pytest.approx([0.0, 10.0, 0.0])
    # 第 3 帧：根旋转为 0，Spine 位置 = root(20,0,0) + offset(0,10,0)
    # （关节自身旋转只影响其子链，不影响自身位置）
    assert gpos[2, 1] == pytest.approx([20.0, 10.0, 0.0], abs=1e-6)
    # Spine 朝向 = Rz(90°)：局部 (0,1,0) → 世界 (−1,0,0)
    assert quat_rot_vec(grot[2, 1], np.array([0.0, 1.0, 0.0])) == pytest.approx(
        [-1.0, 0.0, 0.0], abs=1e-6)


def test_euler_zyx_matches_manual():
    # R = Rz·Ry·Rx 约定与手工矩阵一致
    q = euler_to_quat(np.array([np.radians(90.0), 0.0, 0.0]), "zyx")
    v = quat_rot_vec(q, np.array([1.0, 0.0, 0.0]))
    assert v == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_quat_mul_associativity():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 4))
    a, b, c = a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)
    assert np.allclose(quat_mul(quat_mul(a, b), c), quat_mul(a, quat_mul(b, c)), atol=1e-6)
