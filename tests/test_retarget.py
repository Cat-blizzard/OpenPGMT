"""data/retarget_lafan1.py 冒烟测试（需要真实 LAFAN1 数据，缺失则跳过）。

M1 验收的数值版：关节范围物理合理、足底不穿地、接触占比合理、
膝盖重建误差小（欧拉分解正确性的代理指标）。
"""

import os

import numpy as np
import pytest

from data.bvh import load_bvh
from data.retarget_lafan1 import (
    G1_JOINT_NAMES,
    g1_forward_kinematics,
    retarget,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lafan1")
HAS_DATA = os.path.isdir(DATA_DIR) and any(f.endswith(".bvh") for f in os.listdir(DATA_DIR))

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="需要 data/raw/lafan1 真实数据")


@pytest.fixture(scope="module")
def walk():
    return retarget(load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh")))


@pytest.fixture(scope="module")
def ground():
    return retarget(load_bvh(os.path.join(DATA_DIR, "ground1_subject1.bvh")))


def test_lying_reference_lowest_body_point_touches_ground(ground):
    """v2 高度锚定：躺地序列的全身最低体点应触地，而不是足部锚定
    （v1 锚到"最像站立"的帧，ground1_subject1 的参考整体悬浮 0.42 m+）。
    断言源侧隐含最低体点：root_z + (最低源体点 − 髋) ≈ 0。
    """
    from data.retarget_lafan1 import W
    bvh = load_bvh(os.path.join(DATA_DIR, "ground1_subject1.bvh"))
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * float(ground["scale"])
    hip_z = gpos[:, bvh.joint_index("Hips"), 2]
    implied_lowest = ground["root_pos"][:, 2] + (gpos[:, :, 2] - hip_z[:, None]).min(-1)
    assert implied_lowest.min() < 0.10, (
        f"躺地参考最低体点悬浮 {implied_lowest.min():.3f} m（v1 锚定回退）")
    assert implied_lowest.min() > -0.05, f"最低体点穿地 {implied_lowest.min():.3f} m"


def test_walking_reference_lowest_body_point_touches_ground(walk):
    """直立步行的最低体点是足，v2 锚定行为应与 v1 一致（足触地）。"""
    from data.retarget_lafan1 import W
    bvh = load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh"))
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * float(walk["scale"])
    hip_z = gpos[:, bvh.joint_index("Hips"), 2]
    implied_lowest = walk["root_pos"][:, 2] + (gpos[:, :, 2] - hip_z[:, None]).min(-1)
    assert implied_lowest.min() < 0.10
    assert implied_lowest.min() > -0.05


def test_output_shapes(walk):
    assert walk["qpos"].shape[1] == 29
    assert walk["qvel"].shape == walk["qpos"].shape
    assert walk["root_pos"].shape[1] == 3
    assert walk["root_rot"].shape[1] == 4
    assert walk["contacts"].shape[1] == 2
    assert list(walk["joint_names"]) == G1_JOINT_NAMES


def test_joint_ranges_physically_sane(walk):
    deg = np.degrees(walk["qpos"])
    rng = deg.ptp(0)
    # 走路片段：各关节范围应 < 150°（除手腕可略大）
    assert rng[:12].max() < 150, f"腿关节范围异常: {rng[:12].ptp()}"
    assert rng[12:15].max() < 120  # 腰
    # 角度均值接近 0（旋转均值居中后）
    assert np.abs(deg.mean(0)).max() < 30


def test_full_reference_feet_do_not_penetrate_ground(walk):
    """The clearance gate applies to the complete default export pipeline.

    Projecting the rotation-only seed into mechanical limits can lower an ankle
    (walk1 currently reaches -0.057 m before IK). Do not hide that by loosening
    this gate: validate the actual refined reference with the original bound.
    """
    from data.ik_refine import refine_full
    bvh = load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh"))
    reference = refine_full(walk, bvh)
    g1 = g1_forward_kinematics(reference["qpos"], reference["root_pos"], reference["root_rot"])
    min_z = min(g1["left_ankle_roll_link"][:, 2].min(),
                g1["right_ankle_roll_link"][:, 2].min())
    assert min_z > -0.05, f"足底穿地: {min_z:.3f} m"


def test_contact_fraction_reasonable(walk):
    for side in (0, 1):
        frac = walk["contacts"][:, side].mean()
        assert 0.2 < frac < 0.8, f"接触占比异常: {frac:.2f}"


def test_knee_reconstruction_error_small(walk):
    """左膝欧拉分解的可逆性：qpos 重建的旋转应接近源相对旋转。"""
    from data import retarget_lafan1 as R
    bvh = load_bvh(os.path.join(DATA_DIR, "walk1_subject1.bvh"))
    T = bvh.num_frames
    gpos_cm, grot = bvh.fk(unit_scale=1.0)
    scale = R._estimate_scale(gpos_cm, bvh)
    grot_g1 = R.quat_mul(np.tile(R._qW(), (T, 1, 1)), grot)
    Rc = R._mat(grot_g1[:, bvh.joint_index("LeftLeg")])
    Rp = R._mat(grot_g1[:, bvh.joint_index("LeftUpLeg")])
    Rsrc = Rp.transpose(0, 2, 1) @ Rc
    q_base = R._G1_REST["left_hip_yaw_link"][1]
    Ralign = R._mat(q_base) @ R.M_RIG
    R_c = Ralign @ Rsrc @ Ralign.T
    R_c = R_c @ R._mat_mean(R_c).T

    knee = walk["qpos"][:, 3]
    c, s = np.cos(knee), np.sin(knee)
    R_recon = np.zeros((T, 3, 3))
    R_recon[:, 0, 0], R_recon[:, 0, 2] = c, s
    R_recon[:, 1, 1] = 1.0
    R_recon[:, 2, 0], R_recon[:, 2, 2] = -s, c
    err = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", R_recon, R_c) - 1) / 2, -1, 1)))
    assert err.mean() < 15, f"膝盖重建误差 {err.mean():.1f}° 过大"


def test_scale_in_m_per_cm(walk):
    # 身高比例: LAFAN1 cm → G1 米, 比例应在 0.008~0.012 之间
    assert 0.006 < float(walk["scale"]) < 0.014


def test_scale_pose_invariant_lying_down():
    """躺地序列尺度不得爆炸（旧版竖直差法 ground1 scale=0.0318）。

    回归防护：髋高−足底高度的竖直差在躺地时→0，骨链长法则姿态无关。
    """
    bvh = load_bvh(os.path.join(DATA_DIR, "ground1_subject1.bvh"))
    d = retarget(bvh)
    assert float(d["scale"]) < 0.015, f"躺地序列尺度异常: {d['scale']}"


def test_fk_matches_manual_composition():
    """g1_forward_kinematics 与独立手工四元数链组合一致。

    回归防护：曾发现 torso_link 的关节映射缺失（waist_pitch 定义在
    torso_link 元素内，body[:-5] 映射失效 → 腰 pitch 从未应用，
    躯干/手臂位置全部错误）。
    """
    from data import retarget_lafan1 as R
    from data.bvh import quat_mul, quat_rot_vec

    rng = np.random.default_rng(0)
    qpos = rng.uniform(-0.5, 0.5, (4, 29))
    root_pos = rng.uniform(-1, 1, (4, 3))
    root_rot = rng.normal(size=(4, 4))
    root_rot /= np.linalg.norm(root_rot, axis=-1, keepdims=True)

    pos = g1_forward_kinematics(qpos, root_pos, root_rot)

    # 独立手工组合（与函数相同的约定，但独立实现）
    def manual():
        P = {"pelvis": root_pos.copy()}
        Q = {"pelvis": root_rot.copy()}
        for body in R._G1_REST:
            if body == "pelvis":
                continue
            # 相对偏移（_G1_REST 是绝对坐标，FK 叠加须用左表原始相对值）
            if body.startswith("left_"):
                parent, rest_pos, rest_quat = R._G1_REST_LEFT[body]
            elif body.startswith("right_"):
                lb = "left_" + body[len("right_"):]
                lp = R._G1_REST_LEFT[lb][0]
                parent = ("right_" + lp[len("left_"):]) if lp and lp.startswith("left_") else lp
                rest_pos = R._G1_REST_LEFT[lb][1] * np.array([1.0, -1.0, 1.0])
                rest_quat = R._G1_REST_LEFT[lb][2] * np.array([1.0, -1.0, 1.0, -1.0])
            else:
                parent, rest_pos, rest_quat = R._G1_REST_LEFT[body]
            jname = "waist_pitch" if body == "torso_link" else body[:-len("_link")]
            if jname in G1_JOINT_NAMES:
                axis = R._G1_JOINT_SPEC[jname][1]
                th = qpos[:, G1_JOINT_NAMES.index(jname)]
                c, s = np.cos(th / 2), np.sin(th / 2)
                jquat = np.stack([c, s * axis[0], s * axis[1], s * axis[2]], -1)
                q = quat_mul(Q[parent], quat_mul(rest_quat, jquat))
            else:
                q = quat_mul(Q[parent], rest_quat)
            Q[body] = q
            P[body] = P[parent] + quat_rot_vec(Q[parent], rest_pos)
        return P

    manual_pos = manual()
    for body in pos:
        assert np.allclose(pos[body], manual_pos[body], atol=1e-9), f"{body} 位置不一致"


def test_waist_pitch_rotates_shoulder_analytically():
    """解析回归：waist_pitch=90° 时肩部偏移绕 Y 轴精确旋转。

    肩部在 torso 框架内的静止偏移 (0.0039563, 0.10022, 0.24778)，
    Ry(90°) 后应为 (0.24778, 0.10022, −0.0039563)。
    """
    qpos = np.zeros((1, 29))
    qpos[0, G1_JOINT_NAMES.index("waist_pitch")] = np.pi / 2
    pos = g1_forward_kinematics(qpos, np.zeros((1, 3)), np.tile(np.array([1., 0, 0, 0]), (1, 1)))
    d = pos["left_shoulder_pitch_link"][0] - pos["torso_link"][0]
    expected = np.array([0.24778, 0.10022, -0.0039563])
    assert np.allclose(d, expected, atol=1e-4), f"肩部偏移 {d} ≠ {expected}"
