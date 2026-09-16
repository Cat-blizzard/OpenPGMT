"""跟踪残差与分组奖励（M2 待办 #1）。

## 本文件的验证策略

跟踪残差是**纯粹的状态差异度量**，因此可以用"构造状态"做**解析验证**：

  - 机器人状态 == 参考 ⇒ 所有残差**恰为 0**，奖励**恰为 1**（高斯核在 0 处为 1）
  - 把某个量偏移一个**已知量** ⇒ 残差**恰等于**该量（不是"差不多"）

后者是关键：它把"残差公式对不对"变成可判定的等式，而不是模糊的量级检查。
我此前在 `quat_to_mat` 上连错三次期望值，教训就是**期望值必须由构造方式直接给出**。
"""

import math

import numpy as np
import pytest

from pgmt.rewards.spec import SIGMAS, exp_tracking_reward
from pgmt.rewards.tracking import (
    TA_TERMS,
    Partitions,
    ReferenceFrame,
    RobotState,
    align_reference,
    compute_residuals,
    default_partitions,
    group_reward,
    group_term_values,
    joint_pos_residual,
    joint_vel_residual,
    link_ang_vel_residual,
    link_lin_vel_residual,
    link_ori_residual,
    link_pos_residual,
    ref_anchor_from_links,
    relative_rotation_angle,
    residuals_by_partition,
)

BODIES = ("pelvis", "torso_link", "left_knee_link")


def _quat_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _state_and_ref(nq: int = 4, bodies=BODIES, joint_err: float = 0.0,
                   vel_err: float = 0.0):
    """构造"机器人 == 参考"的状态对；`joint_err` 给关节加已知偏移。

    **锚点必须与 `link_pos["pelvis"]` 是同一点** —— 论文的 "reference root
    anchor" 就是 reference 的根 body（pelvis）。我第一版把 `anchor_pos` 随手设成
    基座位置、`pelvis` 另设一个值，两者差 0.37，导致
    `test_align_reference_makes_matching_state_zero_residual` 残差恰为 0.37
    —— 那是**测试数据不自洽**，不是实现错。真实数据里两者必然重合。
    """
    base_q = _quat_z(0.3)
    joint = np.linspace(-0.5, 0.5, nq)
    jvel = np.linspace(-1.0, 1.0, nq)

    pelvis_pos = np.array([0.05, -0.1, 0.75])
    other_pos = {b: np.array([0.1 * i, -0.2 * i, 0.5 + 0.1 * i])
                 for i, b in enumerate(bodies) if b != "pelvis"}
    link_pos = {"pelvis": pelvis_pos, **other_pos}
    link_quat = {b: _quat_z(0.1 * i) for i, b in enumerate(bodies)}
    link_lv = {b: np.array([0.3 * i, 0.1, -0.2]) for i, b in enumerate(bodies)}
    link_av = {b: np.array([0.0, 0.4 * i, 0.2]) for i, b in enumerate(bodies)}

    # 基座与骨盆：真实 G1 里基座(根)与 pelvis body 是同一刚体，取同一点；
    # 但两者朝向都可独立给（此处相同）
    state = RobotState(
        base_pos=pelvis_pos.copy(), base_quat=base_q,
        base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), gravity_z=-1.0,
        joint_pos=joint + joint_err, joint_vel=jvel + vel_err,
        joint_acc=np.zeros(nq),
        link_pos=dict(link_pos), link_quat=dict(link_quat),
        link_lin_vel=dict(link_lv), link_ang_vel=dict(link_av),
    )
    ref = ReferenceFrame(
        joint_pos=joint.copy(), joint_vel=jvel.copy(),
        # 锚点与 pelvis 必须是同一点（见 docstring）
        anchor_pos=pelvis_pos.copy(), anchor_quat=base_q,
        link_pos={k: v.copy() for k, v in link_pos.items()},
        link_quat={k: v.copy() for k, v in link_quat.items()},
        link_lin_vel={k: v.copy() for k, v in link_lv.items()},
        link_ang_vel={k: v.copy() for k, v in link_av.items()},
    )
    # 自洽性守卫：锚点与 pelvis 必须是同一点。这只可能在合成数据里出错
    # （真实参考里 root anchor 就是 pelvis body），但错一次就会让所有
    # root-centric 测试的残差多出一个常数偏移 —— 那正是我第一版的症状（差 0.37）。
    assert np.allclose(ref.anchor_pos, ref.link_pos["pelvis"]), \
        "夹具不自洽：anchor_pos 应等于 link_pos['pelvis']"
    return state, ref


# ---------------------------------------------------------------------------
# 数据结构的自检
# ---------------------------------------------------------------------------

def test_robot_state_validates_shapes():
    s, _ = _state_and_ref()
    assert s.num_joints == 4
    with pytest.raises(ValueError, match="base_pos"):
        RobotState(base_pos=np.zeros(2), base_quat=s.base_quat,
                   base_lin_vel=s.base_lin_vel, base_ang_vel=s.base_ang_vel,
                   gravity_z=-1.0, joint_pos=s.joint_pos, joint_vel=s.joint_vel,
                   joint_acc=s.joint_acc)
    with pytest.raises(ValueError, match="joint_vel"):
        RobotState(base_pos=s.base_pos, base_quat=s.base_quat,
                   base_lin_vel=s.base_lin_vel, base_ang_vel=s.base_ang_vel,
                   gravity_z=-1.0, joint_pos=s.joint_pos,
                   joint_vel=np.zeros(3), joint_acc=s.joint_acc)


def test_robot_state_rejects_empty_joints():
    s, _ = _state_and_ref()
    with pytest.raises(ValueError, match="不能为空"):
        RobotState(base_pos=s.base_pos, base_quat=s.base_quat,
                   base_lin_vel=s.base_lin_vel, base_ang_vel=s.base_ang_vel,
                   gravity_z=-1.0, joint_pos=np.zeros(0), joint_vel=np.zeros(0),
                   joint_acc=np.zeros(0))


def test_reference_frame_validates_lengths():
    _, r = _state_and_ref()
    with pytest.raises(ValueError, match="joint_vel 与 joint_pos"):
        ReferenceFrame(joint_pos=np.zeros(4), joint_vel=np.zeros(3),
                       anchor_pos=r.anchor_pos, anchor_quat=r.anchor_quat)
    with pytest.raises(ValueError, match="anchor_pos"):
        ReferenceFrame(joint_pos=np.zeros(4), joint_vel=np.zeros(4),
                       anchor_pos=np.zeros(2), anchor_quat=r.anchor_quat)


def test_partitions_reject_overlap():
    with pytest.raises(ValueError, match="body 集合不得重叠"):
        Partitions(upper_bodies=("a",), lower_bodies=("a",))
    with pytest.raises(ValueError, match="joint 集合不得重叠"):
        Partitions(upper_bodies=("a",), lower_bodies=("b",),
                   upper_joints=("j",), lower_joints=("j",))
    with pytest.raises(ValueError, match="至少要有若干 body"):
        Partitions(upper_bodies=(), lower_bodies=())


def test_default_partitions_cover_g1_29dof():
    """默认划分必须覆盖 29 个关节，且上下半身不重叠、无遗漏。"""
    p = default_partitions()
    joints = list(p.upper_joints) + list(p.lower_joints)
    assert len(joints) == 29, f"应覆盖 29 个关节，得到 {len(joints)}"
    assert len(set(joints)) == 29
    # 关节名与 G1_JOINT_NAMES 一致（顺序无关，但必须同名）
    from data.retarget_lafan1 import G1_JOINT_NAMES
    assert set(joints) == set(G1_JOINT_NAMES), (
        f"划分与 G1 关节名不一致: {set(joints) ^ set(G1_JOINT_NAMES)}"
    )


def test_default_partitions_lower_body_is_legs_only():
    """论文：lower-body 与平衡/接触强耦合 → 只含腿（髋/膝/踝）。"""
    p = default_partitions()
    for j in p.lower_joints:
        assert j.startswith(("left_hip", "right_hip", "left_knee", "right_knee",
                             "left_ankle", "right_ankle")), f"{j} 不应属 lower"
    for j in p.upper_joints:
        assert not j.startswith(("left_hip", "right_hip", "left_knee",
                                 "right_knee", "left_ankle", "right_ankle"))


# ---------------------------------------------------------------------------
# ★ 解析验证：与参考一致 ⇒ 残差恰为 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,arg", [
    (link_pos_residual, None),
    (link_ori_residual, None),
    (link_lin_vel_residual, None),
    (link_ang_vel_residual, None),
])
def test_link_residuals_are_exactly_zero_when_matching(fn, arg):
    """机器人状态 == 参考 ⇒ 各残差**恰为 0**（不是"接近 0"）。"""
    s, r = _state_and_ref()
    assert fn(s, r, BODIES) == pytest.approx(0.0, abs=1e-12)


def test_joint_residuals_are_exactly_zero_when_matching():
    s, r = _state_and_ref()
    assert joint_pos_residual(s, r) == pytest.approx(0.0, abs=1e-12)
    assert joint_vel_residual(s, r) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# ★ 解析验证：偏移已知量 ⇒ 残差恰等于该量
# ---------------------------------------------------------------------------

def test_link_pos_residual_equals_known_offset():
    """把某个 body 平移已知距离 d ⇒ 该 body 的残差恰为 d。

    用**单个 body** 且只动它，使均值就等于 d（无歧义）。
    推导：残差 = 该 body 的位置差范数 = ‖d‖，均值 over 1 个 body = ‖d‖。
    """
    s, r = _state_and_ref(bodies=("pelvis",))
    d = 0.37
    s = RobotState(**{**s.__dict__,
                      "link_pos": {"pelvis": s.link_pos["pelvis"] + np.array([d, 0.0, 0.0])}})
    assert link_pos_residual(s, r, ("pelvis",)) == pytest.approx(d)


def test_link_pos_residual_averages_over_bodies():
    """两个 body 分别偏 0.1 与 0.3 ⇒ 均值 = 0.2（验证是均值而非求和）。"""
    s, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    s = RobotState(**{**s.__dict__, "link_pos": {
        "pelvis": s.link_pos["pelvis"] + np.array([0.1, 0.0, 0.0]),
        "torso_link": s.link_pos["torso_link"] + np.array([0.3, 0.0, 0.0]),
    }})
    got = link_pos_residual(s, r, ("pelvis", "torso_link"))
    assert got == pytest.approx(0.2), f"应取均值 0.2，得到 {got}"


def test_link_ori_residual_equals_known_angle():
    """把某 body 绕 z 转 θ ⇒ 残差恰为 θ（旋转角公式的解析验证）。"""
    s, r = _state_and_ref(bodies=("pelvis",))
    theta = math.radians(25.0)
    s = RobotState(**{**s.__dict__,
                      "link_quat": {"pelvis": _quat_z(0.1 + theta)}})
    r = ReferenceFrame(joint_pos=r.joint_pos, joint_vel=r.joint_vel,
                       anchor_pos=r.anchor_pos, anchor_quat=r.anchor_quat,
                       link_quat={"pelvis": _quat_z(0.1)})
    got = link_ori_residual(s, r, ("pelvis",))
    assert got == pytest.approx(theta, abs=1e-9)


def test_link_vel_residual_equals_known_difference():
    s, r = _state_and_ref(bodies=("pelvis",))
    dv = np.array([0.0, 0.6, 0.0])
    s = RobotState(**{**s.__dict__,
                      "link_lin_vel": {"pelvis": r.link_lin_vel["pelvis"] + dv}})
    assert link_lin_vel_residual(s, r, ("pelvis",)) == pytest.approx(0.6)


def test_link_ang_vel_residual_equals_known_difference():
    s, r = _state_and_ref(bodies=("torso_link",))
    s = RobotState(**{**s.__dict__,
                      "link_ang_vel": {"torso_link": r.link_ang_vel["torso_link"]
                                       + np.array([0.0, 0.0, 0.8])}})
    assert link_ang_vel_residual(s, r, ("torso_link",)) == pytest.approx(0.8)


def test_joint_pos_residual_equals_known_offset():
    s, r = _state_and_ref(joint_err=0.13)
    assert joint_pos_residual(s, r) == pytest.approx(0.13)


def test_joint_vel_residual_equals_known_offset():
    s, r = _state_and_ref(vel_err=0.27)
    assert joint_vel_residual(s, r) == pytest.approx(0.27)


def test_joint_residual_subsets_select_the_right_entries():
    """按索引取子集时，只有被选中的关节参与 —— 用"只偏一个关节"验证。"""
    s, r = _state_and_ref(nq=6)
    jp = np.asarray(s.joint_pos).copy()
    jp[3] += 0.5
    s = RobotState(**{**s.__dict__, "joint_pos": jp})
    assert joint_pos_residual(s, r, (3,)) == pytest.approx(0.5)
    assert joint_pos_residual(s, r, (0, 1)) == pytest.approx(0.0)
    assert joint_pos_residual(s, r, (0, 3)) == pytest.approx(0.25)  # 均值


def test_residuals_skip_bodies_missing_from_either_side():
    """缺 body 时应**跳过**（不参与均值），而不是报错或当 0 处理。"""
    s, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    r = ReferenceFrame(joint_pos=r.joint_pos, joint_vel=r.joint_vel,
                       anchor_pos=r.anchor_pos, anchor_quat=r.anchor_quat,
                       link_pos={"pelvis": r.link_pos["pelvis"] + np.array([0.4, 0, 0])})
    # 只有 pelvis 两侧都有 → 均值 = 0.4（torso_link 被跳过）
    assert link_pos_residual(s, r, ("pelvis", "torso_link")) == pytest.approx(0.4)


def test_residuals_zero_when_no_bodies_tracked():
    """没有任何可跟踪 body 时返回 0（并集为空 → 均值为 0，不抛异常）。"""
    s, r = _state_and_ref()
    assert link_pos_residual(s, r, ()) == 0.0


def test_joint_residual_rejects_length_mismatch():
    s, r = _state_and_ref(nq=4)
    bad = ReferenceFrame(joint_pos=np.zeros(3), joint_vel=np.zeros(3),
                         anchor_pos=r.anchor_pos, anchor_quat=r.anchor_quat)
    with pytest.raises(ValueError, match="形状不一致"):
        joint_pos_residual(s, bad)


# ---------------------------------------------------------------------------
# relative_rotation_angle
# ---------------------------------------------------------------------------

def test_rotation_angle_analytic():
    for deg in (0.0, 15.0, 90.0, 179.0):
        a = _quat_z(0.0)
        b = _quat_z(math.radians(deg))
        assert relative_rotation_angle(a, b) == pytest.approx(
            math.radians(deg), abs=1e-6), f"{deg}° 反演失败"


def test_rotation_angle_ignores_quaternion_sign():
    """q 与 −q 是同一旋转 → 夹角必须为 0。"""
    q = _quat_z(0.7)
    assert relative_rotation_angle(q, -q) == pytest.approx(0.0, abs=1e-9)


def test_rotation_angle_is_symmetric():
    a, b = _quat_z(0.2), _quat_z(1.1)
    assert relative_rotation_angle(a, b) == pytest.approx(
        relative_rotation_angle(b, a), abs=1e-12)


def test_rotation_angle_max_is_pi():
    a = _quat_z(0.0)
    b = _quat_z(math.pi)
    assert relative_rotation_angle(a, b) == pytest.approx(math.pi, abs=1e-9)


def test_yaw_of_is_consistent_with_the_project_quaternion_convention():
    """偏航提取必须与项目的 `quat_rot_vec` 约定一致。

    `_yaw_of` 第一版用旋转矩阵元素手写 atan2，把符号搞反了 —— 之所以能被抓到，
    是因为 `test_align_reference_makes_matching_state_zero_residual` 里
    残差恰等于"绕 z 转 2ψ"的位移量。这里把它与项目既有的四元数工具**直接对照**，
    使约定一致性成为一个显式断言，而不是靠间接推断。
    """
    from data.bvh import quat_rot_vec
    from pgmt.rewards.tracking import _yaw_of

    for deg in (-170.0, -90.0, -30.0, 0.0, 30.0, 90.0, 170.0):
        a = math.radians(deg)
        q = _quat_z(a)
        assert _yaw_of(q) == pytest.approx(a, abs=1e-9), f"{deg}° 提取失败"
        # 与 quat_rot_vec 独立对照：x 轴转过 a 后方位角应为 a
        v = quat_rot_vec(q, np.array([1.0, 0.0, 0.0]))
        assert math.atan2(v[1], v[0]) == pytest.approx(a, abs=1e-9)


def test_yaw_of_handles_general_quaternion():
    """带横滚/俯仰的四元数，其偏航仍是绕 z 的分量（不是整体旋转角）。"""
    from pgmt.rewards.tracking import _yaw_of

    # 先绕 x 倾 40°，再绕 z 偏 25°：复合后的偏航应仍为 25°
    tilt = math.radians(40.0)
    yaw = math.radians(25.0)
    q_tilt = np.array([math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0])
    q_yaw = _quat_z(yaw)
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_tilt
    q = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    assert _yaw_of(q) == pytest.approx(yaw, abs=1e-9)


def test_yaw_of_rejects_zero_quaternion():
    from pgmt.rewards.tracking import _yaw_of

    with pytest.raises(ValueError, match="零四元数"):
        _yaw_of(np.zeros(4))


# ---------------------------------------------------------------------------
# root-centric 对齐
# ---------------------------------------------------------------------------

def test_align_reference_makes_matching_state_zero_residual():
    """**root-centric 的核心**：机器人与参考是**同一物理配置**时（只是所在
    世界位姿不同），对齐后所有残差应为 0。

    这正是论文 "tracking errors are computed relative to the reference root
    anchor" 的可验证含义 —— 策略从不同初始位姿/朝向都能复现同一运动意图。

    ## 我在这里错了两次，都是构造问题（实现一直是对的）

    正确构造只有一种：**机器人的所有量直接取自世界版参考**。即令
    `robot.base_pos = ref_world.anchor_pos`、`robot.link_pos = ref_world.link_pos`
    —— 这就是"同一物理配置"的定义。

    两次错误构造：
      1. 把 `anchor_pos` 设成基座位置、`link_pos["pelvis"]` 另设一值 → 锚点与
         pelvis 不是同一点，残差恰为那个差值；
      2. 只把参考做了 `Rz(ψ)p + δ`，却把机器人留在**原局部位姿** —— 于是两者的
         "相对锚点位姿"本就不同，对齐后当然不归零。

    第 2 次是靠 `tests/diag_align.py` 打印中间量定位的：手算契约值
    `Rz(-ψ)p_ref + (base_pos - anchor_pos)` 与实现输出**逐位相同**，
    证明实现无误、错在构造。**教训：连续两次手算猜错根因时，停止猜、直接测量。**
    """
    _, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    psi = 0.6
    delta = np.array([1.5, -0.7, 0.0])
    c, sn = math.cos(psi), math.sin(psi)
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])

    def rot_q(q, ang):
        """绕 z 旋转 ang 后与 q 复合（左乘 Rz 的四元数）。"""
        hq = np.array([math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)])
        w1, x1, y1, z1 = hq
        w2, x2, y2, z2 = np.asarray(q, dtype=np.float64)
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    # 世界版参考：把局部参考整体做刚体变换
    r_world = ReferenceFrame(
        joint_pos=r.joint_pos, joint_vel=r.joint_vel,
        anchor_pos=Rz @ r.anchor_pos + delta,
        anchor_quat=rot_q(r.anchor_quat, psi),
        link_pos={k: Rz @ v + delta for k, v in r.link_pos.items()},
        link_quat={k: rot_q(v, psi) for k, v in r.link_quat.items()},
        link_lin_vel={k: Rz @ v for k, v in r.link_lin_vel.items()},
        link_ang_vel={k: Rz @ v for k, v in r.link_ang_vel.items()},
    )
    assert np.allclose(r_world.anchor_pos, r_world.link_pos["pelvis"]), \
        "构造不自洽：锚点应与 pelvis 重合"

    # ★ 机器人 = 世界版参考的同一物理配置（所有量直接取自它）
    s = RobotState(
        base_pos=r_world.anchor_pos.copy(),
        base_quat=r_world.anchor_quat.copy(),
        base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), gravity_z=-1.0,
        joint_pos=np.asarray(r_world.joint_pos).copy(),
        joint_vel=np.asarray(r_world.joint_vel).copy(),
        joint_acc=np.zeros_like(np.asarray(r_world.joint_pos)),
        link_pos={k: v.copy() for k, v in r_world.link_pos.items()},
        link_quat={k: v.copy() for k, v in r_world.link_quat.items()},
        link_lin_vel={k: v.copy() for k, v in r_world.link_lin_vel.items()},
        link_ang_vel={k: v.copy() for k, v in r_world.link_ang_vel.items()},
    )

    r_aligned = align_reference(s, r_world, yaw_only=True)
    res = compute_residuals(s, r_aligned, ("pelvis", "torso_link"))
    for key, val in res.items():
        assert val == pytest.approx(0.0, abs=1e-9), (
            f"{key} 未对齐：{val}；"
            f"aligned_pelvis={r_aligned.link_pos['pelvis']} "
            f"robot_pelvis={s.link_pos['pelvis']}"
        )


def test_align_reference_is_exactly_a_change_of_frame():
    """**对齐就是一次换系**：`p̃ = R(−Δyaw)·(p − anchor) + base`。

    可验证的不变量：**换系是刚体变换，故每个 link 到锚点的距离必须保持**：

        ‖ref.link[b] − ref.anchor‖  ==  ‖aligned.link[b] − aligned.anchor‖

    （两者都在"相对锚点"的向量空间里，只是朝向差一个 Δyaw —— 模长相同。）

    注意**不能**拿 `ref.link − ref.anchor` 与 `aligned.link − aligned.anchor`
    逐分量比较：前者在参考系、后者已被转到机器人朝向，方向本就不同。
    我前两版正是这么写错的。
    """
    _, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    s = RobotState(
        base_pos=np.array([1.0, 2.0, 0.9]), base_quat=_quat_z(-0.4),
        base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), gravity_z=-1.0,
        joint_pos=np.asarray(r.joint_pos).copy(),
        joint_vel=np.asarray(r.joint_vel).copy(),
        joint_acc=np.zeros_like(np.asarray(r.joint_pos)),
    )
    r_aligned = align_reference(s, r, yaw_only=True)
    assert np.allclose(r_aligned.anchor_pos, s.base_pos), "锚点应搬到机器人基座"

    for body in r.link_pos:
        ref_rel = np.asarray(r.link_pos[body]) - np.asarray(r.anchor_pos)
        ali_rel = np.asarray(r_aligned.link_pos[body]) - r_aligned.anchor_pos
        assert np.linalg.norm(ali_rel) == pytest.approx(
            np.linalg.norm(ref_rel), abs=1e-9
        ), f"{body} 相对锚点的距离被改变了"


def test_align_reference_equals_manual_frame_change():
    """与**手算的换系公式**逐点对照（独立实现，只共享公式本身）。"""
    _, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    s = RobotState(
        base_pos=np.array([0.7, -1.3, 1.1]), base_quat=_quat_z(0.9),
        base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), gravity_z=-1.0,
        joint_pos=np.asarray(r.joint_pos).copy(),
        joint_vel=np.asarray(r.joint_vel).copy(),
        joint_acc=np.zeros_like(np.asarray(r.joint_pos)),
    )
    r_aligned = align_reference(s, r, yaw_only=True)

    # 手算：dyaw = yaw(robot) − yaw(ref)
    from pgmt.rewards.tracking import _yaw_of

    dyaw = _yaw_of(s.base_quat) - _yaw_of(r.anchor_quat)
    c, sn = math.cos(dyaw), math.sin(dyaw)
    Rz_inv = np.array([[c, sn, 0.0], [-sn, c, 0.0], [0.0, 0.0, 1.0]])
    for body in r.link_pos:
        manual = Rz_inv @ (np.asarray(r.link_pos[body]) - np.asarray(r.anchor_pos)) \
            + np.asarray(s.base_pos)
        assert np.allclose(r_aligned.link_pos[body], manual, atol=1e-9), (
            f"{body}：实现={r_aligned.link_pos[body]} 手算={manual}"
        )


def test_align_reference_zero_residual_when_headings_match():
    """朝向相同时，对齐只是平移 ⇒ 机器人取参考的同一配置即零残差。

    这是 root-centric 的最简形式（无 yaw 差），也是 Stage 1 平地上最常见的
    情形（参考与机器人都在同一初始朝向下）。
    """
    _, r = _state_and_ref(bodies=("pelvis", "torso_link"))
    delta = np.array([2.0, -1.0, 0.0])
    r_world = ReferenceFrame(
        joint_pos=r.joint_pos, joint_vel=r.joint_vel,
        anchor_pos=r.anchor_pos + delta, anchor_quat=r.anchor_quat,
        link_pos={k: v + delta for k, v in r.link_pos.items()},
        link_quat=dict(r.link_quat),
        link_lin_vel=dict(r.link_lin_vel),
        link_ang_vel=dict(r.link_ang_vel),
    )
    s = RobotState(
        base_pos=r_world.anchor_pos.copy(), base_quat=r_world.anchor_quat.copy(),
        base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), gravity_z=-1.0,
        joint_pos=np.asarray(r_world.joint_pos).copy(),
        joint_vel=np.asarray(r_world.joint_vel).copy(),
        joint_acc=np.zeros_like(np.asarray(r_world.joint_pos)),
        link_pos={k: v.copy() for k, v in r_world.link_pos.items()},
        link_quat={k: v.copy() for k, v in r_world.link_quat.items()},
        link_lin_vel={k: v.copy() for k, v in r_world.link_lin_vel.items()},
        link_ang_vel={k: v.copy() for k, v in r_world.link_ang_vel.items()},
    )
    r_aligned = align_reference(s, r_world, yaw_only=True)
    res = compute_residuals(s, r_aligned, ("pelvis", "torso_link"))
    for key, val in res.items():
        assert val == pytest.approx(0.0, abs=1e-9), f"{key} 未对齐：{val}"


def test_align_reference_preserves_joint_reference():
    """对齐只动位置/朝向，**不得改关节参考**（关节量本就与锚点无关）。"""
    s, r = _state_and_ref()
    r2 = align_reference(s, r)
    assert np.array_equal(r2.joint_pos, r.joint_pos)
    assert np.array_equal(r2.joint_vel, r.joint_vel)


def test_align_reference_moves_anchor_to_robot_base():
    s, r = _state_and_ref()
    r2 = align_reference(s, r)
    assert np.allclose(r2.anchor_pos, s.base_pos)


def test_align_reference_yaw_only_keeps_roll_pitch_of_robot():
    """yaw_only=True 时**不**把机器人的俯仰/横滚搬到参考上。

    构造：机器人绕 x 倾斜 30°，参考水平。对齐后参考仍应水平
    （只有偏航被对齐），因此 link_ori 残差应约为 30° 而不是 0。
    """
    s, r = _state_and_ref(bodies=("pelvis",))
    tilt = math.radians(30.0)
    q_tilt = np.array([math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0])
    s = RobotState(**{**s.__dict__, "base_quat": q_tilt, "link_quat": {"pelvis": q_tilt}})
    r = ReferenceFrame(joint_pos=r.joint_pos, joint_vel=r.joint_vel,
                       anchor_pos=r.anchor_pos, anchor_quat=_quat_z(0.0),
                       link_quat={"pelvis": _quat_z(0.0)})
    r2 = align_reference(s, r, yaw_only=True)
    got = link_ori_residual(s, r2, ("pelvis",))
    assert got == pytest.approx(tilt, abs=1e-6), (
        f"yaw-only 对齐后仍应保留俯仰差 {math.degrees(tilt):.0f}°，得到 {math.degrees(got):.1f}°"
    )


def test_ref_anchor_from_links_requires_pelvis():
    _, r = _state_and_ref()
    pos, quat = ref_anchor_from_links(r)
    assert np.allclose(pos, r.link_pos["pelvis"])
    bad = ReferenceFrame(joint_pos=r.joint_pos, joint_vel=r.joint_vel,
                         anchor_pos=r.anchor_pos, anchor_quat=r.anchor_quat,
                         link_pos={"torso_link": np.zeros(3)})
    with pytest.raises(KeyError, match="pelvis"):
        ref_anchor_from_links(bad)


# ---------------------------------------------------------------------------
# 分组奖励（Eq.5）
# ---------------------------------------------------------------------------

def test_group_reward_is_one_when_perfect():
    """零残差 ⇒ 每项 exp(0) = 1 ⇒ 组奖励 = 该组权重之和。"""
    from pgmt.rewards.spec import LOWER, UPPER

    zero = {k: 0.0 for k in ("link_pos", "link_ori", "link_lin_vel",
                             "link_ang_vel", "joint_pos", "joint_vel")}
    assert group_reward(zero, "upper") == pytest.approx(
        sum(w for _, w in UPPER.terms))
    assert group_reward(zero, "lower") == pytest.approx(
        sum(w for _, w in LOWER.terms))


def test_group_reward_decreases_with_residual():
    """残差增大 ⇒ 奖励单调下降（高斯核的直接后果）。"""
    prev = None
    for e in np.linspace(0.0, 1.0, 11):
        res = {k: float(e) for k in ("link_pos", "link_ori", "link_lin_vel",
                                     "link_ang_vel", "joint_pos", "joint_vel")}
        cur = group_reward(res, "upper")
        if prev is not None:
            assert cur <= prev + 1e-12
        prev = cur


def test_group_term_values_maps_to_table_i_term_names():
    """**直接验证项名映射**：lower 组的输出键必须含三个 `ta_*` 项。

    这比"不抛 KeyError 就说明对"有判别力 —— 若映射漏了前缀，
    `RewardGroup.sum` 才会报错，而这里能提前指出**缺了哪个键**。
    """
    from pgmt.rewards.spec import LOWER, UPPER

    zero = {k: 0.0 for k in ("link_pos", "link_ori", "link_lin_vel",
                             "link_ang_vel", "joint_pos", "joint_vel")}
    up = group_term_values(zero, "upper")
    lo = group_term_values(zero, "lower")
    assert set(up) == {n for n, _ in UPPER.terms}
    assert set(lo) == {n for n, _ in LOWER.terms}
    assert {"ta_link_pos", "ta_link_ori", "ta_joint_pos"} <= set(lo)
    assert not any(k.startswith("ta_") for k in up), "upper 不应有 TA 项"


def test_zero_residual_gives_exactly_one_per_term():
    """零残差 ⇒ 每项恰为 1（高斯核在 0 处为 1）—— 逐项验证，不是只看总和。"""
    zero = {k: 0.0 for k in ("link_pos", "link_ori", "link_lin_vel",
                             "link_ang_vel", "joint_pos", "joint_vel")}
    for grp_name in ("upper", "lower"):
        vals = group_term_values(zero, grp_name)
        for term, v in vals.items():
            assert v == pytest.approx(1.0), f"{grp_name}.{term} 应为 1，得到 {v}"


def test_upper_and_lower_happen_to_have_equal_weight_sums():
    """**记录一个反直觉事实**：两组权重之和都是 4.5。

    我第一版用 `group_reward(zero,"lower") != group_reward(zero,"upper")`
    来验证映射，失败后才算出两个和相等：
      upper 1.0+1.0+0.5+0.5+1.0+0.5 = 4.5
      lower 0.5+2.0+0.5+0.5+0.5+0.5 = 4.5
    因此**不能**用"两组不等"来验证映射。这里把这个事实固定下来，
    免得下次又有人（包括我）据此写出错误的断言。
    """
    from pgmt.rewards.spec import LOWER, UPPER

    zero = {k: 0.0 for k in ("link_pos", "link_ori", "link_lin_vel",
                             "link_ang_vel", "joint_pos", "joint_vel")}
    su = sum(w for _, w in UPPER.terms)
    sl = sum(w for _, w in LOWER.terms)
    assert su == pytest.approx(4.5)
    assert sl == pytest.approx(4.5)
    assert group_reward(zero, "upper") == pytest.approx(group_reward(zero, "lower"))


def test_group_reward_rejects_unknown_group():
    with pytest.raises(ValueError, match="'upper' 或 'lower'"):
        group_reward({}, "middle")


def test_group_reward_requires_all_residuals():
    with pytest.raises(KeyError, match="需要残差"):
        group_reward({"link_pos": 0.0}, "upper")


def test_group_reward_relaxation_only_affects_ta_terms():
    """给地形松弛后，**只有** TA 三项的奖励上升，速度项不变。

    这是 Eq.10 的直接推论（松弛只作用于 lower-body 的 position/orientation/joint
    position），也是 Table I 里 TA 标记的全部含义。
    """
    e = 0.5
    res = {k: e for k in ("link_pos", "link_ori", "link_lin_vel",
                          "link_ang_vel", "joint_pos", "joint_vel")}
    plain = group_reward(res, "lower")
    relaxed = group_reward(res, "lower", terrain_family="stairs", level=9,
                           alpha=1.0, tau_m=0.3, tau_rad=0.3)
    assert relaxed > plain, "松弛后奖励应上升"

    # 逐项核对：TA 三项用松弛后的残差，速度三项用原残差
    from pgmt.rewards.spec import LOWER
    from pgmt.rewards.spec import relaxed_error, chi
    expected = 0.0
    tau_of = {"ta_link_pos": 0.3, "ta_link_ori": 0.3, "ta_joint_pos": 0.3}
    for term, w in LOWER.terms:
        if term in tau_of:
            e_rel = relaxed_error(e, 1.0, chi("stairs"), tau_of[term])
        else:
            e_rel = e
        expected += w * exp_tracking_reward(e_rel, SIGMAS[term])
    assert relaxed == pytest.approx(expected)


def test_group_reward_flat_equals_strict():
    """flat 上 χ=0 → 与不松弛完全一致（即使传了 tau）。"""
    e = 0.4
    res = {k: e for k in ("link_pos", "link_ori", "link_lin_vel",
                          "link_ang_vel", "joint_pos", "joint_vel")}
    assert group_reward(res, "lower", terrain_family="flat", level=9,
                        tau_m=0.3, tau_rad=0.3) == pytest.approx(
        group_reward(res, "lower"))


def test_ta_terms_match_spec_declaration():
    """本模块的 TA 项必须与 `spec.RELAXED_TERMS` 一致（不重复定义）。"""
    from pgmt.rewards.spec import RELAXED_TERMS

    assert set(TA_TERMS) == set(RELAXED_TERMS)


# ---------------------------------------------------------------------------
# 分组残差
# ---------------------------------------------------------------------------

def test_residuals_by_partition_uses_joint_names_not_positions():
    """关节索引必须由**名字**查得，不能假设数组顺序。"""
    from data.retarget_lafan1 import G1_JOINT_NAMES

    parts = default_partitions()
    nq = len(G1_JOINT_NAMES)
    s, r = _state_and_ref(nq=nq)
    up, lo = residuals_by_partition(s, r, parts, G1_JOINT_NAMES)
    assert set(up) == set(lo) == {"link_pos", "link_ori", "link_lin_vel",
                                  "link_ang_vel", "joint_pos", "joint_vel"}
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in up.values())
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in lo.values())


def test_residuals_by_partition_detects_missing_joint_names():
    parts = default_partitions()
    s, r = _state_and_ref(nq=29)
    with pytest.raises(KeyError, match="joint_names 缺少"):
        residuals_by_partition(s, r, parts, ["not_a_joint"] * 29)


def test_residuals_by_partition_only_wrong_joint_shows_up():
    """只把一个 lower 关节偏掉 ⇒ lower 的 joint_pos 残差非 0、upper 的为 0。"""
    from data.retarget_lafan1 import G1_JOINT_NAMES

    parts = default_partitions()
    nq = len(G1_JOINT_NAMES)
    s, r = _state_and_ref(nq=nq)
    idx = G1_JOINT_NAMES.index("left_knee")
    jp = np.asarray(s.joint_pos).copy()
    jp[idx] += 0.2
    s = RobotState(**{**s.__dict__, "joint_pos": jp})
    up, lo = residuals_by_partition(s, r, parts, G1_JOINT_NAMES)
    n_lo = len(parts.lower_joints)
    assert lo["joint_pos"] == pytest.approx(0.2 / n_lo), "lower 应只有 1/n 的均值"
    assert up["joint_pos"] == pytest.approx(0.0), "upper 不应受影响"
