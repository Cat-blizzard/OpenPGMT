"""重定向结果可视化检查（M1 验收：无穿地/关节跳变）。

用法: python -m data.viz_retarget <npz_or_bvh> --frames 100 200 300 ...
生成 PNG（源 Mixamo 与 G1 并排）到 data/processed/viz/。
"""

from __future__ import annotations

import os
import sys

import numpy as np


def plot_frames(bvh_path: str, out_png: str, frames, title: str = ""):
    """源（蓝）与 G1 精修后（绿）并排，双骨架骨盆对齐到原点。

    关键：源 FK 先经 W 变换转到 G1 世界系（上=+z、米）再画——曾直接
    画 rig 原始坐标（上=+x），骨架横躺且超出轴限制，观感全错。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.bvh import load_bvh
    from data.retarget_lafan1 import W, g1_forward_kinematics, retarget
    from data.ik_refine import refine_full

    bvh = load_bvh(bvh_path)
    data = refine_full(retarget(bvh), bvh)
    T = data["qpos"].shape[0]
    frames = [min(f, T - 1) for f in frames]

    # 源 FK（G1 世界系、米）
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos_src = gpos_cm @ W.T * float(data["scale"])

    # G1 FK
    g1_pos = g1_forward_kinematics(data["qpos"], data["root_pos"], data["root_rot"])

    src_joints = [("Hips", "Spine"), ("Spine", "Spine2"), ("Spine2", "Neck"),
                  ("Neck", "Head"),
                  ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
                  ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
                  ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"),
                  ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
                  ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
                  ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
                  ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
                  ("RightLeg", "RightFoot"), ("RightFoot", "RightToe")]
    g1_joints = [("pelvis", "torso_link"),
                 ("torso_link", "left_shoulder_pitch_link"),
                 ("left_shoulder_pitch_link", "left_elbow_link"),
                 ("left_elbow_link", "left_wrist_yaw_link"),
                 ("torso_link", "right_shoulder_pitch_link"),
                 ("right_shoulder_pitch_link", "right_elbow_link"),
                 ("right_elbow_link", "right_wrist_yaw_link"),
                 ("pelvis", "left_hip_pitch_link"),
                 ("left_hip_pitch_link", "left_knee_link"),
                 ("left_knee_link", "left_ankle_pitch_link"),
                 ("left_ankle_pitch_link", "left_ankle_roll_link"),
                 ("pelvis", "right_hip_pitch_link"),
                 ("right_hip_pitch_link", "right_knee_link"),
                 ("right_knee_link", "right_ankle_pitch_link"),
                 ("right_ankle_pitch_link", "right_ankle_roll_link")]

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9), subplot_kw={"projection": "3d"})
    for col, f in enumerate(frames):
        # 源（骨盆对齐原点）
        ax = axes[0, col]
        src_pos = {name: gpos_src[f, bvh.joint_index(name)] for name in bvh.names}
        root = src_pos["Hips"]
        for a, b in src_joints:
            ax.plot([src_pos[a][0] - root[0], src_pos[b][0] - root[0]],
                    [src_pos[a][1] - root[1], src_pos[b][1] - root[1]],
                    [src_pos[a][2] - root[2], src_pos[b][2] - root[2]], "b-", lw=2)
        ax.plot([0], [0], [0], "ro", ms=6)
        ax.set_title(f"source f={f}")
        # G1（骨盆对齐原点）
        ax = axes[1, col]
        g1_root = g1_pos["pelvis"][f]
        for a, b in g1_joints:
            ax.plot([g1_pos[a][f, 0] - g1_root[0], g1_pos[b][f, 0] - g1_root[0]],
                    [g1_pos[a][f, 1] - g1_root[1], g1_pos[b][f, 1] - g1_root[1]],
                    [g1_pos[a][f, 2] - g1_root[2], g1_pos[b][f, 2] - g1_root[2]], "g-", lw=2)
        ax.plot([0], [0], [0], "ro", ms=6)
        ax.set_title(f"G1 f={f}")
    for ax in axes.flat:
        # 骨盆在原点：足 ≈ −0.8m，源头顶 ≈ +0.65m
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.9, 0.7)
        ax.set_box_aspect((1.2, 1.2, 1.6))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"[viz] {out_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh")
    ap.add_argument("--frames", nargs="+", type=int, default=[100, 500, 1000, 1500])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"data/processed/viz/{os.path.basename(args.bvh)[:-4]}.png"
    plot_frames(args.bvh, out, args.frames, title=os.path.basename(args.bvh))
