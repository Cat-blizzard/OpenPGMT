"""重定向结果可视化检查（M1 验收：无穿地/关节跳变）。

用法: python -m data.viz_retarget <npz_or_bvh> --frames 100 200 300 ...
生成 PNG（源 Mixamo 与 G1 并排）到 data/processed/viz/。
"""

from __future__ import annotations

import os
import sys

import numpy as np


def plot_frames(bvh_path: str, out_png: str, frames, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.bvh import load_bvh
    from data.retarget_lafan1 import G1_JOINT_NAMES, g1_forward_kinematics, retarget

    bvh = load_bvh(bvh_path)
    data = retarget(bvh)
    T = data["qpos"].shape[0]
    frames = [min(f, T - 1) for f in frames]

    # 源 FK（米，世界系）
    gpos_src, _ = bvh.fk(unit_scale=float(data["scale"]))

    # G1 FK
    q = data["qpos"]
    g1_pos = g1_forward_kinematics(q, data["root_pos"], data["root_rot"])

    def skeleton_lines(pos: dict, joints):
        lines = []
        for a, b in joints:
            if a in pos and b in pos:
                lines.append((pos[a], pos[b]))
        return lines

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
        # 源
        ax = axes[0, col]
        src_pos = {name: gpos_src[f, bvh.joint_index(name)] for name in bvh.names}
        for a, b in src_joints:
            if a in src_pos and b in src_pos:
                ax.plot([src_pos[a][0], src_pos[b][0]],
                        [src_pos[a][1], src_pos[b][1]],
                        [src_pos[a][2], src_pos[b][2]], "b-", lw=2)
        ax.plot([src_pos["Hips"][0]], [src_pos["Hips"][1]], [src_pos["Hips"][2]], "ro", ms=6)
        ax.set_title(f"source f={f}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        # G1
        ax = axes[1, col]
        for a, b in g1_joints:
            ax.plot([g1_pos[a][f, 0], g1_pos[b][f, 0]],
                    [g1_pos[a][f, 1], g1_pos[b][f, 1]],
                    [g1_pos[a][f, 2], g1_pos[b][f, 2]], "g-", lw=2)
        ax.plot([g1_pos["pelvis"][f, 0]], [g1_pos["pelvis"][f, 1]], [g1_pos["pelvis"][f, 2]], "ro", ms=6)
        ax.set_title(f"G1 f={f}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
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
