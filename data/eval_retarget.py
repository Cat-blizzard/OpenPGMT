"""重定向质量评估（M1 验收数值版）。

指标（每序列）：
  1. FK 对齐误差：源骨架与 G1 骨架逐帧骨盆平移对齐后，20 个关键点的
     位置误差（cm）——姿势保真的标准代理指标（OmniH2O/PHC 同款思路）
  2. 足底最低高度（m）：穿地/悬空检查
  3. 速度尖峰占比：|q̇| > 30 rad/s 的帧（2π 跳变标志）
  4. 接触一致率：重定向 G1 足速接触标签 vs 源接触标签的重合率
  5. 限位超限占比：超出 G1 关节范围（g1_29dof_rev_1_0.xml）的帧占比

输出：控制台报告表 + data/processed/quality_report.csv

用法: python -m data.eval_retarget
"""

from __future__ import annotations

import os

import numpy as np

from data.bvh import load_bvh
from data.retarget_lafan1 import G1_JOINT_LIMITS, W, g1_forward_kinematics, retarget

SRC_DIR = "data/raw/lafan1"
NPZ_DIR = "data/processed/lafan1_g1"

# 关键点匹配：Mixamo 源关节 ↔ G1 body
KEYPOINTS = [
    ("Hips", "pelvis"),
    ("Spine2", "torso_link"),
    ("LeftShoulder", "left_shoulder_pitch_link"),
    ("LeftArm", "left_elbow_link"),
    ("LeftForeArm", "left_wrist_roll_link"),
    ("LeftHand", "left_wrist_yaw_link"),
    ("RightShoulder", "right_shoulder_pitch_link"),
    ("RightArm", "right_elbow_link"),
    ("RightForeArm", "right_wrist_roll_link"),
    ("RightHand", "right_wrist_yaw_link"),
    ("LeftUpLeg", "left_hip_pitch_link"),
    ("LeftLeg", "left_knee_link"),
    ("LeftFoot", "left_ankle_pitch_link"),
    ("LeftToe", "left_ankle_roll_link"),
    ("RightUpLeg", "right_hip_pitch_link"),
    ("RightLeg", "right_knee_link"),
    ("RightFoot", "right_ankle_pitch_link"),
    ("RightToe", "right_ankle_roll_link"),
]

_LEFT_LIMITS = G1_JOINT_LIMITS


def _motion_type(name: str) -> str:
    for t in ("walk", "run", "sprint", "dance", "jump", "aim", "fight", "ground",
              "fall", "push", "obstacle", "multiple"):
        if name.startswith(t):
            return t
    return "other"


def evaluate(seq_name: str) -> dict:
    bvh = load_bvh(os.path.join(SRC_DIR, seq_name + ".bvh"))
    data_raw = retarget(bvh)
    from data.ik_refine import refine_full
    data = refine_full(data_raw, bvh)
    T = bvh.num_frames
    scale = float(data["scale"])

    # 源骨架（G1 系，米）
    gpos_cm, _ = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale  # (T,J,3)
    # G1 FK
    g1 = g1_forward_kinematics(data["qpos"], data["root_pos"], data["root_rot"])

    # 1. FK 对齐误差：逐帧骨盆平移对齐
    pelvis_delta = gpos[:, bvh.joint_index("Hips")] - g1["pelvis"]
    kp_errs = {}
    for src_name, g1_body in KEYPOINTS:
        d = gpos[:, bvh.joint_index(src_name)] - pelvis_delta - g1[g1_body]
        kp_errs[f"{src_name}->{g1_body}"] = np.linalg.norm(d, axis=-1).mean()
    fk_err_cm = float(np.mean(list(kp_errs.values())) * 100)
    worst_kp = max(kp_errs, key=kp_errs.get)
    worst_kp_cm = float(kp_errs[worst_kp] * 100)

    # 2. 足底最低
    min_z = float(min(g1["left_ankle_roll_link"][:, 2].min(),
                      g1["right_ankle_roll_link"][:, 2].min()))

    # 3. 速度尖峰
    spike = float((np.abs(data["qvel"]) > 30).mean())

    # 4. 接触一致率：G1 FK 统一协议标签 vs 源统一协议标签（data_raw
    # contacts = 源足端速度+高度+滤波；精修后重算同协议，非循环度量）
    from data.retarget_lafan1 import contact_labels
    g1_foot = np.stack([g1["left_ankle_roll_link"], g1["right_ankle_roll_link"]], axis=1)
    g1_contact = contact_labels(g1_foot, float(data["frame_time"]), 0.15)
    agree = (g1_contact == data_raw["contacts"]).mean()
    contact_agree = float(agree * 100)

    # 5. 限位超限
    over = 0.0
    for i, name in enumerate(data["joint_names"]):
        lo, hi = _LEFT_LIMITS[str(name)]
        over += ((data["qpos"][:, i] < lo) | (data["qpos"][:, i] > hi)).sum()
    limit_over = float(over / (T * 29) * 100)

    return {
        "seq": seq_name, "type": _motion_type(seq_name), "frames": T,
        "fk_err_cm": fk_err_cm, "worst_kp": worst_kp, "worst_kp_cm": worst_kp_cm,
        "min_foot_z": min_z, "spike_pct": spike * 100,
        "contact_agree": contact_agree, "limit_over_pct": limit_over,
    }


def main():
    import csv
    names = sorted(f[:-4] for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    rows = []
    for n in names:
        try:
            rows.append(evaluate(n))
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {n}: {e}")

    rows.sort(key=lambda r: r["fk_err_cm"])
    print(f"{'序列':26s} {'类型':8s} {'FK误差cm':>8s} {'最差关键点':>22s} "
          f"{'足底m':>6s} {'尖峰%':>6s} {'接触%':>6s} {'限位%':>6s}  评级")
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    # 阈值按 v1 现实预期（FK 误差含骨骼比例差异的系统分量）：
    # A ≤18cm / B ≤25cm / C ≤35cm；足底 >−5cm；尖峰 <1%；接触 >60%
    for r in rows:
        if r["fk_err_cm"] <= 18 and r["min_foot_z"] > -0.05 and r["spike_pct"] < 0.5 \
                and r["contact_agree"] > 60:
            grade = "A"
        elif r["fk_err_cm"] <= 25 and r["min_foot_z"] > -0.05 and r["spike_pct"] < 1 \
                and r["contact_agree"] > 55:
            grade = "B"
        elif r["fk_err_cm"] <= 35 and r["min_foot_z"] > -0.1 and r["spike_pct"] < 5:
            grade = "C"
        else:
            grade = "D"
        grades[grade] += 1
        print(f"{r['seq']:26s} {r['type']:8s} {r['fk_err_cm']:8.2f} "
              f"{r['worst_kp']:>22s} {r['min_foot_z']:6.3f} {r['spike_pct']:6.2f} "
              f"{r['contact_agree']:6.1f} {r['limit_over_pct']:6.2f}  {grade}")

    # 汇总
    agg = {}
    for r in rows:
        agg.setdefault(r["type"], []).append(r)
    print("\n== 按类型汇总（FK 误差 cm 均值 / 尖峰% / 接触%）==")
    for t in sorted(agg):
        rs = agg[t]
        print(f"{t:10s} n={len(rs):2d}  FK={np.mean([r['fk_err_cm'] for r in rs]):5.2f} "
              f"尖峰={np.mean([r['spike_pct'] for r in rs]):5.2f} "
              f"接触={np.mean([r['contact_agree'] for r in rs]):5.1f} "
              f"限位={np.mean([r['limit_over_pct'] for r in rs]):5.2f}")
    print(f"\n评级分布: A={grades['A']} B={grades['B']} C={grades['C']} D={grades['D']}")
    print(f"总体 FK 误差: {np.mean([r['fk_err_cm'] for r in rows]):.2f} cm")

    out_csv = "data/processed/quality_report.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {out_csv}")


if __name__ == "__main__":
    main()
