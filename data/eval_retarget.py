"""重定向质量评估（M1 验收数值版）。

指标（每序列）：
  1. **拟合关键点误差**（`fitted_err_cm`）：源骨架与 G1 骨架逐帧骨盆平移对齐后，
     13 个"被 IK 优化过"的关键点位置误差（cm）。
     ⚠️ 这是 IK 精修的**训练残差**，不是独立度量 —— 只能用于回归监控。
  2. **留出关键点误差**（`holdout_err_cm`）：4 个 **`ik_refine` 从未优化**的
     关键点（左/右髋、左/右趾）的同类误差（cm）。这是独立性的主要来源。
  3. **绝对位置误差**（`abs_pos_err_cm`）与**根朝向误差**（`root_rot_err_deg`）：
     不做骨盆平移对齐、直接比较世界系位置/朝向。骨盆对齐会掩盖全局漂移与
     朝向错误，这两项补上该盲点。
  4. 足底最低高度（m）：穿地/悬空检查
  5. 速度尖峰占比：|q̇| > 30 rad/s 的帧（2π 跳变标志）
  6. 接触一致率：重定向 G1 足速接触标签 vs 源接触标签的重合率
  7. 限位超限占比：超出 G1 关节范围（g1_29dof_rev_1_0.xml）的帧占比

输出：控制台报告表 + data/processed/quality_report.csv

用法:
  python -m data.eval_retarget                  # 全部序列
  python -m data.eval_retarget --only ground    # 只看 ground 类（A17 复核用）

## A17 复核口径（为什么需要本模块的改造）

`ik_refine.refine_full` 的优化目标 `FULL_KEYPOINTS` 有 13 个关键点，而旧版
`KEYPOINTS` 有 18 个且**这 13 个全部包含在内** —— 旧 `fk_err_cm` 实质上
是"在训练集上测训练损失"；再加上逐帧骨盆平移对齐对全局漂移与根朝向完全不
敏感，因此它**不能**用来判断 ground 类参考是否可用（而 A17 原始裁定的依据
正是这个自证指标）。现在 `holdout_err_cm` + `abs_pos_err_cm` +
`root_rot_err_deg` 组成独立证据，用于重裁 A17。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from data.bvh import load_bvh
from data.retarget_lafan1 import G1_JOINT_LIMITS, W, g1_forward_kinematics, retarget

SRC_DIR = "data/raw/lafan1"
NPZ_DIR = "data/processed/lafan1_g1"

# 关键点匹配：Mixamo 源关节 ↔ G1 body，附类别标签。
#   fused   = 被 ik_refine.refine_full 优化过（FULL_KEYPOINTS）→ 训练残差
#   holdout = **从未**被优化 → 独立证据
#   ref     = 仅作对齐基准（Hips/pelvis），不参与任何误差均值
KEYPOINTS: List[Tuple[str, str, str]] = [
    ("Hips", "pelvis", "ref"),
    ("Spine2", "torso_link", "fused"),
    ("LeftShoulder", "left_shoulder_pitch_link", "fused"),
    ("LeftArm", "left_elbow_link", "fused"),
    ("LeftForeArm", "left_wrist_roll_link", "fused"),
    ("LeftHand", "left_wrist_yaw_link", "fused"),
    ("RightShoulder", "right_shoulder_pitch_link", "fused"),
    ("RightArm", "right_elbow_link", "fused"),
    ("RightForeArm", "right_wrist_roll_link", "fused"),
    ("RightHand", "right_wrist_yaw_link", "fused"),
    ("LeftUpLeg", "left_hip_pitch_link", "holdout"),
    ("LeftLeg", "left_knee_link", "fused"),
    ("LeftFoot", "left_ankle_pitch_link", "fused"),
    ("LeftToe", "left_ankle_roll_link", "holdout"),
    ("RightUpLeg", "right_hip_pitch_link", "holdout"),
    ("RightLeg", "right_knee_link", "fused"),
    ("RightFoot", "right_ankle_pitch_link", "fused"),
    ("RightToe", "right_ankle_roll_link", "holdout"),
]

FITTED_KPS = [k for k in KEYPOINTS if k[2] == "fused"]
HOLDOUT_KPS = [k for k in KEYPOINTS if k[2] == "holdout"]

MOTION_TYPES = ("walk", "run", "sprint", "dance", "jump", "aim", "fight", "ground",
                "fall", "push", "obstacle", "multiple")


def _motion_type(name: str) -> str:
    for t in MOTION_TYPES:
        if name.startswith(t):
            return t
    return "other"


def _mean_err_cm(kp_errs: Dict[str, np.ndarray],
                 kps: Sequence[Tuple[str, str, str]]) -> float:
    """给定关键点子集的平均位置误差（cm）。空集返回 nan。"""
    vals = [kp_errs[f"{s}->{g}"] for s, g, _ in kps if f"{s}->{g}" in kp_errs]
    if not vals:
        return float("nan")
    return float(np.mean([v.mean() for v in vals]) * 100)


def evaluate(seq_name: str) -> dict:
    bvh = load_bvh(os.path.join(SRC_DIR, seq_name + ".bvh"))
    data_raw = retarget(bvh)
    from data.ik_refine import refine_full
    data = refine_full(data_raw, bvh)
    T = bvh.num_frames
    scale = float(data["scale"])

    # 源骨架（G1 系，米）
    gpos_cm, grot_src = bvh.fk(unit_scale=1.0)
    gpos = gpos_cm @ W.T * scale  # (T,J,3)
    # G1 FK（含朝向，供根朝向误差用）
    g1, g1_quat = g1_forward_kinematics(data["qpos"], data["root_pos"],
                                        data["root_rot"], return_quats=True)

    # ---- 1/2. 骨盆对齐后的关键点误差（分组统计）----
    pelvis_delta = gpos[:, bvh.joint_index("Hips")] - g1["pelvis"]
    kp_errs: Dict[str, np.ndarray] = {}
    for src_name, g1_body, _ in KEYPOINTS:
        if src_name == "Hips":
            continue  # 对齐基准，误差恒为 0，不参与均值
        d = gpos[:, bvh.joint_index(src_name)] - pelvis_delta - g1[g1_body]
        kp_errs[f"{src_name}->{g1_body}"] = np.linalg.norm(d, axis=-1)

    fitted_err = _mean_err_cm(kp_errs, FITTED_KPS)
    holdout_err = _mean_err_cm(kp_errs, HOLDOUT_KPS)
    # 兼容旧口径：18 个关键点（含 Hips 的 0 误差）的总均值
    fk_err_cm = float(np.mean([v.mean() for v in kp_errs.values()]) * 100)
    worst_kp = max(kp_errs, key=lambda k: kp_errs[k].mean())
    worst_kp_cm = float(kp_errs[worst_kp].mean() * 100)
    worst_holdout = max((f"{s}->{g}" for s, g, _ in HOLDOUT_KPS),
                        key=lambda k: kp_errs[k].mean())
    worst_holdout_cm = float(kp_errs[worst_holdout].mean() * 100)

    # ---- 3. 绝对位置误差（不做骨盆对齐）----
    # 把 G1 放到源 Hips 的世界位置（零朝向），只衡量"各 body 在空间中是否
    # 落在源关节处"。骨盆平移对齐会把这一项整体吸收掉，因此必须单独报告。
    g1_abs = g1_forward_kinematics(
        data["qpos"],
        np.tile(gpos[:, bvh.joint_index("Hips")], (1, 1)),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1)),
    )
    abs_errs = []
    for src_name, g1_body, kind in KEYPOINTS:
        if kind == "ref":
            continue
        d = gpos[:, bvh.joint_index(src_name)] - g1_abs[g1_body]
        abs_errs.append(np.linalg.norm(d, axis=-1).mean())
    abs_pos_err_cm = float(np.mean(abs_errs) * 100)

    # ---- 3b. 根朝向误差 ----
    # 独立于 IK：IK 只优化关节角，root_pos/root_rot 由 retarget 冻结后再不改动，
    # 因此根朝向误差完全由旋转映射（而非 IK）决定 —— 用于判断地面类姿态分解
    # 是否真的退化。换算走 `source_root_quat_to_g1_base`（唯一出处，带单测），
    # 避免在这里重复推导乘法顺序。
    from data.retarget_lafan1 import relative_rotation_angle_deg, source_root_quat_to_g1_base
    src_q = source_root_quat_to_g1_base(grot_src[:, bvh.joint_index("Hips")])
    root_rot_err_deg = float(relative_rotation_angle_deg(
        g1_quat["pelvis"], src_q).mean())

    # ---- 4. 足底最低 ----
    min_z = float(min(g1["left_ankle_roll_link"][:, 2].min(),
                      g1["right_ankle_roll_link"][:, 2].min()))

    # ---- 5. 速度尖峰 ----
    spike = float((np.abs(data["qvel"]) > 30).mean())

    # ---- 6. 接触一致率（G1 FK 统一协议标签 vs 源统一协议标签）----
    from data.retarget_lafan1 import contact_labels
    g1_foot = np.stack([g1["left_ankle_roll_link"], g1["right_ankle_roll_link"]], axis=1)
    g1_contact = contact_labels(g1_foot, float(data["frame_time"]), 0.15)
    contact_agree = float((g1_contact == data_raw["contacts"]).mean() * 100)

    # ---- 7. 限位超限 ----
    over = 0.0
    for i, name in enumerate(data["joint_names"]):
        lo, hi = G1_JOINT_LIMITS[str(name)]
        over += ((data["qpos"][:, i] < lo) | (data["qpos"][:, i] > hi)).sum()
    limit_over = float(over / (T * 29) * 100)

    return {
        "seq": seq_name, "type": _motion_type(seq_name), "frames": T,
        "fk_err_cm": fk_err_cm,
        "fitted_err_cm": fitted_err,
        "holdout_err_cm": holdout_err,
        "abs_pos_err_cm": abs_pos_err_cm,
        "root_rot_err_deg": root_rot_err_deg,
        "worst_kp": worst_kp, "worst_kp_cm": worst_kp_cm,
        "worst_holdout_kp": worst_holdout, "worst_holdout_cm": worst_holdout_cm,
        "min_foot_z": min_z, "spike_pct": spike * 100,
        "contact_agree": contact_agree, "limit_over_pct": limit_over,
    }


def _pf(x: float, width: int = 7) -> str:
    return f"{'nan':>{width}}" if not np.isfinite(x) else f"{x:>{width}.2f}"


def main():
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="重定向质量评估")
    ap.add_argument("--only", default=None,
                    help="只评估名字以该前缀开头的序列（如 ground）")
    ap.add_argument("--out", default="data/processed/quality_report.csv")
    args = ap.parse_args()

    names = sorted(f[:-4] for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
        if not names:
            print(f"[!] 没有匹配 --only {args.only} 的序列")
            return

    rows = []
    for n in names:
        try:
            rows.append(evaluate(n))
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {n}: {e}")
    if not rows:
        print("[!] 没有成功评估的序列")
        return

    rows.sort(key=lambda r: (r["holdout_err_cm"] if np.isfinite(r["holdout_err_cm"])
                             else r["fk_err_cm"]))

    hdr = (f"{'序列':26s} {'类型':9s} {'拟合cm':>7s} {'留出cm':>7s} {'绝对cm':>7s} "
           f"{'根朝向°':>8s} {'足底m':>7s} {'尖峰%':>6s} {'接触%':>6s} 最差留出点")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['seq']:26s} {r['type']:9s} {_pf(r['fitted_err_cm'])} "
              f"{_pf(r['holdout_err_cm'])} {_pf(r['abs_pos_err_cm'])} "
              f"{_pf(r['root_rot_err_deg'], 8)} {r['min_foot_z']:7.3f} "
              f"{r['spike_pct']:6.2f} {r['contact_agree']:6.1f} {r['worst_holdout_kp']}")

    agg: Dict[str, List[dict]] = {}
    for r in rows:
        agg.setdefault(r["type"], []).append(r)

    def _m(rs, key):
        vals = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n== 按类型汇总（均值）==")
    print(f"{'类型':10s} {'n':>3s} {'拟合cm':>7s} {'留出cm':>7s} {'绝对cm':>7s} "
          f"{'根朝向°':>8s} {'尖峰%':>6s} {'接触%':>6s}")
    for t in sorted(agg):
        rs = agg[t]
        print(f"{t:10s} {len(rs):3d} {_pf(_m(rs, 'fitted_err_cm'))} "
              f"{_pf(_m(rs, 'holdout_err_cm'))} {_pf(_m(rs, 'abs_pos_err_cm'))} "
              f"{_pf(_m(rs, 'root_rot_err_deg'), 8)} {_m(rs, 'spike_pct'):6.2f} "
              f"{_m(rs, 'contact_agree'):6.1f}")

    print(f"\n总体（n={len(rows)}）: 拟合 {_pf(_m(rows, 'fitted_err_cm'))}cm | "
          f"留出 {_pf(_m(rows, 'holdout_err_cm'))}cm | "
          f"绝对 {_pf(_m(rows, 'abs_pos_err_cm'))}cm | "
          f"根朝向 {_pf(_m(rows, 'root_rot_err_deg'))}°")
    print("注意：'拟合' 是 IK 训练残差（非独立，仅回归监控）；"
          "'留出' 的 4 个关键点未被 IK 优化，才是参考可用性的独立证据。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {args.out}")


if __name__ == "__main__":
    main()
