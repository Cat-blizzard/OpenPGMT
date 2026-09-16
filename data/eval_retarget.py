"""重定向质量评估（M1 验收数值版）。

指标（每序列）：
  1. **拟合关键点误差**（`fitted_err_cm`）：源骨架与 G1 骨架逐帧骨盆平移对齐后，
     13 个"被 IK 优化过"的关键点位置误差（cm）。
     ⚠️ 这是 IK 精修的**训练残差**，不是独立度量 —— 只能用于回归监控。
  2. **留出关键点误差**（`holdout_err_cm`）：4 个**未进入 IK 优化目标**的关键点
     （左/右髋、左/右趾）的同类误差（cm）。
  3. **单位根朝向对照**（`upright_holdout_err_cm`）：把根朝向强制为单位
     朝向、其余不变。正常转向/躺倒也会增大此值，不能据此认定根朝向错误。
     ⚠️ 另有 `root_rot_err_deg`（直接比较根朝向角）—— **已实测确认它是恒等式、
     不是指标**：`retarget()` 里 `root_rot = (qw ⊗ grot_src[Hips]) ⊗ Q_MRIG_INV`
     与这里调用的 `source_root_quat_to_g1_base(...)` 是**同一个式子**，故按构造
     相等；77 个序列实测恒为 0.01°（`root_rot` 存 float32 的舍入）。它**测不出
     参考质量**，不得用作 A17 的证据。该换算的正确性由
     `tests/test_root_frame.py` 对真实数据的闭环测试保证，不需要在报告里再列一列。
  4. 双踝原点最低高度（m）：高度异常线索，非足底碰撞几何或全身接触判定
  5. 速度尖峰占比：全部帧×关节样本中 |q̇| > 30 rad/s 的比例
  6. 接触一致率：G1 FK 推断 vs 源推断；另报存储标签 vs G1 FK 推断一致率
  7. 限位超限占比：帧×关节样本中超出 G1 范围超过 1e-6 rad 的比例

输出：控制台报告表 + CSV。增量结果写独立 `.partial.csv`；仅全部选中序列
成功后原子发布正式报告，失败/中断保留旧报告并返回非零状态。
**不带 `--only` 才写完整报告**
`data/processed/quality_report.csv`；带 `--only` 写
`data/processed/quality_report_<only>.csv`，完整报告不受影响
（`resolve_out_path`；此前 `--only ground` 会静默把 77 行报告覆盖成 5 行）。

默认读取现存 NPZ；`--regenerate` 显式评估重新计算的候选，默认另存
`quality_report_regenerated[_<only>].csv`，不改写 NPZ。CSV 的
`reference_source` 列区分两种来源。

用法:
  python -m data.eval_retarget                  # 全部序列
  python -m data.eval_retarget --only ground    # 只看 ground 类（A17 复核用）

## A17 复核口径（为什么需要本模块的改造）

`ik_refine.refine_full` 的优化目标 `FULL_KEYPOINTS` 有 13 个关键点，而旧版
`KEYPOINTS` 有 18 个且**这 13 个全部包含在内** —— 旧 `fk_err_cm` 实质上
主要复用了拟合目标；逐帧骨盆平移对齐还会移除全局平移误差（并不会移除
根朝向误差），因此它**不能单独**用来判断 ground 类参考是否可用
（而 A17 原始裁定的依据正是这个自证指标）。

## 独立性的**准确**表述（本轮审计修正，别过度声称）

  - **`root_rot_err_deg`：恒等式，不是证据（2026 复核实测确认）。**
    它在 IK 意义下确实"独立"（`root_rot` 由 `retarget` 冻结，`refine_full`
    只把它当输入、从不写回），但 `retarget()` 的根朝向本就由源髋朝向经同一
    公式算出，故与参考值**按构造相等** —— 77 个序列恒为 0.01°，
    **无法区分任何序列**。不得用作 A17 的证据。
  - `upright_holdout_err_cm` 是根朝向的消融对照，`min_foot_z` 是踝原点高度。
    两者提供诊断线索，不能直接证明根朝向错误或全身悬空。
  - **`holdout_err_cm` / `upright_holdout_err_cm`：未被拟合，但不独立。**
    这 4 个关键点不在 `FULL_KEYPOINTS` 里（没有目标牵引它们），但
    `refine_full` 优化的是 `list(range(29))`——**全部 29 个关节**。改动髋/踝
    关节角会移动部分留出点，它们与拟合点共享运动链。髋原点位置则主要取决于
    根姿态和固定偏移。这是 **held-out 诊断**；高误差也可能来自骨架比例或
    关键点匹配差异，不能全部归因于优化造成扭曲。
  - **`fitted_err_cm`：IK 的训练残差**，只能用于回归监控。

同时注意 `holdout_err_cm` 与 `upright_holdout_err_cm` 的**平移部分完全相同**
（两者都把每帧骨盆对到源 Hips），唯一差别是根朝向用重定向值还是单位值 ——
所以后者是固定单位根朝向的对照误差，既不是真实根朝向误差，
也不是全局位置漂移。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.bvh import load_bvh
from data.retarget_lafan1 import G1_JOINT_LIMITS, W, g1_forward_kinematics, retarget

SRC_DIR = "data/raw/lafan1"
NPZ_DIR = "data/processed/lafan1_g1"

# 关键点匹配：Mixamo 源关节 ↔ G1 body，附类别标签。
#   fused   = 被 ik_refine.refine_full 优化过（FULL_KEYPOINTS）→ 训练残差
#   holdout = 不在位置拟合目标中，但可能受同一运动链优化影响
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


#: 完整报告的固定路径。`--only` **不得**写入此处（见 `resolve_out_path`）。
FULL_REPORT_PATH = "data/processed/quality_report.csv"


def resolve_out_path(only: Optional[str], out: Optional[str],
                     *, regenerate: bool = False) -> str:
    """决定输出 CSV 路径 —— **`--only` 绝不覆盖完整报告**。

    不加这条保护时，`python -m data.eval_retarget --only ground` 会把仓库里
    77 行的 `quality_report.csv` 静默改写成 5 行，而 README 恰好推荐用这条
    命令做 A17 复核 —— 第一次执行就会发生，且没有任何提示。

    Args:
        only: `--only` 的值（前缀）或 None
        out:  `--out` 的值或 None（None ⇒ 按 only 自动选择）

    Raises:
        ValueError: 显式给的 `--out` 指向完整报告路径、同时又有 `--only`
    """
    if out is None:
        if regenerate:
            suffix = f"_{only}" if only else ""
            return f"data/processed/quality_report_regenerated{suffix}.csv"
        if not only:
            return FULL_REPORT_PATH
        return f"data/processed/quality_report_{only}.csv"
    if regenerate and os.path.abspath(out) == os.path.abspath(FULL_REPORT_PATH):
        raise ValueError("再生成候选不能覆盖实际 NPZ 的完整报告，请另给 --out")
    if only and os.path.abspath(out) == os.path.abspath(FULL_REPORT_PATH):
        raise ValueError(
            f"拒绝写入：--only {only} 会把完整报告 {FULL_REPORT_PATH} 覆盖成"
            f"子集。请另给 --out，或省略 --out 让它自动写到 "
            f"quality_report_{only}.csv")
    return out


def legacy_fk_err_cm(kp_errs: Dict[str, np.ndarray]) -> float:
    """旧口径 `fk_err_cm`：**分母是 18（含骨盆）而非 17**。

    骨盆（Hips）是对齐基准、不参与误差统计，故 `kp_errs` 只有 17 条；但旧口径
    把它作为 0 计入了分母。若改用 17，均值会被抬高 18/17 ≈ 5.9%，**新旧 CSV
    将不可比** —— 而保留这一列的唯一意义就是对比改造前后的数值变化。

    注：旧版实现已随改造重写，此处按改造时的记录口径复现。可用
    `git show 5fd33f4:data/eval_retarget.py` 核对（若不符则改本函数）。
    """
    if not kp_errs:
        return float("nan")
    return float(sum(v.mean() for v in kp_errs.values()) / len(KEYPOINTS) * 100)


def load_reference(seq_name: str, bvh, *, regenerate: bool = False,
                   npz_dir: Optional[str] = None) -> dict:
    """Load the actual exported reference; regeneration is an explicit opt-in.

    Reject incompatible artifacts instead of silently reporting fresh motion as
    the quality of the file consumed by training.
    """
    if regenerate:
        from data.ik_refine import refine_full
        return refine_full(retarget(bvh), bvh)
    path = os.path.join(NPZ_DIR if npz_dir is None else npz_dir, seq_name + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    from data.retarget_lafan1 import G1_JOINT_NAMES
    if list(data.get("joint_names", [])) != G1_JOINT_NAMES:
        raise ValueError(f"{path}: joint_names must match the canonical G1 order")
    expected = {"qpos": (bvh.num_frames, 29), "qvel": (bvh.num_frames, 29),
                "root_pos": (bvh.num_frames, 3), "root_rot": (bvh.num_frames, 4),
                "contacts": (bvh.num_frames, 2)}
    for key, shape in expected.items():
        if key not in data or data[key].shape != shape:
            raise ValueError(f"{path}: {key} must have shape {shape}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
    if not np.isin(data["contacts"], (0, 1)).all():
        raise ValueError(f"{path}: contacts must contain only boolean labels")
    for key in ("scale", "frame_time"):
        if key not in data or data[key].shape != () or not np.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f"{path}: {key} must be a positive finite scalar")
    if not np.isclose(float(data["frame_time"]), bvh.frame_time, rtol=1e-5):
        raise ValueError(f"{path}: frame_time does not match the source BVH")
    return data


def evaluate(seq_name: str, *, regenerate: bool = False,
             bvh_dir: Optional[str] = None, npz_dir: Optional[str] = None) -> dict:
    """Evaluate stored NPZ by default, or explicitly regenerate a candidate."""
    bvh = load_bvh(os.path.join(SRC_DIR if bvh_dir is None else bvh_dir,
                                seq_name + ".bvh"))
    data = load_reference(seq_name, bvh, regenerate=regenerate, npz_dir=npz_dir)
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
    fk_err_cm = legacy_fk_err_cm(kp_errs)   # 旧口径，分母含骨盆（见该函数）
    worst_kp = max(kp_errs, key=lambda k: kp_errs[k].mean())
    worst_kp_cm = float(kp_errs[worst_kp].mean() * 100)
    worst_holdout = max((f"{s}->{g}" for s, g, _ in HOLDOUT_KPS),
                        key=lambda k: kp_errs[k].mean())
    worst_holdout_cm = float(kp_errs[worst_holdout].mean() * 100)

    # ---- 3. 根朝向的厘米级影响（**只用留出点**）----
    # 与上面的 `kp_errs` **平移部分完全相同**（两者都把每帧骨盆对到源 Hips），
    # 唯一差别是这里强制单位根朝向；正常转向/躺倒也会增大这一对照误差。
    #
    # ⚠️ 分母用 HOLDOUT_KPS（4 点）而**不是**全部 17 点：后者含 13 个 IK 拟合点，
    # 会被训练残差主导、失去"未拟合"的意义（改造前的 `abs_pos_err_cm` 正是
    # 如此，README 却把它标成独立指标 —— 本轮审计修正）。
    g1_upright = g1_forward_kinematics(
        data["qpos"],
        np.tile(gpos[:, bvh.joint_index("Hips")], (1, 1)),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1)),
    )
    upright_errs: Dict[str, np.ndarray] = {}
    for src_name, g1_body, kind in KEYPOINTS:
        if kind == "ref":
            continue
        d = gpos[:, bvh.joint_index(src_name)] - g1_upright[g1_body]
        upright_errs[f"{src_name}->{g1_body}"] = np.linalg.norm(d, axis=-1)
    upright_holdout_err_cm = _mean_err_cm(upright_errs, HOLDOUT_KPS)

    # ---- 3b. 根朝向误差 ----
    # 检查产物根朝向与约定换算是否一致。对当前 retarget 输出按构造为零；
    # 可发现损坏/不兼容产物，不能证明重定向姿态的物理正确性。
    from data.retarget_lafan1 import relative_rotation_angle_deg, source_root_quat_to_g1_base
    src_q = source_root_quat_to_g1_base(grot_src[:, bvh.joint_index("Hips")])
    root_rot_err_deg = float(relative_rotation_angle_deg(
        g1_quat["pelvis"], src_q).mean())

    # ---- 4. 双踝原点最低（不是足底碰撞几何）----
    min_z = float(min(g1["left_ankle_roll_link"][:, 2].min(),
                      g1["right_ankle_roll_link"][:, 2].min()))

    # ---- 5. 速度尖峰 ----
    spike = float((np.abs(data["qvel"]) > 30).mean())

    # ---- 6. 接触一致率（G1 FK 统一协议标签 vs 源统一协议标签）----
    from data.retarget_lafan1 import contact_labels
    g1_foot = np.stack([g1["left_ankle_roll_link"], g1["right_ankle_roll_link"]], axis=1)
    g1_contact = contact_labels(g1_foot, float(data["frame_time"]), 0.15)
    source_foot = np.stack([gpos[:, bvh.joint_index("LeftFoot")],
                            gpos[:, bvh.joint_index("RightFoot")]], axis=1)
    source_contact = contact_labels(source_foot, float(data["frame_time"]), 0.15)
    contact_agree = float((g1_contact == source_contact).mean() * 100)
    contact_label_agree = float((g1_contact == data["contacts"]).mean() * 100)

    # ---- 7. 限位超限 ----
    over = 0.0
    for i, name in enumerate(data["joint_names"]):
        lo, hi = G1_JOINT_LIMITS[str(name)]
        # Ignore float32 representation error at a hard joint boundary.
        over += ((data["qpos"][:, i] < lo - 1e-6)
                 | (data["qpos"][:, i] > hi + 1e-6)).sum()
    limit_over = float(over / (T * 29) * 100)

    return {
        "seq": seq_name, "type": _motion_type(seq_name), "frames": T,
        "reference_source": "regenerated" if regenerate else "stored_npz",
        "fk_err_cm": fk_err_cm,
        "fitted_err_cm": fitted_err,
        "holdout_err_cm": holdout_err,
        "upright_holdout_err_cm": upright_holdout_err_cm,
        "root_rot_err_deg": root_rot_err_deg,
        "worst_kp": worst_kp, "worst_kp_cm": worst_kp_cm,
        "worst_holdout_kp": worst_holdout, "worst_holdout_cm": worst_holdout_cm,
        "min_foot_z": min_z, "spike_pct": spike * 100,
        "contact_agree": contact_agree, "contact_label_agree": contact_label_agree,
        "limit_over_pct": limit_over,
    }


def _pf(x: float, width: int = 7) -> str:
    return f"{'nan':>{width}}" if not np.isfinite(x) else f"{x:>{width}.2f}"


def main() -> int:
    import argparse
    import csv
    import tempfile
    import time

    ap = argparse.ArgumentParser(description="重定向质量评估")
    ap.add_argument("--bvh-dir", default=SRC_DIR, help="源 BVH 目录")
    ap.add_argument("--npz-dir", default=NPZ_DIR, help="待评估的实际 NPZ 目录")
    ap.add_argument("--only", default=None,
                    help="只评估名字以该前缀开头的序列（如 ground）")
    ap.add_argument("--out", default=None,
                    help="输出 CSV；省略时自动选择 —— 不带 --only 写完整报告，"
                         "带 --only 写 quality_report_<only>.csv（绝不覆盖完整报告）")
    ap.add_argument("--regenerate", action="store_true",
                    help="显式重算候选参考；默认读取现存 NPZ，候选报告另存")
    args = ap.parse_args()

    try:
        out_path = resolve_out_path(args.only, args.out, regenerate=args.regenerate)
    except ValueError as e:
        raise SystemExit(f"[!] {e}") from None

    input_dir, extension = ((args.bvh_dir, ".bvh") if args.regenerate
                            else (args.npz_dir, ".npz"))
    try:
        names = sorted(f[:-4] for f in os.listdir(input_dir) if f.endswith(extension))
    except OSError as e:
        print(f"[!] 无法读取输入目录 {input_dir}: {e}")
        return 1
    print(f"[i] 参考来源：{'显式再生成候选' if args.regenerate else '实际存储 NPZ'}")
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
        if not names:
            print(f"[!] 没有匹配 --only {args.only} 的序列")
            return 1
        print(f"[i] 只评估前缀 {args.only!r} 的 {len(names)} 个序列 —— "
              f"输出 {out_path}，完整报告 {FULL_REPORT_PATH} 不受影响")

    if not names:
        print(f"[!] 输入目录 {input_dir} 没有 {extension} 序列")
        return 1

    # A run owns a unique partial file next to the final destination. Never
    # truncate the previous complete report while evaluating a new one.
    output_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix=os.path.basename(out_path) + ".",
            suffix=".partial.csv", delete=False) as partial:
        partial_path = partial.name
    print(f"[i] 增量结果暂存：{partial_path}", flush=True)

    def _write_rows(rs: List[dict]) -> None:
        """Checkpoint only this run's partial report; final publication is atomic."""
        with open(partial_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)

    rows: List[dict] = []
    failures: List[str] = []
    t0 = time.time()
    try:
        for i, n in enumerate(names, 1):
            try:
                rows.append(evaluate(n, regenerate=args.regenerate,
                                     bvh_dir=args.bvh_dir, npz_dir=args.npz_dir))
            except Exception as e:  # noqa: BLE001
                failures.append(n)
                print(f"[失败] {n}: {e}", flush=True)
                continue
            _write_rows(rows)
            # Explicit regeneration also includes LM and can take longer.
            el = time.time() - t0
            print(f"[{i:3d}/{len(names)}] {n:28s} {el:7.1f}s "
                  f"(均 {el / i:5.1f}s/序列，预计共 {el / i * len(names):5.0f}s)",
                  flush=True)
    except KeyboardInterrupt:
        print(f"\n[!] 评估中断，原报告保留；已完成 {len(rows)}/{len(names)}。"
              f"部分结果：{partial_path}", flush=True)
        return 130
    if failures:
        print(f"[!] {len(failures)} 个序列失败，原报告保留；"
              f"已完成 {len(rows)}/{len(names)}。部分结果：{partial_path}")
        return 1
    print(f"[i] 评估完成，用时 {time.time() - t0:.1f}s", flush=True)

    rows.sort(key=lambda r: (r["holdout_err_cm"] if np.isfinite(r["holdout_err_cm"])
                             else r["fk_err_cm"]))

    hdr = (f"{'序列':26s} {'类型':9s} {'拟合cm':>7s} {'留出cm':>7s} {'直立cm':>7s} "
           f"{'根朝向°':>8s} {'踝高m':>7s} {'尖峰%':>6s} {'接触%':>6s} 最差留出点")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['seq']:26s} {r['type']:9s} {_pf(r['fitted_err_cm'])} "
              f"{_pf(r['holdout_err_cm'])} {_pf(r['upright_holdout_err_cm'])} "
              f"{_pf(r['root_rot_err_deg'], 8)} {r['min_foot_z']:7.3f} "
              f"{r['spike_pct']:6.2f} {r['contact_agree']:6.1f} {r['worst_holdout_kp']}")

    agg: Dict[str, List[dict]] = {}
    for r in rows:
        agg.setdefault(r["type"], []).append(r)

    def _m(rs, key):
        vals = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n== 按类型汇总（均值）==")
    print(f"{'类型':10s} {'n':>3s} {'拟合cm':>7s} {'留出cm':>7s} {'直立cm':>7s} "
          f"{'根朝向°':>8s} {'尖峰%':>6s} {'接触%':>6s}")
    for t in sorted(agg):
        rs = agg[t]
        print(f"{t:10s} {len(rs):3d} {_pf(_m(rs, 'fitted_err_cm'))} "
              f"{_pf(_m(rs, 'holdout_err_cm'))} "
              f"{_pf(_m(rs, 'upright_holdout_err_cm'))} "
              f"{_pf(_m(rs, 'root_rot_err_deg'), 8)} {_m(rs, 'spike_pct'):6.2f} "
              f"{_m(rs, 'contact_agree'):6.1f}")

    print(f"\n总体（n={len(rows)}）: 拟合 {_pf(_m(rows, 'fitted_err_cm'))}cm | "
          f"留出 {_pf(_m(rows, 'holdout_err_cm'))}cm | "
          f"直立留出 {_pf(_m(rows, 'upright_holdout_err_cm'))}cm | "
          f"根朝向 {_pf(_m(rows, 'root_rot_err_deg'))}°")
    print("注意：'拟合' 是 IK 训练残差（非独立，仅回归监控）。'留出'与'直立'用的 4 个"
          "关键点未进入 IK 目标，但 IK 优化全部 29 个关节、会移动同一条运动链 —— "
          "它们是留出诊断，误差还受骨架比例和关键点匹配影响。"
          "根朝向°只检查坐标换算一致性；直立列是强制单位根朝向的对照，"
          "不能单独证明真实根朝向错误。min_foot_z 为踝原点，非足底碰撞几何。")

    _write_rows(rows)                      # Final sorted checkpoint.
    os.replace(partial_path, out_path)     # Publish only a fully successful run.
    print(f"[csv] {out_path}（已按留出误差升序，完整报告已原子发布）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
