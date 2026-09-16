"""检查源骨架与 G1 参考的高度剖面。

默认读取实际存储 NPZ，与 eval_retarget 的默认来源一致。
--regenerate 显式检查当前重定向 + IK 候选，不修改存储产物。

root 高度使用序列髋高与最小踝髋高度差重锚；躺倒、跨障等动作需要结合
全身接触与地形检查这一启发式。这里报告的是双踝 body 原点，不能把踝高
直接当成足底碰撞几何，或据此认定整个机器人悬空。

用法：
    python -m data.probe_heights
    python -m data.probe_heights ground1_subject1 --regenerate
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence

import numpy as np

from data.bvh import load_bvh
from data.retarget_lafan1 import W, g1_forward_kinematics
from data.eval_retarget import load_reference

SRC_DIR = "data/raw/lafan1"

#: 无参数时的默认样本：两个异常 ground + 一个穿地 obstacle + 一个正常 walk 作对照
DEFAULT_SEQS: Sequence[str] = (
    "ground1_subject1", "ground1_subject4",
    "obstacles5_subject2", "walk1_subject1",
)


def _rng(x: np.ndarray) -> str:
    return f"[{x.min():7.3f}, {x.max():7.3f}] 均 {x.mean():7.3f}"


def probe(seq_name: str, *, regenerate: bool = False) -> dict:
    """算出该序列的垂直剖面（**不做判断**，只报数）。"""
    bvh = load_bvh(os.path.join(SRC_DIR, seq_name + ".bvh"))
    data = load_reference(seq_name, bvh, regenerate=regenerate)
    scale = float(data["scale"])

    gpos = bvh.fk(unit_scale=1.0)[0] @ W.T * scale      # 源（G1 系，米）
    hip = gpos[:, bvh.joint_index("Hips")]
    lfoot = gpos[:, bvh.joint_index("LeftFoot")]
    rfoot = gpos[:, bvh.joint_index("RightFoot")]
    src_foot = np.minimum(lfoot[:, 2], rfoot[:, 2])

    # 锚定假设用的那个量：逐帧 (最低足高 − 髋高)
    foot_rel = src_foot - hip[:, 2]
    i_min = int(np.argmin(foot_rel))

    pos = g1_forward_kinematics(data["qpos"], data["root_pos"], data["root_rot"])
    g1_foot = np.minimum(pos["left_ankle_roll_link"][:, 2],
                         pos["right_ankle_roll_link"][:, 2])

    return {
        "seq": seq_name,
        "reference_source": "regenerated" if regenerate else "stored_npz",
        "T": bvh.num_frames,
        "src_hip_z": hip[:, 2],
        "src_foot_z": src_foot,
        "foot_rel": foot_rel,
        "argmin_foot_rel": i_min,
        # 该帧上源足是否真的触地（相对全序列最低足高）
        "foot_at_argmin_above_lowest": float(src_foot[i_min] - src_foot.min()),
        "g1_pelvis_z": data["root_pos"][:, 2],
        "g1_foot_z": g1_foot,
    }


def report(r: dict) -> None:
    print(f"\n== {r['seq']}  (T={r['T']}, {r['reference_source']}) ==")
    print(f"  源 髋高 z            {_rng(r['src_hip_z'])}")
    print(f"  源 最低足 z          {_rng(r['src_foot_z'])}")
    print(f"  源 足相对髋（锚定量）{_rng(r['foot_rel'])}")
    print(f"  → 锚定取 min 的帧 t={r['argmin_foot_rel']}；"
          f"该帧源足高出全序列最低点 {r['foot_at_argmin_above_lowest']:.4f} m")
    print(f"  重定向 骨盆 z        {_rng(r['g1_pelvis_z'])}")
    print(f"  G1 最低踝原点 z     {_rng(r['g1_foot_z'])}")

    # 仅提示踝原点高度异常；单个负最小值只说明存在该帧，非全段穿地。
    src_low = float(r["src_foot_z"].min())
    g1_low = float(r["g1_foot_z"].min())
    flags: List[str] = []
    if r["foot_at_argmin_above_lowest"] > 1e-6:
        flags.append("⚠ 锚定帧并非源足最低帧")
    if g1_low > 0.10:
        flags.append(f"⚠ 双踝原点全段偏高（最低 {g1_low:+.3f} m，需核对全身接触）")
    if g1_low < -0.05:
        flags.append(f"⚠ 存在踝原点低于零平面的帧（最低 {g1_low:+.3f} m）")
    print(f"  源足最低 {src_low:+.3f} m；" +
          ("；".join(flags) if flags else "无明显异常"))


def main(argv: Sequence[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="*")
    parser.add_argument("--regenerate", action="store_true",
                        help="显式检查重新生成的候选，默认读存储 NPZ")
    args = parser.parse_args(argv)
    names = args.sequences or list(DEFAULT_SEQS)
    missing = [n for n in names
               if not os.path.exists(os.path.join(SRC_DIR, n + ".bvh"))]
    if missing:
        print(f"[!] 找不到源 BVH: {missing}（先跑 data/download_lafan1.sh）")
        return 1
    failed = False
    for n in names:
        try:
            report(probe(n, regenerate=args.regenerate))
        except Exception as e:  # noqa: BLE001
            failed = True
            print(f"[跳过] {n}: {e}")
    print("\n提示：判据在 report() 里以 ⚠ 标出，但结论需人工核对 —— "
          "本模块只报数，不替你做 A17 的裁定。")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
