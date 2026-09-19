"""奖励尺度的实测探针（A18 的 σ 取值依据，服务器可跑）。

## 为什么需要它

`pgmt/rewards/spec.py` 的核是高斯式 `exp(−e²/σ)`，其中 σ 论文未给。σ 不是
可以随便拍的"调参量"：取小了该项在任何误差下都奖励≈0（**项是死的**，没有
梯度），取大了所有误差都奖励≈1（**项无区分度**，等价于从奖励里消失）。
两条退化都由 σ 与"该目标的真实误差分布"的相对关系决定，因此**可以测量**。

本脚本做两件事：

1. **量出真实参考运动的误差尺度**：从 77 个重定向序列取若干，测量
   关节速度、连杆位置的变化范围、根位置一帧位移等，给出真实量的量级。
   这不能直接给出"跟踪误差"（那需要仿真中策略的实际误差），
   但给出了**误差的物理上限参考**（策略误差不可能远小于参考量本身的变化）。
2. **给出 σ 的响应曲线**：对每个跟踪项，报告在候选 σ 下
   "哪些误差落在可响应区（reward ∈ [0.1, 0.9]）"，以及半衰误差 `√σ`。

## 判据（写进报告）

对每个 σ：
  - `σ_min`：小于它则连"很小但真实"的误差都压到 0.1 以下 → 项失效
  - `σ_max`：大于它则连"明显失误"的误差都还 >0.9 → 项无区分度
  取 σ 落在 [σ_min, σ_max] 内，并在报告中说明理由。

用法:
  python -m data.probe_reward_scales              # 全部项
  python -m data.probe_reward_scales --only pos   # 只看位置类
  python -m data.probe_reward_scales --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

NPZ_DIR = "data/processed/lafan1_g1"

#: 每个跟踪项的"物理参考量级"来源说明，供报告引用
#: (项名, 误差单位, 参考量来源描述)
TRACKED_TERMS: Tuple[Tuple[str, str, str], ...] = (
    ("link_pos", "m", "连杆位置误差：参考相邻帧连杆位移与骨架尺度"),
    ("ta_link_pos", "m", "下半身连杆位置误差（同上）"),
    ("link_ori", "rad", "连杆朝向误差：参考相邻帧朝向变化"),
    ("ta_link_ori", "rad", "下半身连杆朝向误差（同上）"),
    ("link_lin_vel", "m/s", "连杆线速度误差：参考连杆速度量级"),
    ("link_ang_vel", "rad/s", "连杆角速度误差：参考连杆角速度量级"),
    ("joint_pos", "rad", "关节位置误差：关节限位跨度与参考帧间变化"),
    ("ta_joint_pos", "rad", "下半身关节位置误差（同上）"),
    ("joint_vel", "rad/s", "关节速度误差：参考关节速度量级"),
)


def _load_reference_stats(n_seq: int = 12, stride: int = 3) -> Dict[str, np.ndarray]:
    """从真实重定向序列采集参考运动的统计量。

    stride 用于降采样（50 Hz 控制频率远高于 30 fps 源数据的独立信息量）。
    """
    if not os.path.isdir(NPZ_DIR):
        raise SystemExit(f"[!] 找不到 {NPZ_DIR}（先跑 data.retarget_lafan1）")
    names = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    if not names:
        raise SystemExit(f"[!] {NPZ_DIR} 里没有 npz")
    # 等间隔取样，避免只取到 walk 类
    idx = np.linspace(0, len(names) - 1, min(n_seq, len(names))).astype(int)
    picked = [names[i] for i in idx]

    qpos, qvel, rpos, rrot, ftime = [], [], [], [], []
    for nm in picked:
        d = np.load(os.path.join(NPZ_DIR, nm))
        qpos.append(np.asarray(d["qpos"])[::stride])
        qvel.append(np.asarray(d["qvel"])[::stride])
        rpos.append(np.asarray(d["root_pos"])[::stride])
        rrot.append(np.asarray(d["root_rot"])[::stride])
        ftime.append(float(np.asarray(d["frame_time"])) * stride)
    return {
        "names": np.array(picked),
        "qpos": np.concatenate(qpos, axis=0),
        "qvel": np.concatenate(qvel, axis=0),
        "root_pos": np.concatenate(rpos, axis=0),
        "root_rot": np.concatenate(rrot, axis=0),
        "dt": np.array([float(np.mean(ftime))]),
    }


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """两批四元数之间的最小夹角（度），用 |dot| 消符号歧义。"""
    a = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    b = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def reference_scales(stats: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """从真实参考运动量出各物理量的量级（用于标定误差上限参考）。"""
    q, qd = stats["qpos"], stats["qvel"]
    rp, rr, dt = stats["root_pos"], stats["root_rot"], float(stats["dt"][0])

    # 根位置一帧位移（平面）→ 速度量级
    step = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=-1) / dt
    # 根朝向一帧变化
    rot_step = _quat_angle_deg(rr[:-1], rr[1:])
    # 关节帧间变化
    jstep = np.abs(np.diff(q, axis=0))
    # 关节活动范围（限位跨度的代理）
    jrange = np.percentile(q, 99, axis=0) - np.percentile(q, 1, axis=0)

    def pct(a, p):
        return float(np.percentile(np.asarray(a).ravel(), p))

    return {
        "root_lin_speed": {"p50": pct(step, 50), "p95": pct(step, 95), "max": pct(step, 100)},
        "root_rot_step_deg": {"p50": pct(rot_step, 50), "p95": pct(rot_step, 95), "max": pct(rot_step, 100)},
        "root_rot_step_rad": {
            "p50": math.radians(pct(rot_step, 50)),
            "p95": math.radians(pct(rot_step, 95)),
            "max": math.radians(pct(rot_step, 100)),
        },
        "joint_vel": {"p50": pct(qd, 50), "p95": pct(qd, 95), "max": pct(np.abs(qd), 100)},
        "joint_step_rad": {"p50": pct(jstep, 50), "p95": pct(jstep, 95), "max": pct(jstep, 100)},
        "joint_range_rad": {"p50": pct(jrange, 50), "p95": pct(jrange, 95), "max": pct(jrange, 100)},
    }


def responsive_band(sigma: float, lo: float = 0.1, hi: float = 0.9) -> Tuple[float, float]:
    """reward ∈ [lo, hi] 对应的误差区间（该项的"可响应区"）。

    解 `exp(−e²/σ) = r` → `e = √(−σ·ln r)`。
    """
    if sigma <= 0.0:
        raise ValueError("σ 必须为正")
    e_hi = math.sqrt(-sigma * math.log(lo))   # reward 降到 lo 时的误差
    e_lo = math.sqrt(-sigma * math.log(hi))   # reward 降到 hi 时的误差
    return e_lo, e_hi


def report_sigma(term: str, unit: str, sigma: float) -> Dict[str, float]:
    e_lo, e_hi = responsive_band(sigma)
    return {
        "term": term,
        "unit": unit,
        "sigma": sigma,
        "half_decay_error": math.sqrt(sigma),   # reward = 1/e 处
        "responsive_lo": e_lo,                  # reward ≤ 0.9 起
        "responsive_hi": e_hi,                  # reward ≥ 0.1 止
    }


def main():
    ap = argparse.ArgumentParser(description="奖励 σ 的实测探针")
    ap.add_argument("--only", default=None, help="只报告名字含该子串的项")
    ap.add_argument("--seq", type=int, default=12, help="采样序列数")
    ap.add_argument("--stride", type=int, default=3, help="帧降采样步长")
    ap.add_argument("--json", default=None, help="把结果写入 JSON")
    args = ap.parse_args()

    from pgmt.rewards.spec import SIGMAS

    print("=" * 78)
    print("奖励尺度探针（A18）—— 核：exp(−e²/σ)，σ 量纲 = 误差平方")
    print("=" * 78)

    stats = _load_reference_stats(args.seq, args.stride)
    print(f"\n[数据] {len(stats['names'])} 个序列，共 {stats['qpos'].shape[0]} 帧（降采样后）")
    print("       " + ", ".join(stats["names"][:6]) + (" ..." if len(stats["names"]) > 6 else ""))

    ref = reference_scales(stats)
    print("\n== 参考运动的物理量级（跟踪误差的参照上限）==")
    print(f"{'量':28s} {'p50':>10s} {'p95':>10s} {'max':>10s}")
    for k, v in ref.items():
        print(f"{k:28s} {v['p50']:10.4f} {v['p95']:10.4f} {v['max']:10.4f}")

    print("\n== 各跟踪项的 σ 与可响应区 ==")
    print(f"{'项':16s} {'σ':>9s} {'√σ(=1/e误差)':>13s} {'reward≤0.9 起':>14s} {'reward≥0.1 止':>14s}  单位")
    rows: List[Dict[str, float]] = []
    for term, unit, _desc in TRACKED_TERMS:
        if args.only and args.only not in term:
            continue
        if term not in SIGMAS:
            print(f"{term:16s} {'(未登记)':>9s}")
            continue
        r = report_sigma(term, unit, SIGMAS[term])
        rows.append(r)
        print(f"{term:16s} {r['sigma']:9.4f} {r['half_decay_error']:13.4f} "
              f"{r['responsive_lo']:14.4f} {r['responsive_hi']:14.4f}  {unit}")

    print("\n== 判读 ==")
    print("· 'reward≤0.9 起' = 误差超过它才开始明显扣分；'reward≥0.1 止' = 误差超过它该项基本归零。")
    print("· 若真实误差落在 [起, 止] 之外：小于'起'→ 无区分度；大于'止'→ 项是死的。")
    print("· 对比上表的参考量级：σ 的响应区应与该目标的误差量级同数量级。")
    print("  （跟踪误差还取决于仿真中策略的实际表现，本表只给出量级上界参考。）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"reference_scales": ref, "sigmas": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[写] {args.json}")


if __name__ == "__main__":
    main()
