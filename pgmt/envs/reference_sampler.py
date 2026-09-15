"""参考帧采样与运动段采样（M1 交付，M2 环境层调用）。

- MotionDatabase：加载重定向 npz 序列（data/retarget_lafan1.py 输出）
- ref_at：任意时刻（小数帧）参考——关节位置/速度线性插值，供 50 Hz
  控制频率重采样（LAFAN1 为 30 fps）
- future_refs：C^K 未来参考帧（论文 Eq.1，61 维 = q^r(29)+q̇^r(29)+ṽ^r(3)），
  偏移 τ = t + 2^k − 1（A2），越界截断到序列末帧
- correct_anchor_velocity：A13 全局位置修正速度接口（Stage 2 用）
- AdaptiveSampler：失败频次软加权采样（A16），保留全覆盖

坐标：重定向后的 G1 系（z-up），q ∈ R^29 与
G1_JOINT_NAMES 顺序一致（见 data/retarget_lafan1.py）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from pgmt.cfg.assumptions import get

_REF_DIM = 61  # q^r(29) + q̇^r(29) + ṽ^r(3)


class MotionDatabase:
    """重定向运动序列集合。

    每个序列 dict：qpos (T,29), qvel (T,29), root_pos (T,3),
    root_rot (T,4), contacts (T,2), frame_time, joint_names, name。
    """

    def __init__(self, npz_dir: str):
        self.seqs: List[Dict[str, np.ndarray]] = []
        for f in sorted(os.listdir(npz_dir)):
            if not f.endswith(".npz"):
                continue
            d = {k: v for k, v in np.load(os.path.join(npz_dir, f)).items()}
            d["name"] = f[:-4]
            self.seqs.append(d)
        if not self.seqs:
            raise ValueError(f"{npz_dir} 中没有 npz 序列（先跑 data.retarget_lafan1）")

    @property
    def num_sequences(self) -> int:
        return len(self.seqs)

    def lengths(self) -> np.ndarray:
        return np.array([s["qpos"].shape[0] for s in self.seqs])

    def seq_len(self, seq_idx: int) -> int:
        return int(self.seqs[seq_idx]["qpos"].shape[0])

    def _lerp(self, seq_idx: int, t: float, key: str) -> np.ndarray:
        arr = self.seqs[seq_idx][key]
        t = float(np.clip(t, 0.0, arr.shape[0] - 1))
        i0 = int(np.floor(t))
        w = t - i0
        if i0 >= arr.shape[0] - 1:
            return arr[-1].copy()
        return arr[i0] * (1 - w) + arr[i0 + 1] * w

    def ref_at(self, seq_idx: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """时刻 t（帧，可为小数）的参考 (q, q̇)。"""
        return self._lerp(seq_idx, t, "qpos"), self._lerp(seq_idx, t, "qvel")

    def anchor_velocity(self, seq_idx: int, t: float) -> np.ndarray:
        """参考锚点速度 v^r（平面 xy，z=0），root_pos 有限差分（m/s）。

        速度按帧差分换算为每秒（数据帧率 ≠ 控制帧率）。
        """
        rp = self.seqs[seq_idx]["root_pos"]
        t = float(np.clip(t, 0.0, rp.shape[0] - 1))
        i0 = int(np.floor(t))
        if i0 >= rp.shape[0] - 1:
            i0 = rp.shape[0] - 2
        v = (rp[i0 + 1] - rp[i0]) / float(self.seqs[seq_idx]["frame_time"])
        return np.array([v[0], v[1], 0.0])

    def future_refs(self, seq_idx: int, t: float, e_p: Optional[np.ndarray] = None,
                    lambda_pos: Optional[float] = None) -> np.ndarray:
        """C^K 未来参考帧 (K, 61)。

        Args:
            e_p: 参考锚点系平面位置误差 (2,)，None = 零（Stage 1）
            lambda_pos: A13 增益，None = 取 A13 默认
        """
        cfg = get("A2").value
        K = cfg.K
        T = self.seq_len(seq_idx)
        out = np.zeros((K, _REF_DIM))
        for k, tau in enumerate(cfg.offsets):
            tt = min(t + tau, T - 1)  # 越界截断
            q, qd = self.ref_at(seq_idx, tt)
            v = self.anchor_velocity(seq_idx, tt)
            if e_p is not None:
                v[:2] = correct_anchor_velocity(v[:2], e_p, lambda_pos)[:2]
            out[k] = np.concatenate([q, qd, v])
        return out

    def sample_segment(self, rng: np.random.Generator, seg_len: int) -> Tuple[int, int]:
        """均匀采样运动段：按序列长度加权的 (seq_idx, start_frame)。"""
        lengths = self.lengths()
        usable = np.maximum(lengths - seg_len + 1, 1)
        seq_idx = int(rng.choice(len(lengths), p=usable / usable.sum()))
        start = int(rng.integers(0, usable[seq_idx]))
        return seq_idx, start


def correct_anchor_velocity(v_xy: np.ndarray, e_p: np.ndarray,
                            lambda_pos: Optional[float] = None) -> np.ndarray:
    """A13 全局位置修正：ṽ = v + clip(g(‖v‖)·λ_pos·e^p, ±v̄)。

    Args:
        v_xy: 名义平面参考速度 (2,)
        e_p:  参考锚点系平面位置误差 (2,)（环境运行时计算）
    Returns:
        修正后平面速度 (2,)
    """
    cfg = get("A13").value
    lam = cfg.lambda_pos if lambda_pos is None else lambda_pos
    v = np.asarray(v_xy, dtype=np.float64)
    e = np.asarray(e_p, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    # smoothstep 门控：静止归零
    g = _smoothstep(cfg.gate_v0, cfg.gate_v1, speed)
    corr = g * lam * e
    return v + np.clip(corr, -cfg.clip_v, cfg.clip_v)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = float(np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class AdaptiveSampler:
    """A16：失败频次软加权运动段采样。

    段 = (seq_idx, start)（起点网格化保证键稳定），权重 = 1 + β·失败次数：
    失败多的段采样概率高，权重下限 1 保留全数据集覆盖。
    """

    def __init__(self, db: MotionDatabase, seg_len: int, fail_boost: Optional[float] = None):
        self.db = db
        self.seg_len = seg_len
        self.fail_boost = get("A16").value.fail_boost if fail_boost is None else fail_boost
        self.fails: Dict[Tuple[int, int], float] = {}

    def _key(self, seq_idx: int, start: int) -> Tuple[int, int]:
        return (int(seq_idx), int(start) // self.seg_len * self.seg_len)

    def _candidates(self) -> List[Tuple[int, int]]:
        """全部候选段（网格起点）。"""
        cands = []
        for i, L in enumerate(self.db.lengths()):
            for s in range(0, max(L - self.seg_len + 1, 1), self.seg_len):
                cands.append((i, s))
        return cands

    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        cands = self._candidates()
        w = np.array([1.0 + self.fail_boost * self.fails.get(self._key(*c), 0.0)
                      for c in cands])
        idx = int(rng.choice(len(cands), p=w / w.sum()))
        return cands[idx]

    def update(self, seq_idx: int, start: int, failed: bool):
        """训练回调：episode 跟踪失败时累计该段失败次数。"""
        if failed:
            key = self._key(seq_idx, start)
            self.fails[key] = self.fails.get(key, 0.0) + 1.0

    def failure_rate(self) -> float:
        """失败段占比（监控全覆盖是否被破坏）。"""
        if not self.fails:
            return 0.0
        return float(len(self.fails) / max(len(self._candidates()), 1))
