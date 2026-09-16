"""参考帧采样与运动段采样（M1 交付，M2 环境层调用）。

- MotionDatabase：加载重定向 npz 序列（data/retarget_lafan1.py 输出）
- ref_at：任意时刻（小数帧）参考——关节位置/速度线性插值，供 50 Hz
  控制频率重采样（LAFAN1 为 30 fps）
- ref_rot：任意时刻（小数帧）参考锚点朝向——四元数**最短弧插值**，
  供 `o_t` 的 `e_t`（参考相对 anchor 朝向）使用
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
from pgmt.policy.rotation import normalize_quat, quat_slerp

_REF_DIM = 61  # q^r(29) + q̇^r(29) + ṽ^r(3)


class MotionDatabase:
    """重定向运动序列集合。

    每个序列 dict：qpos (T,29), qvel (T,29), root_pos (T,3),
    root_rot (T,4), contacts (T,2), frame_time, joint_names, name。
    """

    #: `lengths()` 的缓存（构造后序列结构不变）。类级默认值使
    #: `__init__` 与 `from_sequences` 两条构造路径都无需各自初始化。
    _lengths_cache: Optional[np.ndarray] = None

    def __init__(self, npz_dir: str):
        self.seqs: List[Dict[str, np.ndarray]] = []
        excluded = tuple(get("A17").value.excluded_prefixes)
        for f in sorted(os.listdir(npz_dir)):
            if not f.endswith(".npz"):
                continue
            name = f[:-4]
            if name.startswith(excluded):
                continue  # A17：ground 类重定向退化，排除训练集
            d = {k: v for k, v in np.load(os.path.join(npz_dir, f)).items()}
            d["name"] = name
            self.seqs.append(d)
        if not self.seqs:
            raise ValueError(f"{npz_dir} 中没有 npz 序列（先跑 data.retarget_lafan1）")

    @classmethod
    def from_sequences(cls, sequences: List[Dict[str, np.ndarray]],
                       apply_a17_filter: bool = False) -> "MotionDatabase":
        """从内存中的序列列表构造（测试与 M2 环境单测用，免去落盘 npz）。

        Args:
            sequences: 每个 dict 须含 qpos/qvel/root_pos/root_rot/frame_time，
                       可选 contacts；`name` 用于 A17 过滤。
            apply_a17_filter: 是否按 A17 的前缀规则过滤（默认关闭，
                       便于测试显式控制数据集合）。
        """
        obj = cls.__new__(cls)
        obj.seqs = []
        excluded = tuple(get("A17").value.excluded_prefixes)
        required = ("qpos", "qvel", "root_pos", "root_rot", "frame_time")
        for d in sequences:
            name = str(d.get("name", f"seq{len(obj.seqs)}"))
            if apply_a17_filter and name.startswith(excluded):
                continue
            missing = [k for k in required if k not in d]
            if missing:
                raise ValueError(f"序列 {name} 缺字段: {missing}")
            e = dict(d)
            e["name"] = name
            obj.seqs.append(e)
        if not obj.seqs:
            raise ValueError("from_sequences: 过滤后没有可用序列")
        return obj

    @property
    def num_sequences(self) -> int:
        return len(self.seqs)

    def lengths(self) -> np.ndarray:
        """各序列帧数（缓存：采样热路径每步都调用）。"""
        if self._lengths_cache is None:
            self._lengths_cache = np.array([s["qpos"].shape[0] for s in self.seqs])
        return self._lengths_cache

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

    def ref_rot(self, seq_idx: int, t: float) -> np.ndarray:
        """时刻 t（帧，可为小数）的参考锚点朝向，[w,x,y,z]。

        四元数须按**最短弧**插值：直接线性插值在 q 与 −q（同一旋转的两种
        表示）之间会跳出 360° 假旋转 —— 30 fps 源数据在 50 Hz 控制下必然
        出现小数帧，因此这里是必需路径而非可选优化。
        越界截断到序列末帧（与 `_lerp` 一致）。
        """
        if "root_rot" not in self.seqs[seq_idx]:
            raise KeyError(f"序列 {self.seqs[seq_idx].get('name', seq_idx)} 缺 root_rot 字段")
        q = self.seqs[seq_idx]["root_rot"]
        t = float(np.clip(t, 0.0, q.shape[0] - 1))
        i0 = int(np.floor(t))
        if i0 >= q.shape[0] - 1:
            return normalize_quat(q[-1])
        return quat_slerp(normalize_quat(q[i0]), normalize_quat(q[i0 + 1]), t - i0)

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

    `_candidates()` 结果按序列结构缓存（构造后 DB 不再变化），只在第一次
    调用时枚举 —— 采样发生在**每次环境 reset**，15k 并行环境下这是热路径，
    每次重建列表 + 逐项查字典会造成可观的固定开销。权重也改成向量化构建，
    语义与逐项 `1 + β·fails[k]` 完全一致。
    """

    def __init__(self, db: MotionDatabase, seg_len: int, fail_boost: Optional[float] = None):
        self.db = db
        self.seg_len = int(seg_len)
        if self.seg_len <= 0:
            raise ValueError(f"seg_len 必须为正，得到 {seg_len}")
        self.fail_boost = get("A16").value.fail_boost if fail_boost is None else fail_boost
        self.fails: Dict[Tuple[int, int], float] = {}
        self._cands: Optional[List[Tuple[int, int]]] = None
        self._keys: Optional[np.ndarray] = None  # (N,2) 整数，供向量化查权重
        self._w_cache: Optional[np.ndarray] = None

    def _key(self, seq_idx: int, start: int) -> Tuple[int, int]:
        return (int(seq_idx), int(start) // self.seg_len * self.seg_len)

    def _candidates(self) -> List[Tuple[int, int]]:
        """全部候选段（网格起点）。首次调用后缓存。"""
        if self._cands is None:
            cands = []
            for i, L in enumerate(self.db.lengths()):
                for s in range(0, max(L - self.seg_len + 1, 1), self.seg_len):
                    cands.append((i, s))
            if not cands:
                raise ValueError("没有候选运动段（检查 seg_len 与序列长度）")
            self._cands = cands
            self._keys = np.asarray(cands, dtype=np.int64)
        return self._cands

    def _weights(self) -> np.ndarray:
        """向量化权重 1 + β·失败次数（未在候选集中的失败键被忽略）。

        结果缓存，`update()` 时失效 —— 采样发生在每次环境 reset，15k 并行
        环境下这是热路径，每次重算会带来 O(|失败键|) 的 Python 循环开销。
        """
        if self._w_cache is not None:
            return self._w_cache
        self._candidates()  # 确保 _keys 就绪
        n_boost = np.zeros(self._keys.shape[0], dtype=np.float64)
        if self.fails:
            for (si, st), f in self.fails.items():
                lo = np.searchsorted(self._keys[:, 0], si, side="left")
                hi = np.searchsorted(self._keys[:, 0], si, side="right")
                if hi <= lo:
                    continue
                seg = slice(lo, hi)
                m = self._keys[seg, 1] == st
                n_boost[seg] += np.where(m, self.fail_boost * f, 0.0)
        self._w_cache = 1.0 + n_boost
        return self._w_cache

    def _invalidate_weights(self) -> None:
        self._w_cache = None

    def sample(self, rng: np.random.Generator) -> Tuple[int, int]:
        cands = self._candidates()
        w = self._weights()
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            # 极端防御：权重异常时退回均匀采样，保持全覆盖而不是崩溃
            idx = int(rng.integers(0, len(cands)))
        else:
            idx = int(rng.choice(len(cands), p=w / total))
        return cands[idx]

    def update(self, seq_idx: int, start: int, failed: bool):
        """训练回调：episode 跟踪失败时累计该段失败次数。"""
        if failed:
            key = self._key(seq_idx, start)
            self.fails[key] = self.fails.get(key, 0.0) + 1.0
            self._invalidate_weights()

    def failure_rate(self) -> float:
        """失败段占比（监控全覆盖是否被破坏）。"""
        if not self.fails:
            return 0.0
        return float(len(self.fails) / max(len(self._candidates()), 1))
