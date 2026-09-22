"""参考帧采样与运动段采样（M1 交付，M2 环境层调用）。

- MotionDatabase：加载重定向 npz 序列（data/retarget_lafan1.py 输出）
- ref_at：任意时刻（小数帧）参考——关节位置/速度线性插值，供 50 Hz
  控制频率重采样（LAFAN1 为 30 fps）
- ref_rot：任意时刻（小数帧）参考锚点朝向——四元数**最短弧插值**，
  供 `o_t` 的 `e_t`（参考相对 anchor 朝向）使用
- future_refs：C^K 未来参考帧（论文 Eq.1，61 维 = q^r(29)+q̇^r(29)+ṽ^r(3)），
  偏移 2^k − 1 按控制步计（A1/A2），换算成源帧后采样并截断到序列末帧
- correct_anchor_velocity：A13 全局位置修正速度接口（Stage 2 用）
- AdaptiveSampler：失败频次软加权采样（A16），保留全覆盖

坐标：重定向后的 G1 系（z-up），q ∈ R^29 与
G1_JOINT_NAMES 顺序一致（见 data/retarget_lafan1.py）。
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from pgmt.cfg.assumptions import get
from pgmt.contracts import ACT_DIM, REF_FRAME_DIM
from pgmt.policy.rotation import normalize_quat, quat_slerp, quat_to_mat

_REF_DIM = REF_FRAME_DIM  # q^r + q̇^r + ṽ^r


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
            with np.load(os.path.join(npz_dir, f)) as npz:
                d = {k: v for k, v in npz.items()}
            self.seqs.append(self._validate_sequence(d, name))
        if not self.seqs:
            raise ValueError(f"{npz_dir} 中没有 npz 序列（先跑 data.retarget_lafan1）")
        if any(str(s.get("contact_source", "")) == "offline_reference_mesh_v2" for s in self.seqs):
            manifest_path = os.path.join(npz_dir, "contact_manifest.json")
            if not os.path.isfile(manifest_path):
                raise ValueError("mesh-contact generation is incomplete: contact_manifest.json is missing")
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            if manifest["sequence_names"] != [s["name"] for s in self.seqs]:
                raise ValueError("motion corpus differs from the complete mesh-contact manifest")

    def pose_fingerprint(self):
        """Bind recovery states to the exact poses and sequence ordering."""
        digest = hashlib.sha256()
        for seq in self.seqs:
            digest.update(seq["name"].encode())
            for key in ("qpos", "qvel", "root_pos", "root_rot", "frame_time"):
                value = np.asarray(seq[key])
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
            # Preserve legacy fingerprints, but never reuse a recovery pool
            # across different ground-placement or asset contracts.
            if "reference_frame_contract" in seq:
                for key in ("reference_frame_contract", "reference_ground_z", "kinematic_urdf_sha256"):
                    digest.update(key.encode())
                    digest.update(str(np.asarray(seq[key]).item()).encode())
        return digest.hexdigest()

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
        for d in sequences:
            name = str(d.get("name", f"seq{len(obj.seqs)}"))
            if apply_a17_filter and name.startswith(excluded):
                continue
            obj.seqs.append(cls._validate_sequence(d, name))
        if not obj.seqs:
            raise ValueError("from_sequences: 过滤后没有可用序列")
        return obj

    @staticmethod
    def _validate_sequence(d: Dict, name: str) -> Dict:
        """两种加载入口共享的帧数、维度和时间契约。"""
        required = ("qpos", "qvel", "root_pos", "root_rot", "frame_time")
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError(f"序列 {name} 缺字段: {missing}")
        out = dict(d, name=name)
        qpos = np.asarray(d["qpos"])
        if qpos.ndim != 2 or qpos.shape[0] == 0:
            raise ValueError(f"序列 {name} qpos 必须是非空二维帧数组")
        n_frames = qpos.shape[0]
        for key, dim in (("qpos", ACT_DIM), ("qvel", ACT_DIM),
                         ("root_pos", 3), ("root_rot", 4)):
            arr = np.asarray(d[key])
            if arr.shape != (n_frames, dim) or not np.isfinite(arr).all():
                raise ValueError(f"序列 {name} {key} 应为有限数组 ({n_frames},{dim})，得到 {arr.shape}")
            out[key] = arr
        dt = np.asarray(d["frame_time"])
        if dt.size != 1 or not np.isfinite(dt).all() or float(dt.reshape(-1)[0]) <= 0:
            raise ValueError(f"序列 {name} frame_time 必须为正的有限标量")
        out["frame_time"] = float(dt.reshape(-1)[0])
        if np.any(np.linalg.norm(out["root_rot"], axis=-1) < 1e-8):
            raise ValueError(f"序列 {name} root_rot 含无效零四元数")
        contract = str(d.get("reference_frame_contract", ""))
        if contract:
            if contract != "flat_ground_v1":
                raise ValueError(f"unknown reference frame contract: {contract}")
            ground = np.asarray(d.get("reference_ground_z", np.nan))
            fingerprint = str(d.get("kinematic_urdf_sha256", ""))
            if ground.shape != () or not np.isfinite(ground) or len(fingerprint) != 64:
                raise ValueError("flat_ground_v1 requires a finite ground height and URDF fingerprint")
            if np.asarray(d.get("contacts")).shape != (n_frames, 2):
                raise ValueError("flat_ground_v1 requires offline foot contacts")
        return out

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

    def anchor_velocity(self, seq_idx: int, t: float, *,
                        anchor_t: float | None = None) -> np.ndarray:
        """时刻 t 的参考速度，表达在 anchor_t 的参考锚点系，z=0。

        root_pos 为世界系位置；先以源数据帧率差分得到世界速度，再按
        R_ref(anchor_t).T 转入参考锚点系，最后提取 xy 分量。采用完整
        参考旋转，与论文位置误差 [R_ref(t).T @ (p_ref-p_robot)]_xy 一致。
        anchor_t 默认 t；future_refs 显式传当前 t，使所有未来速度与
        当前 e_p 处于同一坐标系。单帧序列没有位移信息，速度定义为零。
        """
        rp = self.seqs[seq_idx]["root_pos"]
        t = float(np.clip(t, 0.0, rp.shape[0] - 1))
        if rp.shape[0] == 1:
            return np.zeros(3, dtype=np.float64)
        i0 = min(int(np.floor(t)), rp.shape[0] - 2)
        v_world = (rp[i0 + 1] - rp[i0]) / float(self.seqs[seq_idx]["frame_time"])
        R_ref = quat_to_mat(self.ref_rot(seq_idx, t if anchor_t is None else anchor_t))
        v_anchor = R_ref.T @ v_world
        return np.array([v_anchor[0], v_anchor[1], 0.0])

    def future_refs(self, seq_idx: int, t: float, e_p: Optional[np.ndarray] = None,
                    lambda_pos: Optional[float] = None) -> np.ndarray:
        """C^K 未来参考帧 (K, 61)。

        坐标约定：全部未来速度统一表达在当前 t 的参考锚点系。论文明确
        位置误差使用该坐标系，但没有细化未来 token 的换基时刻；这里
        固定用当前锚点，避免转弯时将不同帧坐标系的速度与同一个 e_p 相加。

        时间约定：t 的单位为源帧，可为小数；A2 的偏移以 A1 控制步为
        单位，按 control_dt / source_frame_time 换算。默认最长前瞻
        为 31 × 0.02 = 0.62 秒，与源数据帧率无关。

        Args:
            t: 当前源帧下标（可为小数）
            e_p: 当前 t 参考锚点系平面位置误差 (2,)，None = 零（Stage 1）
            lambda_pos: A13 增益，None = 取 A13 默认
        """
        cfg = get("A2").value
        K = cfg.K
        T = self.seq_len(seq_idx)
        source_frames_per_step = get("A1").value.dt / float(self.seqs[seq_idx]["frame_time"])
        out = np.zeros((K, _REF_DIM))
        for k, tau in enumerate(cfg.offsets):
            tt = min(t + tau * source_frames_per_step, T - 1)  # 控制步 → 源帧
            q, qd = self.ref_at(seq_idx, tt)
            v = self.anchor_velocity(seq_idx, tt, anchor_t=t)
            if e_p is not None:
                v[:2] = correct_anchor_velocity(v[:2], e_p, lambda_pos)[:2]
            out[k] = np.concatenate([q, qd, v])
        return out

    def sample_segment(self, rng: np.random.Generator, seg_len: int) -> Tuple[int, int]:
        """均匀采样运动段：按序列长度加权的 (seq_idx, start_frame)。"""
        if not isinstance(seg_len, (int, np.integer)) or seg_len <= 0:
            raise ValueError(f"seg_len 必须为正整数，得到 {seg_len}")
        lengths = self.lengths()
        usable = np.maximum(lengths - seg_len + 1, 1)
        seq_idx = int(rng.choice(len(lengths), p=usable / usable.sum()))
        start = int(rng.integers(0, usable[seq_idx]))
        return seq_idx, start


def correct_anchor_velocity(v_xy: np.ndarray, e_p: np.ndarray,
                            lambda_pos: Optional[float] = None) -> np.ndarray:
    """A13 全局位置修正：ṽ = v + clip(g(‖v‖)·λ_pos·e^p, ±v̄)。

    Args:
        v_xy: 当前参考锚点系名义平面速度 (2,)，须与 e_p 同系
        e_p:  参考锚点系平面位置误差 (2,)（环境运行时计算）
    Returns:
        同一参考锚点系的修正后平面速度 (2,)
    """
    cfg = get("A13").value
    lam = cfg.lambda_pos if lambda_pos is None else lambda_pos
    v = np.asarray(v_xy, dtype=np.float64)
    e = np.asarray(e_p, dtype=np.float64)
    if v.shape != (2,) or e.shape != (2,):
        raise ValueError(f"v_xy 与 e_p 必须为形状 (2,)，得到 {v.shape}/{e.shape}")
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
        if not isinstance(seg_len, (int, np.integer)) or seg_len <= 0:
            raise ValueError(f"seg_len 必须为正整数，得到 {seg_len}")
        self.seg_len = int(seg_len)
        self.fail_boost = get("A16").value.fail_boost if fail_boost is None else fail_boost
        if not np.isfinite(self.fail_boost) or self.fail_boost < 0:
            raise ValueError(f"fail_boost 必须非负且有限，得到 {self.fail_boost}")
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
