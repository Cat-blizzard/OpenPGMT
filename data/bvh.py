"""BVH 解析器（LAFAN1 / Mixamo 格式）。

支持：层级（ROOT/JOINT/End Site）、每关节 3 或 6 通道
（LAFAN1 为每关节 6 通道：位移 + 旋转）、任意欧拉旋转序、
前向运动学（FK）与世界位置/朝向输出。

依赖仅 numpy。解析逻辑参考官方 lafan1/extract.py，但保留
每关节通道信息与 End Site 偏移（重定向需要手部方向）。

坐标：原始单位（LAFAN1 为厘米，米制缩放由调用方负责）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 四元数工具（[w, x, y, z] 顺序）
# ---------------------------------------------------------------------------

def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_rot_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """单位四元数 q 旋转向量 v。q: (...,4)，v: (...,3)。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
        2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
        2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ], axis=-1)


def euler_to_quat(e: np.ndarray, order: str) -> np.ndarray:
    """欧拉角（弧度）→ 四元数。order 如 'zxy' 表示 R = Rz·Rx·Ry。

    与官方 lafan1/utils.py 约定一致：q = q0⊗(q1⊗q2)，即矩阵积 R0·R1·R2。
    """
    axis_vec = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
                "z": np.array([0, 0, 1.0])}
    c = np.cos(e[..., 0] / 2.0)[..., None]
    s = np.sin(e[..., 0] / 2.0)[..., None]
    q = np.concatenate([c, s * axis_vec[order[0]]], axis=-1)
    for i in (1, 2):
        c = np.cos(e[..., i] / 2.0)[..., None]
        s = np.sin(e[..., i] / 2.0)[..., None]
        q = quat_mul(q, np.concatenate([c, s * axis_vec[order[i]]], axis=-1))
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """四元数 → 旋转矩阵。q: (...,4) → (...,3,3)。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def remove_quat_discontinuities(q: np.ndarray) -> np.ndarray:
    """沿时间轴翻转符号保证相邻帧四元数连续（同官方实现思路）。q: (T, ...)。"""
    out = q.copy()
    for t in range(1, q.shape[0]):
        flip = (out[t] * out[t - 1]).sum(-1) < 0
        out[t] = np.where(flip[..., None], -out[t], out[t])
    return out


# ---------------------------------------------------------------------------
# BVH 解析
# ---------------------------------------------------------------------------

_CHANNEL_AXIS = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


@dataclass
class BVH:
    """解析后的 BVH 动画。

    Attrs:
        names:        关节名（J,）
        parents:      父关节下标（J,），根为 -1
        offsets:      各关节相对父的静止偏移（J,3），原始单位
        end_sites:    {关节名: 末端偏移}，End Site 不属于关节列表
        channels:     每关节通道名列表（J, 3|6）
        euler_order:  每关节旋转序（J, 如 'zxy'）
        frame_time:   帧时长（秒）
        root_pos:     根位移（T,3），原始单位
        rotations:    各关节旋转（T,J,3），弧度（已去跳变？否，原始）
        local_trans:  每关节每帧位移通道（T,J,3），原始单位
    """

    names: List[str]
    parents: np.ndarray
    offsets: np.ndarray
    end_sites: Dict[str, np.ndarray]
    channels: List[List[str]]
    euler_order: List[str]
    frame_time: float
    root_pos: np.ndarray
    rotations: np.ndarray
    local_trans: np.ndarray = field(default=None)

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    def joint_index(self, name: str) -> int:
        return self.names.index(name)

    def local_quats(self) -> np.ndarray:
        """各关节局部旋转四元数 (T,J,4)，含符号连续化。"""
        q = np.stack([euler_to_quat(self.rotations[:, j], self.euler_order[j])
                      for j in range(self.num_joints)], axis=1)
        return remove_quat_discontinuities(q)

    def local_positions(self, unit_scale: float = 1.0) -> np.ndarray:
        """各关节局部位置 (T,J,3) = 静止偏移 + 每帧位移通道，乘 unit_scale。"""
        trans = np.zeros_like(self.rotations) if self.local_trans is None else self.local_trans
        return (self.offsets[None] + trans) * unit_scale

    def fk(self, unit_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """前向运动学：世界位置 (T,J,3) 与世界朝向四元数 (T,J,4)。"""
        T, J = self.num_frames, self.num_joints
        lpos = self.local_positions(unit_scale)
        lrot = self.local_quats()
        gpos = np.zeros((T, J, 3))
        grot = np.zeros((T, J, 4))
        gpos[:, 0] = self.root_pos * unit_scale
        grot[:, 0] = lrot[:, 0]
        for j in range(1, J):
            p = self.parents[j]
            gpos[:, j] = gpos[:, p] + quat_rot_vec(grot[:, p], lpos[:, j])
            grot[:, j] = quat_mul(grot[:, p], lrot[:, j])
        return gpos, grot


# ---------------------------------------------------------------------------
# 文本解析
# ---------------------------------------------------------------------------

def parse_bvh(text: str) -> BVH:
    """解析 BVH 文本。"""
    names: List[str] = []
    parents: List[int] = []
    offsets: List[np.ndarray] = []
    channels: List[List[str]] = []
    end_sites: Dict[str, np.ndarray] = {}

    stack: List[int] = []  # 当前关节栈（下标）
    in_end_site = False
    end_site_offset = None
    frame_time = None
    frames: List[List[float]] = []

    lines = text.splitlines()
    i = 0
    in_motion = False
    n_frames = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.upper() == "MOTION":
            in_motion = True
            i += 1
            continue
        if in_motion:
            m = re.match(r"Frames:\s*(\d+)", line, re.IGNORECASE)
            if m:
                n_frames = int(m.group(1))
                i += 1
                continue
            m = re.match(r"Frame Time:\s*([\d.eE+-]+)", line, re.IGNORECASE)
            if m:
                frame_time = float(m.group(1))
                i += 1
                continue
            vals = line.split()
            if vals and all(_is_float(v) for v in vals):
                frames.append([float(v) for v in vals])
            i += 1
            continue

        m = re.match(r"ROOT\s+(\S+)", line)
        if m:
            names.append(m.group(1))
            parents.append(-1)
            offsets.append(np.zeros(3))
            channels.append([])
            stack.append(len(names) - 1)
            i += 1
            continue
        m = re.match(r"JOINT\s+(\S+)", line)
        if m:
            names.append(m.group(1))
            parents.append(stack[-1])
            offsets.append(np.zeros(3))
            channels.append([])
            stack.append(len(names) - 1)
            i += 1
            continue
        m = re.match(r"OFFSET\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", line)
        if m:
            off = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            if in_end_site:
                end_site_offset = off
            else:
                offsets[stack[-1]] = off
            i += 1
            continue
        m = re.match(r"CHANNELS\s+(\d+)\s+(.+)", line)
        if m:
            if not in_end_site:
                channels[stack[-1]] = m.group(2).split()
            i += 1
            continue
        if "End Site" in line:
            in_end_site = True
            end_site_offset = np.zeros(3)
            i += 1
            continue
        if "}" in line:
            if in_end_site:
                end_sites[names[stack[-1]]] = end_site_offset
                in_end_site = False
            else:
                stack.pop()
            i += 1
            continue
        i += 1

    if frame_time is None or not frames:
        raise ValueError("BVH 解析失败：缺少 MOTION 数据")

    J = len(names)
    euler_order = []
    for chans in channels:
        rot_chans = [c for c in chans if c in _CHANNEL_AXIS]
        euler_order.append("".join(_CHANNEL_AXIS[c] for c in rot_chans))

    data = np.array(frames, dtype=np.float64)
    assert data.shape[0] == n_frames, f"帧数不符: {data.shape[0]} vs {n_frames}"
    # 按每关节通道数切分
    root_pos = np.zeros((n_frames, 3))
    rotations = np.zeros((n_frames, J, 3))
    local_trans = np.zeros((n_frames, J, 3))
    col = 0
    for j in range(J):
        chans = channels[j]
        pos_chans = [k for k, c in enumerate(chans) if c.endswith("position")]
        rot_chans = [k for k, c in enumerate(chans) if c in _CHANNEL_AXIS]
        for k in pos_chans:
            name = chans[k]
            if j == 0:
                root_pos[:, _axis_idx(name[0])] = data[:, col + k]
            local_trans[:, j, _axis_idx(name[0])] = data[:, col + k]
        rotations[:, j, :] = np.radians(data[:, col + np.array(rot_chans)])
        col += len(chans)

    return BVH(
        names=names, parents=np.array(parents, dtype=int),
        offsets=np.stack(offsets), end_sites=end_sites,
        channels=channels, euler_order=euler_order,
        frame_time=frame_time, root_pos=root_pos,
        rotations=rotations, local_trans=local_trans,
    )


def _axis_idx(c: str) -> int:
    return {"x": 0, "X": 0, "y": 1, "Y": 1, "z": 2, "Z": 2}[c]


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_bvh(path: str) -> BVH:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_bvh(f.read())
