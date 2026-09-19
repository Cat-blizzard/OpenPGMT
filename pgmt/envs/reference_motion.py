"""Batched G1 reference kinematics for the simulator reward boundary.

The motion database stores joint and root trajectories, while the paper's
tracking terms are defined on links.  This module keeps that conversion
deterministic and simulator independent by parsing the licensed G1 URDF once
and evaluating its fixed joint tree with Torch.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pgmt.contracts import ACT_DIM
from pgmt.envs.g1_env import G1_JOINT_NAMES, REQUIRED_BODY_NAMES
from pgmt.train.stage1 import TorchMotionDatabase


def _rpy(rpy: Sequence[float], device: torch.device) -> torch.Tensor:
    r, p, y = torch.tensor(rpy, dtype=torch.float32, device=device).unbind()
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    return torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr)).reshape(3, 3)


def _qmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def _matq(m: torch.Tensor) -> torch.Tensor:
    # Branch on the largest diagonal term; the trace-only formula loses the
    # axis at pi (its scalar component is zero).
    d0, d1, d2 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = d0 + d1 + d2
    q_trace = torch.stack((torch.sqrt((1 + tr).clamp_min(1e-8)) / 2,
        (m[...,2,1]-m[...,1,2]), (m[...,0,2]-m[...,2,0]),
        (m[...,1,0]-m[...,0,1])), -1)
    q_trace[...,1:] /= (4*q_trace[..., :1]).clamp_min(1e-6)
    qx = torch.sqrt((1+d0-d1-d2).clamp_min(1e-8))/2
    qy = torch.sqrt((1-d0+d1-d2).clamp_min(1e-8))/2
    qz = torch.sqrt((1-d0-d1+d2).clamp_min(1e-8))/2
    qxv = torch.stack(((m[...,2,1]-m[...,1,2])/(4*qx).clamp_min(1e-6), qx,
                       (m[...,0,1]+m[...,1,0])/(4*qx).clamp_min(1e-6),
                       (m[...,0,2]+m[...,2,0])/(4*qx).clamp_min(1e-6)), -1)
    qyv = torch.stack(((m[...,0,2]-m[...,2,0])/(4*qy).clamp_min(1e-6),
                       (m[...,0,1]+m[...,1,0])/(4*qy).clamp_min(1e-6), qy,
                       (m[...,1,2]+m[...,2,1])/(4*qy).clamp_min(1e-6)), -1)
    qzv = torch.stack(((m[...,1,0]-m[...,0,1])/(4*qz).clamp_min(1e-6),
                       (m[...,0,2]+m[...,2,0])/(4*qz).clamp_min(1e-6),
                       (m[...,1,2]+m[...,2,1])/(4*qz).clamp_min(1e-6), qz), -1)
    idx = torch.stack((d0, d1, d2), -1).argmax(-1)
    q = torch.where((idx == 0)[...,None], qxv, torch.where((idx == 1)[...,None], qyv, qzv))
    q = torch.where((tr > 0)[...,None], q_trace, q)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_delta(a: torch.Tensor, b: torch.Tensor, dt: float) -> torch.Tensor:
    """Angular velocity taking ``a`` to ``b`` in the parent/world frame."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    q = torch.stack((bw*aw + bx*ax + by*ay + bz*az,
                     bw*(-ax) + bx*aw + by*az - bz*ay,
                     bw*(-ay) - bx*az + by*aw + bz*ax,
                     bw*(-az) + bx*ay - by*ax + bz*aw), -1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    angle = 2 * torch.atan2(v.norm(dim=-1), q[..., :1].squeeze(-1).clamp_min(1e-8))
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) * (angle / dt).unsqueeze(-1)


class ReferenceMotion:
    """URDF FK view over a :class:`MotionDatabase` or TorchMotionDatabase."""

    def __init__(self, database: Any, urdf_path: str | Path, device="cpu", control_dt: float = .02):
        self.device = torch.device(device)
        self.control_dt = float(control_dt)
        self.db = database if isinstance(database, TorchMotionDatabase) else TorchMotionDatabase(database, self.device)
        if self.db.device != self.device:
            raise ValueError("database and ReferenceMotion must use the same device")
        self._parse_urdf(Path(urdf_path))
        self._root_link = self._find_root()
        self._body_order = tuple(REQUIRED_BODY_NAMES)
        missing = [n for n in self._body_order if n not in self._links]
        if missing:
            raise ValueError(f"URDF missing required reference bodies: {missing}")

    def _parse_urdf(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        root = ET.parse(path).getroot()
        self._links = {x.attrib["name"] for x in root.findall("link")}
        joints = {}
        actuated = []
        for j in root.findall("joint"):
            name, typ = j.attrib["name"], j.attrib.get("type", "fixed")
            parent, child = j.find("parent").attrib["link"], j.find("child").attrib["link"]
            origin = j.find("origin")
            xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split()) if origin is not None else (0.,0.,0.)
            rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split()) if origin is not None else (0.,0.,0.)
            axis = tuple(float(v) for v in (j.find("axis").attrib.get("xyz", "1 0 0") if j.find("axis") is not None else "1 0 0").split())
            joints[child] = (parent, name, typ, xyz, rpy, axis)
            if typ in ("revolute", "continuous", "prismatic"):
                actuated.append(name)
        canonical = [n[:-6] if n.endswith("_joint") else n for n in actuated]
        if len(actuated) != ACT_DIM or set(canonical) != set(G1_JOINT_NAMES):
            raise ValueError("URDF actuated joints do not match the 29-joint G1 contract")
        self._joints = joints
        self._actuator_index = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        for raw in actuated:
            self._actuator_index[raw] = self._actuator_index[raw[:-6] if raw.endswith("_joint") else raw]

    def _find_root(self):
        children = set(self._joints)
        return next(iter(self._links - children))

    @property
    def body_names(self): return self._body_order
    @property
    def lengths(self): return self.db.lengths
    @property
    def frame_time(self): return self.db.frame_time

    def sample_segments(self, batch_size: int, seg_len: int = 1, generator=None):
        return self.db.sample_segments(batch_size, seg_len, generator)

    def _fk(self, q: torch.Tensor):
        shape = q.shape[:-1]; n = q.reshape(-1, ACT_DIM).shape[0]
        R, p = {}, {}
        eye = torch.eye(3, device=self.device).expand(n,3,3)
        zero = torch.zeros((n,3), device=self.device)
        R[self._root_link], p[self._root_link] = eye, zero
        pending = set(self._joints)
        qflat = q.reshape(-1, ACT_DIM)
        while pending:
            progressed = False
            for child in tuple(pending):
                parent, name, typ, xyz, rpy, axis = self._joints[child]
                if parent not in R: continue
                T = _rpy(rpy, self.device).expand(n,3,3)
                if typ in ("revolute", "continuous", "prismatic"):
                    qi = qflat[:, self._actuator_index[name]]
                    if typ == "prismatic":
                        trans = torch.tensor(axis, device=self.device).expand(n,3) * qi[:,None]
                        J = torch.eye(3, device=self.device).expand(n,3,3)
                    else:
                        ax = torch.tensor(axis, device=self.device); ax = ax / ax.norm().clamp_min(1e-8)
                        x,y,z = ax
                        c,s = torch.cos(qi), torch.sin(qi); C=1-c
                        J = torch.stack((c+x*x*C,x*y*C-z*s,x*z*C+y*s,
                                         y*x*C+z*s,c+y*y*C,y*z*C-x*s,
                                         z*x*C-y*s,z*y*C+x*s,c+z*z*C),-1).reshape(n,3,3)
                        trans = torch.zeros((n,3), device=self.device)
                else: J, trans = torch.eye(3,device=self.device).expand(n,3,3), torch.zeros((n,3),device=self.device)
                R[child] = R[parent] @ T @ J
                xyz_t = torch.tensor(xyz, device=self.device).view(1, 3, 1).expand(n, 3, 1)
                p[child] = p[parent] + torch.bmm(R[parent], T @ trans[...,None]).squeeze(-1) + torch.bmm(R[parent], xyz_t).squeeze(-1)
                pending.remove(child); progressed=True
            if not progressed: raise ValueError("URDF joint graph is disconnected")
        return torch.stack([p[x] for x in self._body_order],1).reshape(*shape,len(self._body_order),3), torch.stack([_matq(R[x]) for x in self._body_order],1).reshape(*shape,len(self._body_order),4)

    def kinematics(self, q, qd, root_pos, root_quat, root_lin_vel, root_ang_vel):
        """Return world body pose/velocity for a batch of robot states.

        Velocities use an epsilon FK integration, which includes the complete
        joint Jacobian without requiring Isaac Sim.  Root velocities are added
        in world coordinates; accelerations are provided as zero because a
        single state has no temporal derivative.
        """
        q, qd = q.to(self.device).float(), qd.to(self.device).float()
        rp, rr = root_pos.to(self.device).float(), root_quat.to(self.device).float()
        rv, rw = root_lin_vel.to(self.device).float(), root_ang_vel.to(self.device).float()
        p0, q0 = self._fk(q); eps = 1e-4
        p1, q1 = self._fk(q + qd * eps)
        R = _qmat(rr)
        body_pos = rp[:, None] + torch.einsum("bij,bnj->bni", R, p0)
        body_pos_1 = rp[:, None] + rv[:, None] * eps + torch.einsum("bij,bnj->bni", _qmat(rr + 0.5 * eps * torch.cat((torch.zeros_like(rw[:, :1]), rw), -1)), p1)
        body_quat = _matq(R[:, None] @ _qmat(q0))
        body_quat_1 = _matq(R[:, None] @ _qmat(q1))
        body_lin = (body_pos_1 - body_pos) / eps
        body_ang = _quat_delta(body_quat, body_quat_1, eps)
        return dict(body_pos=body_pos, body_quat=body_quat,
                    body_lin_vel=body_lin, body_ang_vel=body_ang,
                    body_accel=torch.zeros_like(body_pos),
                    root_lin_vel=rv, root_ang_vel=rw)

    def sample(self, seq_ids: torch.Tensor, frames: torch.Tensor, *, placement_pos=None, placement_quat=None, robot_root_pos=None):
        seq_ids, frames = seq_ids.to(self.device).long().reshape(-1), frames.to(self.device).float().reshape(-1)
        q, qd, rp, rr = self.db.ref_at(seq_ids, frames)
        body_local_p, body_local_q = self._fk(q)
        step = self.control_dt / self.db.frame_time[seq_ids]
        max_frame = (self.db.lengths[seq_ids] - 1).float()
        # At the final frame use the preceding interval, preserving the last
        # physical velocity instead of silently returning zero.
        forward = (frames < max_frame)
        nxt = torch.where(forward, frames + step, frames - step).clamp_min(0)
        qn, _, rpn, rrn = self.db.ref_at(seq_ids, nxt)
        body_local_p_n, body_local_q_n = self._fk(qn)
        prev = (frames - step).clamp_min(0)
        qp, _, rpp, rrp = self.db.ref_at(seq_ids, prev)
        body_local_p_p, _ = self._fk(qp)
        dt = self.control_dt
        root_R, root_Rn = _qmat(rr), _qmat(rrn)
        body_p = rp[:,None] + torch.einsum("bij,bnj->bni", root_R, body_local_p)
        bp_n = rpn[:,None] + torch.einsum("bij,bnj->bni", root_Rn, body_local_p_n)
        body_q = _matq(root_R[:,None] @ _qmat(body_local_q))
        bq_n = _matq(root_Rn[:,None] @ _qmat(body_local_q_n))
        root_lin = (rpn-rp)/dt; root_ang = _quat_delta(rr, rrn, dt)
        body_lin=(bp_n-body_p)/dt; body_ang=_quat_delta(body_q,bq_n,dt)
        body_p_p = rpp[:, None] + torch.einsum("bij,bnj->bni", _qmat(rrp), body_local_p_p)
        body_lin_p = (body_p - body_p_p) / dt
        body_accel = (body_lin - body_lin_p) / dt
        root_lin_p = (rp - rpp) / dt
        root_accel = (root_lin - root_lin_p) / dt
        position_error = None
        if robot_root_pos is not None:
            # This is the global-position-correction signal, not a reset-time
            # teleport of the reference trajectory.
            delta = robot_root_pos.to(self.device) - rp
            position_error = torch.einsum("bij,bj->bi", root_R.transpose(-1, -2), delta)[..., :2]
        if placement_pos is not None:
            pp = placement_pos.to(self.device)
            rp, body_p = rp + pp, body_p + pp[:,None]
        if placement_quat is not None:
            R = _qmat(placement_quat.to(self.device)); rp = torch.einsum("bij,bj->bi", R, rp); body_p=torch.einsum("bij,bnj->bni", R, body_p); rr=_matq(R@_qmat(rr)); body_q=_matq(R[:,None]@_qmat(body_q))
            root_lin=torch.einsum("bij,bj->bi", R, root_lin); root_ang=torch.einsum("bij,bj->bi", R, root_ang); body_lin=torch.einsum("bij,bnj->bni", R, body_lin); body_ang=torch.einsum("bij,bnj->bni", R, body_ang)
            body_accel=torch.einsum("bij,bnj->bni", R, body_accel); root_accel=torch.einsum("bij,bj->bi", R, root_accel)
        future = self.db.future_refs(seq_ids, frames, e_p=position_error)
        contacts = getattr(self.db, "contacts", None)
        if contacts is not None:
            _, ci0, ci1, cw = self.db._indices(seq_ids, frames)
            contact = self.db.contacts[seq_ids, ci0] * (1 - cw[:, None]) + self.db.contacts[seq_ids, ci1] * cw[:, None]
            contact = contact > 0.5
        else:
            # Legacy sequences without labels use a conservative ankle-height
            # proxy; the simulator's contact sensor remains authoritative for
            # the robot side of the reward.
            feet = torch.stack((body_p[:, self._body_order.index("left_ankle_roll_link"), 2], body_p[:, self._body_order.index("right_ankle_roll_link"), 2]),-1)
            contact = (feet < feet.mean(-1,keepdim=True) + .04)
        return dict(joint_pos=q, joint_vel=qd, root_pos=rp, root_quat=rr, root_lin_vel=root_lin, root_ang_vel=root_ang, body_pos=body_p, body_quat=body_q, body_lin_vel=body_lin, body_ang_vel=body_ang, body_accel=body_accel, root_accel=root_accel, position_error=position_error, future=future, foot_contact=contact)
