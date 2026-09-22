"""Batched Torch implementation of the PGMT reward contract.

The simulator supplies tensors in world coordinates.  Body tracking is
computed in each root's yaw frame, which makes a common world translation and
yaw transform disappear from the residuals.  This is the batched counterpart
of :mod:`pgmt.rewards.tracking` and deliberately keeps all undocumented
choices in the existing A21/A22 configuration.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from pgmt.cfg.assumptions import get
from pgmt.contracts import ACT_DIM
from pgmt.rewards.spec import AUX, LOWER, TERRAIN, UPPER, SIGMAS
from pgmt.rewards.tracking import default_partitions
from data.retarget_lafan1 import G1_JOINT_LIMITS


_REQUIRED_STATE = ("joint_pos", "joint_vel", "root_pos", "root_quat",
                   "root_lin_vel", "root_ang_vel", "body_pos", "body_quat",
                   "body_lin_vel", "body_ang_vel")
_TERRAIN_FAMILIES = {"flat": 0, "slopes": 1, "stairs": 2, "boxes": 3, "rough": 4}


def _t(x, device, dtype=torch.float32):
    return x if isinstance(x, torch.Tensor) and x.device == device and x.dtype == dtype else torch.as_tensor(x, device=device, dtype=dtype)


def _check_batch(d: Mapping[str, torch.Tensor], name: str, n: int, shape_tail):
    if name not in d:
        raise KeyError(f"missing {name}")
    x = d[name]
    if x.ndim != len(shape_tail) + 1 or x.shape[0] != n or tuple(x.shape[1:]) != tuple(shape_tail):
        raise ValueError(f"{name} must have shape (N,{','.join(map(str, shape_tail))}), got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf")


def _qnorm(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def _qconj(q):
    return q * q.new_tensor([1., -1., -1., -1.])


def _qrot(q, v):
    z = q.new_zeros(v.shape[:-1] + (1,))
    return _qmul(_qmul(q, torch.cat((z, v), -1)), _qconj(q))[..., 1:]


def _yaw(q):
    q = _qnorm(q); w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _yaw_quat(angle):
    h = angle * .5
    return torch.stack((torch.cos(h), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h)), -1)


def _angle(a, b):
    rel = _qmul(_qconj(_qnorm(a)), _qnorm(b))
    return 2 * torch.atan2(rel[..., 1:].norm(dim=-1), rel[..., 0].abs())


def _exp(e, sigma):
    return torch.exp(-e.square() / float(sigma))


def root_relative_body_positions(root_pos, root_quat, body_pos):
    """Shared yaw-aligned body coordinates for tracking and termination."""
    inv = _qconj(_yaw_quat(_yaw(root_quat)))
    return _qrot(inv[:, None, :], body_pos - root_pos[:, None, :])


class BatchedRewardComputer:
    """Compute Stage 1 or Stage 2 rewards for a batch of G1 states.

    ``body_pos`` and related tensors follow ``body_names`` order.  Reference
    body tensors follow the same order.  All body tracking is reduced by the
    element mean, matching the NumPy oracle.  Terrain TA relaxation is applied
    to each body/joint error before that mean, as required by Eq.10.
    """

    def __init__(self, body_names: Sequence[str], joint_names: Sequence[str], device="cpu", dt: float = .02):
        self.device = torch.device(device); self.body_names = tuple(body_names); self.joint_names = tuple(joint_names); self.dt = float(dt)
        if len(set(self.body_names)) != len(self.body_names): raise ValueError("body_names must be unique")
        if len(self.joint_names) != ACT_DIM or len(set(self.joint_names)) != ACT_DIM: raise ValueError("G1 needs 29 unique joints")
        p = default_partitions()
        missing_bodies = (set(p.upper_bodies) | set(p.lower_bodies)) - set(self.body_names)
        missing_joints = (set(p.upper_joints) | set(p.lower_joints)) - set(self.joint_names)
        if missing_bodies or missing_joints:
            raise ValueError(f"missing required G1 partition names: bodies={sorted(missing_bodies)}, joints={sorted(missing_joints)}")
        self.upper_bodies = tuple(p.upper_bodies); self.lower_bodies = tuple(p.lower_bodies)
        self.upper_joints = tuple(p.upper_joints); self.lower_joints = tuple(p.lower_joints)
        self.bi = {n:i for i,n in enumerate(self.body_names)}; self.ji = {n:i for i,n in enumerate(self.joint_names)}

    def _validate(self, d, n, reference=False):
        for k in _REQUIRED_STATE:
            if k not in d and (not reference or k not in ("root_lin_vel", "root_ang_vel")):
                raise KeyError(f"{'reference.' if reference else 'state.'}{k} is required")
        for k, tail in (("joint_pos", (ACT_DIM)), ("joint_vel", (ACT_DIM)), ("root_pos", (3,)), ("root_quat", (4,)), ("root_lin_vel", (3,)), ("root_ang_vel", (3,))):
            if k in d: _check_batch(d, k, n, (tail,) if isinstance(tail, int) else tail)
        b = len(self.body_names)
        for k, tail in (("body_pos", (b,3)), ("body_quat", (b,4)), ("body_lin_vel", (b,3)), ("body_ang_vel", (b,3))): _check_batch(d, k, n, tail)

    def _root_frame(self, d):
        # yaw-only root frame, preserving the existing tracking oracle semantics
        q = _yaw_quat(_yaw(d["root_quat"]))
        inv = _qconj(q)
        pos = root_relative_body_positions(d["root_pos"], d["root_quat"], d["body_pos"])
        lin = _qrot(inv[:,None,:], d["body_lin_vel"])
        ang = _qrot(inv[:,None,:], d["body_ang_vel"])
        ori = _qmul(_qconj(q)[:,None,:], d["body_quat"])
        return pos, ori, lin, ang

    def _idx(self, names, mapping): return torch.tensor([mapping[x] for x in names], device=self.device, dtype=torch.long)

    def compute(self, state: Mapping, reference: Mapping, *, previous_state=None, action=None, previous_action=None, corrected_velocity=None, recovery_mask=None, terrain=None, stage2=False, alpha=1.0):
        state = {k: _t(v, self.device, torch.bool if k == "contact" else torch.float32) for k,v in state.items()}
        reference = {k: _t(v, self.device, torch.float32) for k,v in reference.items()}
        n = state["joint_pos"].shape[0]; self._validate(state,n); self._validate(reference,n,True)
        if "contact_forces" not in state:
            raise KeyError("state.contact_forces is required: contact measurements cannot be inferred")
        _check_batch(state, "contact_forces", n, (len(self.body_names), 3))
        if action is None: raise ValueError("action is required for auxiliary action_rate")
        action = _t(action,self.device); previous_action = _t(previous_action,self.device) if previous_action is not None else torch.zeros_like(action)
        if action.shape != (n,ACT_DIM) or previous_action.shape != action.shape: raise ValueError("action and previous_action must be (N,29)")
        sp, so, sv, sw = self._root_frame(state); rp, ro, rv, rw = self._root_frame(reference)
        def body_err(a,b,names,kind):
            ix=self._idx(names,self.bi)
            if kind=="pos": return (a[:,ix]-b[:,ix]).norm(dim=-1)
            if kind=="ori": return _angle(a[:,ix],b[:,ix])
            return (a[:,ix]-b[:,ix]).norm(dim=-1)
        def joint_err(a,b,names):
            ix=self._idx(names,self.ji); return (a[:,ix]-b[:,ix]).abs()
        def group(names_b,names_j, lower=False):
            vals={}
            pos=body_err(sp,rp,names_b,"pos"); ori=body_err(so,ro,names_b,"ori"); lin=body_err(sv,rv,names_b,"lin"); ang=body_err(sw,rw,names_b,"ang")
            jp=joint_err(state["joint_pos"],reference["joint_pos"],names_j); jv=joint_err(state["joint_vel"],reference["joint_vel"],names_j)
            if lower and terrain is not None:
                fam=terrain.get("family_ids"); lev=terrain.get("level")
                if fam is None or lev is None: raise KeyError("terrain family_ids and level are required for TA relaxation")
                fam=_t(fam,self.device,torch.long); lev=_t(lev,self.device,torch.float32)
                active=((fam==1)|(fam==2)|(fam==3)).float()
                sat=get("A18").value.tau_saturation
                tau_m=lev.clamp(0,9)/9*float(sat["ta_link_pos"])
                tau_r=lev.clamp(0,9)/9*float(sat["ta_link_ori"])
                tau_j=lev.clamp(0,9)/9*float(sat["ta_joint_pos"])
                pos=(pos-alpha*active[:,None]*tau_m[:,None]).clamp_min(0); ori=(ori-alpha*active[:,None]*tau_r[:,None]).clamp_min(0); jp=(jp-alpha*active[:,None]*tau_j[:,None]).clamp_min(0)
            vals["ta_link_pos" if lower else "link_pos"]=_exp(pos.mean(1),SIGMAS["ta_link_pos" if lower else "link_pos"])
            vals["ta_link_ori" if lower else "link_ori"]=_exp(ori.mean(1),SIGMAS["ta_link_ori" if lower else "link_ori"])
            vals["link_lin_vel"]=_exp(lin.mean(1),SIGMAS["link_lin_vel"]); vals["link_ang_vel"]=_exp(ang.mean(1),SIGMAS["link_ang_vel"])
            vals["ta_joint_pos" if lower else "joint_pos"]=_exp(jp.mean(1),SIGMAS["ta_joint_pos" if lower else "joint_pos"]); vals["joint_vel"]=_exp(jv.mean(1),SIGMAS["joint_vel"])
            terms=LOWER.terms if lower else UPPER.terms; return sum(w*vals[k] for k,w in terms), vals
        upper, um = group(self.upper_bodies,self.upper_joints); lower, lm = group(self.lower_bodies,self.lower_joints,True)
        root_ori=_exp(_angle(state["root_quat"],reference["root_quat"]),get("A21").value.sigma_root_ori)
        if corrected_velocity is None: corrected_velocity=reference["root_lin_vel"]
        corrected_velocity=_t(corrected_velocity,self.device); rv_err=(state["root_lin_vel"]-corrected_velocity).norm(dim=-1)
        anchor_err=(state["root_pos"]-reference["root_pos"]).norm(dim=-1)
        upward=(float(get("A21").value.recovery_target_upward_vel)-state["root_lin_vel"][:,2]).clamp_min(0)
        # Recovery is an explicit curriculum state.  Outside a recovery
        # episode its term is neutral, rather than silently rewarding upward
        # velocity during every ordinary tracking episode.
        mask=torch.zeros(n,device=self.device) if recovery_mask is None else _t(recovery_mask,self.device)
        auxv={"root_ori":root_ori,"corrected_root_vel":_exp(rv_err,get("A21").value.sigma_corrected_root_vel),"floating_anchor_pos":_exp(anchor_err,get("A21").value.sigma_floating_anchor),"recovery_upward_vel":mask*_exp(upward,get("A21").value.sigma_recovery_upward),"action_rate":(action-previous_action).square().sum(-1),"joint_limit":torch.zeros(n,device=self.device),"pelvis_vert_accel":torch.zeros(n,device=self.device),"ee_accel_mismatch":torch.zeros(n,device=self.device),"undesired_contact":torch.zeros(n,device=self.device),"head_torso_impact":torch.zeros(n,device=self.device)}
        lo = _t(state.get("joint_low", [G1_JOINT_LIMITS[x][0] for x in self.joint_names]), self.device)
        hi = _t(state.get("joint_high", [G1_JOINT_LIMITS[x][1] for x in self.joint_names]), self.device)
        auxv["joint_limit"] = ((lo[None] - state["joint_pos"]).clamp_min(0).square() + (state["joint_pos"] - hi[None]).clamp_min(0).square()).sum(-1)
        if "body_accel" in state:
            _check_batch(state, "body_accel", n, (len(self.body_names), 3))
            auxv["pelvis_vert_accel"] = state["body_accel"][:, self.bi["pelvis"], 2].square()
        elif previous_state is not None:
            prev={k:_t(v,self.device) for k,v in previous_state.items()}; accel=(state["root_lin_vel"]-prev["root_lin_vel"])/self.dt; auxv["pelvis_vert_accel"]=accel[:,2].square()
        if "body_accel" in state or "body_accel" in reference:
            if "body_accel" not in state or "body_accel" not in reference:
                raise KeyError("state and reference must both provide body_accel")
            _check_batch(reference, "body_accel", n, (len(self.body_names), 3))
            ei=self._idx(("left_ankle_roll_link","right_ankle_roll_link","left_wrist_yaw_link","right_wrist_yaw_link"),self.bi)
            auxv["ee_accel_mismatch"]=(state["body_accel"][:,ei]-reference["body_accel"][:,ei]).square().sum(dim=(-1, -2))
        force = state["contact_forces"].norm(dim=-1)
        allowed = torch.tensor([x in get("A21").value.allowed_contact_bodies for x in self.body_names], device=self.device)
        threshold=float(get("A21").value.contact_force_threshold)
        auxv["undesired_contact"]=((force > threshold) & ~allowed[None]).sum(-1).float()
        hit=torch.tensor([("head" in x or "torso" in x) for x in self.body_names], device=self.device)
        impact_threshold=float(get("A21").value.head_torso_impact_threshold)
        auxv["head_torso_impact"]=((force-impact_threshold).clamp_min(0).square()*hit[None]).sum(-1)
        aux=sum(w*auxv[k] for k,w in AUX.terms)
        rewards=[upper,lower,aux]; metrics={"upper":um,"lower":lm,"aux":auxv,
                                            "body_pos_error": torch.cat(((
                                                sp[:, self._idx(self.upper_bodies,self.bi)] -
                                                rp[:, self._idx(self.upper_bodies,self.bi)]).norm(dim=-1),
                                                (sp[:, self._idx(self.lower_bodies,self.bi)] -
                                                 rp[:, self._idx(self.lower_bodies,self.bi)]).norm(dim=-1)), dim=1).mean(1)}
        if stage2:
            if terrain is None: raise ValueError("terrain inputs are required for Stage 2")
            terrain_reward, tm=self._terrain(state, reference, previous_state, terrain); rewards.insert(2,terrain_reward); metrics["terrain"]=tm
        return torch.stack(rewards,-1), metrics

    def _terrain(self,state,reference,previous_state,terrain):
        force=state.get("contact_forces"); refc=reference.get("foot_contact",terrain.get("reference_contact"))
        if force is None or refc is None: raise KeyError("Stage 2 requires state.contact_forces and reference.foot_contact")
        fi=self._idx(("left_ankle_roll_link","right_ankle_roll_link"),self.bi); f=force[:,fi]; sim=f.norm(dim=-1)>1.0
        prevf=None if previous_state is None else previous_state.get("contact_forces"); prev= torch.zeros_like(sim) if prevf is None else _t(prevf,self.device)[:,fi].norm(dim=-1)>1.0
        h=terrain.get("foot_height_samples");
        if h is None: raise KeyError("terrain.foot_height_samples is required")
        h=_t(h,self.device); q=torch.exp(-h.std(-1, unbiased=False).square()/float(get("A22").value.sigma_touchdown_quality)); td=sim & ~prev; touchdown=torch.where(td,q,torch.zeros_like(q)).sum(-1)/td.sum(-1).clamp_min(1)
        match=(sim==_t(refc,self.device,torch.bool)).float().mean(-1); vel=state["body_lin_vel"][:,fi,:2]; slip=(vel.square().sum(-1)*sim).sum(-1)
        ff=f; horiz=ff[:,:,:2].norm(dim=-1); vert=ff[:,:,2].clamp_min(0); stumble=(horiz.square()*(horiz>float(get("A22").value.stumble_force_ratio)*vert)).sum(-1)
        age=_t(terrain.get("contact_age",torch.zeros_like(sim,dtype=torch.long)),self.device,torch.float32); switching=((sim!=prev).float()*(1-age.clamp_max(get("A22").value.contact_switching_min_dwell)/get("A22").value.contact_switching_min_dwell)).sum(-1)
        over=(ff.norm(dim=-1)-float(get("A22").value.contact_force_max)).clamp_min(0); force_cost=(over.square()*sim).sum(-1)
        vals={"touchdown_quality":touchdown,"reference_contact_match":match,"slip":slip,"stumble":stumble,"contact_switching":switching,"contact_force":force_cost}
        return sum(w*vals[k] for k,w in TERRAIN.terms),vals
