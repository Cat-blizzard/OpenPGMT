"""Flat-floor support of URDF collision geometry for an opt-in reset probe.

Uses mesh convex-hull vertices (identical support on a plane), exact spheres
and cylinders. Fixed collision-only links are folded into their tracked parent.
This is geometric clearance, not PhysX's contact/rest-offset distance.
"""
from pathlib import Path
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import torch

from pgmt.envs.reference_motion import _qmat, _rpy


class CollisionFloor:
    def __init__(self, urdf, body_names, device="cpu"):
        import trimesh

        path = Path(urdf)
        root = ET.parse(path).getroot()
        self.device = torch.device(device)
        bodies = {name: i for i, name in enumerate(body_names)}
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}

        def origin(node):
            o = node.find("origin")
            xyz = [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0., 0., 0.]
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0., 0., 0.]
            return _rpy(rpy, torch.device("cpu")).numpy(), np.array(xyz)

        def parent_pose(name):
            if name in bodies:
                return bodies[name], np.eye(3), np.zeros(3)
            joint = joints[name]
            if joint.get("type") != "fixed":
                raise ValueError(f"untracked moving collision link: {name}")
            bi, R, p = parent_pose(joint.find("parent").get("link"))
            r, t = origin(joint)
            return bi, R @ r, p + R @ t

        points, indices, radii, cylinders = [], [], [], []
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                bi, R, p = parent_pose(link.get("name"))
                r, t = origin(collision)
                p, R = p + R @ t, R @ r
                shape = list(collision.find("geometry"))[0]
                radius = 0.
                if shape.tag == "mesh":
                    mesh = trimesh.load_mesh(path.parent / shape.get("filename"), process=False)
                    vertices = np.asarray(mesh.convex_hull.vertices) * np.array(
                        [float(x) for x in shape.get("scale", "1 1 1").split()])
                elif shape.tag == "sphere":
                    vertices = np.zeros((1, 3))
                    radius = float(shape.get("radius"))
                elif shape.tag == "box":
                    half = np.array([float(x) for x in shape.get("size").split()]) / 2
                    vertices = np.array(list(itertools.product((-1, 1), repeat=3))) * half
                elif shape.tag == "cylinder":
                    cylinders.append((bi, p, R[:, 2], float(shape.get("radius")), float(shape.get("length"))/2))
                    continue
                else:
                    raise ValueError(f"unsupported reset collision geometry: {shape.tag}")
                points.extend(vertices @ R.T + p)
                indices.extend([bi] * len(vertices))
                radii.extend([radius] * len(vertices))
        if not points and not cylinders:
            raise ValueError("reset clearance requires collision geometry")
        self.points = torch.tensor(np.asarray(points).reshape(-1, 3), device=self.device, dtype=torch.float32)
        self.indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.radii = torch.tensor(radii, device=self.device, dtype=torch.float32)
        self.cylinders = [(bi, torch.tensor(p, device=self.device, dtype=torch.float32),
                           torch.tensor(axis, device=self.device, dtype=torch.float32), r, h)
                          for bi, p, axis, r, h in cylinders]

    def min_height(self, body_pos, body_quat):
        rotation = _qmat(body_quat)
        minima = []
        if len(self.points):
            z = body_pos[:, self.indices, 2] + (rotation[:, self.indices, 2] * self.points).sum(-1) - self.radii
            minima.append(z.min(-1).values)
        for bi, center, axis, radius, half_length in self.cylinders:
            z = body_pos[:, bi, 2] + (rotation[:, bi, 2] * center).sum(-1)
            cosine = (rotation[:, bi, 2] * axis).sum(-1).clamp(-1, 1)
            support = half_length*cosine.abs() + radius*(1-cosine.square()).clamp_min(0).sqrt()
            minima.append(z-support)
        return torch.stack(minima).min(0).values
