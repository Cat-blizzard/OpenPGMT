"""Replace heuristic contact labels by offline reference-mesh queries.

The paper does not publish the reference terrain meshes. The explicit default
is a flat source mesh at z=0 for LAFAN1, independent of the Stage 2 terrain.
An external regular heightfield NPZ (heights, origin_xy, spacing) is supported.
Foot geometry is taken from the supplied URDF's collision spheres. No velocity
threshold is used. Query tolerance (2 cm) remains an implementation assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from pgmt.envs.reference_sampler import MotionDatabase
from pgmt.envs.reference_motion import ReferenceMotion, _qmat

CONTACT_SOURCE = "offline_reference_mesh_v2"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class ReferenceMesh:
    def __init__(self, path=None):
        if path is None:
            self.heights = torch.zeros(2, 2)
            self.origin = torch.tensor([-10000., -10000.])
            self.spacing = 20000.
            self.provenance = {"kind": "flat_reference_mesh", "z": 0., "author_asset": False}
        else:
            with np.load(path) as d:
                self.heights = torch.as_tensor(d["heights"], dtype=torch.float32)
                self.origin = torch.as_tensor(d["origin_xy"], dtype=torch.float32)
                self.spacing = float(d["spacing"])
            if self.heights.ndim != 2 or min(self.heights.shape) < 2 or self.origin.shape != (2,) or self.spacing <= 0:
                raise ValueError("invalid heightfield mesh")
            self.provenance = {"kind": "heightfield_mesh", "path": str(Path(path).resolve()),
                               "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        if not torch.isfinite(self.heights).all():
            raise ValueError("non-finite reference mesh")

    def query(self, xy):
        pixel = (xy.cpu() - self.origin) / self.spacing
        h, w = self.heights.shape
        if ((pixel < 0).any() or (pixel[..., 0] >= w - 1).any() or (pixel[..., 1] >= h - 1).any()):
            raise ValueError("reference foot lies outside the supplied mesh")
        ij = pixel.floor().long()
        x, y = ij.unbind(-1)
        u, v = (pixel - ij).unbind(-1)
        a, b, c, d = self.heights[y, x], self.heights[y, x+1], self.heights[y+1, x], self.heights[y+1, x+1]
        return torch.where(u >= v, a + u*(b-a) + v*(d-b), a + v*(c-a) + u*(d-c))


def sphere_geometry(urdf):
    tree = ET.parse(urdf).getroot()
    result = []
    for name in FEET:
        link = tree.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"missing foot: {name}")
        centers, radii = [], []
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is None:
                raise ValueError("this offline tool requires the G1 collision-sphere feet")
            origin = collision.find("origin")
            centers.append([float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.])
            radii.append(float(sphere.get("radius")))
        if not centers:
            raise ValueError(f"no foot collision geometry: {name}")
        result.append((torch.tensor(centers), torch.tensor(radii)))
    return result


def contact_labels(body_pos, body_quat, body_names, geometry, mesh, tolerance=.02):
    labels = []
    for name, (centers, radii) in zip(FEET, geometry):
        i = body_names.index(name)
        points = body_pos[:, i, None] + torch.einsum("bij,kj->bki", _qmat(body_quat[:, i]), centers)
        clearance = points[..., 2] - radii - mesh.query(points[..., :2])
        labels.append((clearance <= tolerance).any(-1))
    return torch.stack(labels, -1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--mesh", type=Path)
    p.add_argument("--tolerance", type=float, default=.02)
    a = p.parse_args(argv)
    if a.input.resolve() == a.output.resolve() or a.output.exists():
        p.error("output must be a new directory; source motions are preserved")
    if a.tolerance < 0:
        p.error("tolerance must be nonnegative")
    mesh, geometry = ReferenceMesh(a.mesh), sphere_geometry(a.urdf)
    database = MotionDatabase(str(a.input))
    a.output.mkdir(parents=True)
    provenance = {"source": CONTACT_SOURCE, "mesh": mesh.provenance, "tolerance_m": a.tolerance,
                  "urdf_sha256": hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
                  "sequence_names": [s["name"] for s in database.seqs]}
    for sequence in database.seqs:
        single = MotionDatabase.from_sequences([sequence])
        reference = ReferenceMotion(single, a.urdf)
        n = len(sequence["qpos"])
        chunks = []
        for start in range(0, n, 1024):
            frames = torch.arange(start, min(start + 1024, n), dtype=torch.float32)
            body = reference.sample(torch.zeros(len(frames), dtype=torch.long), frames)
            chunks.append(contact_labels(body["body_pos"], body["body_quat"], list(reference._body_order),
                                         geometry, mesh, a.tolerance))
        data = {k: v for k, v in sequence.items() if k != "name"}
        if "contacts" in data:
            data["contacts_heuristic"] = data["contacts"]
        data["contacts"] = torch.cat(chunks).numpy()
        data["contact_source"] = np.array(CONTACT_SOURCE)
        data["contact_provenance"] = np.array(json.dumps(provenance))
        np.savez_compressed(a.output / (sequence["name"] + ".npz"), **data)
        print(sequence["name"], n, flush=True)
    (a.output / "contact_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
