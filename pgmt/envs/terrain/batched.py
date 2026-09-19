"""Device-resident queries against the same triangle mesh used by physics.

The atlas has five columns (families) and ten rows (levels). Robots belonging
to different collision groups may share a tile, as in standard vectorized
locomotion environments. Mesh resolution is an engineering assumption: sharp
step faces are ramps one grid cell wide (default 2.5 cm), not analytic steps.
"""
from __future__ import annotations

import numpy as np
import torch

from pgmt.envs.terrain.generators import FAMILIES, TILE_SIZE, params_for, sample_grid


class TerrainAtlas:
    def __init__(self, device="cpu", stage=2, resolution=.025):
        self.device = torch.device(device)
        self.stage = stage
        self.size = TILE_SIZE
        self.n = round(TILE_SIZE / resolution) + 1
        if self.n < 3:
            raise ValueError("terrain resolution must be smaller than half a tile")
        self.spacing = TILE_SIZE / (self.n - 1)
        grids = [sample_grid(params_for(f, l), self.n)[2]
                 for l in range(10) for f in FAMILIES]
        self.heights = torch.tensor(np.stack(grids), dtype=torch.float32, device=self.device)

    def origins(self, families, levels):
        return torch.stack((families * self.size, levels * self.size,
                            torch.zeros_like(levels)), -1).to(self.device, torch.float32)

    def query(self, xy):
        """World XY -> world Z, using the exact mesh triangles (not bilinear)."""
        if self.stage == 1:
            return torch.zeros_like(xy[..., 0])
        tile = torch.floor((xy + self.size / 2) / self.size).long()
        family, level = tile.unbind(-1)
        valid = (family >= 0) & (family < 5) & (level >= 0) & (level < 10)
        local = xy - tile * self.size
        pixel = ((local + self.size / 2) / self.spacing).clamp(0, self.n - 1 - 1e-5)
        ij = pixel.floor().long()
        x, y = ij.unbind(-1)
        u, v = (pixel - ij).unbind(-1)
        grid = (level.clamp(0, 9) * 5 + family.clamp(0, 4))
        z00 = self.heights[grid, y, x]
        z10 = self.heights[grid, y, x+1]
        z01 = self.heights[grid, y+1, x]
        z11 = self.heights[grid, y+1, x+1]
        lo = z00 + u * (z10-z00) + v * (z11-z10)
        hi = z00 + v * (z01-z00) + u * (z11-z01)
        return torch.where(valid, torch.where(u >= v, lo, hi), torch.zeros_like(u))

    def elevation(self, root_pos, root_quat, *, families=None, levels=None, corrupt=False):
        if families is None:
            families = torch.zeros(root_pos.shape[0], device=self.device, dtype=torch.long)
        if levels is None:
            levels = torch.zeros(root_pos.shape[0], device=self.device, dtype=torch.long)
        families = families.to(self.device, torch.long)
        levels = levels.to(self.device, torch.long)
        origin = torch.stack((families * self.size, levels * self.size), -1).float()
        local_root = root_pos[:, :2] - origin
        axis = torch.linspace(-1, 1, 21, device=self.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        w, qx, qy, qz = root_quat.unbind(-1)
        yaw = torch.atan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
        xy = torch.stack((local_root[:, 0, None, None]+c*x-s*y + origin[:, 0, None, None],
                          local_root[:, 1, None, None]+s*x+c*y + origin[:, 1, None, None]), -1)
        out = self.query(xy) - root_pos[:, 2, None, None]
        if corrupt:
            from pgmt.cfg.assumptions import get
            cfg = get("A8").value
            f = levels.float()[:, None, None] / 9
            out = out + torch.randn_like(out) * (cfg.sigma_min + f*(cfg.sigma_max-cfg.sigma_min))
            out = out.masked_fill(torch.rand_like(out) < cfg.dropout_prob_max*f, 0)
        return out

    def foot_samples(self, foot_pos):
        axis = torch.linspace(-.1, .1, 5, device=self.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        offset = torch.stack((x.flatten(), y.flatten()), -1)
        return self.query(foot_pos[:, :, None, :2] + offset)

    def mesh(self):
        """CPU vertices/faces, shared between physics and CPU query tests."""
        axis = np.linspace(-self.size/2, self.size/2, self.n, dtype=np.float32)
        y, x = np.meshgrid(axis, axis, indexing="ij")
        ids = np.arange(self.n*self.n).reshape(self.n, self.n)
        a, b = ids[:-1, :-1].flatten(), ids[:-1, 1:].flatten()
        c, d = ids[1:, :-1].flatten(), ids[1:, 1:].flatten()
        faces = np.concatenate((np.stack((a,b,d),-1), np.stack((a,d,c),-1)))
        grids = self.heights.cpu().numpy()
        vertices, triangles = [], []
        for tile, h in enumerate(grids):
            vertices.append(np.stack((x+tile%5*self.size, y+tile//5*self.size, h), -1).reshape(-1,3))
            triangles.append(faces + tile*self.n*self.n)
        return np.concatenate(vertices), np.concatenate(triangles)
