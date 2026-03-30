import torch
import torch.nn as nn
import os
import numpy as np
import pyvista as pv
from math import exp, log
from plyfile import PlyData, PlyElement
from bvh_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class PV(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        mesh = pv.read(os.path.join(opt["path_to_load"], opt["mesh_file"]))
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        self.mesh = mesh

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        self.vol_min = [xmin, ymin, zmin]
        self.vol_max = [xmax, ymax, zmax]
        self.vol_extent = [xmax - xmin, ymax - ymin, zmax - zmin]
        self.register_buffer(
            "volume_min",
            torch.tensor([self.opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([self.opt['data_max']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
    
    def set_default_timestep(self, timestep:int):
        pass

    def get_default_timestep(self):
        return 0

    def prepare_timestep(self, timestep:int):
        pass

    def unload_timestep(self, timestep:int):
        pass

    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max
    
    def get_volume_extents(self):
        return [
            float(self.opt["full_shape"][0] * self.vol_extent[2]),
            float(self.opt["full_shape"][1] * self.vol_extent[1]),
            float(self.opt["full_shape"][2] * self.vol_extent[0]),
        ]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        x[:, 0] = x[:, 0] * self.vol_extent[0] + self.vol_min[0]
        x[:, 1] = x[:, 1] * self.vol_extent[1] + self.vol_min[1]
        x[:, 2] = x[:, 2] * self.vol_extent[2] + self.vol_min[2]
        probe_mesh = pv.PolyData(x.cpu().numpy())
        probed = probe_mesh.sample(self.mesh)
        y = probed[self.mesh.array_names[0]]
        valid_mask = probed['vtkValidPointMask'].astype(bool)
        y[~valid_mask] = 0
        y = torch.from_numpy(y).float().to(x.device)
        return y.reshape(-1, 1)
