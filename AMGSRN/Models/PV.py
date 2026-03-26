import torch
import torch.nn as nn
import os
import numpy as np
import pyvista as pv
from math import exp, log
import tinycudann as tcnn
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

        mesh = pv.read(opt["mesh_file"])
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        self.mesh = mesh
    
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
    
    def _apply_cap(self, s):
        r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        r_soft = self.max_scale * torch.tanh(r / self.max_scale)
        return s * (r_soft / r)

    @property
    def get_scaling(self):
        return self._apply_cap(self.scaling_activation(self._scaling))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_weight(self):
        return self.weight_activation(self._weight)

    @property
    def get_values(self):
        return self.values_activation(self._values)
    
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
        return self.opt['full_shape']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        probe_mesh = pv.PolyData(x.cpu().numpy())
        probed = probe_mesh.sample(self.mesh)
        y = probed[self.mesh.array_names[0]]
        valid_mask = probed['vtkValidPointMask'].astype(bool)
        y[~valid_mask] = 0
        y = torch.from_numpy(y).to(x.device)
        print(y.shape)
        print(np.count_nonzero(valid_mask))
        return y.reshape(-1, 1)
