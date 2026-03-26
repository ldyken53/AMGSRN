import torch
import torch.nn as nn
import os
import numpy as np
from math import exp, log
import tinycudann as tcnn
from plyfile import PlyData, PlyElement
from bvh_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class VEGS(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        plydata = PlyData.read(os.path.join(opt['path_to_load'], opt["ply_file"]))
        print(
            f"Number of points at initialisation : {plydata.elements[0]['x'].shape[0]}"
        )
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        weights = np.asarray(plydata.elements[0]["weight"])[..., np.newaxis]

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        values = np.asarray(plydata.elements[0]["value"])[..., np.newaxis]

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._weight = nn.Parameter(
            torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._values = nn.Parameter(
            torch.tensor(values, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        raster_settings = GaussianRasterizationSettings(
            volume_mins=[0, 0, 0],
            volume_maxes=[1, 1, 1],
            cell_count=100,
            bg=-1.0,
            scale_modifier=1.0,
            use_gaussian_bvh=False,
            debug=False,
        )
        self._rasterizer = GaussianRasterizer(
            raster_settings=raster_settings,
        )

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

        self.scaling_activation = torch.exp
        self.inverse_scaling_activation = torch.log

        self.weight_activation = torch.sigmoid
        self.inverse_weight_activation = inverse_sigmoid

        self.values_activation = torch.sigmoid
        self.inverse_value_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.max_scale = 0.02
    
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
        # Sort spatially via Morton code (Z-order curve)
        # def part1by2_torch(n: torch.Tensor) -> torch.Tensor:
        #     # n: int64
        #     n = n & 0x1fffff
        #     n = (n | (n << 32)) & 0x1f00000000ffff
        #     n = (n | (n << 16)) & 0x1f0000ff0000ff
        #     n = (n | (n << 8))  & 0x100f00f00f00f00f
        #     n = (n | (n << 4))  & 0x10c30c30c30c30c3
        #     n = (n | (n << 2))  & 0x1249249249249249
        #     return n

        # scale = (1 << 21) - 1
        # norm = torch.clamp(x, 0.0, 1.0)
        # q = (norm * scale).to(torch.int64)
        # morton = (
        #     part1by2_torch(q[:, 0])
        #     | (part1by2_torch(q[:, 1]) << 1)
        #     | (part1by2_torch(q[:, 2]) << 2)
        # )     
        # order = morton.argsort(dim=0)
        # x = x[order]
        # def morton3d_unit(xyz: torch.Tensor) -> torch.Tensor:
        #     """Morton codes for points already in [0, 1]^3."""
        #     norm = (xyz * ((1 << 21) - 1)).long()

        #     def part1by2(n):
        #         n = n & 0x1fffff
        #         n = (n | (n << 32)) & 0x1f00000000ffff
        #         n = (n | (n << 16)) & 0x1f0000ff0000ff
        #         n = (n | (n << 8))  & 0x100f00f00f00f00f
        #         n = (n | (n << 4))  & 0x10c30c30c30c30c3
        #         n = (n | (n << 2))  & 0x1249249249249249
        #         return n

        #     return part1by2(norm[:, 0]) | (part1by2(norm[:, 1]) << 1) | (part1by2(norm[:, 2]) << 2)
        
        self._rasterizer.build_bvh(x, False, False)
        means3D = self.get_xyz
        scales = self.get_scaling
        rotations = self.get_rotation
        values = self.get_values
        weights = self.get_weight

        # Sort Gaussians by Morton code for spatial coherence
        # order = torch.argsort(morton3d_unit(means3D))
        # means3D = means3D[order]
        # scales = scales[order]
        # rotations = rotations[order]
        # values = values[order]
        # weights = weights[order]

        y, __ = self._rasterizer(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            values=values,
            weights=weights,
            debug=False
        )
        print(torch.count_nonzero(y < 0))
        return y.reshape(-1, 1)
