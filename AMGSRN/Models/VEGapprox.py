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

class VEG(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        mesh = pv.read(os.path.join(opt['path_to_load'], opt["mesh_file"]))
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0 / (global_max - global_min), inplace=True)
        self.mesh = mesh

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
        
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        self.vol_min = [xmin, ymin, zmin]
        self.vol_max = [xmax, ymax, zmax]
        self.vol_extent = [xmax - xmin, ymax - ymin, zmax - zmin]
        print(self.vol_min, self.vol_max)

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
       
        xyz_t = torch.tensor(xyz, dtype=torch.float, device="cuda")
        weight_t = torch.tensor(weights, dtype=torch.float, device="cuda")
        scaling_t = torch.tensor(scales, dtype=torch.float, device="cuda")
        rotation_t = torch.tensor(rots, dtype=torch.float, device="cuda")
        values_t = torch.tensor(values, dtype=torch.float, device="cuda")

        # Sort by Morton code for spatial coherence
        def morton3d_unit(xyz: torch.Tensor) -> torch.Tensor:
            """Morton codes for points already in [0, 1]^3."""
            norm = (xyz * ((1 << 21) - 1)).long()

            def part1by2(n):
                n = n & 0x1fffff
                n = (n | (n << 32)) & 0x1f00000000ffff
                n = (n | (n << 16)) & 0x1f0000ff0000ff
                n = (n | (n << 8))  & 0x100f00f00f00f00f
                n = (n | (n << 4))  & 0x10c30c30c30c30c3
                n = (n | (n << 2))  & 0x1249249249249249
                return n

            return part1by2(norm[:, 0]) | (part1by2(norm[:, 1]) << 1) | (part1by2(norm[:, 2]) << 2)

        order = torch.argsort(morton3d_unit(xyz_t))
        xyz_t = xyz_t[order]
        weight_t = weight_t[order]
        scaling_t = scaling_t[order]
        rotation_t = rotation_t[order]
        values_t = values_t[order]

        # Now wrap in nn.Parameter
        self._xyz = nn.Parameter(xyz_t.requires_grad_(True))
        self._weight = nn.Parameter(weight_t.requires_grad_(True))
        self._scaling = nn.Parameter(scaling_t.requires_grad_(True))
        self._rotation = nn.Parameter(rotation_t.requires_grad_(True))
        self._values = nn.Parameter(values_t.requires_grad_(True))

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

        # --- Precompute validity mask on the full grid (one-time cost) ---
        # full_shape is [Nz, Ny, Nx]
        fs = opt['full_shape']
        self._grid_shape = (int(fs[0]), int(fs[1]), int(fs[2]))
        Nz, Ny, Nx = self._grid_shape

        print(f"Precomputing validity mask on grid {Nz}x{Ny}x{Nx} ...")

        # Build the full grid in volume coordinates, matching forward()'s mapping:
        #   col 0 (x-coord) uses vol_extent[0], col 1 (y) uses [1], col 2 (z) uses [2]
        # full_shape dim ordering: [z, y, x]
        x_lin = np.linspace(0, 1, Nx, dtype=np.float32) * self.vol_extent[0] + self.vol_min[0]
        y_lin = np.linspace(0, 1, Ny, dtype=np.float32) * self.vol_extent[1] + self.vol_min[1]
        z_lin = np.linspace(0, 1, Nz, dtype=np.float32) * self.vol_extent[2] + self.vol_min[2]

        # meshgrid with indexing='ij' gives shape [Nz, Ny, Nx] for each
        zz, yy, xx = np.meshgrid(z_lin, y_lin, x_lin, indexing='ij')
        grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Probe the mesh in chunks to limit peak memory
        chunk_size = 2_000_000
        n_pts = grid_pts.shape[0]
        mask_full = np.empty(n_pts, dtype=bool)
        for start in range(0, n_pts, chunk_size):
            end = min(start + chunk_size, n_pts)
            probe_mesh = pv.PolyData(grid_pts[start:end])
            probed = probe_mesh.sample(self.mesh)
            mask_full[start:end] = probed['vtkValidPointMask'].astype(bool)

        # Store as a flat bool tensor on GPU. Flat index = iz*Ny*Nx + iy*Nx + ix
        self.register_buffer(
            "_valid_mask",
            torch.tensor(mask_full, dtype=torch.bool, device="cuda"),
            persistent=False,
        )
        print(f"Validity mask precomputed: {mask_full.sum()}/{n_pts} valid points")

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
        return [
            float(self.opt['full_shape'][0] * self.vol_extent[2]), 
            float(self.opt['full_shape'][1] * self.vol_extent[1]), 
            float(self.opt['full_shape'][2] * self.vol_extent[0]), 
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nz, Ny, Nx = self._grid_shape

        # Map from [-1, 1] to grid indices (entirely on GPU, no CPU roundtrip)
        x01 = (x + 1.0) * 0.5  # [-1,1] -> [0,1]
        ix = (x01[:, 0] * (Nx - 1)).round().long()
        iy = (x01[:, 1] * (Ny - 1)).round().long()
        iz = (x01[:, 2] * (Nz - 1)).round().long()

        # Clamp to valid range (handles floating point edge cases)
        ix = ix.clamp(0, Nx - 1)
        iy = iy.clamp(0, Ny - 1)
        iz = iz.clamp(0, Nz - 1)

        flat_idx = iz * (Ny * Nx) + iy * Nx + ix
        valid_mask = self._valid_mask[flat_idx]

        # Transform x to volume coordinates for the rasterizer
        x[:, 0] = x01[:, 0] * self.vol_extent[0] + self.vol_min[0]
        x[:, 1] = x01[:, 1] * self.vol_extent[1] + self.vol_min[1]
        x[:, 2] = x01[:, 2] * self.vol_extent[2] + self.vol_min[2]

        self._rasterizer.build_bvh(x, False, False)
        means3D = self.get_xyz
        scales = self.get_scaling
        rotations = self.get_rotation
        values = self.get_values
        weights = self.get_weight

        y, __ = self._rasterizer(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            values=values,
            weights=weights,
            debug=False
        )
        y[~valid_mask] = 0
        return y.reshape(-1, 1)