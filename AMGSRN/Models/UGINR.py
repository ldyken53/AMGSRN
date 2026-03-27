import os
import json
import pickle
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
from sklearn.cluster import KMeans


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
            #
        #
    #

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    #
#

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
        #
    #

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)
    #
#

def compute_num_neurons(opt,target_size):
    # relevant options
    d_in = opt.d_in
    d_out = opt.d_out

    def network_size(neurons):
        layers = [d_in]
        layers.extend([neurons]*opt.n_layers)
        layers.append(d_out)
        n_layers = len(layers)-1

        n_params = 0
        for ndx in np.arange(n_layers):
            layer_in = layers[ndx]
            layer_out = layers[ndx+1]
            og_layer_in = max(layer_in,layer_out)

            if ndx==0 or ndx==(n_layers-1):
                n_params += ((layer_in+1)*layer_out)
            #
            else:
                if opt.is_residual:
                    is_shortcut = layer_in != layer_out
                    if is_shortcut:
                        n_params += (layer_in*layer_out)+layer_out
                    n_params += (layer_in*og_layer_in)+og_layer_in
                    n_params += (og_layer_in*layer_out)+layer_out
                else:
                    n_params += ((layer_in+1)*layer_out)
                #
            #
        #

        return n_params
    #

    min_neurons = 3
    while network_size(min_neurons) < target_size:
        min_neurons+=1
    min_neurons-=1

    return min_neurons
#

class FieldNet(nn.Module):
    def __init__(self, opt):
        super(FieldNet, self).__init__()

        self.d_in = opt.d_in
        self.layers = [self.d_in]
        self.layers.extend(opt.layers)
        self.d_out = opt.d_out
        self.layers.append(self.d_out)
        self.n_layers = len(self.layers)-1
        self.w0 = opt.w0
        self.is_residual = opt.is_residual

        self.net_layers = nn.ModuleList()
        for ndx in np.arange(self.n_layers):
            layer_in = self.layers[ndx]
            layer_out = self.layers[ndx+1]
            if ndx != self.n_layers-1:
                if not self.is_residual:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                    continue
                #

                if ndx==0:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=ndx==0))
                else:
                    self.net_layers.append(ResidualSineLayer(layer_in,bias=True,ave_first=ndx>1,ave_second=ndx==(self.n_layers-2)))
                #
            else:
                final_linear = nn.Linear(layer_in,layer_out)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / 30.0, np.sqrt(6 / (layer_in)) / 30.0)
                self.net_layers.append(final_linear)
            #
        #
    #

    def forward(self,input):
        batch_size = input.shape[0]
        out = input
        for ndx,net_layer in enumerate(self.net_layers):
            out = net_layer(out)
        #
        return out
    #
#


class UGINR(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.device = torch.device(opt.get("device", "cpu"))

        mesh = pv.read(os.path.join(opt["path_to_load"], opt["mesh_file"]))
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0 / (global_max - global_min), inplace=True)
        self.mesh = mesh

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        self.vol_extent = [xmax, ymax, zmax]

        points = torch.from_numpy(mesh.points.astype(np.float32))

        kmeans_path = os.path.join(opt["path_to_load"], opt["kmeans_file"])
        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)

        net_path = os.path.join(opt["path_to_load"], opt["net_file"])
        net_value = torch.load(net_path, map_location=self.device)

        config = None
        if "config_file" in opt:
            config_path = os.path.join(opt["path_to_load"], opt["config_file"])
            with open(config_path, "r") as f:
                config = json.load(f)

        self.n_clusters = self.kmeans.n_clusters
        self.cluster_norm_params = {}
        self.nets = nn.ModuleDict()

        density_regions = self._build_cluster_regions(points, self.kmeans)

        for cid in range(self.n_clusters):
            pointdata4 = density_regions[cid]

            min_bb, max_bb, scales = self._prepare_normalized_data(pointdata4, self.device)

            self.cluster_norm_params[str(cid)] = {
                "min_bb": min_bb,
                "max_bb": max_bb,
                "scales": scales,
            }

            net = self._build_net_for_cluster(cid, config)

            state_dict = net_value[cid] if cid in net_value else net_value[str(cid)]
            net.load_state_dict(state_dict)
            net.to(self.device)
            net.eval()

            self.nets[str(cid)] = net
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

    def _build_cluster_regions(self, points, kmeans):
        density_regions = {}
        groups = kmeans.labels_

        for i in range(kmeans.n_clusters):
            group_indices = np.where(groups == i)[0]
            density_regions[i] = points[group_indices]

        return density_regions

    def _prepare_normalized_data(self, points, device):
        pts = points.to(device)

        min_bb = pts.min(dim=0).values
        max_bb = pts.max(dim=0).values

        denominator = max_bb - min_bb
        denominator = torch.where(
            denominator == 0,
            torch.tensor(1e-9, device=device, dtype=pts.dtype),
            denominator,
        )

        scales = denominator / denominator.max()
        return min_bb, max_bb, scales

    def _make_opt_for_net(self, layers_for_cluster, config=None):
        class NetOpt:
            pass

        net_opt = NetOpt()
        net_opt.d_in = self.opt.get("d_in", 3)
        net_opt.d_out = self.opt.get("d_out", 1)
        net_opt.w0 = config.get("w0", self.opt.get("w0", 30.0)) if config else self.opt.get("w0", 30.0)
        net_opt.is_residual = (
            config.get("is_residual", self.opt.get("is_residual", True))
            if config else self.opt.get("is_residual", True)
        )
        net_opt.n_layers = config.get("n_layers", len(layers_for_cluster)) if config else len(layers_for_cluster)
        net_opt.layers = layers_for_cluster
        return net_opt

    def _build_net_for_cluster(self, cid, config=None):
        if config is None:
            if "layers" not in self.opt:
                raise ValueError(
                    "Either opt['config_file'] or opt['layers'] must be provided to rebuild FieldNet."
                )
            layers_total = self.opt["layers"]
        else:
            layers_total = config["layers"]

        layers_for_cluster = (
            layers_total[str(cid)] if str(cid) in layers_total else layers_total[cid]
        )

        net_opt = self._make_opt_for_net(layers_for_cluster, config)
        return FieldNet(net_opt)

    def set_default_timestep(self, timestep: int):
        pass

    def get_default_timestep(self):
        return 0

    def prepare_timestep(self, timestep: int):
        pass

    def unload_timestep(self, timestep: int):
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
        device = x.device
        original_shape = x.shape[:-1]
        x = x.reshape(-1, 3)

        x = (x + 1.0) / 2.0
        x = x.clone()
        x[:, 0] *= self.vol_extent[0]
        x[:, 1] *= self.vol_extent[1]
        x[:, 2] *= self.vol_extent[2]

        x_np = x.detach().cpu().numpy().astype(np.float64)
        cluster_labels = self.kmeans.predict(x_np)

        y = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=device)

        for cid in range(self.n_clusters):
            mask = cluster_labels == cid
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            cluster_xyz = x[idx].to(self.device)

            nparams = self.cluster_norm_params[str(cid)]
            min_bb = nparams["min_bb"]
            max_bb = nparams["max_bb"]
            scales = nparams["scales"]

            denominator = max_bb - min_bb
            denominator = torch.where(
                denominator == 0,
                torch.tensor(1e-9, device=self.device, dtype=cluster_xyz.dtype),
                denominator,
            )

            normalized_xyz = 2.0 * (
                (cluster_xyz - min_bb.unsqueeze(0)) / denominator.unsqueeze(0)
            ) - 1.0
            normalized_xyz = scales.unsqueeze(0) * normalized_xyz

            net = self.nets[str(cid)]
            preds = net(normalized_xyz)

            y[idx] = preds.to(device=device, dtype=y.dtype)

        return y.reshape(*original_shape, 1)