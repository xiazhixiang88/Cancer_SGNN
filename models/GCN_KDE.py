import torch
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
import numpy as np

class GCN_KDE(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 dropout,
                 graph_pooling_type="att",
                 kde_quantiles=np.linspace(0, 1, 20),
                 kde_grid_points=500):
        super(GCN_KDE, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        self.kde_quantiles = torch.tensor(kde_quantiles)
        self.kde_grid_points = kde_grid_points
        self.num_quantiles = len(kde_quantiles)

        self.kde_projectors = nn.ModuleList()
        self.kde_projectors.append(nn.Linear(in_dim * self.num_quantiles, out_dim))
        for _ in range(n_layers):
            self.kde_projectors.append(nn.Linear(hidden_dim * self.num_quantiles, out_dim))

        self.pools = []
        self.linears_prediction = nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, out_dim))

            if graph_pooling_type == 'sum':
                self.pools.append(global_add_pool)
            elif graph_pooling_type == 'mean':
                self.pools.append(global_mean_pool)
            elif graph_pooling_type == 'max':
                self.pools.append(global_max_pool)
            elif graph_pooling_type == 'att':
                gate_nn = torch.nn.Linear(in_dim if layer == 0 else hidden_dim, 1)
                gate_nn.to("cuda" if torch.cuda.is_available() else "cpu")
                self.pools.append(GlobalAttention(gate_nn))

        self.h0 = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(out_dim))

    def _kde_per_graph(self, x, quantiles, grid_points):
        if x.size(0) < 2: 
            return torch.zeros(x.size(1) * len(quantiles), device=x.device)

        min_val = x.min(dim=0)[0] - 1e-6
        max_val = x.max(dim=0)[0] + 1e-6
        grid = torch.linspace(0, 1, grid_points, device=x.device).unsqueeze(1) * (max_val - min_val) + min_val  # [grid_points, hidden_dim]

        std = x.std(dim=0, unbiased=False) + 1e-8 / 3 # [hidden_dim]
        h = 1.06 * std * (x.size(0) ** -0.2) # [hidden_dim]

        u = (grid.unsqueeze(1) - x.unsqueeze(0)) / h.unsqueeze(0)  # [grid_points, num_nodes, hidden_dim]
        kernel = torch.exp(-0.5 * (u ** 2)) / (h * math.sqrt(2 * math.pi)).unsqueeze(0)  # [grid_points, num_nodes, hidden_dim]
        density = kernel.mean(dim=1)  # [grid_points, hidden_dim]

        cdf = torch.cumsum(density, dim=0)  # [grid_points, hidden_dim]
        cdf = cdf / cdf[-1:].clamp(min=1e-8)  # [grid_points, hidden_dim]

        distance = (cdf.unsqueeze(2) - quantiles.to(x.device)).abs()  # [grid_points, hidden_dim, Q]
        weights = torch.sigmoid(- 100 * distance)  # [grid_points, hidden_dim, Q]
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)  # [grid_points, hidden_dim, Q]

        quantile_vals = (grid.unsqueeze(-1) * weights).sum(dim=0)  # [hidden_dim, Q]
        return quantile_vals.flatten()  # [hidden_dim * Q]

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        # Not GNNExplain process, get x, edge_idnex and batch from data.
        if (x is None) and (edge_index is None):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        # Testing process and GNNExplain process, only 1 inputed graph.
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_list = []
        kde_list = []
        current_x = x

        num_graphs = batch.max().item() + 1

        for i in range(self.n_layers):
            current_x = self.dropout(current_x) if i != 0 else current_x

            pooled = self.pools[i](current_x, batch)
            h = self.linears_prediction[i](pooled)
            h_list.append(h)

            kde_features = []
            for graph_idx in range(num_graphs):
                graph_x = current_x[batch == graph_idx]
                kde_feat = self._kde_per_graph(graph_x, self.kde_quantiles, self.kde_grid_points).float()
                kde_features.append(kde_feat)
            kde_features = torch.stack(kde_features)
            projected = self.kde_projectors[i](kde_features)
            kde_list.append(projected)

            current_x = self.layers[i](current_x, edge_index)

        pooled_final = self.pools[-1](current_x, batch)
        h_final = self.classify(pooled_final)
        h_list.append(h_final)

        main_out = torch.stack(h_list).mean(dim=0)
        kde_out = torch.stack(kde_list).mean(dim=0)

        risk_score = torch.matmul(main_out + kde_out, self.beta) + self.h0
        return risk_score