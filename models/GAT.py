"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

# import torch
# import torch.nn as nn

# import dgl
# from dgl.nn import GATConv
# from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
# from torch_geometric.data import Data


# class GAT(nn.Module):
#     def __init__(self,
#                  n_layers,
#                  in_dim,
#                  hidden_dim,
#                  out_dim,
#                  heads,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual,
#                  graph_pooling_type="att"):
#         super(GAT, self).__init__()
#         self.n_layers = n_layers
#         self.layers = nn.ModuleList()
#         self.activation = activation

#         for l in range(n_layers + 1):
#             if l == 0:
#                 # input projection (no residual)
#                 self.layers.append(GATConv(
#                     in_dim, hidden_dim, heads[0],
#                     feat_drop, attn_drop, negative_slope, False, self.activation))
#             elif l == n_layers:  # hidden layers
#                 # output projection
#                 self.layers.append(GATConv(
#                     hidden_dim * heads[-2], out_dim, heads[-1],
#                     feat_drop, attn_drop, negative_slope, residual, None))
#             else:
#                 # due to multi-head, the in_dim = num_hidden * num_heads
#                 self.layers.append(GATConv(
#                     hidden_dim * heads[l-1], hidden_dim, heads[l],
#                     feat_drop, attn_drop, negative_slope, residual, self.activation))

#         # Linear function for graph poolings of output of each layer
#         # which maps the output of different layers into a prediction score
#         self.linears_prediction = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         for layer in range(n_layers + 1):
#             if layer == 0:
#                 self.linears_prediction.append(
#                     nn.Linear(in_dim, out_dim))
#             else:
#                 self.linears_prediction.append(
#                     nn.Linear(hidden_dim * heads[layer-1], out_dim))

#             if graph_pooling_type == 'sum':
#                 self.pools.append(SumPooling())
#             elif graph_pooling_type == 'mean':
#                 self.pools.append(AvgPooling())
#             elif graph_pooling_type == 'max':
#                 self.pools.append(MaxPooling())
#             elif graph_pooling_type == 'att':
#                 if layer == 0:
#                     gate_nn = torch.nn.Linear(in_dim, 1)
#                 else:
#                     gate_nn = torch.nn.Linear(hidden_dim * heads[layer-1], 1)
#                 self.pools.append(GlobalAttentionPooling(gate_nn))
#             else:
#                 raise NotImplementedError

#     def forward(self, g, h=None):
#         if h is None:
#             h = g.ndata['feat']

#         h_list = []
#         for i, layer in enumerate(self.layers):
#             pool_h = self.pools[i](g, h)
#             pool_h = self.linears_prediction[i](pool_h)
#             h_list.append(pool_h)
#             h = layer(g, h).flatten(1)

#         out = torch.stack(h_list).mean(0)

#         return out

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Data

class GAT(torch.nn.Module):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 activation=torch.nn.ReLU(),
                 feat_drop=0.2,
                 attn_drop=0.2,
                 n_heads=4,
                 graph_pooling_type="att"):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.feat_drop = nn.Dropout(p=feat_drop)
        heads = [n_heads] * (n_layers)

        self.convs = torch.nn.ModuleList()
        for l in range(n_layers):
            in_channels = in_dim if l == 0 else hidden_dim * heads[l-1]
            out_channels = hidden_dim
            heads_current = heads[l]
            self.convs.append(
                GATConv(in_channels, out_channels, heads=heads_current,
                        dropout=attn_drop, concat=True)
            )

        self.pools = []
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(n_layers + 1):
            if graph_pooling_type == 'sum':
                self.pools.append(global_add_pool)
            elif graph_pooling_type == 'mean':
                self.pools.append(global_mean_pool)
            elif graph_pooling_type == 'max':
                self.pools.append(global_max_pool)
            elif graph_pooling_type == 'att':
                if layer == 0:
                    gate_nn = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, 1),
                        torch.nn.Sigmoid()
                    )
                else:
                    gate_nn = torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim * heads[layer-1], 1),
                        torch.nn.Sigmoid()
                    )
                gate_nn.to("cuda" if torch.cuda.is_available() else "cpu")
                self.pools.append(GlobalAttention(gate_nn))

            linear_in = in_dim if layer == 0 else hidden_dim * heads[layer-1]
            self.linears_prediction.append(torch.nn.Linear(linear_in, out_dim))
        
        self.h_weights = torch.nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))

        self.h0 = torch.nn.Parameter(torch.randn(1))
        self.beta = torch.nn.Parameter(torch.randn(out_dim))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h_list = []
        for i in range(self.n_layers):
            current_x = self.feat_drop(x) if i != 0 else x
            h = self.linears_prediction[i](self.pools[i](current_x, batch))
            h_list.append(h)
            x = self.convs[i](x, edge_index)
            if self.activation is not None and i != self.n_layers-1:
                x = self.activation(x)
        h_list.append(self.linears_prediction[-1](self.pools[-1](x, batch)))

        out = torch.sum(torch.stack(h_list).view(self.n_layers+1, -1) * self.h_weights.view(-1, 1), dim=0)
        risk_score = torch.matmul(out, self.beta) + self.h0
        
        return risk_score