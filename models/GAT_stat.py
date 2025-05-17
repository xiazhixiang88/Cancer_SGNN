import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, GlobalAttention
import scipy.stats as stats

class GAT_stat(torch.nn.Module):
    def __init__(self,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 pooling_list,
                 activation=torch.nn.ReLU(),
                 feat_drop=0.2,
                 attn_drop=0.2,
                 n_heads=4):
        super(GAT_stat, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.feat_drop = nn.Dropout(p=feat_drop)
        heads = [n_heads] * (n_layers)
        self.pooling_list = pooling_list

        self.convs = torch.nn.ModuleList()
        for l in range(n_layers):
            in_channels = in_dim if l == 0 else hidden_dim * heads[l - 1]
            out_channels = hidden_dim
            heads_current = heads[l]
            self.convs.append(
                GATConv(in_channels, out_channels, heads=heads_current,
                        dropout=attn_drop, concat=True)
            )

        self.att_gate_nns = torch.nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 1),
                    torch.nn.Sigmoid()
                )
            else:
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * heads[layer - 1], 1),
                    torch.nn.Sigmoid()
                )
            self.att_gate_nns.append(gate_nn)

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(n_layers + 1):
            linear_in = in_dim if layer == 0 else hidden_dim * heads[layer - 1]
            self.linears_prediction.append(torch.nn.Linear(linear_in, out_dim))

        self.h_weights = torch.nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))

        self.h0 = torch.nn.Parameter(torch.randn(1))
        self.beta = torch.nn.Parameter(torch.randn(out_dim))

        self.pooling_functions = {
            'att': lambda x, i, batch: GlobalAttention(self.att_gate_nns[i])(x, batch),
            'mean': global_mean_pool,
            'max': global_max_pool,
            'min': lambda x, _, batch: torch.cat([x[batch == i].min(dim=0)[0].unsqueeze(0) for i in range(batch.max().item() + 1)]),
            'skewness': lambda x, _, batch: torch.tensor(stats.skew(x.detach().cpu().numpy(), axis=0), dtype=torch.float32).unsqueeze(0).to(x.device),
            'kurtosis': lambda x, _, batch: torch.tensor(stats.kurtosis(x.detach().cpu().numpy(), axis=0), dtype=torch.float32).unsqueeze(0).to(x.device)
        }

        self.pooling_weights = nn.Parameter(torch.ones(len(pooling_list)) / len(pooling_list))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 提取batch

        h_list = []
        for i in range(self.n_layers):
            current_x = self.feat_drop(x) if i != 0 else x

            pooled_features = []
            for pooling in self.pooling_list:
                if pooling == 'att':
                    pooled = self.pooling_functions[pooling](current_x, i, batch)
                else:
                    pooled = self.pooling_functions[pooling](current_x, batch)
                pooled_features.append(pooled)
            pooled_features = torch.stack(pooled_features, dim=0)  # [num_pooling, num_graphs, hidden_dim]

            weighted_pooled = torch.sum(pooled_features * self.pooling_weights.unsqueeze(1).unsqueeze(2), dim=0)

            h = self.linears_prediction[i](weighted_pooled)
            h_list.append(h)

            x = self.convs[i](x, edge_index)
            if self.activation is not None and i != self.n_layers - 1:
                x = self.activation(x)

        pooled_features = []
        for pooling in self.pooling_list:
            if pooling == 'att':
                pooled = self.pooling_functions[pooling](x, -1, batch)
            else:
                pooled = self.pooling_functions[pooling](x, batch)
            pooled_features.append(pooled)
        pooled_features = torch.stack(pooled_features, dim=0)
        weighted_pooled = torch.sum(pooled_features * self.pooling_weights.unsqueeze(1).unsqueeze(2), dim=0)
        h_list.append(self.linears_prediction[-1](weighted_pooled))

        h_list = torch.stack(h_list)  # [n_layers+1, num_graphs, out_dim]
        out = torch.sum(h_list * self.h_weights.view(-1, 1, 1), dim=0)  # [num_graphs, out_dim]

        risk_score = torch.matmul(out, self.beta) + self.h0  # [num_graphs, 1]
        return risk_score