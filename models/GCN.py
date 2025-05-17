import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, GlobalAttention
from torch_geometric.data import Data

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 graph_pooling_type="att"):
        super(GCN, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GCNConv(in_dim, hidden_dim))
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        # Define pooling layers
        self.linears_prediction = nn.ModuleList()
        self.pools = []  # Use a regular list for pooling functions

        for layer in range(n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, out_dim))

            # Use functions directly, not Module subclasses
            if graph_pooling_type == 'sum':
                self.pools.append(global_add_pool)
            elif graph_pooling_type == 'mean':
                self.pools.append(global_mean_pool)
            elif graph_pooling_type == 'max':
                self.pools.append(global_max_pool)
            elif graph_pooling_type == 'att':
                if layer == 0:
                    gate_nn = torch.nn.Linear(in_dim, 1)
                else:
                    gate_nn = torch.nn.Linear(hidden_dim, 1)
                gate_nn.to("cuda" if torch.cuda.is_available() else "cpu")
                self.pools.append(GlobalAttention(gate_nn))  # This one can stay in ModuleList
            else:
                raise NotImplementedError
        self.h0 = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(out_dim))
        
        self.risk_layer = nn.Linear(out_dim, 1)

    def forward(self, data=None, x=None, edge_index=None):
        """
        Forward function for both PyTorch Geometric and DGL graph formats.
        """
        if (x is None) and (edge_index is None):
            if isinstance(data, Data):  # PyTorch Geometric format
                x, edge_index, batch = data.x, data.edge_index, data.batch
            elif hasattr(data, "ndata") and hasattr(data, "edata"):  # DGL format
                # Extract node features and edges from DGL graph
                x = data.ndata["feat"]
                edge_index = torch.stack(data.edges(), dim=0)
                # DGL doesn't provide batch directly; use default batch=0 for single graph
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                raise ValueError("Unsupported graph data format. Expecting PyTorch Geometric Data or DGLGraph.")
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            h_list.append(self.linears_prediction[i](self.pools[i](x, batch)))
            x = layer(x, edge_index)

        h_list.append(self.classify(self.pools[-1](x, batch)))

        # Aggregate outputs from all layers
        out = torch.stack(h_list).mean(0)
        
        # h(t) = h0 * exp(beta^T * out)
        risk_score = torch.exp(self.h0) * torch.exp(torch.matmul(out, self.beta))
        # risk_score = self.risk_layer(out)

        return risk_score