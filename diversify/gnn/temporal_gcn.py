import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.input_dim = input_dim  # node feature length (e.g. 200)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # graph embedding size
        self.graph_builder = graph_builder

        # Spatial GCN layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Final projection to desired output_dim
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, edge_index=None):
        # Accept PyG Data or Batch
        x = data.x  # [num_nodes, input_dim]
        e = data.edge_index if edge_index is None else edge_index

        h = F.relu(self.gcn1(x, e))
        h = F.relu(self.gcn2(h, e))

        # create batch vector for pooling
        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # global mean pooling to get graph-level embedding
        hg = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        out = self.project(hg)           # [batch_size, output_dim]
        return out

    def reconstruct(self, features):
        raise NotImplementedError("Reconstruction not implemented")
