import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_builder = graph_builder

        # two GCN layers (they produce hidden_dim features per node)
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # **project pooled graph-level features to output_dim**
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # downstream code reads featurizer.in_features to size its bottlenecks
        self.in_features = output_dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # flatten any extra dims on x (e.g. [N,1,200] → [N,200])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # pool per-graph
        hg = global_mean_pool(h, batch)   # [batch_size, hidden_dim]

        # **project up to output_dim**
        out = self.classifier(hg)         # [batch_size, output_dim]
        return out

    def reconstruct(self, features):
        return self.recon(features)
