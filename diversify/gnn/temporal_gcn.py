import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_builder = graph_builder
        
        # GCN layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Project pooled features to desired output_dim
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Expose in_features for downstream blocks
        self.in_features = output_dim

    def forward(self, data):
        # data is a torch_geometric.data.Data or Batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # now batch has length == h.size(0)
        hg = global_mean_pool(h, batch)   # [num_graphs, hidden_dim]
        return hg

    def reconstruct(self, features):
        # Optional pretraining reconstruction
        return self.recon(features)
