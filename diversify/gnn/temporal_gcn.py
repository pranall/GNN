# gnn/temporal_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GCN layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Final classifier (unused if only featurizer)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, edge_index=None):
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            node_feat = data.x  # [num_nodes, features]
            edge_index = data.edge_index if edge_index is None else edge_index
        else:
            raise ValueError(f"Expected PyG Data object as input, got {type(data)}")

        h = F.relu(self.gcn1(node_feat, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        return h  # [num_nodes, hidden_dim]

    #def reconstruct(self, features):
        #"""Reconstruct mean input features for pretraining"""
        #return self.recon(features)
