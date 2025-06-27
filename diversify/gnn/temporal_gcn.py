# gnn/temporal_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.graph_builder = graph_builder
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Temporal feature extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Spatial GCN layers
        self.gcn1 = GCNConv(32, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Final classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index=None, batch=None):
        # --- allow PyG Data object as input ---
        if isinstance(x, Data):
            data = x
            x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)

        # --- normalize raw Tensor shapes ---
        if isinstance(x, torch.Tensor):
            # collapse [B, C, 1, T] -> [B, C, T]
            if x.dim() == 4 and x.size(2) == 1:
                x = x.squeeze(2)
            # permute [B, T, C] -> [B, C, T] if channels mismatch
            if x.dim() == 3 and x.size(1) != self.input_dim:
                x = x.permute(0, 2, 1)

        # now expect [B, C, T]
        B, C, T = x.shape

        # temporal convolution -> [B, 32, T//4]
        x_temp = self.temporal_conv(x)
        _, feat_dim, new_T = x_temp.shape

        # flatten for graph conv: treat each time step as node
        x_temp = x_temp.permute(0, 2, 1).reshape(-1, feat_dim)  # [B*new_T, feat_dim]

        # if no edge_index, build via graph_builder
        if edge_index is None:
            if self.graph_builder is None:
                raise ValueError("No edge_index or graph_builder provided to TemporalGCN.forward")
            gb_data = self.graph_builder(x_temp.view(B, new_T, feat_dim).cpu().numpy())
            edge_index = gb_data.edge_index.to(x_temp.device)

        # graph convolutions
        h = F.relu(self.gcn1(x_temp, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # pool back to graph-level
        if batch is not None:
            h = global_mean_pool(h, batch)  # [B, hidden_dim]
        else:
            h = h.view(B, new_T, self.hidden_dim).mean(dim=1)

        # classifier
        return self.classifier(h)

    def reconstruct(self, features):
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
