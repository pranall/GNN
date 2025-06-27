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

        # Temporal feature extractor (for raw EMG tensors)
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
        # --- Case 1: input is a PyG Data object, skip temporal conv ---
        if isinstance(x, Data):
            data = x
            node_feat, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
            h = F.relu(self.gcn1(node_feat, edge_index))
            h = F.relu(self.gcn2(h, edge_index))
            if batch is not None:
                h = global_mean_pool(h, batch)
            else:
                # assume one graph: mean over nodes
                h = h.mean(dim=0, keepdim=True)
            return self.classifier(h)

        # --- Case 2: raw tensor input [B, C, T] or variants ---
        # normalize raw Tensor shapes
        if isinstance(x, torch.Tensor):
            # collapse [B, C, 1, T] -> [B, C, T]
            if x.dim() == 4 and x.size(2) == 1:
                x = x.squeeze(2)
            # swap [B, T, C] -> [B, C, T] if needed
            if x.dim() == 3 and x.size(1) != self.input_dim:
                x = x.permute(0, 2, 1)

        # now x should be [B, C, T]
        B, C, T = x.shape
        # temporal convolution -> [B, 32, T//4]
        x_temp = self.temporal_conv(x)
        _, feat_dim, new_T = x_temp.shape
        # flatten for GCN: each time step is a node
        x_temp = x_temp.permute(0, 2, 1).reshape(-1, feat_dim)

        # build edge_index if missing
        if edge_index is None:
            if self.graph_builder is None:
                raise ValueError("No edge_index or graph_builder for TemporalGCN")
            gb_data = self.graph_builder(x_temp.view(B, new_T, feat_dim).cpu().numpy())
            edge_index = gb_data.edge_index.to(x_temp.device)

        # GCN layers
        h = F.relu(self.gcn1(x_temp, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        # pool back to graph-level
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.view(B, new_T, self.hidden_dim).mean(dim=1)
        return self.classifier(h)

    def reconstruct(self, features):
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
