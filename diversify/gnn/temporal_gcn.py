import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

        # Final classifier (you can tweak this to your needs)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # x might come in as [B, C, 1, T] or [B, C, T]; force it to 3-D:
        if x.dim() == 4 and x.size(2) == 1:
            x = x.squeeze(2)                # now [B, C, T]
        elif x.dim() == 4:
            B, C, D, T = x.shape
            x = x.view(B, C * D, T)         # flatten middle dims if necessary

        # Now it's safe to unpack
        B, C, T = x.shape

        # Temporal feature extraction
        # expects (batch_size, in_channels, seq_len)
        x_temp = self.temporal_conv(x)      # → [B, 32, T//4]
        _, feat_dim, new_T = x_temp.shape

        # reshape to node-wise features for GCN:
        # if you have N nodes per graph, you'd need to break this up
        # but assuming each channel is a node:
        x_temp = x_temp.permute(0, 2, 1)     # [B, new_T, feat_dim]
        x_temp = x_temp.reshape(-1, feat_dim)  # [(B*new_T), feat_dim]

        # apply graph convolutions
        h = F.relu(self.gcn1(x_temp, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # global pooling back to graph-level (if using batch)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(h, batch)   # [B, hidden_dim]
        else:
            h = h.view(B, new_T, self.hidden_dim).mean(dim=1)

        # final classifier
        out = self.classifier(h)            # [B, output_dim]
        return out


    def reconstruct(self, features):
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
