import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gnn.graph_builder import GraphBuilder

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for sensor-based activity recognition
    Combines 1D convolutions for temporal features with GCN for spatial features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.graph_builder = graph_builder or GraphBuilder()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim

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
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Reconstruction layer for pretraining
        self.recon = nn.Linear(output_dim, input_dim)

        # Cache for edge indices
        self.graph_cache = {}

    def forward(self, x):
        # 1) If this is a PyG Batch, unpack its tensor
        if hasattr(x, 'x'):
            x = x.x

        # 2) Fix stray shape [N,1,T] -> [B, C=input_dim, T]
        if x.dim() == 3 and x.size(1) == 1 and x.size(0) % self.input_dim == 0:
            B = x.size(0) // self.input_dim
            x = x.view(B, self.input_dim, x.size(2))

        # 3) Now enforce [B, C, T]
        B, C, T = x.shape
        if C != self.input_dim:
            raise ValueError(f"Expected C={self.input_dim}, got {C}")

        # 4) Temporal convolutions -> [B, 32, T//4]
        x = self.temporal_conv(x)
        _, feat_dim, T4 = x.shape

        # 5) Flatten for GCN: [B, 32, T4] -> [B*T4, 32]
        x = x.permute(0, 2, 1).reshape(B * T4, feat_dim)

        # 6) Build or fetch edge_index for this T4
        key = str(T4)
        if key not in self.graph_cache:
            # build graph on mean features per time node
            mean_feat = x.view(B, T4, feat_dim).mean(dim=0).cpu().numpy()
            raw_ei = self.graph_builder.build_graph(mean_feat)
            # convert to torch.LongTensor and clamp
            edge_index = torch.tensor(raw_ei, dtype=torch.long)
            max_i = T4 - 1
            edge_index = edge_index.clamp(0, max_i)
            self.graph_cache[key] = edge_index
        edge_index = self.graph_cache[key].to(x.device)

        # 7) GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # 8) Pool back to [B, hidden_dim]
        x = x.view(B, T4, self.hidden_dim).mean(dim=1)

        return self.fc(x)

    def reconstruct(self, features):
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
