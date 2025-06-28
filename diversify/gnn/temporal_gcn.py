import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.gcn1       = GCNConv(input_dim,  hidden_dim)
        self.gcn2       = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.in_features = output_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        # Ensure proper input dimensions
        if x.dim() == 3:  # [batch*num_nodes, 1, timesteps]
            x = x.squeeze(1)  # [batch*num_nodes, timesteps]
    
        # Add residual connections
        identity = x
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
    return x + identity  # Residual connection

    def reconstruct(self, features):
        return self.recon(features)
