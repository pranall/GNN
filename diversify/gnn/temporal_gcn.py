import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TemporalConv
from torch_geometric.utils import to_dense_batch

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(TemporalGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Temporal convolution for EMG time-series
        self.temporal_conv = TemporalConv(hidden_dim, hidden_dim, kernel_size=3)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # Graph Convolution
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Temporal Processing
        if batch is not None:
            x, mask = to_dense_batch(x, batch)  # [B, T, D]
            x = self.temporal_conv(x)  # Temporal convolution
            x = x.mean(dim=1)  # Global temporal pooling
        else:
            x = x.unsqueeze(0)  # [1, N, D] if no batch
            x = self.temporal_conv(x)
            x = x.squeeze(0)
        
        return self.fc(x)
