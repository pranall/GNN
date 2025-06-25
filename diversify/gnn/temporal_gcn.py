import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # Better than GCN for EMG
from torch.nn import LSTM

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, heads=4):
        super().__init__()
        # Spatial aggregation
        self.conv1 = GATv2Conv(
            input_dim, hidden_dim, 
            heads=heads, 
            edge_dim=1,  # Use correlation strength
            dropout=0.3
        )
        
        # Temporal processing
        self.lstm = LSTM(
            hidden_dim*heads, hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.in_features = output_dim  # For downstream compatibility

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Spatial aggregation
        x = F.elu(self.conv1(x, edge_index, edge_attr))  # (N, heads*hidden)
        
        # Temporal processing
        if batch is not None:
            x, _ = self.lstm(x.unsqueeze(0))  # (1, N, hidden*2)
            x = x.squeeze(0)
        else:
            # Handle single sample
            x, _ = self.lstm(x.unsqueeze(0))
            x = x.squeeze(0)
        
        return self.fc(x)
