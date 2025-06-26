import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(TemporalGCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.lstm = nn.LSTM(hidden_dim, output_dim, 
                           num_layers=2, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch_size):
        # GCN processing
        for layer in self.gcn_layers[:-1]:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.gcn_layers[-1](x, edge_index)
        x = self.norm(x)
        
        # Temporal processing
        x = x.view(batch_size, -1, x.size(1))
        x, (h_n, _) = self.lstm(x)
        return h_n[-1]
