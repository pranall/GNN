# /gnn/temporal_gcn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dim, lstm_hidden_dim, output_dim):
        super(TemporalGCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.lstm = nn.LSTM(input_size=gcn_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.proj = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x, edge_index, batch_size):
        # x: [num_nodes, input_dim]
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))

        # x: [num_nodes, gcn_hidden_dim]
        # Group nodes per sample (assumes equal-sized subgraphs)
        seq_len = x.size(0) // batch_size
        x = x.view(batch_size, seq_len, -1)  # [B, T, F]

        _, (h_n, _) = self.lstm(x)  # h_n: [1, B, lstm_hidden_dim]
        h_n = h_n.squeeze(0)  # [B, lstm_hidden_dim]
        return self.proj(h_n)  # [B, output_dim]
