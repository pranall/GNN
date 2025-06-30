import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=64, output_dim=128):
        super().__init__()
        self.time_encoder = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.time_encoder(x))  # [batch*8, hidden]
        x = F.relu(self.gcn1(x, edge_index))
        x = global_mean_pool(self.gcn2(x, edge_index), batch)
        return x

    def reconstruct(self, features):
        return self.recon(features)
