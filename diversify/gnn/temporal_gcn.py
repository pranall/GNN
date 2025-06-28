import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = 0.3
        self.in_features = output_dim

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        
        if x.dim() == 3:
            x = x.squeeze(1)
            
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gcn2(x, edge_index)
        out = self.classifier(x)
        
        if return_embeddings:
            return out, x
        return out

    def reconstruct(self, features):
        return self.classifier(features)
