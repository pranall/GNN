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
        # 1. Extract node features and edge connections
        x, edge_index = data.x, data.edge_index  
    
        # 2. Handle batch dimension if present
        if x.dim() == 3:  # [batch_size, num_nodes, features]
            x = x.squeeze(1)
    
        # 3. First graph convolution
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        # 4. Second graph convolution (embeddings)
        x = self.gcn2(x, edge_index)  # This is the embedding we want
    
        # 5. Final classification layer
        out = self.classifier(x)  # Prediction outputs
    
        # 6. Return logic
        if return_embeddings:
            return out, x  # Return (predictions, embeddings)
        return out

    def reconstruct(self, features):
        return self.classifier(features)
