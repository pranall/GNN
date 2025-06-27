import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_builder = graph_builder
        
        # GCN layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Project pooled features to desired output_dim
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Expose in_features for downstream blocks
        self.in_features = output_dim

    def forward(self, data):
        # Accept PyG Data or Batch
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
            raise ValueError(f"TemporalGCN expects a PyG Data/Batch, got {type(data)}")
        x = data.x  # [num_nodes, node_feat_dim]
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1) Graph convolutions
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # 2) Global pooling to get graph-level embeddings
        hg = global_mean_pool(h, batch)  # [batch_size, hidden_dim]

        # 3) Final projection
        out = self.classifier(hg)        # [batch_size, output_dim]
        return out

    def reconstruct(self, features):
        # Optional pretraining reconstruction
        return self.recon(features)
