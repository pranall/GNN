import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        #self.gcn1       = GCNConv(input_dim,  hidden_dim)
        self.gcn1 = GCNConv(200, hidden_dim) 
        self.gcn2       = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.in_features = output_dim

    def forward(self, data):
        print(f"\n🔥 Input shape: {data.x.shape} | Edge index: {data.edge_index.shape}")
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # if x has extra dims (e.g. [N,1,200]), flatten to [N,200]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # two graph-conv layers
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))

        # **global mean pooling over each graph in the batch**
        hg = global_mean_pool(h, batch)     # → [batch_size, hidden_dim]

        # project to output dim
        out = self.classifier(hg)           # → [batch_size, output_dim]
        return out

    def reconstruct(self, features):
        return self.recon(features)
