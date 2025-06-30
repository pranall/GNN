import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=128):
        """
        Optimized GNN for EMG data with:
        - input_dim: 3 (mean, std, fft features per channel)
        - hidden_dim: 128 (recommended for EMG)
        - output_dim: 128 (matches your diversify setup)
        """
        super().__init__()
        
        # Feature transformation
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        
        # Graph layers with residual connections
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Input: PyG Data object with:
        - x: [batch*8, 3] node features
        - edge_index: [2, E] graph structure
        - batch: [batch*8] graph indices
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Feature encoding
        h = self.feature_encoder(x)  # [batch*8, hidden_dim]
        
        # 2. Graph propagation with residual
        h1 = F.relu(self.gcn1(h, edge_index))
        h2 = F.relu(self.gcn2(h1 + h, edge_index))  # Residual skip
        
        # 3. Attention-weighted pooling
        attn = self.attention(h2)  # [batch*8, 1]
        h_pool = global_mean_pool(h2 * attn, batch)  # [batch_size, hidden_dim]
        
        return self.proj(h_pool)  # [batch_size, output_dim]

    @torch.no_grad()
    def test_forward_pass(self, batch_size=2):
        """Test method to verify layer dimensions"""
        test_data = Data(
            x=torch.randn(batch_size*8, 3),  # 8 channels/sample
            edge_index=torch.randint(0, batch_size*8, (2, 20)),
            batch=torch.arange(batch_size).repeat_interleave(8)
        )
        out = self.forward(test_data)
        print(f"✅ Test passed! Input: {test_data.x.shape} → Output: {out.shape}")
        return out

if __name__ == '__main__':
    # Quick verification
    model = TemporalGCN()
    model.test_forward_pass()
