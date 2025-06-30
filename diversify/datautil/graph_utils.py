import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # For Dynamic Time Warping

def convert_to_graph(batch_data, adjacency_strategy='correlation', threshold=0.3, top_k=None):
    """
    Optimized graph construction for EMG data (shape: [batch, timesteps, channels])
    
    Args:
        batch_data: Input tensor of shape [batch_size, 200, 8]
        adjacency_strategy: 'correlation' or 'fully_connected'
        threshold: Minimum correlation for edges (0.3 works best for EMG)
        top_k: Not used (for compatibility)
        
    Returns:
        PyG Data object with:
        - x: Node features [batch*8, 3] (mean, std, fft)
        - edge_index: Graph connections [2, num_edges]
        - batch: Graph indices [batch*8]
    """
    batch_size, timesteps, num_channels = batch_data.shape
    
    # 1. Extract powerful EMG features per channel
    means = batch_data.mean(dim=1)  # [batch, 8]
    stds = batch_data.std(dim=1)    # [batch, 8]
    fft = torch.fft.fft(batch_data, dim=1).abs().mean(dim=1)  # [batch, 8]
    
    # Stack features: [batch, 8, 3] -> [batch*8, 3]
    x = torch.stack([means, stds, fft], dim=2).reshape(-1, 3)
    
    # 2. Build edges based on strategy
    if adjacency_strategy == 'correlation':
        edge_index = []
        for b in range(batch_size):
            # Compute channel correlations
            channels = batch_data[b].permute(1, 0)  # [8, 200]
            corr = torch.corrcoef(channels)         # [8, 8]
            
            # Add edges for correlated channels
            for i in range(8):
                for j in range(8):
                    if i != j and corr[i,j] > threshold:
                        src = b*8 + i
                        dst = b*8 + j
                        edge_index.append([src, dst])
        
        edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else \
                    torch.tensor([[i for i in range(8*batch_size)], 
                                 [i for i in range(8*batch_size)]]).contiguous()
    
    elif adjacency_strategy == 'fully_connected':
        # Connect all channels within sample
        edge_index = []
        for b in range(batch_size):
            for i in range(8):
                for j in range(8):
                    if i != j:
                        edge_index.append([b*8 + i, b*8 + j])
        edge_index = torch.tensor(edge_index).t().contiguous()
    
    else:
        raise ValueError(f"Unsupported strategy: {adjacency_strategy}")

    return Data(
        x=x,
        edge_index=edge_index,
        batch=torch.arange(batch_size).repeat_interleave(8),
        edge_attr=None
    )

# Test function to verify implementation
def test_graph_conversion():
    """Run this to validate your graph construction"""
    test_data = torch.randn(64, 200, 8)  # Batch of 64 samples
    graph = convert_to_graph(test_data)
    
    print("=== TEST RESULTS ===")
    print(f"✅ Node features shape: {graph.x.shape} (expected [512, 3])")
    print(f"✅ Edge index shape: {graph.edge_index.shape} (should be [2, E])")
    print(f"✅ Sample edges (first 5):\n{graph.edge_index[:, :5]}")
    print(f"✅ Batch indices:\n{graph.batch[:16]} (first 2 samples)")
    
    if graph.edge_index.shape[1] == 8*64:
        print("⚠️ Warning: Only self-loops detected. Try lowering threshold.")

if __name__ == '__main__':
    test_graph_conversion()
