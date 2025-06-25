import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data

def build_emg_graph(signal, threshold=0.4, dynamic_threshold=True):
    """
    Enhanced EMG graph builder with MYO armband-specific features.
    
    Args:
        signal: (T, 8) array for MYO's 8 sensors
        threshold: Correlation threshold (0.3-0.5 works best for EMG)
    
    Returns:
        Data: PyG Data object with:
        - x: (8, T) normalized sensor readings
        - edge_index: (2, E) connectivity
        - edge_attr: (E,) correlation strengths
    """
    assert signal.shape[1] == 8, "MYO armband requires 8 channels"
    
    # 1. Compute sensor correlations
    corr = np.corrcoef(signal.T)  # (8,8)
    np.fill_diagonal(corr, 0)  # Remove self-loops
    
    # 2. Adaptive thresholding (ensure minimum connectivity)
    if dynamic_threshold:
        for _ in range(5):  # Max 5 attempts
            edges = np.argwhere(np.abs(corr) > threshold)
            if len(edges) >= 8:  # At least 1 edge per sensor
                break
            threshold *= 0.9  # Reduce threshold
        
    # 3. Normalize features
    x = torch.FloatTensor(signal.T)  # (8, T)
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
    
    # 4. Build graph
    edge_attr = torch.FloatTensor(np.abs(corr[edges[:,0], edges[:,1]]))
    
    return Data(
        x=x,
        edge_index=torch.LongTensor(edges.T).contiguous(),
        edge_attr=edge_attr,
        num_nodes=8
    )
