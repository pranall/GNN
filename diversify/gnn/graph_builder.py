import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data

def build_emg_graph(signal, threshold=0.4, dynamic_threshold=True):
    """
    Build graph from EMG signal with adaptive thresholding.
    
    Args:
        signal: np.ndarray of shape (T, C) — time steps x channels
        threshold: float — baseline correlation threshold
        dynamic_threshold: bool — auto-adjust threshold if no edges found
    
    Returns:
        Data: PyG Data object with x, edge_index, edge_attr
    """
    assert signal.ndim == 2, "Input must be 2D (T, C)"
    
    # 1. Compute dynamic time warping (DTW) or correlation
    corr = np.corrcoef(signal.T)
    dist = 1 - np.abs(corr)  # Distance metric
    
    # 2. Adaptive thresholding
    if dynamic_threshold:
        while True:
            edges = np.argwhere(dist < threshold)
            if len(edges) > 0 or threshold >= 1.0:
                break
            threshold += 0.1
    
    # 3. Create PyG Data object
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    edge_attr = torch.tensor(1 - dist[edges[:,0], edges[:,1]], dtype=torch.float)
    
    return Data(
        x=torch.tensor(signal, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr
    )
