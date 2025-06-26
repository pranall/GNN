import torch
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform

def build_correlation_graph(data_sample, threshold=0.3):
    """
    Build edge_index from a single EMG sample of shape (T, C).
    
    Args:
        data_sample: np.ndarray of shape (T, C) — time steps x channels
        threshold: float — minimum absolute correlation to form an edge

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]
        edge_weights: torch.FloatTensor of shape [num_edges]
    """
    assert len(data_sample.shape) == 2, "Input must be 2D (T, C)"
    sensors = data_sample.shape[1]

    # Compute correlation matrix with fallback
    try:
        corr = np.corrcoef(data_sample.T)
    except:
        corr = np.eye(sensors)
    
    # Create edges and weights
    edge_index = []
    edge_weights = []
    for i, j in itertools.product(range(sensors), repeat=2):
        if i != j and abs(corr[i, j]) > threshold:
            edge_index.append([i, j])
            edge_weights.append(abs(corr[i, j]))

    # Fallback: fully connected graph with uniform weights
    if len(edge_index) == 0:
        edge_index = [[i, j] for i in range(sensors) for j in range(sensors) if i != j]
        edge_weights = [1.0] * len(edge_index)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    
    return edge_index, edge_weights

def build_distance_graph(data_sample, threshold=0.5):
    """
    Alternative graph builder using Euclidean distance.
    """
    dist_matrix = squareform(pdist(data_sample.T, 'euclidean'))
    normalized_dist = 1 - (dist_matrix / dist_matrix.max())
    
    edge_index = []
    edge_weights = []
    for i, j in itertools.product(range(data_sample.shape[1]), repeat=2):
        if i != j and normalized_dist[i, j] > threshold:
            edge_index.append([i, j])
            edge_weights.append(normalized_dist[i, j])
    
    if len(edge_index) == 0:
        edge_index = [[i, j] for i in range(data_sample.shape[1]) 
                     for j in range(data_sample.shape[1]) if i != j]
        edge_weights = [1.0] * len(edge_index)
    
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), \
           torch.tensor(edge_weights, dtype=torch.float32)
