import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # For Dynamic Time Warping

def convert_to_graph(batch_data, threshold=0.2):
    # Input: [batch_size, 200, 8]
    batch_size, timesteps, num_channels = batch_data.shape
    
    # Node features: [batch*8, 200]
    x = batch_data.permute(0, 2, 1).reshape(-1, timesteps)
    
    # Edges: Correlated channels within same sample
    edge_index = []
    for b in range(batch_size):
        channels = batch_data[b].T  # [8, 200]
        corr = torch.corrcoef(channels)
        for i in range(8):
            for j in range(8):
                if i != j and corr[i,j] > threshold:
                    src = b*8 + i
                    dst = b*8 + j
                    edge_index.append([src, dst])
    
    edge_index = torch.tensor(edge_index).t() if edge_index else \
                torch.tensor([[i for i in range(8*batch_size)], 
                             [i for i in range(8*batch_size)]])  # Fallback
    
    return Data(x=x, edge_index=edge_index, 
               batch=torch.arange(batch_size).repeat_interleave(8))

# def convert_to_graph(sensor_data, adjacency_strategy='fully_connected', threshold=0.5, top_k=None):
#     """
#     Convert sensor data to graph representation for GNN models
#     Args:
#         sensor_data: Tensor of shape (num_sensors, timesteps, features)
#         adjacency_strategy: Graph construction method ('fully_connected', 'correlation', 'knn', 'top_k_correlation', 'dtw')
#         threshold: Correlation threshold for 'correlation' strategy
#         top_k: Number of top neighbors for 'top_k_correlation' or 'dtw' strategy
#     Returns:
#         PyG Data object with node features, edge indices, and edge attributes
#     """
#     num_nodes = sensor_data.shape[0]
#     timesteps = sensor_data.shape[1]
#     num_features = sensor_data.shape[2]

#     # Node features: flatten time series
#     x = sensor_data.reshape(num_nodes, -1)  # Shape: [num_nodes, timesteps*features]
#     flat_data_np = x.cpu().numpy()

#     edge_index = torch.empty((2, 0), dtype=torch.long)
#     edge_attr = None

#     if adjacency_strategy == 'fully_connected':
#         edge_index = []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:
#                     edge_index.append([i, j])
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_attr = None

#     elif adjacency_strategy == 'correlation':
#         corr_matrix = np.corrcoef(flat_data_np)
#         edge_index = []
#         edge_weight = []
#         for i in range(num_nodes):
#             for j in range(i+1, num_nodes):
#                 if abs(corr_matrix[i, j]) > threshold:
#                     edge_index.append([i, j])
#                     edge_index.append([j, i])  # Undirected
#                     weight = abs(corr_matrix[i, j])
#                     edge_weight.extend([weight, weight])
#         if edge_index:
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = None

#     elif adjacency_strategy == 'top_k_correlation':
#         corr_matrix = np.corrcoef(flat_data_np)
#         abs_corr = np.abs(corr_matrix)
#         np.fill_diagonal(abs_corr, 0)
#         edge_index = []
#         edge_weight = []
#         for i in range(num_nodes):
#             top_k_indices = np.argsort(abs_corr[i])[-top_k:]
#             for j in top_k_indices:
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])
#                 weight = abs_corr[i, j]
#                 edge_weight.extend([weight, weight])
#         if edge_index:
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = None

#     elif adjacency_strategy == 'knn':
#         from sklearn.neighbors import kneighbors_graph
#         knn_graph = kneighbors_graph(flat_data_np, n_neighbors=3, mode='distance', include_self=False)
#         edge_index = []
#         edge_weight = []
#         rows, cols = knn_graph.nonzero()
#         for i, j in zip(rows, cols):
#             dist = knn_graph[i, j]
#             weight = 1.0 / (1.0 + dist)
#             edge_index.append([i, j])
#             edge_index.append([j, i])
#             edge_weight.extend([weight, weight])
#         if edge_index:
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = None

#     elif adjacency_strategy == 'dtw':
#         edge_index = []
#         edge_weight = []
#         dtw_matrix = np.zeros((num_nodes, num_nodes))
#         for i in range(num_nodes):
#             for j in range(i+1, num_nodes):
#                 dist, _ = fastdtw(
#                     sensor_data[i].cpu().numpy(),
#                     sensor_data[j].cpu().numpy(),
#                     dist=euclidean
#                 )
#                 dtw_matrix[i, j] = dist
#                 dtw_matrix[j, i] = dist
#         max_dist = np.max(dtw_matrix)
#         if max_dist > 0:
#             dtw_sim = 1.0 - (dtw_matrix / max_dist)
#         else:
#             dtw_sim = np.ones_like(dtw_matrix)
#         for i in range(num_nodes):
#             top_k_indices = np.argsort(dtw_sim[i])[-top_k:]
#             for j in top_k_indices:
#                 if i == j:
#                     continue
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])
#                 weight = dtw_sim[i, j]
#                 edge_weight.extend([weight, weight])
#         if edge_index:
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)
#             edge_attr = None

#     else:
#         raise ValueError(f"Unknown adjacency strategy: {adjacency_strategy}")

#     # Spam protection: Print "No edges found" only ONCE
#     if edge_index.numel() == 0:
#         if not hasattr(convert_to_graph, "_printed_no_edges"):
#             print(f"[convert_to_graph] No edges found. Added self-loops for {num_nodes} nodes.")
#             setattr(convert_to_graph, "_printed_no_edges", True)
#         edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
#         edge_attr = None

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
