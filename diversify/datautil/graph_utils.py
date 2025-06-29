import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def convert_to_graph(sensor_data, adjacency_strategy='fully_connected', threshold=0.5, top_k=None):
    num_nodes = sensor_data.shape[0]
    timesteps = sensor_data.shape[1]
    num_features = sensor_data.shape[2]

    x = sensor_data.reshape(num_nodes, -1)
    flat_data_np = x.cpu().numpy()

    if adjacency_strategy == 'fully_connected':
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = None

    elif adjacency_strategy == 'correlation':
        corr_matrix = np.corrcoef(flat_data_np)

        edge_index = []
        edge_weight = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if abs(corr_matrix[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    weight = abs(corr_matrix[i, j])
                    edge_weight.extend([weight, weight])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1) if edge_weight else None

    elif adjacency_strategy == 'top_k_correlation':
        corr_matrix = np.corrcoef(flat_data_np)
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)

        edge_index = []
        edge_weight = []
        for i in range(num_nodes):
            top_k_indices = np.argsort(abs_corr[i])[-top_k:]
            for j in top_k_indices:
                edge_index.append([i, j])
                edge_index.append([j, i])
                weight = abs_corr[i, j]
                edge_weight.extend([weight, weight])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1) if edge_weight else None

    elif adjacency_strategy == 'knn':
        from sklearn.neighbors import kneighbors_graph
        knn_graph = kneighbors_graph(flat_data_np, n_neighbors=3, mode='distance', include_self=False)

        edge_index = []
        edge_weight = []
        rows, cols = knn_graph.nonzero()
        for i, j in zip(rows, cols):
            dist = knn_graph[i, j]
            weight = 1.0 / (1.0 + dist)
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weight.extend([weight, weight])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1) if edge_weight else None

    elif adjacency_strategy == 'dtw':
        edge_index = []
        edge_weight = []
        dtw_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                dist, _ = fastdtw(
                    sensor_data[i].cpu().numpy(),
                    sensor_data[j].cpu().numpy(),
                    dist=euclidean
                )
                dtw_matrix[i, j] = dist
                dtw_matrix[j, i] = dist

        max_dist = np.max(dtw_matrix)
        if max_dist > 0:
            dtw_sim = 1.0 - (dtw_matrix / max_dist)
        else:
            dtw_sim = np.ones_like(dtw_matrix)

        for i in range(num_nodes):
            top_k_indices = np.argsort(dtw_sim[i])[-top_k:]
            for j in top_k_indices:
                if i == j:
                    continue
                edge_index.append([i, j])
                edge_index.append([j, i])
                weight = dtw_sim[i, j]
                edge_weight.extend([weight, weight])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1) if edge_weight else None

    else:
        raise ValueError(f"Unknown adjacency strategy: {adjacency_strategy}")

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
