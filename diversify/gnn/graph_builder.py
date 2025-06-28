import torch
import numpy as np
from typing import Union, List
from torch_geometric.data import Data

class GraphBuilder:
    """
    Optimized graph builder for EMG data that creates sensor-level graphs.
    Nodes represent EMG sensors (8 nodes), edges represent functional connections.
    
    Key Features:
    - EMG-specific correlation thresholds
    - Guaranteed edge creation
    - Batch processing support
    - Comprehensive validation
    
    Args:
        method: Similarity metric ('correlation', 'covariance', 'euclidean')
        threshold_type: 'fixed' or 'adaptive'
        default_threshold: For fixed threshold (0.3-0.7 works well for EMG)
        adaptive_factor: Multiplier for adaptive threshold (1.2-2.0 recommended)
    """
    
    def __init__(self,
                 method: str = 'correlation',
                 threshold_type: str = 'adaptive',
                 default_threshold: float = 0.5,
                 adaptive_factor: float = 1.5):
        self.method = method
        self.threshold_type = threshold_type
        self.default_threshold = default_threshold
        self.adaptive_factor = adaptive_factor
        
        if method not in {'correlation', 'covariance', 'euclidean'}:
            raise ValueError(f"Invalid method '{method}'. Choose from 'correlation', 'covariance', or 'euclidean'")
            
        if threshold_type not in {'fixed', 'adaptive'}:
            raise ValueError(f"Invalid threshold_type '{threshold_type}'. Choose 'fixed' or 'adaptive'")

    def build_graph(self, feature_sequence: Union[torch.Tensor, np.ndarray]) -> torch.LongTensor:
        """
        Build graph from EMG data.
        
        Args:
            feature_sequence: (time_steps, 8) or (batch, time_steps, 8)
            
        Returns:
            edge_index: Shape [2, num_edges] with sensor connections
        """
        # Convert and validate input
        if isinstance(feature_sequence, np.ndarray):
            feature_sequence = torch.from_numpy(feature_sequence).float()
            
        if feature_sequence.ndim == 3:
            feature_sequence = feature_sequence[0]  # Use first sample if batched
            
        if feature_sequence.shape[1] != 8:
            raise ValueError(f"EMG data must have 8 sensors, got {feature_sequence.shape[1]}")
            
        # Build the graph
        edge_index = self._build_sensor_graph(feature_sequence.t())  # [8, time_steps]
        
        # Ensure we have edges
        if edge_index.shape[1] == 0:
            edge_index = self._create_default_edges(8, feature_sequence.device)
            
        return edge_index

    def _build_sensor_graph(self, sensor_data: torch.Tensor) -> torch.LongTensor:
        """Core graph construction between sensors"""
        # Compute similarity matrix [8, 8]
        sim_matrix = self._compute_similarity(sensor_data)
        
        # Get adaptive threshold
        threshold = self._determine_threshold(sim_matrix)
        
        # Create edges
        return self._create_edges(sim_matrix, threshold, sensor_data.device)

    def _compute_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """Compute sensor similarity matrix"""
        if self.method == 'correlation':
            return self._correlation_matrix(data)
        elif self.method == 'covariance':
            return self._covariance_matrix(data)
        else:
            return self._euclidean_similarity(data)

    def _correlation_matrix(self, data: torch.Tensor) -> torch.Tensor:
        """Pearson correlation between sensors"""
        centered = data - data.mean(dim=1, keepdim=True)
        cov = centered @ centered.t() / (data.shape[1] - 1)
        stds = torch.std(data, dim=1, keepdim=True)
        return cov / (stds @ stds.t()).clamp(min=1e-8)

    def _covariance_matrix(self, data: torch.Tensor) -> torch.Tensor:
        """Covariance between sensors"""
        centered = data - data.mean(dim=1, keepdim=True)
        return (centered @ centered.t()) / (data.shape[1] - 1)

    def _euclidean_similarity(self, data: torch.Tensor) -> torch.Tensor:
        """Convert distances to similarities"""
        dists = torch.cdist(data, data)
        return 1 / (1 + dists)

    def _determine_threshold(self, matrix: torch.Tensor) -> float:
        """EMG-optimized threshold calculation"""
        if self.threshold_type == 'fixed':
            return self.default_threshold
            
        # Focus on positive correlations for EMG
        pos_corrs = matrix.clamp(min=0).flatten()
        pos_corrs = pos_corrs[pos_corrs > 0.1]  # Filter weak connections
        
        if len(pos_corrs) == 0:
            return 0.4  # Fallback threshold
            
        # Use lower quartile for stable connections
        return torch.quantile(pos_corrs, 0.25).item() * self.adaptive_factor

    def _create_edges(self, matrix: torch.Tensor, 
                     threshold: float, device: torch.device) -> torch.LongTensor:
        """Create edges between sensors"""
        if torch.sum(matrix > threshold) < 4:  # If too sparse
            threshold = torch.median(matrix[matrix > 0.1]) * 0.8  # Lower threshold dynamically                
        n = matrix.shape[0]
        rows, cols = torch.where(matrix.abs() > threshold)
        
        # Remove self-loops and duplicates
        mask = (rows < cols)
        rows, cols = rows[mask], cols[mask]
        
        # Bidirectional edges
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ], dim=0).to(device)
        print(f"Similarity matrix range: {matrix.min():.2f} to {matrix.max():.2f}")
        print(f"Active threshold: {threshold:.2f}")
        return edge_index.unique(dim=1)  # Remove any duplicates

    def _create_default_edges(self, num_sensors: int, device: torch.device) -> torch.LongTensor:
        """Fallback topology when no edges meet threshold"""
        # Create a ring graph as reasonable default
        edges = []
        for i in range(num_sensors):
            j = (i + 1) % num_sensors
            edges.append([i, j])
            edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long, device=device).t()

    def build_graph_for_batch(self, batch_data: torch.Tensor) -> List[torch.LongTensor]:
        """Process batch of EMG samples"""
        return [self.build_graph(sample) for sample in batch_data]


# Quick test function
def test_graph_builder():
    """Sanity check the graph builder"""
    builder = GraphBuilder()
    emg_data = torch.randn(200, 8)  # [time_steps, sensors]
    
    edge_index = builder.build_graph(emg_data)
    print(f"Created graph with {edge_index.shape[1]} edges")
    print("First 5 edges:", edge_index[:, :5].t())

if __name__ == '__main__':
    test_graph_builder()
