import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset

# Maximum person-IDs per dataset (needed by cross_people.py)
Nmax = {
    'emg':   36,
    'pamap': 10,
    'dsads':  8
}

class GraphDatasetMixin:
    """Enhanced mixin for graph data support with EMG-specific features"""
    def __init__(self):
        self.edge_indices = None
        self.batches = None
        self.graphs = None  # Store full PyG Data objects
        
    def set_graph_attributes(self, edge_indices=None, batches=None, graphs=None):
        """Handle both legacy and PyG-native graph storage"""
        if graphs is not None:
            self.graphs = graphs
            self.edge_indices = [g.edge_index for g in graphs]
            self.batches = [g.batch if hasattr(g, 'batch') else None for g in graphs]
        else:
            self.edge_indices = edge_indices
            self.batches = batches

class mydataset(GraphDatasetMixin):
    """Base dataset class with EMG-aware graph handling"""
    def __init__(self, args):
        super().__init__()
        self.x = None          # EMG sensor data
        self.labels = None     # Gesture labels
        self.dlabels = None    # Domain labels
        self.pclabels = None   # Pseudo-class labels  
        self.pdlabels = None   # Pseudo-domain labels
        self.args = args
        self.sensor_count = 8  # MYO armband specific
        
    def __getitem__(self, index):
        """Returns either graph or raw data based on initialization"""
        # Base items
        item = [
            self.input_trans(self.x[index]),
            self.target_trans(self.labels[index]),
            self.target_trans(self.dlabels[index]),
            self.target_trans(self.pclabels[index]) if self.pclabels is not None else -1,
            self.target_trans(self.pdlabels[index]) if self.pdlabels is not None else -1,
            index  # Original sample index
        ]
        
        # Graph mode additions
        if self.graphs is not None:
            return self.graphs[index]  # Return full PyG Data object
        elif self.edge_indices is not None:
            item.append(self.edge_indices[index])
            
        return tuple(item)

class combindataset(mydataset):
    """Dataset combiner with graph-aware merging"""
    def __init__(self, args, datalist):
        super().__init__(args)
        self.domain_num = len(datalist)
        
        # Combine all data fields
        self.x = torch.vstack([d.x for d in datalist])
        self.labels = np.hstack([d.labels for d in datalist])
        self.dlabels = np.hstack([d.dlabels for d in datalist])
        self.pclabels = np.hstack([d.pclabels for d in datalist]) if datalist[0].pclabels is not None else None
        self.pdlabels = np.hstack([d.pdlabels for d in datalist]) if datalist[0].pdlabels is not None else None
        
        # Handle graph data merging
        if all(hasattr(d, 'graphs') for d in datalist):
            self.graphs = [g for d in datalist for g in d.graphs]
            self.set_graph_attributes(graphs=self.graphs)
        elif all(hasattr(d, 'edge_indices') for d in datalist):
            self.edge_indices = [ei for d in datalist for ei in d.edge_indices]
            self.batches = [b for d in datalist for b in d.batches]

def graph_collate_fn(batch):
    """Smart collator handling all EMG data formats"""
    # Case 1: PyG Data objects (recommended)
    if isinstance(batch[0], Data):
        batch = Batch.from_data_list(batch)
        batch.batch_idx = torch.arange(len(batch))  # Preserve original indices
        return batch
    
    # Case 2: Legacy tuple format
    elif isinstance(batch[0], (tuple, list)):
        # Extract components
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        domains = [item[2] for item in batch]
        pclabels = [item[3] for item in batch]
        pdlabels = [item[4] for item in batch]
        indices = [item[5] for item in batch]
        
        # Build graph batch if edge indices exist
        if len(batch[0]) > 6:
            edge_indices = [item[6] for item in batch]
            graphs = [
                Data(
                    x=x.float(),
                    y=y.long(),
                    edge_index=eidx.long(),
                    domain=d.long(),
                    pclabel=pcl.long() if pcl != -1 else None,
                    pdlabel=pdl.long() if pdl != -1 else None,
                    batch_idx=idx
                )
                for x, y, d, pcl, pdl, idx, eidx in zip(
                    xs, ys, domains, pclabels, pdlabels, indices, edge_indices
                )
            ]
            return Batch.from_data_list(graphs)
        
        # Non-graph fallback
        return torch.utils.data.default_collate(batch)
    
    # Case 3: Unknown format
    raise ValueError(f"Unsupported batch type: {type(batch[0])}")

def validate_emg_batch(batch):
    """EMG-specific batch validation"""
    if isinstance(batch, Batch):
        # Shape checks
        assert batch.x.dim() == 2, f"Features should be 2D (got {batch.x.shape})"
        assert batch.y.dim() == 1, f"Labels should be 1D (got {batch.y.shape})"
        
        # MYO armband specific
        if batch.x.size(1) != 8:
            print(f"⚠️ Unexpected sensor count: {batch.x.size(1)} (expected 8)")
        
        # Graph connectivity checks
        if batch.edge_index.size(1) == 0:
            print("⚠️ Empty edge_index - check graph construction")
            
    return True

def get_graph_metrics(batch):
    """Compute EMG graph statistics"""
    metrics = {}
    if isinstance(batch, Batch):
        metrics.update({
            'sensors_used': batch.x.abs().mean(dim=0),  # Per-sensor activation
            'edge_density': batch.num_edges / (batch.num_nodes ** 2),
            'cross_domain_edges': (
                batch.domain[batch.edge_index[0]] != 
                batch.domain[batch.edge_index[1]]
            ).float().mean().item() if hasattr(batch, 'domain') else None
        })
    return metrics
    
class subdataset(Dataset):
    """Light‐wrapper around another dataset + index list."""
    def __init__(self, args, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = np.array(indices, dtype=np.int64)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        # if it's a PyG Data it comes back whole
        if hasattr(item, 'edge_index'):
            return item
        # else it's a tuple (x,c,p,s,pd,idx,[edge])
        # preserve the real_idx override at position 5
        out = list(item)
        out[5] = real_idx
        return tuple(out)
