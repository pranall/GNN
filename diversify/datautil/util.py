import numpy as np
import torch
from torch_geometric.data import Batch

class combindataset(mydataset):
    def __init__(self, args, datalist):
        super(combindataset, self).__init__(args)
        self.domain_num = len(datalist)
        self.loader = datalist[0].loader
        xlist = [item.x for item in datalist]
        cylist = [item.labels for item in datalist]
        dylist = [item.dlabels for item in datalist]
        pcylist = [item.pclabels for item in datalist]
        pdylist = [item.pdlabels for item in datalist]
        self.dataset = datalist[0].dataset
        self.task = datalist[0].task
        self.transform = datalist[0].transform
        self.target_transform = datalist[0].target_transform
        self.x = torch.vstack(xlist)
        self.labels = np.hstack(cylist)
        self.dlabels = np.hstack(dylist)
        self.pclabels = np.hstack(pcylist) if pcylist[0] is not None else None
        self.pdlabels = np.hstack(pdylist) if pdylist[0] is not None else None
        
class GraphDatasetMixin:
    """Mixin class for graph data support"""
    def __init__(self):
        self.edge_indices = None
        self.batches = None
        
    def set_graph_attributes(self, edge_indices, batches):
        self.edge_indices = edge_indices
        self.batches = batches

class basedataset(GraphDatasetMixin):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        if self.edge_indices is not None:
            return self.x[index], self.y[index], self.edge_indices[index]
        return self.x[index], self.y[index]

class mydataset(GraphDatasetMixin):
    def __init__(self, args):
        super().__init__()
        self.x = None
        self.labels = None
        self.dlabels = None
        self.pclabels = None
        self.pdlabels = None
        self.args = args
        
    def __getitem__(self, index):
        item = [
            self.input_trans(self.x[index]),
            self.target_trans(self.labels[index]),
            self.target_trans(self.dlabels[index]),
            self.target_trans(self.pclabels[index]) if self.pclabels is not None else -1,
            self.target_trans(self.pdlabels[index]) if self.pdlabels is not None else -1,
            index
        ]
        
        if self.edge_indices is not None:
            item.append(self.edge_indices[index])
        return tuple(item)

import torch
from torch_geometric.data import Data, Batch

def graph_collate_fn(batch):
    """Unified collate function handling both graph and non-graph EMG data"""
    # Case 1: PyG Data objects (pre-converted graphs)
    if isinstance(batch[0], Data):
        return Batch.from_data_list(batch)
    
    # Case 2: Tuple format with edge indices (legacy)
    elif isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 7:  # Has edge_index at [6]
        graphs = []
        for item in batch:
            g = Data(
                x=item[0].float(),  # EMG sensor readings
                y=item[1].long(),   # Gesture labels
                edge_index=item[6].long(),  # Graph connectivity
                domain=item[2].long(),  # Source domain
                pclabel=item[3].long() if item[3] is not None else None,
                pdlabel=item[4].long() if item[4] is not None else None,
                batch_idx=item[5] if len(item) > 5 else None  # Original sample index
            )
            graphs.append(g)
        return Batch.from_data_list(graphs)
    
    # Case 3: Regular tensor batches (non-graph)
    else:
        return torch.utils.data.default_collate(batch)

def get_graph_stats(batch):
    """Extract graph statistics for debugging"""
    stats = {}
    if isinstance(batch, Batch):
        stats.update({
            'num_nodes': batch.num_nodes,
            'num_edges': batch.num_edges,
            'avg_degree': batch.num_edges / batch.num_nodes,
            'has_isolated': (torch.bincount(batch.edge_index[0]) == 0).any().item()
        })
    return stats

def validate_batch(batch):
    """Sanity check for EMG graph batches"""
    if isinstance(batch, Batch):
        assert batch.x.dim() == 2, f"Node features must be 2D (got {batch.x.shape})"
        assert batch.edge_index.dim() == 2, f"Edge_index must be 2D (got {batch.edge_index.shape})"
        assert batch.y.dim() == 1, f"Labels must be 1D (got {batch.y.shape})"
        
        # MYO armband specific checks
        if batch.x.size(1) != 8:
            print(f"⚠️ Expected 8 sensors, got {batch.x.size(1)}")
