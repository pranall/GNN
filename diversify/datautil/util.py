import numpy as np
import torch
from torch_geometric.data import Batch

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

def graph_collate_fn(batch):
    """Handle both graph and non-graph batches"""
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) > 6:  # Has edge_index
        return Batch.from_data_list([Data(
            x=item[0], 
            y=item[1], 
            edge_index=item[6],
            domain=item[2],
            pclabel=item[3],
            pdlabel=item[4]
        ) for item in batch])
    return torch.utils.data.default_collate(batch)
