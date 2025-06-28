import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import datautil.actdata.util as actutil
import datautil.actdata.cross_people as cross_people
from datautil.util import combindataset, subdataset
import os

# System-aware worker count calculation
MAX_WORKERS = min(2, os.cpu_count() // 2)  # Never exceeds system recommendations

def collate_gnn(batch):
    """Optimized collate function for graph data"""
    graphs, labels, domains = zip(*batch)
    
    for graph, y, d in zip(graphs, labels, domains):
        # Efficient tensor conversion without warnings
        graph.y = y.clone().detach().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        graph.domain = d.clone().detach().long() if torch.is_tensor(d) else torch.tensor(d, dtype=torch.long)
    
    batched = Batch.from_data_list(graphs)
    
    # Handle empty edge_index
    if batched.edge_index.shape[1] == 0:
        num_nodes = batched.num_nodes
        batched.edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    
    return batched

def get_dataloader_single(args, tr, val, targetdata):
    """Returns dataloaders with system-optimized worker count"""
    train_loader = DataLoader(
        tr, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(MAX_WORKERS, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    train_ns_loader = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(MAX_WORKERS, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(MAX_WORKERS, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    test_loader = DataLoader(
        targetdata,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(MAX_WORKERS, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    return train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata
