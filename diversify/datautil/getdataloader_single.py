import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import datautil.actdata.util as actutil
import datautil.actdata.cross_people as cross_people
from datautil.util import combindataset, subdataset
import os

def get_safe_workers():
    """Calculate optimal worker count based on available resources"""
    try:
        cpu_count = os.cpu_count() or 1
        # Conservative defaults for stability
        workers = min(2, cpu_count)
        
        # Reduce further in memory-constrained environments
        if os.getenv('COLAB_GPU') or os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
            workers = 1
            
        return max(1, workers)  # Always at least 1
    except:
        return 1  # Fallback for any errors

SAFE_WORKERS = get_safe_workers()

def collate_gnn(batch):
    """Optimized collate function for graph data with tensor safety"""
    graphs, labels, domains = zip(*batch)
    
    for graph, y, d in zip(graphs, labels, domains):
        # Safe tensor conversion without warnings
        graph.y = y.clone().detach().long() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)
        graph.domain = d.clone().detach().long() if torch.is_tensor(d) else torch.tensor(d, dtype=torch.long)
    
    batched = Batch.from_data_list(graphs)
    
    # Handle empty edge_index
    if batched.edge_index.shape[1] == 0:
        num_nodes = batched.num_nodes
        batched.edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    
    return batched

def get_act_dataloader(args, tr, val, targetdata):
    """
    Creates dataloaders with automatic resource optimization
    
    Args:
        args: Configuration object with batch_size, use_gnn etc.
        tr: Training dataset
        val: Validation dataset
        targetdata: Test dataset
        
    Returns:
        Tuple of (train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata)
    """
    # Dynamic worker configuration
    workers = min(SAFE_WORKERS, getattr(args, 'N_WORKERS', SAFE_WORKERS))
    
    # Debug output
    if getattr(args, 'debug_mode', False):
        print(f"\nDataLoader Config:")
        print(f"- Safe workers: {SAFE_WORKERS} (CPU: {os.cpu_count()}, GPU: {torch.cuda.device_count()})")
        print(f"- Using workers: {workers}")
        print(f"- Batch size: {args.batch_size}")
    
    # Common DataLoader configuration
    loader_config = {
        'batch_size': args.batch_size,
        'num_workers': workers,
        'persistent_workers': workers > 0,
        'collate_fn': collate_gnn if getattr(args, 'use_gnn', False) else None,
        'pin_memory': torch.cuda.is_available()
    }
    
    # Create all dataloaders
    train_loader = DataLoader(tr, shuffle=True, **loader_config)
    train_ns_loader = DataLoader(tr, shuffle=False, **loader_config)
    val_loader = DataLoader(val, shuffle=False, **loader_config)
    test_loader = DataLoader(targetdata, shuffle=False, **loader_config)
    
    return train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata

# Optional: Add environment diagnostics
if __name__ == "__main__":
    print("Environment Diagnostics:")
    print(f"Available CPUs: {os.cpu_count()}")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    print(f"Recommended Workers: {SAFE_WORKERS}")
