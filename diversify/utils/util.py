# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL
from torch_geometric.data import Data, Batch
from typing import Union, List, Dict, Any

def set_random_seed(seed: int = 0) -> None:
    """Enhanced seeding for reproducibility across libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For PyTorch Geometric if available
    try:
        import torch_geometric
        torch_geometric.seed_everything(seed)
    except ImportError:
        pass

def train_valid_target_eval_names(args) -> Dict[str, List[str]]:
    """Generate evaluation split names (unchanged)"""
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append('eval%d_in' % i)
            eval_name_dict['valid'].append('eval%d_out' % i)
        else:
            eval_name_dict['target'].append('eval%d_out' % i)
    return eval_name_dict

def alg_loss_dict(args) -> List[str]:
    """Extended loss tracking for GNN mode"""
    losses = {
        'diversify': ['class', 'dis', 'total'],
        'gnn': ['class', 'dis', 'graph', 'total']
    }
    return losses.get(args.algorithm.lower(), ['class', 'total'])

def print_args(args, print_list: List[str] = []) -> str:
    """Enhanced argument printer with GNN-specific params"""
    s = "==========================================\n"
    s += f"Algorithm: {args.algorithm}\n"
    
    # Print all if no list specified
    print_all = len(print_list) == 0
    for arg, content in args.__dict__.items():
        if print_all or arg in print_list:
            if arg == 'test_envs' and isinstance(content, list):
                content = ','.join(map(str, content))
            s += f"{arg}: {content}\n"
    
    # Add GNN-specific info if relevant
    if hasattr(args, 'gnn_hidden'):
        s += f"\nGNN Config:\n  Hidden: {args.gnn_hidden}\n  Layers: {args.gnn_layers}\n"
    return s

def print_row(row: List[Any], colwidth: int = 10, latex: bool = False) -> None:
    """Unchanged from original"""
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

def print_environ() -> None:
    """Enhanced environment printer with PyG info"""
    print("Environment:")
    print("\tPython:", sys.version.split(" ")[0])
    print("\tPyTorch:", torch.__version__)
    print("\tTorchvision:", torchvision.__version__)
    print("\tCUDA:", torch.version.cuda)
    print("\tCUDNN:", torch.backends.cudnn.version())
    
    try:
        import torch_geometric
        print("\tPyTorch Geometric:", torch_geometric.__version__)
    except ImportError:
        print("\tPyTorch Geometric: Not installed")

class Tee:
    """Unchanged from original"""
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def act_param_init(args) -> argparse.Namespace:
    """Extended with GNN default parameters"""
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9+j for j in range(9)] for i in range(4)]}
    tmp = {'emg': ((8, 1, 200), 6, 10)}
    args.num_classes, args.input_shape, args.grid_size = tmp[args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]
    
    # Add GNN defaults if not specified
    if not hasattr(args, 'gnn_hidden'):
        args.gnn_hidden = 64
    if not hasattr(args, 'gnn_layers'):
        args.gnn_layers = 2
    if not hasattr(args, 'graph_threshold'):
        args.graph_threshold = 0.3
        
    return args

def get_args() -> argparse.Namespace:
    """Updated argument parser with GNN options"""
    parser = argparse.ArgumentParser(description='Domain Generalization')
    
    # Base arguments (unchanged)
    parser.add_argument('--algorithm', type=str, default="diversify", choices=['diversify', 'gnn'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--seed', type=int, default=0)
    
    # GNN-specific arguments
    parser.add_argument('--gnn_hidden', type=int, default=64, help='Hidden dimension for GNN layers')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--graph_threshold', type=float, default=0.3, 
                      help='Correlation threshold for graph construction')
    
    # Existing arguments...
    args = parser.parse_args()
    args = act_param_init(args)
    
    # Auto-configure batch size for GNN
    if args.algorithm.lower() == 'gnn':
        args.batch_size = max(16, args.batch_size // 2)
        
    return args

# --------------------------
# New GNN Utility Functions
# --------------------------

def validate_graph_data(data: Union[Data, Batch]) -> bool:
    """Check if data contains required graph attributes"""
    required = ['x', 'edge_index', 'y']
    return all(hasattr(data, attr) for attr in required)

def print_graph_stats(data: Union[Data, Batch]) -> None:
    """Print summary statistics for graph data"""
    if isinstance(data, (Data, Batch)):
        print(f"Graph Batch Stats:")
        print(f"  - Num Graphs: {getattr(data, 'num_graphs', 1)}")
        print(f"  - Num Nodes: {data.num_nodes}")
        print(f"  - Num Edges: {data.num_edges}")
        if hasattr(data, 'batch'):
            print(f"  - Nodes per Graph: ~{data.num_nodes // data.num_graphs}")
    else:
        print("Not a graph data object")

def graph_collate_fn(batch: List) -> Union[Batch, Dict]:
    """Custom collate for mixed graph/non-graph batches"""
    if isinstance(batch[0], (Data, dict)):  # Graph data
        try:
            return Batch.from_data_list(batch)
        except Exception as e:
            print(f"Graph collate error: {e}")
            return torch.utils.data.default_collate(batch)
    return torch.utils.data.default_collate(batch)

def normalize_graph_features(data: Data) -> Data:
    """Normalize node features in graph"""
    if hasattr(data, 'x'):
        data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
    return data
