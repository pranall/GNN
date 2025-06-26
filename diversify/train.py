import time
import os
import random
import numpy as np
import torch
from alg.opt import get_optimizer
from alg.alg import get_algorithm_class
from utils.util import set_random_seed, get_args, print_row, print_environ
from datautil.getdataloader_single import get_act_dataloader
from gnn.graph_builder import build_emg_graph
import argparse
# Initialize seed - use only one method
# set_random_seed(42)  # Either use this
# OR use the existing set_random_seed(args.seed) in main()

def prepare_graph_data(batch, device):
    """Optimized graph data preparation"""
    try:
        # Handle both tuple-style and Data-style batches
        if isinstance(batch, (list, tuple)):  # Traditional batch
            x_np = batch[0].cpu().numpy().squeeze()  # (B, C, T)
            y = batch[1]
            d = batch[4] if len(batch) >=5 else None
        else:  # PyG Data batch
            x_np = batch.x.cpu().numpy().squeeze()
            y = batch.y
            d = getattr(batch, 'domain', None)
        
        g = build_emg_graph(x_np)
        return (
            g.x.to(device),
            g.edge_index.to(device),
            y.to(device),
            d.to(device) if d is not None else None,
            g.batch.to(device) if hasattr(g, 'batch') else None
        )
    except Exception as e:
        print(f"Error preparing graph data: {e}")
        raise
        
def patch_args(args):
    """Add missing arguments that may be required by data loaders"""
    if not hasattr(args, 'act_people'):
        args.act_people = 36  # Default value for EMG dataset
    return args
    
def main(args):
    # 1. Initialization
    set_random_seed(args.seed if hasattr(args, 'seed') else 42)
    args.num_classes = 6  # For your 6 EMG gestures
    print_environ()

    # 2. Data Loading
    try:
        train_loader, train_noshuf, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Model Setup
    try:
        model = get_algorithm_class(args.algorithm)(args).cuda()
        opt_d = get_optimizer(model, args, nettype='Diversify-adv')
        opt_c = get_optimizer(model, args, nettype='Diversify-cls')
        opt_a = get_optimizer(model, args, nettype='Diversify-all')
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 4. Training Loop
    for epoch in range(args.max_epoch):
        print(f"\nEPOCH {epoch}/{args.max_epoch}")
        
        try:
            # Phase 1: Feature Update
            model.train()
            for _ in range(args.local_epoch):
                for batch in train_loader:
                    x, ei, y, d, _ = prepare_graph_data(batch, 'cuda')
                    model.update_a((x, y, None, None, d), opt_a)

            # Phase 2: Domain Characterization
            for _ in range(args.local_epoch):
                for batch in train_loader:
                    x, ei, y, d, _ = prepare_graph_data(batch, 'cuda')
                    model.update_d((x, y, None, None, d), opt_d)
            model.set_dlabel(train_loader)

            # Phase 3: Main Training
            for _ in range(args.local_epoch):
                for batch in train_loader:
                    x, ei, y, d, _ = prepare_graph_data(batch, 'cuda')
                    model.update((x, y, None, None, d), opt_c)

            # 5. Basic Evaluation (simplified)
            train_acc = compute_accuracy(model, train_noshuf)
            valid_acc = compute_accuracy(model, valid_loader)
            print(f"Train Acc: {train_acc:.2f} | Valid Acc: {valid_acc:.2f}")

        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue

# modify the __main__ block at the bottom:
if __name__ == "__main__":
    args = get_args()
    args = patch_args(args)  # Add this line right after get_args()
    args.use_gnn = args.algorithm.lower() == 'gnn'  # Existing line
    main(args)
