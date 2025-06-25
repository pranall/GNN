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

# Initialize seed - use only one method
# set_random_seed(42)  # Either use this
# OR use the existing set_random_seed(args.seed) in main()

def prepare_graph_data(batch, device):
    """Optimized graph data preparation"""
    try:
        x_np = batch[0].cpu().numpy().squeeze()  # Remove singleton dim (B, C, T)
        g = build_emg_graph(x_np)
        return (
            g.x.to(device),
            g.edge_index.to(device),
            batch[1].to(device),  # y
            batch[4].to(device),  # d
            g.batch.to(device) if hasattr(g, 'batch') else None
        )
    except Exception as e:
        print(f"Error preparing graph data: {e}")
        raise

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

if __name__ == "__main__":
    args = get_args()
    args.use_gnn = args.algorithm.lower() == 'gnn'  # Auto-set GNN flag
    main(args)
