import time
import os
import torch
from alg.opt import get_optimizer
from alg.alg import get_algorithm_class
from utils.util import set_random_seed, get_args, print_row, print_environ
from datautil.getdataloader_single import get_act_dataloader
from gnn.graph_builder import build_emg_graph

# Metrics - Only what you need
from eval.metrics import (
    compute_accuracy,
    compute_h_divergence,
    compute_gesture_separability,
    compute_sensor_importance,
    compute_edge_consistency
)

def prepare_graph_data(batch, device):
    """Optimized graph data preparation"""
    x_np = batch[0].cpu().numpy().squeeze()  # Remove singleton dim (B, C, T)
    g = build_emg_graph(x_np)
    return (
        g.x.to(device),
        g.edge_index.to(device),
        batch[1].to(device),  # y
        batch[4].to(device),  # d
        g.batch.to(device) if hasattr(g, 'batch') else None
    )

def main(args):
    # 1. Initialization
    set_random_seed(args.seed)
    args.num_classes = 6  # For your 6 EMG gestures
    print_environ()

    # 2. Data Loading
    train_loader, train_noshuf, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    # 3. Model Setup
    model = get_algorithm_class(args.algorithm)(args).cuda()
    opt_d = get_optimizer(model, args, nettype='Diversify-adv')
    opt_c = get_optimizer(model, args, nettype='Diversify-cls')
    opt_a = get_optimizer(model, args, nettype='Diversify-all')

    # 4. Training Loop
    for epoch in range(args.max_epoch):
        print(f"\nEPOCH {epoch}/{args.max_epoch}")
        
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

        # 5. Evaluation
        metrics = {
            'train_acc': compute_accuracy(model, train_noshuf),
            'valid_acc': compute_accuracy(model, valid_loader),
            'target_acc': compute_accuracy(model, target_loader),
            'h_divergence': compute_h_divergence(...),  # Your implementation
            'sensor_importance': compute_sensor_importance(model, train_loader)
        }
        print_row([metrics.values()], colwidth=12)

if __name__ == "__main__":
    args = get_args()
    args.use_gnn = args.algorithm.lower() == 'gnn'  # Auto-set GNN flag
    main(args)
