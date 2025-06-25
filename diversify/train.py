# train.py
import time, sys, os
import torch
import torch.nn.functional as F

from alg.opt import get_optimizer
from alg.alg import get_algorithm_class
from alg.modelopera import accuracy as compute_accuracy
from utils.util import set_random_seed, get_args, print_row, print_environ
from datautil.getdataloader_single import get_act_dataloader
from gnn.graph_builder import build_emg_graph

# The five metrics you requested:
from eval.metrics import (
    extract_features_labels,
    compute_h_divergence,
    compute_gesture_separability,
    compute_sensor_importance,
    compute_edge_consistency
)

def prepare_graph_data(batch, device):
    # batch comes as a tuple: (x_tensor, y, domain, ..., index)
    x_np = batch[0].cpu().numpy()        # shape: (B, C, 1, T)
    y    = batch[1].to(device)
    d    = batch[4].to(device)
    g    = build_emg_graph(x_np)         # returns a PyG Data object
    x    = g.x.to(device)
    ei   = g.edge_index.to(device)
    ba   = g.batch.to(device) if hasattr(g, 'batch') else None
    return x, ei, y, d, ba

def main(args):
    set_random_seed(args.seed)
    print_environ()
    # adapt batch size heuristics
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    # load data
    train_loader, train_loader_nosf, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    # build & cuda model
    Alg = get_algorithm_class(args.algorithm)
    model = Alg(args).cuda().train()

    # optimizers
    opt_d = get_optimizer(model, args, nettype='Diversify-adv')
    opt_c = get_optimizer(model, args, nettype='Diversify-cls')
    opt_a = get_optimizer(model, args, nettype='Diversify-all')

    history = []

    for epoch in range(args.max_epoch):
        print(f"\n======== EPOCH {epoch} ========")

        # 1) Feature update (abottleneck/classifier)
        for _ in range(args.local_epoch):
            for batch in train_loader:
                x, ei, y, d, ba = prepare_graph_data(batch, 'cuda')
                model.update_a((x, y, None, None, d), opt_a)

        # 2) Latent-domain characterization
        for _ in range(args.local_epoch):
            for batch in train_loader:
                x, ei, y, d, ba = prepare_graph_data(batch, 'cuda')
                model.update_d((x, y, None, None, d), opt_d)
        model.set_dlabel(train_loader)

        # 3) Domain-invariant feature learning
        for _ in range(args.local_epoch):
            for batch in train_loader:
                x, ei, y, d, ba = prepare_graph_data(batch, 'cuda')
                model.update((x, y, None, None, d), opt_c)

        # 4) Gather all metrics
        train_acc  = compute_accuracy(model, train_loader_nosf, None)
        valid_acc  = compute_accuracy(model, valid_loader,   None)
        target_acc = compute_accuracy(model, target_loader,  None)

        feats, labels = extract_features_labels(model, train_loader)
        h_div        = compute_h_divergence(feats, feats[len(feats)//2:], model.discriminator)
        gest_sep     = compute_gesture_separability(feats, labels)
        sens_imp     = compute_sensor_importance(model, train_loader)
        edge_cons    = compute_edge_consistency(train_loader)

        # print them in a nice row
        print_row([
            epoch,
            f"{train_acc:.4f}", f"{valid_acc:.4f}", f"{target_acc:.4f}",
            f"{h_div:.4f}", f"{gest_sep:.4f}",
            f"{sens_imp:.4f}", f"{edge_cons:.4f}"
        ], colwidth=12, latex=False)

        history.append({
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'target_acc': target_acc,
            'h_divergence': h_div,
            'gesture_sep': gest_sep,
            'sensor_importance': sens_imp,
            'edge_consistency': edge_cons
        })

    # Optionally plot/save history
    plot_dir = os.path.join(args.output, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    from eval.metrics import plot_metrics
    plot_metrics({'run': history}, save_dir=plot_dir)
    print(f"\nAll done! Plots → {plot_dir}")

if __name__ == "__main__":
    args = get_args()
    # if you invoked with --algorithm gnn, flip the flag and still use Diversify
    if args.algorithm.lower() == 'gnn':
        args.use_gnn = True
        args.algorithm = 'diversify'
    main(args)
