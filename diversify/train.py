import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import time
import torch

from alg.opt import *
from alg import alg, modelopera
from utils.util import (
    set_random_seed,
    get_args,
    print_row,
    print_args,
    train_valid_target_eval_names,
    alg_loss_dict,
    print_environ
)
from datautil.getdataloader_single import get_act_dataloader

# now that 'eval' and 'gnn' are on the path:
from gnn.graph_builder import build_emg_graph

# metric helpers
from eval.metrics import (
    extract_features_labels,
    compute_silhouette,
    compute_davies_bouldin,
    compute_h_divergence,
    plot_metrics
)

# batch validation / graph stats
from datautil.util import validate_emg_batch, get_graph_metrics


def prepare_graph_data(batch, device):
    """Convert batch to PyG Data object with edge_index."""
    x, y, d = batch[0], batch[1], batch[4]
    graph_data = build_emg_graph(x.cpu().numpy())
    return (
        graph_data.x.to(device),
        graph_data.edge_index.to(device),
        y.to(device),
        d.to(device),
        graph_data.batch.to(device) if hasattr(graph_data, 'batch') else None
    )

def main(args):
    # Initialization (unchanged)
    set_random_seed(args.seed)
    args.num_classes = 36
    print_environ()
    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    # Data loading
    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
    batch = next(iter(train_loader))
    validate_emg_batch(batch)  # Will raise AssertionError if invalid
    print("Graph stats:", get_graph_metrics(batch))
    # Model setup
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    
    # Optimizers
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    # Metrics history
    history = {
        'train_acc': [], 'valid_acc': [], 'target_acc': [],
        'silhouette': [], 'davies_bouldin': [], 'h_divergence': []
    }

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        
        # Feature Update Phase
        print('====Feature update====')
        for step in range(args.local_epoch):
            for data in train_loader:
                x, edge_index, y, d, batch = prepare_graph_data(data, 'cuda')
                loss_result = algorithm.update_a((x, y, None, None, d), opta)
            print_row([step, loss_result['class']], colwidth=15)

        # Domain Characterization
        print('====Latent domain characterization====')
        for step in range(args.local_epoch):
            for data in train_loader:
                x, edge_index, y, d, batch = prepare_graph_data(data, 'cuda')
                loss_result = algorithm.update_d((x, y, None, None, d), optd)
            print_row([step, loss_result['total'], loss_result['dis'], loss_result['ent']], colwidth=15)

        algorithm.set_dlabel(train_loader)

        # Main Training Loop
        print('====Domain-invariant feature learning====')
        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                x, edge_index, y, d, batch = prepare_graph_data(data, 'cuda')
                step_vals = algorithm.update((x, y, None, None, d), opt)

            # Evaluation
            results = {
                'epoch': step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
                'total_cost_time': time.time() - sss
            }
            results.update({f'{k}_loss': step_vals[k] for k in alg_loss_dict(args)})

            # Graph Metrics (New)
            feats, labels = extract_features_labels(algorithm, train_loader)
            results.update({
                'silhouette': compute_silhouette(feats, labels),
                'davies_bouldin': compute_davies_bouldin(feats, labels),
                'h_divergence': compute_h_divergence(
                    feats[:len(feats)//2], feats[len(feats)//2:], algorithm.discriminator)
            })
            
            # Update history
            for k in history.keys():
                if k in results:
                    history[k].append(results[k])

            print_row([results[k] for k in [
                'epoch', 'train_acc', 'valid_acc', 'target_acc',
                'class_loss', 'dis_loss', 'silhouette', 'total_cost_time'
            ]], colwidth=15)

    # Final output
    print(f'Best Valid Acc: {max(history["valid_acc"]):.4f}')
    print(f'Final Target Acc: {history["target_acc"][-1]:.4f}')
    plot_metrics({'training': history})

if __name__ == '__main__':
    args = get_args()
    # if the user selects the 'gnn' variant, 
    # flip on the GNN‐mode flag and then fall back to the
    # standard Diversify class (which checks args.use_gnn)
    if args.algorithm.lower() == 'gnn':
        args.use_gnn = True
        args.algorithm = 'diversify'
    main(args)
