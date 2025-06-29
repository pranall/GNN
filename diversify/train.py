import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx

# Absolute imports
from diversify.alg import Diversify
from diversify.utils.util import set_random_seed, get_args, print_args, print_environ
from diversify.utils.monitor import TrainingMonitor
from diversify.gnn.temporal_gcn import TemporalGCN
from diversify.eval.evaluate import evaluate_model, visualize_results
from diversify.datautil import get_act_dataloader
from diversify.datautil.actdata import load_datasets

# PyG imports
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

tr, val, targetdata = cross_people.load_datasets(args)  # Note the module prefix
train_loader = getdataloader_single.get_act_dataloader(args, tr, val, targetdata)


def unpack_batch(batch_item):
    """Robust batch unpacking handling multiple input formats"""
    if isinstance(batch_item, (list, tuple)):
        if len(batch_item) == 2:  # (data, y)
            return batch_item[0], batch_item[1], None
        return batch_item  # (data, y, d)
    return batch_item, batch_item.y, getattr(batch_item, 'domain', None)

class DomainAdversarialLoss(nn.Module):
    """Enhanced domain classifier with gradient reversal"""
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, features, domain_labels):
        if domain_labels.dim() == 1:
            domain_labels = domain_labels.float().unsqueeze(1)
        return self.loss_fn(self.discriminator(features), domain_labels)

def main(args):
    # Initialization
    cfg_path = os.path.join(os.path.dirname(__file__), 'emg_gnn.yml')
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    torch.cuda.empty_cache()

    # Data loading
    try:
        train_loader, train_ns_loader, val_loader, test_loader, _, _, _ = get_act_dataloader(
            args, 
            tr, 
            val, 
            targetdata
    )
            
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}")

    # Data sanity check
    sample_batch = unpack_batch(next(iter(train_loader)))[0]
    assert sample_batch.edge_index.size(1) > 0, "No edges detected in graph!"
    print("\n=== DATA SANITY ===")
    print(f"Features: {sample_batch.x.shape}")
    print(f"Edges:    {sample_batch.edge_index.size(1)}")
    print(f"Labels:   {torch.bincount(sample_batch.y)}")

    # Model initialization
    algorithm = alg.get_algorithm_class(args.algorithm)(args).to(device)
    if args.use_gnn:
        algorithm.featurizer = TemporalGCN(
            input_dim=args.input_shape[-1],
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim
        ).to(device)
        # Visualize sample graph
        sample_graph = Data(x=sample_batch.x[:8], edge_index=sample_batch.edge_index)
        plt.figure(figsize=(6,6))
        nx.draw(to_networkx(sample_graph), with_labels=True, node_color='lightblue')
        plt.title("EMG Sensor Graph")
        plt.show()

    # Adversarial loss if configured
    if getattr(args, 'domain_adv_weight', 0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(int(args.bottleneck)).to(device)

    # Optimization setup
    optimizer = optim.AdamW(
        algorithm.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training state
    logs = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'h_divergence': [], 'silhouette': []
    }
    best_val = 0.0
    best_h_div = float('inf')
    early_stop_counter = 0

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        epoch_start = time.time()
        algorithm.train()
        epoch_class_loss, epoch_dis_loss = [], []

        for batch_idx, (batch_src, batch_adv) in enumerate(zip(train_loader, train_loader), 1):
            # Source batch
            data, y, d = unpack_batch(batch_src)
            data, y = data.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if d is not None:
                d = d.to(device, non_blocking=True)

            # Adversarial batch
            adv_data, y_adv, d_adv = unpack_batch(batch_adv)
            adv_data, y_adv = adv_data.to(device, non_blocking=True), y_adv.to(device, non_blocking=True)
            if d_adv is not None:
                d_adv = d_adv.to(device, non_blocking=True)

            # Training steps
            res_a = algorithm.update_a([data, y, d, y, d], optimizer)
            res_d = algorithm.update_d([adv_data, y_adv, d_adv], optimizer)
            _ = algorithm.update((adv_data, y_adv), optimizer)

            epoch_class_loss.append(res_a.get('class', 0))
            epoch_dis_loss.append(res_d.get('dis', 0))

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(algorithm.parameters(), max_norm=1.0)

            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # Evaluation
        train_acc = modelopera.accuracy(algorithm, train_ns_loader, device)
        val_acc   = modelopera.accuracy(algorithm, val_loader,        device)
        scheduler.step(val_acc)

        # Domain metrics every 10 epochs
        if epoch % 10 == 0:
            try:
                eval_res = evaluate_model(
                    algorithm,
                    {'source': train_ns_loader, 'target': val_loader},
                    device
                )
                h_div = eval_res['domain_metrics']['h_divergence']
                sil   = eval_res['domain_metrics']['silhouette']
                if h_div < best_h_div * 0.95:
                    best_h_div = h_div
                    torch.save(
                        algorithm.state_dict(),
                        os.path.join(args.output, f'best_domain_epoch{epoch}.pth')
                    )
                logs['h_divergence'].append(h_div)
                logs['silhouette'].append(sil)
                print(f"Epoch {epoch}: H-Div: {h_div:.3f}, Silhouette: {sil:.3f}")
            except Exception as e:
                print(f"Domain evaluation failed: {e}")

        # Logging & checkpointing
        logs['epoch'].append(epoch)
        logs['train_loss'].append(float(np.mean(epoch_class_loss)))
        logs['val_loss'].append(float(np.mean(epoch_dis_loss)))
        logs['train_acc'].append(train_acc)
        logs['val_acc'].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            early_stop_counter = 0
            torch.save(
                algorithm.state_dict(),
                os.path.join(args.output, 'best_model.pth')
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= getattr(args, 'early_stop_patience', 10):
            print(f"Early stopping at epoch {epoch}")
            break

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.max_epoch} | "
              f"Train {train_acc:.4f} (L {np.mean(epoch_class_loss):.4f}) | "
              f"Val {val_acc:.4f} (L {np.mean(epoch_dis_loss):.4f}) | "
              f"Time {epoch_time:.1f}s")

    # Finalization
    torch.save({
        'logs': logs,
        'config': config,
        'args': vars(args)
    }, os.path.join(args.output, 'training_logs.pt'))

    try:
        final_res = evaluate_model(
            algorithm,
            {'source': train_ns_loader, 'target': test_loader},
            device
        )
        visualize_results(final_res, args.output)
        np.savez(os.path.join(args.output, 'final_metrics.npz'), **final_res)
    except Exception as e:
        print(f"Final evaluation failed: {e}")

    print(f"\nTraining complete. Best validation accuracy: {best_val:.4f}")

if __name__ == '__main__':
    args = get_args()
    args.use_gnn            = getattr(args, 'use_gnn', False)
    args.gnn_hidden_dim     = getattr(args, 'gnn_hidden_dim', 64)
    args.gnn_output_dim     = getattr(args, 'gnn_output_dim', 128)
    args.latent_domain_num  = getattr(args, 'latent_domain_num', 4)
    args.early_stop_patience = getattr(args, 'early_stop_patience', 10)
    main(args)
