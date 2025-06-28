import os
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_args, print_environ
from utils.monitor import TrainingMonitor
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from gnn.temporal_gcn import TemporalGCN
from eval.evaluate import evaluate_model, visualize_results

def unpack_batch(batch_item):
    """
    Given either a tuple (DataBatch, y, d) or a DataBatch with .y/.domain,
    returns (data, y, d).
    """
    if isinstance(batch_item, tuple):
        data, y, d = batch_item
    else:
        data = batch_item
        y    = data.y
        d    = data.domain
    return data, y, d

class DomainAdversarialLoss(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, features, labels):
        preds = self.classifier(features).squeeze()
        return self.loss_fn(preds, labels.float())

def main(args):
    # load config
    with open('configs/emg_gnn.yaml') as f:
        config = yaml.safe_load(f)

    # setup
    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    monitor = TrainingMonitor()

    # data
    train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata = \
        get_act_dataloader(args)

    # graph sanity
    batch_item   = next(iter(train_loader))
    sample_batch = batch_item[0] if isinstance(batch_item, tuple) else batch_item
    assert sample_batch.edge_index.size(1) > 0, "No edges in graph! Graph builder misconfigured."
    print("\n=== GRAPH SANITY CHECK ===")
    print(f"Edges in first batch: {sample_batch.edge_index.size(1)}")
    print(f"Edge examples:\n{sample_batch.edge_index[:, :5].t()}")

    # data sanity
    batch_item   = next(iter(train_loader))
    debug_batch  = batch_item[0] if isinstance(batch_item, tuple) else batch_item
    labels       = batch_item[1] if isinstance(batch_item, tuple) else debug_batch.y
    print("\n=== DATA SANITY CHECK ===")
    print(f"Batch Features: {debug_batch.x.shape}")
    print(f"Edges:          {debug_batch.edge_index.size(1)}")
    print(f"Labels:         {torch.bincount(labels)}")

    # model init
    algorithm = alg.get_algorithm_class(args.algorithm)(args).to(device)
    if args.use_gnn:
        gnn = TemporalGCN(
            input_dim=args.input_shape[-1],
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim
        ).to(device)
        algorithm.featurizer = gnn

        # visualize sensor graph
        sample_graph = Data(x=debug_batch.x[:8], edge_index=debug_batch.edge_index)
        nx_graph = to_networkx(sample_graph, to_undirected=True)
        plt.figure(figsize=(6,6))
        nx.draw(nx_graph, with_labels=True, node_color='lightblue')
        plt.title("EMG Sensor Graph")
        plt.show()

    # domain adversarial loss if used
    if getattr(args, 'domain_adv_weight', 0) > 0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(int(args.bottleneck)).to(device)

    # optimizer & scheduler
    optimizer = optim.AdamW(algorithm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    best_val = 0.0
    best_h_div = float('inf')
    logs = {'class_loss': [], 'dis_loss': []}

    # training
    for epoch in range(1, args.max_epoch+1):
        start_time = time.time()
        algorithm.train()
        epoch_class_loss = []
        epoch_dis_loss = []

        for batch_src, batch_adv in zip(train_loader, train_loader):
            # source batch
            data, y, d = unpack_batch(batch_src)
            data, y, d = data.to(device), y.to(device), d.to(device)
            # adv batch
            adv_data, y_adv, d_adv = unpack_batch(batch_adv)
            adv_data, y_adv, d_adv = adv_data.to(device), y_adv.to(device), d_adv.to(device)

            # 1) feature/class update
            res_a = algorithm.update_a([data, y, d, y, d], optimizer)
            epoch_class_loss.append(res_a.get('class', 0))
            # 2) domain‐discriminator update
            res_d = algorithm.update_d([adv_data, y_adv, d], optimizer)
            epoch_dis_loss.append(res_d.get('dis', 0))
            # 3) domain‐invariant update
            _ = algorithm.update((adv_data, y_adv), optimizer)

        # eval
        train_acc = modelopera.accuracy(algorithm, train_ns_loader, device)
        val_acc   = modelopera.accuracy(algorithm, val_loader,        device)

        logs['class_loss'].extend(epoch_class_loss)
        logs['dis_loss'].extend(epoch_dis_loss)
        monitor.update('train', float(np.mean(epoch_class_loss)), train_acc)
        monitor.update('val',   float(np.mean(epoch_dis_loss)),   val_acc)

        # domain metrics every 10 epochs
        if epoch % 10 == 0:
            loaders = {'source': train_ns_loader, 'target': val_loader}
            eval_res = evaluate_model(algorithm, loaders, device)
            h_div    = eval_res['domain_metrics']['h_divergence']
            sil      = eval_res['domain_metrics']['silhouette']
            if h_div < best_h_div and eval_res['source']['accuracy'] > 0.7:
                best_h_div = h_div
                torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_domain_model.pth'))
            print(f"Epoch {epoch}: H-Div: {h_div:.3f}, Silhouette: {sil:.3f}")

        # save best val
        if val_acc > best_val:
            best_val = val_acc
            torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))

        # log epoch
        print(f"Epoch {epoch}/{args.max_epoch} | "
              f"Train: {train_acc:.4f} (Loss: {np.mean(epoch_class_loss):.4f}) | "
              f"Val:   {val_acc:.4f} (Loss: {np.mean(epoch_dis_loss):.4f}) | "
              f"Time:  {time.time()-start_time:.1f}s")

        scheduler.step()

    # finalize
    monitor.plot(args.output)
    print(f"\nTraining complete. Best validation accuracy: {best_val:.4f}")

    torch.save({'logs': logs, 'config': config},
               os.path.join(args.output, 'training_logs.pt'))

    final_loaders = {'source': train_ns_loader, 'target': test_loader}
    final_results = evaluate_model(algorithm, final_loaders, device)
    visualize_results(final_results, args.output)
    np.savez(os.path.join(args.output, 'final_metrics.npz'), **final_results)


if __name__ == '__main__':
    args = get_args()
    args.use_gnn         = getattr(args, 'use_gnn', False)
    args.gnn_hidden_dim  = getattr(args, 'gnn_hidden_dim', 64)
    args.gnn_output_dim  = getattr(args, 'gnn_output_dim', 128)
    args.latent_domain_num = getattr(args, 'latent_domain_num', 4)
    main(args)
