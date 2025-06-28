import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_args, print_environ
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.utils import to_networkx
import networkx as nx

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
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata = get_act_dataloader(args)
    
    # Batch verification
    debug_batch = next(iter(train_loader))
    print(f"\nBatch Features: {debug_batch.x.shape}")
    print(f"Edges: {debug_batch.edge_index.shape[1]}")
    print(f"Labels: {torch.bincount(debug_batch.y)}")

    # Initialize algorithm
    algorithm = alg.get_algorithm_class(args.algorithm)(args).to(device)
    
    if args.use_gnn:
        gnn = TemporalGCN(
            input_dim=args.input_shape[-1],
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim
        ).to(device)
        algorithm.featurizer = gnn
        
        # Visualize graph
        sample_graph = Data(x=debug_batch.x[:8], edge_index=debug_batch.edge_index)
        nx_graph = to_networkx(sample_graph, to_undirected=True)
        nx.draw(nx_graph, with_labels=True, node_color='lightblue')
        plt.show()

    optimizer = optim.AdamW(algorithm.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    
    best_val = 0.0
    for epoch in range(1, args.max_epoch+1):
        start_time = time.time()
        algorithm.train()
        
        for (x, y, d), (x_adv, y_adv, _) in zip(train_loader, train_loader):
            x, y, d = x.to(device), y.to(device), d.to(device)
            x_adv, y_adv = x_adv.to(device), y_adv.to(device)
            
            algorithm.update_a([x, y, d, y, d], optimizer)
            algorithm.update_d([x_adv, y_adv, d], optimizer)
            algorithm.update((x_adv, y_adv), optimizer)

        train_acc = modelopera.accuracy(algorithm, train_ns_loader, device)
        val_acc = modelopera.accuracy(algorithm, val_loader, device)
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))

        print(f"Epoch {epoch}/{args.max_epoch} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {time.time()-start_time:.1f}s")
        scheduler.step()

    print(f"Best Val Acc: {best_val:.4f}")

if __name__ == '__main__':
    args = get_args()
    args.use_gnn = getattr(args, 'use_gnn', False)
    args.gnn_hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
    args.gnn_output_dim = getattr(args, 'gnn_output_dim', 128)
    args.latent_domain_num = getattr(args, 'latent_domain_num', 4)
    main(args)
