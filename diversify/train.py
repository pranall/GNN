import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_args, print_environ
from datautil.getdataloader_single import get_act_dataloader
from torch_geometric.data import Data
from gnn.temporal_gcn import TemporalGCN

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
        return self.loss_fn(self.classifier(features).squeeze(), labels.float())

def safe_update(optimizer, model, loss, max_norm=1.0):
    """Unified update with gradient clipping"""
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad()

def main(args):
    # Initialize
    args.N_WORKERS = 0
    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("✅ GNN active" if args.use_gnn else "⚠️ Using CNN baseline")
    
    os.makedirs(args.output, exist_ok=True)
    args.steps_per_epoch = min(100, args.batch_size * 10)
    
    # Load data
    train_loader, train_ns_loader, val_loader, test_loader, *_ = get_act_dataloader(args)
    
    # Inspect first batch
    batch = next(iter(train_loader))
    x, y, d = batch
    print("🔍 BATCH INSPECTION:")
    if hasattr(x, 'x'):
        print(f"  Nodes: {x.x.shape} (should be [batch*8, 3])")
        print(f"  Edges: {x.edge_index.shape} (should be [2, E] where E > 0)")
        print(f"  Edge samples:\n{x.edge_index[:, :5]}")
    else:
        print(f"  Input shape: {x.shape}")
    print(f"  Labels: {y.shape}, Domains: {d.shape}")

    # Initialize model
    AlgoClass = alg.get_algorithm_class(args.algorithm)
    algorithm = AlgoClass(args).to(device)
    
    # GNN setup
    if args.use_gnn:
        feat_dim = x.x.size(1)
        gnn = TemporalGCN(
            input_dim=feat_dim,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim
        ).to(device)
        algorithm.featurizer = gnn
        
        # Validate GNN
        test_input = Data(
            x=torch.randn(64*8, feat_dim).to(device),
            edge_index=torch.randint(0, 64*8, (2, 64*8*3)).to(device),
            batch=torch.arange(64).repeat_interleave(8).to(device)
        )
        out = algorithm.featurizer(test_input)
        assert out.shape == (64, args.gnn_output_dim), \
               f"GNN shape mismatch! Got {out.shape}, expected {(64, args.gnn_output_dim)}"

    # Training setup
    optimizer = optim.AdamW(algorithm.parameters(), lr=0.0)  # Start with 0 for warmup
    best_val = 0.0
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        start = time.time()
        algorithm.train()
        
        # LR warmup (5 epochs)
        warmup_lr = args.lr * min(1.0, epoch/5)
        for g in optimizer.param_groups:
            g['lr'] = warmup_lr

        # Training phase
        total_loss = 0.0
        for x, y, d in train_loader:
            x, y, d = x.to(device), y.to(device), d.to(device)
            
            # Combined update with gradient clipping
            optimizer.zero_grad()
            loss = algorithm.update_a([x, y, d, y, d])
            safe_update(optimizer, algorithm, loss)
            total_loss += loss.item()

        # Evaluation
        algorithm.eval()
        with torch.no_grad():
            train_acc = modelopera.accuracy(algorithm, train_ns_loader, device)
            val_acc = modelopera.accuracy(algorithm, val_loader, device)
            test_acc = modelopera.accuracy(algorithm, test_loader, device)

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save(algorithm.state_dict(), os.path.join(args.output, 'best_model.pth'))
            if val_acc > 0.80 and epoch >= 5:  # Early stopping
                print(f"🎯 Early stopping at epoch {epoch} (val_acc={val_acc:.2%})")
                break

        # Logging
        epoch_time = time.time() - start
        print(f"Epoch {epoch}/{args.max_epoch} - "
              f"LR: {warmup_lr:.1e} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.2%} | "
              f"Val: {val_acc:.2%} | "
              f"Time: {epoch_time:.1f}s")

    print(f"Training complete. Best val acc: {best_val:.2%}")

if __name__ == '__main__':
    args = get_args()
    if not hasattr(args, 'use_gnn'): 
        args.use_gnn = False
    main(args)
