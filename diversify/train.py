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

def main(args):
    # force single‐worker if needed
    args.N_WORKERS = args.N_WORKERS
    set_random_seed(args.seed)
    print_environ()
    print(print_args(args, []))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("✅ GNN active" if args.use_gnn else "⚠️ Using CNN baseline")

    os.makedirs(args.output, exist_ok=True)
    args.steps_per_epoch = min(100, args.batch_size * 10)

    # ——— load data ———
    train_loader, train_ns_loader, val_loader, test_loader, *_ = get_act_dataloader(args)

    # ——— inspect one batch ———
    batch = next(iter(train_loader))
    x, y, d = batch
    print("🔎 BATCH X type:", type(x))
    if hasattr(x, 'x'):
        print("  x.x.shape:", x.x.shape,
              "  edge_index:", x.edge_index.shape,
              "  batch:", x.batch.shape)
    else:
        print("  x.shape:", x.shape)
    print("  y.shape:", y.shape, "  d.shape:", d.shape)

    # ——— algorithm init ———
    AlgoClass = alg.get_algorithm_class(args.algorithm)
    algorithm = AlgoClass(args).to(device)

    # ——— GNN integration ———
    if args.use_gnn:
        # after dataset transform, each sample is a DataBatch; node-features are in x.x
        feat_dim = x.x.size(1)
        gnn = TemporalGCN(
            input_dim=feat_dim,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim
        ).to(device)
        algorithm.featurizer = gnn

        # quick smoke test
        demo = Data(x=torch.randn(10, feat_dim, device=device),
                    edge_index=torch.zeros((2,0),dtype=torch.long,device=device))
        print("✅ GNN smoke out:", algorithm.featurizer(demo).shape)

    # ——— training loop ———
    algorithm.train()
    optimizer = optim.AdamW(algorithm.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    if args.domain_adv_weight>0:
        algorithm.domain_adv_loss = DomainAdversarialLoss(args.bottleneck).to(device)

    logs, best_val = {k:[] for k in
        ['train_acc','val_acc','test_acc','class_loss','dis_loss','ent_loss','total_loss']
    }, 0.0

    for epoch in range(1, args.max_epoch+1):
        start = time.time()
        # 1) feature‐extractor update
        for x,y,d in train_loader:
            x,y,d = x.to(device), y.to(device), d.to(device)
            res = algorithm.update_a([x,y,d,y,d], optimizer)
            logs['class_loss'].append(res['class'])
        # 2) discriminator update
        for x,y,d in train_loader:
            x,y,d = x.to(device), y.to(device), d.to(device)
            res = algorithm.update_d([x,y,d], optimizer)
            logs['dis_loss'].append(res['dis'])
            logs['ent_loss'].append(res['ent'])
            logs['total_loss'].append(res['total'])
        # 3) classifier update
        for x,y,_ in train_loader:
            x,y = x.to(device), y.to(device)
            _ = algorithm.update((x,y), optimizer)

        # eval
        acc_fn = modelopera.accuracy
        logs['train_acc'].append(acc_fn(algorithm, train_ns_loader, device))
        logs['val_acc'].append  (acc_fn(algorithm, val_loader,      device))
        logs['test_acc'].append (acc_fn(algorithm, test_loader,     device))

        scheduler.step()
        if logs['val_acc'][-1] > best_val:
            best_val = logs['val_acc'][-1]
            torch.save(algorithm.state_dict(), os.path.join(args.output,'best_model.pth'))

        print(f"Epoch {epoch}/{args.max_epoch} — "
              f"Train: {logs['train_acc'][-1]:.4f}, "
              f"Val:   {logs['val_acc'][-1]:.4f}, "
              f"Time: {time.time()-start:.1f}s")

    print(f"Done. Best val acc: {best_val:.4f}")

if __name__=='__main__':
    args = get_args()
    if not hasattr(args,'use_gnn'): args.use_gnn=False
    if args.use_gnn:
        args.gnn_hidden_dim = args.gnn_hidden_dim
        args.gnn_output_dim = args.gnn_output_dim
    if args.latent_domain_num is None:
        args.latent_domain_num = 4
    main(args)
