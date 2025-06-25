import torch
from network import act_network

def get_fea(args):
    if getattr(args, 'use_gnn', False):
        from gnn.temporal_gcn import TemporalGCN
        
        # MYO armband defaults
        return TemporalGCN(
            input_dim=args.time_steps if hasattr(args, 'time_steps') else 100,
            hidden_dim=getattr(args, 'gnn_hidden_dim', 64),
            output_dim=getattr(args, 'gnn_output_dim', 128),
            heads=getattr(args, 'gnn_heads', 4)
        )
    else:
        return act_network.ActNetwork(args.dataset)

def accuracy(model, loader, weights=None):
    """GNN-compatible accuracy calculation"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Handle both graph and non-graph batches
            if hasattr(batch, 'edge_index'):  # PyG Data
                x = batch.x.cuda().float()
                y = batch.y.cuda().long()
                preds = model(x, batch.edge_index.cuda())
            else:  # Traditional batch
                x = batch[0].cuda().float()
                y = batch[1].cuda().long()
                preds = model.predict(x)
            
            # Weighted accuracy
            w = weights.cuda() if weights is not None else torch.ones_like(y).float()
            correct += ((preds.argmax(1) == y).float().mul(w).sum().item())
            total += w.sum().item()
    
    return correct / max(total, 1e-8)
