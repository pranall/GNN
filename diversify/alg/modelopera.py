# modelopera.py

import torch
from network import act_network

def get_fea(args):
    if hasattr(args, 'use_gnn') and args.use_gnn:
        from gnn.temporal_gcn import TemporalGCN

        input_dim = 8  # EMG input channels
        hidden_dim = getattr(args, 'gnn_hidden_dim', 32)
        output_dim = getattr(args, 'gnn_output_dim', 128)

        net = TemporalGCN(input_dim, hidden_dim, output_dim)
        net.in_features = output_dim  # Used by bottleneck layer

        return net
    else:
        return act_network.ActNetwork(args.dataset)

def accuracy(model, loader, weights=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].cuda().float()
            y = batch[1].cuda().long()
            batch_weights = weights if weights is not None else torch.ones_like(y).float()

            preds = model.predict(x)
            if preds.dim() > 2:
                preds = preds.squeeze(1)

            pred_labels = torch.argmax(preds, dim=1)
            correct += ((pred_labels == y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    return correct / total
