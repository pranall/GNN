import torch
import torch.nn.functional as F
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

def compute_accuracy(model, loader):
    """GNN‐compatible accuracy calculation."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'edge_index'):
                x, y = batch.x.cuda(), batch.y.cuda()
                preds = model(x, batch.edge_index.cuda(), getattr(batch, 'batch', None))
            else:
                x, y = batch[0].cuda().float(), batch[1].cuda().long()
                preds = model.predict(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)

def extract_features_labels(model, loader):
    """Pull out (features, labels) from train/target loaders."""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'edge_index'):
                x = batch.x.cuda().float()
                edge_index = batch.edge_index.cuda()
                batch_idx = getattr(batch, 'batch', None)
                if batch_idx is not None:
                    batch_idx = batch_idx.cuda()
                f = model.extract_features(x, edge_index, batch_idx).cpu()
                l = batch.y.cpu()
            else:
                x = batch[0].cuda().float()
                f = model.extract_features(x).cpu()
                l = batch[1].cpu()
            feats.append(f)
            labels.append(l)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def compute_h_divergence(source_loader, target_loader, discriminator):
    """
    H‐divergence: train a 2‐way discriminator on 
    source vs target and report its CE loss.
    """
    source_feats, _ = extract_features_labels(discriminator, source_loader)
    target_feats, _ = extract_features_labels(discriminator, target_loader)
    
    device = next(discriminator.parameters()).device
    src = torch.from_numpy(source_feats).float().to(device)
    tgt = torch.from_numpy(target_feats).float().to(device)
    inputs = torch.cat([src, tgt], dim=0)
    
    with torch.no_grad():
        logits = discriminator(inputs)
    
    domains = torch.cat([
        torch.zeros(len(src), dtype=torch.long),
        torch.ones(len(tgt), dtype=torch.long)
    ], dim=0).to(device)
    
    return F.cross_entropy(logits, domains).item()

def compute_gesture_separability(features, labels):
    """How well a simple LDA separates the gesture classes."""
    lda = LinearDiscriminantAnalysis()
    return lda.fit(features, labels).score(features, labels)

def compute_sensor_importance(model, loader):
    """Variance of first‐layer attention/weights across sensors."""
    device = next(model.parameters()).device
    weights = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'edge_index'):
                x = batch.x.cuda().float()
                ei = batch.edge_index.cuda()
                conv1 = getattr(model.featurizer, 'conv1', None)
                if conv1 is None:
                    break
                out, att = conv1(x, ei, return_attention_weights=True)
                alpha = att[1]  # [num_edges]
                src = att[0][0]
                per_node = torch.zeros(model.featurizer.in_features, device=device)
                per_node = per_node.scatter_add(0, src, alpha)
                weights.append(per_node.cpu())
    if not weights:
        return 0.0
    all_w = torch.stack(weights)
    return all_w.var(dim=0).mean().item()

def compute_edge_consistency(loader):
    """Stability of the inferred graph across samples."""
    adjs = []
    for batch in loader:
        if hasattr(batch, 'edge_index'):
            ei = batch.edge_index.cpu().numpy()
            n = batch.num_nodes
            mat = csr_matrix(
                (np.ones(ei.shape[1]), (ei[0], ei[1])),
                shape=(n, n)
            )
            adjs.append(mat)
    scores = []
    for i in range(len(adjs)):
        for j in range(i+1, len(adjs)):
            a1 = adjs[i].toarray().flatten()
            a2 = adjs[j].toarray().flatten()
            scores.append(1.0 - jaccard(a1, a2))
    return float(np.mean(scores)) if scores else 0.0
