import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import jensenshannon
import os
import torch.nn.functional as F

def compute_accuracy(model, loader):
    """GNN-compatible accuracy calculation"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            try:
                # Handle both graph and non-graph data
                if hasattr(batch, 'edge_index'):  # PyG Data object
                    x, y = batch.x, batch.y
                    edge_index, batch_idx = batch.edge_index, batch.batch
                    preds = model.predict(x, edge_index, batch_idx)
                else:  # Traditional batch
                    x, y = batch[0].cuda().float(), batch[1].long().cuda()
                    preds = model.predict(x)
                
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)
            except Exception as e:
                print(f"⚠️ Accuracy calc error: {e}")
                continue
    return correct / max(total, 1)

def extract_features_labels(model, loader):
    """Unified feature extraction for graphs and regular data"""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            try:
                if hasattr(batch, 'edge_index'):
                    x = batch.x.cuda().float()
                    edge_index = batch.edge_index.cuda()
                    batch_idx = batch.batch.cuda() if hasattr(batch, 'batch') else None
                    feats.append(model.extract_features(x, edge_index, batch_idx).cpu())
                    labels.append(batch.y.cpu())
                else:
                    x = batch[0].cuda().float()
                    feats.append(model.extract_features(x).cpu())
                    labels.append(batch[1].cpu())
            except Exception as e:
                print(f"⚠️ Feature extraction error: {e}")
                continue
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def compute_h_divergence(source_feats, target_feats, discriminator):
    """Improved domain divergence metric"""
    source = torch.FloatTensor(source_feats).cuda()
    target = torch.FloatTensor(target_feats).cuda()
    domain_preds = discriminator(torch.cat([source, target], dim=0))

    # Fix the mismatched brackets here:
    domains = torch.cat([
        torch.zeros(len(source)),
        torch.ones(len(target))
    ], dim=0).cuda().long()

    return F.cross_entropy(domain_preds, domains).item()

def compute_js_divergence(p, q):
    """Additional metric for distribution alignment"""
    p, q = np.asarray(p), np.asarray(q)
    return jensenshannon(p, q) ** 2  # Scipy returns sqrt(JS)

# Visualization (unchanged)
def plot_metrics(history_dict, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    for metric in ["train_acc", "valid_acc", "target_acc", "class_loss", "dis_loss"]:
        plt.figure()
        for label, values in history_dict.items():
            if metric in values:
                plt.plot(values[metric], label=label)
        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/{metric}.png")
        plt.close()
