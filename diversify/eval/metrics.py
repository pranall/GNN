import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 🔍 Label sanity check at file level
y_all = np.load('/content/GNN/diversify/data/emg/emg_y.npy')
print("🎯 Unique labels in dataset:", np.unique(y_all))


def compute_silhouette(features, labels):
    try:
        return silhouette_score(features, labels)
    except Exception as e:
        print(f"Silhouette error: {e}")
        return -1


def compute_davies_bouldin(features, labels):
    try:
        return davies_bouldin_score(features, labels)
    except Exception as e:
        print(f"Davies-Bouldin error: {e}")
        return -1
        
def compute_h_divergence(source_feats, target_feats, discriminator):
    source = torch.tensor(source_feats).cuda()
    target = torch.tensor(target_feats).cuda()
    feats = torch.cat([source, target], dim=0)
    labels = torch.cat([
        torch.zeros(source.shape[0], dtype=torch.long),
        torch.ones(target.shape[0], dtype=torch.long)
    ]).cuda()
    preds = discriminator(feats)
    return F.cross_entropy(preds, labels).item()


def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]  # ✅ keep on CPU

            # ✅ Validate labels BEFORE moving to CUDA
            if y.min() < 0 or y.max() >= model.args.num_classes:
                print(f"⚠️ Invalid labels found! min={y.min().item()}, max={y.max().item()}, expected 0 to {model.args.num_classes - 1}")
                print("🚫 Skipping this batch to avoid crash.")
                continue

            # ✅ Safe to move to GPU
            x, y = x.cuda().float(), y.cuda().long()
            batch_size = x.size(0)
            device = x.device

            try:
                # ✅ Handle GNN featurizers
                featurizer_params = model.featurizer.forward.__code__.co_varnames
                if 'edge_index' in featurizer_params and 'batch_size' in featurizer_params:
                    edge_index = torch.tensor([
                        list(range(batch_size - 1)),
                        list(range(1, batch_size))
                    ], dtype=torch.long).to(device)
                    preds = model.predict(x, edge_index=edge_index, batch_size=batch_size)
                else:
                    preds = model.predict(x)
            except Exception as e:
                print(f"🚨 Prediction failed: {e}")
                continue

            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0
