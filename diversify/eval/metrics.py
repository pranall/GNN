import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

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

def compute_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].cuda().float(), batch[1].cuda().long()
            batch_size = x.size(0)
            device = x.device

            # Create a dummy full edge_index
            edge_index = torch.stack([
                torch.arange(batch_size).repeat_interleave(batch_size),
                torch.arange(batch_size).repeat(batch_size)
            ], dim=0).to(device)

            try:
                preds = model.predict(x, edge_index=edge_index, batch_size=batch_size)
            except TypeError:
                preds = model.predict(x)

            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


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

def extract_features_labels(model, loader):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            x, y = x.cuda().float(), y.cuda().long()
            feats = model.extract_features(x)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)

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
