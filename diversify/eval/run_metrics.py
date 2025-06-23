import sys
import os
import argparse
from pathlib import Path
import torch
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard

sys.path.append(os.path.abspath("."))  # Make sure ./eval etc. are importable

# Custom imports
from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg

# --------------------------
# Core Metrics for EMG/GNN
# --------------------------

def compute_h_divergence(source_feats, target_feats, discriminator):
    """Quantifies domain shift between source and target domains"""
    with torch.no_grad():
        inputs = torch.cat([source_feats, target_feats])
        outputs = discriminator(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 0]  # Prob of being source domain
        return abs(probs.mean() - 0.5).item() * 2  # Normalized to [0,1]

def compute_gesture_separability(features, labels):
    """Measures how well gestures are separated in feature space"""
    lda = LinearDiscriminantAnalysis()
    X = features.cpu().numpy()
    y = labels.cpu().numpy()
    return lda.fit(X, y).score(X, y)  # Accuracy of LDA classifier

def compute_sensor_importance(model, loader):
    """Quantifies variance in GNN attention across sensors"""
    sensor_weights = []
    with torch.no_grad():
        for data in loader:
            x, edge_index, _, _, _ = data
            x = x.cuda().float()
            edge_index = edge_index.cuda()
            # Extract first GNN layer's attention weights (assuming GAT)
            if hasattr(model.featurizer.conv1, 'att_src'):
                _, att_weights = model.featurizer.conv1(x, edge_index, return_attention_weights=True)
                sensor_weights.append(att_weights[1].mean(dim=0).cpu())
    if not sensor_weights:
        return 0.0
    return torch.stack(sensor_weights).var(dim=0).mean().item()

def compute_edge_consistency(loader):
    """Measures stability of sensor connections across samples"""
    adj_matrices = []
    for data in loader:
        edge_index = data.edge_index.cpu().numpy()
        adj = csr_matrix((np.ones(edge_index.shape[1]), 
                         (edge_index[0], edge_index[1]),
                         shape=(8,8))  # 8 sensors in MYO armband
        adj_matrices.append(adj)
    
    consistency_scores = []
    for i, adj1 in enumerate(adj_matrices):
        for adj2 in adj_matrices[i+1:]:
            consistency_scores.append(1 - jaccard(
                adj1.toarray().flatten(),
                adj2.toarray().flatten()
            ))
    return np.mean(consistency_scores) if consistency_scores else 0.0

def extract_features_labels(model, loader):
    """Extracts features and labels for metric computation"""
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            x, edge_index, y, _, _ = data
            x = x.cuda().float()
            edge_index = edge_index.cuda()
            feats = model.extract_features(x, edge_index)
            features.append(feats.cpu())
            labels.append(y.cpu())
    return torch.cat(features), torch.cat(labels)

# --------------------------
# Main Evaluation Function
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='emg')
    args_extra = parser.parse_args()

    # Load core configuration
    args = get_args()
    args.num_classes = 6  # Using 6 gesture classes as per your paper
    args.data_dir = './data/'
    args.dataset = args_extra.dataset
    args.output = args_extra.output_dir
    args.test_envs = [args_extra.test_env]
    args.use_gnn = True

    # Prepare data loaders
    train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

    # Load trained model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # Load training history if available
    history_path = Path(args.output) / "training_history.pkl"
    history = {}
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    print("\n=== EMG-Specific Evaluation Metrics ===")

    # 1. Basic Accuracy
    def simple_accuracy(model, loader):
        correct = total = 0
        with torch.no_grad():
            for data in loader:
                x, edge_index, y, _, _ = data
                x = x.cuda().float()
                edge_index = edge_index.cuda()
                preds = model(x, edge_index).argmax(dim=1)
                correct += (preds == y.cuda()).sum().item()
                total += len(y)
        return correct / total

    print(f"Test Accuracy: {simple_accuracy(model, target_loader):.4f}")

    # 2. Advanced EMG/GNN Metrics
    try:
        train_feats, train_labels = extract_features_labels(model, train_loader)
        target_feats, target_labels = extract_features_labels(model, target_loader)

        print("\n--- Domain Generalization ---")
        print(f"H-Divergence: {compute_h_divergence(train_feats, target_feats, model.discriminator):.4f}")

        print("\n--- Gesture Separability ---")
        print(f"Gesture Separability (LDA): {compute_gesture_separability(train_feats, train_labels):.4f}")

        print("\n--- Sensor-Level Metrics ---")
        print(f"Sensor Importance Variance: {compute_sensor_importance(model, train_loader):.4f}")
        print(f"Edge Consistency: {compute_edge_consistency(train_loader):.4f}")

    except Exception as e:
        print(f"\n⚠️ Metric computation failed: {str(e)}")

    # Plot training curves if history exists
    if history:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['valid_acc'], label='Validation')
        plt.title("Accuracy")
        plt.legend()
        
        plt.subplot(132)
        plt.plot(history['h_divergence'])
        plt.title("H-Divergence")
        
        plt.subplot(133)
        plt.plot(history['sensor_variance'])
        plt.title("Sensor Importance Variance")
        
        plt.tight_layout()
        plt.savefig(Path(args.output) / "metrics_summary.png")
        print("\nSaved metrics visualization to metrics_summary.png")

if __name__ == "__main__":
    main()
