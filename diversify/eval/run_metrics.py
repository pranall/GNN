# eval/run_metrics.py

import sys
import os
sys.path.append(os.path.abspath("."))  # ✅ Ensure correct base path

import argparse
from pathlib import Path
import torch
import pickle

# ✅ Import after sys.path is updated
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)
from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='emg')
    args_extra = parser.parse_args()

    # ✅ Load original args
    args = get_args()
    args.num_classes = 36  # ✅ Use 36 for EMG full labels
    args.data_dir = './data/'
    args.dataset = args_extra.dataset
    args.output = args_extra.output_dir
    args.test_envs = [args_extra.test_env]
    args.use_gnn = True
    args.layer = 'ln'

    # ✅ Load data
    train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

    # ✅ Load model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # ✅ Load history (optional)
    history_path = Path(args.output) / "training_history.pkl"
    history = {}
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    # ✅ Evaluation
    print("\n=== Evaluation Metrics on Target Domain ===")
    print("Test Accuracy (OOD):", compute_accuracy(model, target_loader))

    train_feats, train_labels = extract_features_labels(model, train_loader)
    target_feats, target_labels = extract_features_labels(model, target_loader)

    print("Silhouette Score:", compute_silhouette(train_feats, train_labels))
    print("Davies-Bouldin Score:", compute_davies_bouldin(train_feats, train_labels))
    print("H-divergence:", compute_h_divergence(
        torch.tensor(train_feats).cuda(),
        torch.tensor(target_feats).cuda(),
        model.discriminator
    ))

    if history:
        print("Plotting training metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)

if __name__ == "__main__":
    main()
