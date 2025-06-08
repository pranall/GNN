import sys
sys.path.append('./')

import argparse
import pickle
from pathlib import Path
import torch

from eval.metrics import (
    compute_silhouette, compute_davies_bouldin,
    compute_accuracy, compute_h_divergence,
    extract_features_labels
)
from eval.plotter import plot_metrics
from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg


def run_evaluation(output_dir, test_env):

    # === Load training args ===
    args = get_args()
    args.use_gnn = True
    args.test_envs = [test_env]
    args.output = output_dir

    # === Rebuild model and dataloaders ===
    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # === Load training history ===
    with open(Path(output_dir) / "training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # === Feature extraction ===
    train_feats, train_labels = extract_features_labels(model, train_loader)
    target_feats, target_labels = extract_features_labels(model, target_loader)

    # === Metric Outputs ===
    print("\n====== EVALUATION METRICS ======")
    print(f"Test Accuracy (OOD): {compute_accuracy(model, target_loader):.4f}")
    print(f"Silhouette Score   : {compute_silhouette(train_feats, train_labels):.4f}")
    print(f"Davies-Bouldin     : {compute_davies_bouldin(train_feats, train_labels):.4f}")
    print(f"H-divergence       : {compute_h_divergence(torch.tensor(train_feats).cuda(), torch.tensor(target_feats).cuda(), model.discriminator):.4f}")

    # === Plots ===
    print("\nGenerating accuracy and loss plots...")
    plot_metrics({f"Diversify-GNN-TE{test_env}": history}, save_dir=str(Path(output_dir) / "plots"))
    print("Plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory containing training_history.pkl")
    parser.add_argument("--test_env", type=int, required=True, help="Test environment index used in training")
    args = parser.parse_args()

    run_evaluation(args.output_dir, args.test_env)
