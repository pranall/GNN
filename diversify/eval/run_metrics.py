# eval/run_metrics.py

import argparse
from pathlib import Path
import torch
import pickle
import sys

sys.path.append('./')  # Ensures root directory is in path

from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg, modelopera
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    args_extra = parser.parse_args()

    # Load training arguments
    args = get_args()
    args.output = args_extra.output_dir
    args.test_envs = [args_extra.test_env]
    args.use_gnn = True  # Ensures GNN is used for evaluation
    args.layer = 'ln'

    # Load dataloaders
    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    # Initialize model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # Load training history
    history_path = Path(args.output) / "training_history.pkl"
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    else:
        history = {}

    # === Evaluation Metrics ===
    print("\n=== Evaluation Metrics on Target Domain ===")
    test_acc = compute_accuracy(model, target_loader)
    print("Test Accuracy (OOD):", test_acc)

    train_feats, train_labels = extract_features_labels(model, train_loader)
    target_feats, target_labels = extract_features_labels(model, target_loader)

    print("Silhouette Score:", compute_silhouette(train_feats, train_labels))
    print("Davies-Bouldin Score:", compute_davies_bouldin(train_feats, train_labels))
    print("H-divergence:", compute_h_divergence(
        torch.tensor(train_feats).cuda(),
        torch.tensor(target_feats).cuda(),
        model.discriminator
    ))

    # === Plot metrics ===
    if history:
        print("Generating plots from training history...")
        plot_metrics({"GNN": history}, save_dir=args.output)
        print("Saved training plots to:", args.output)

if __name__ == "__main__":
    main()
