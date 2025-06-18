import argparse
import pickle
from pathlib import Path
import torch
import sys

sys.path.append('./')  # Make sure root dir is accessible

from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    args_extra = parser.parse_args()

    args = get_args()
    args.output = args_extra.output_dir
    args.test_envs = [args_extra.test_env]
    args.use_gnn = True
    args.layer = 'ln'

    # Load model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # Load dataloaders
    train_loader, _, _, target_loader, *_ = get_act_dataloader(args)

    # Load training history
    hist_path = Path(args.output) / "training_history.pkl"
    if hist_path.exists():
        with open(hist_path, "rb") as f:
            history = pickle.load(f)
    else:
        history = {}

    # === Run metrics ===
    print("\n=== Evaluation Metrics ===")
    print("Test Accuracy (OOD):", compute_accuracy(model, target_loader))

    train_feats, train_labels = extract_features_labels(model, train_loader)
    target_feats, _ = extract_features_labels(model, target_loader)

    print("Silhouette Score:", compute_silhouette(train_feats, train_labels))
    print("Davies-Bouldin Score:", compute_davies_bouldin(train_feats, train_labels))
    print("H-divergence:", compute_h_divergence(train_feats, target_feats, model.discriminator))

    # === Plotting ===
    if history:
        print("Plotting metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)
        print("Saved plots to:", args.output)

if __name__ == "__main__":
    main()
