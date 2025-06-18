# eval/run_metrics.py

import argparse
from pathlib import Path
import torch
import pickle
import sys

sys.path.append('./')

from utils.util import get_args
from datautil.getdataloader_single import get_act_dataloader
from alg import alg
from eval.metrics import (
    compute_accuracy, compute_silhouette, compute_davies_bouldin,
    compute_h_divergence, extract_features_labels, plot_metrics
)

def main():
    # ✅ These are CLI-only arguments
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--output_dir', type=str, required=True)
    cli_parser.add_argument('--test_env', type=int, required=True)
    cli_args, _ = cli_parser.parse_known_args()

    # ✅ Full config from get_args
    args = get_args()
    args.output = cli_args.output_dir
    args.test_envs = [cli_args.test_env]
    args.use_gnn = True
    args.layer = 'ln'

    # ✅ Dataloaders
    train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

    # ✅ Model
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.eval()

    # ✅ Training history (optional)
    history_path = Path(args.output) / "training_history.pkl"
    history = {}
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

    # ✅ Metrics
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

    # ✅ Plot if history available
    if history:
        print("Plotting training metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)

if __name__ == "__main__":
    main()
