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
    # ✅ Only parse the additional custom args
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument('--output_dir', type=str, required=True)
    extra_parser.add_argument('--test_env', type=int, required=True)
    extra_args, _ = extra_parser.parse_known_args()

    # ✅ Load training config arguments
    args = get_args()
    args.output = extra_args.output_dir
    args.test_envs = [extra_args.test_env]
    args.use_gnn = True
    args.layer = 'ln'

    # ✅ Load dataloaders
    train_loader, _, _, target_loader, _, _, _ = get_act_dataloader(args)

    # ✅ Load trained model
    model_class = alg.get_algorithm_class(args.algorithm)
    model = model_class(args).cuda()
    model.eval()

    # ✅ Load training history (optional)
    history_path = Path(args.output) / "training_history.pkl"
    history = {}
    if history_path.exists():
        with open(history_path, "rb") as f:
            history = pickle.load(f)

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

    # ✅ Plot training curves if available
    if history:
        print("Plotting training metrics...")
        plot_metrics({"GNN": history}, save_dir=args.output)

if __name__ == "__main__":
    main()
