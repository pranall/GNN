# diversify/utils/util.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import random
import argparse
from typing import Any, List, Dict

import numpy as np
import torch
import torchvision
import PIL


def set_random_seed(seed: int = 0) -> None:
    """Make runs reproducible across PyTorch, NumPy, Python RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ensure determinism in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_environ() -> None:
    """Print out versions of key libraries."""
    print("Environment:")
    print(f"\tPython:    {sys.version.split()[0]}")
    print(f"\tPyTorch:   {torch.__version__}")
    print(f"\tTorchVision: {torchvision.__version__}")
    print(f"\tCUDA:      {torch.version.cuda}")
    print(f"\tcuDNN:     {torch.backends.cudnn.version()}")
    print(f"\tNumPy:     {np.__version__}")
    print(f"\tPIL:       {PIL.__version__}")


def print_row(row: List[Any], colwidth: int = 10, latex: bool = False) -> None:
    """Nicely print a single row of values in fixed-width columns."""
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}".ljust(colwidth)
        return str(x).ljust(colwidth)
    sep = "  " if not latex else " & "
    end = "\n" if not latex else " \\\\\n"
    print(sep.join(_fmt(x) for x in row), end=end)


def print_args(args: argparse.Namespace, keys: List[str] = None) -> str:
    """Return a multi-line string of args. If keys is given, only print those."""
    out = []
    d = vars(args)
    if keys is None:
        for k in sorted(d):
            out.append(f"{k}: {d[k]}")
    else:
        for k in keys:
            out.append(f"{k}: {d.get(k)}")
    return "\n".join(out)


def train_valid_target_eval_names(args: argparse.Namespace) -> Dict[str, int]:
    """
    Map split names to indices for your 'valid' / 'target' evaluation.
    (Used in train.py to align your accuracies.)
    """
    return {"train": 0, "valid": 1, "target": 2}


def alg_loss_dict(args: argparse.Namespace) -> List[str]:
    """Which losses your main update() will return."""
    # For both Diversify and GNN variant we track these two:
    return ["class", "dis"]


def get_args() -> argparse.Namespace:
    """Parse all command-line flags needed by train.py (Diversify + GNN)."""
    parser = argparse.ArgumentParser(description="Domain Generalization / GNN EMG")

    # === Data / task ===
    parser.add_argument("--data_dir",    type=str,   default="./data/",
                        help="Root folder for your datasets")
    parser.add_argument("--task",        type=str,   default="cross_people",
                        help="ACT task (e.g. cross_people)")
    parser.add_argument("--dataset",     type=str,   default="emg",
                        help="Dataset name")
    parser.add_argument("--test_envs",   type=int,   nargs="+", default=[0],
                        help="Which environment(s) to hold out")
    parser.add_argument("--output",      type=str,   default="./data/train_output/",
                        help="Where to save history & metrics")

    # === Diversify algorithm flags ===
    parser.add_argument("--algorithm",        type=str,
                        choices=["diversify", "gnn"],
                        default="diversify",
                        help="Use 'diversify' or its GNN variant")
    parser.add_argument("--latent_domain_num", type=int,   default=5,
                        help="Number of latent domains (Diversify)")
    parser.add_argument("--alpha1",            type=float, default=0.1,
                        help="ReverseGrad α₁ for domain discriminator")
    parser.add_argument("--alpha",             type=float, default=1.0,
                        help="ReverseGrad α for class discriminator")
    parser.add_argument("--lam",               type=float, default=0.1,
                        help="Entropy regularization weight")
    parser.add_argument("--local_epoch",       type=int,   default=1,
                        help="Inner steps per round")
    parser.add_argument("--max_epoch",         type=int,   default=1,
                        help="Number of outer rounds")
    parser.add_argument("--batch_size",        type=int,   default=32,
                        help="Batch size for training")
    parser.add_argument("--lr",                type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay",      type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--layer",             type=str,
                        choices=["ln", "bn", "linear", "wn"],
                        default="ln",
                        help="Normalization in bottlenecks")
    parser.add_argument("--bottleneck",        type=int,   default=256,
                        help="Feature bottleneck dimension")
    parser.add_argument("--classifier",        type=str,
                        choices=["linear", "bn", "wn"],
                        default="linear",
                        help="Classifier head style")
    parser.add_argument("--dis_hidden",        type=int,   default=128,
                        help="Discriminator hidden size")
    parser.add_argument("--beta1",             type=float, default=0.9,
                        help="Adam β₁")
    parser.add_argument("--lr_decay1",         type=float, default=0.01,
                        help="LR decay for classifier")
    parser.add_argument("--lr_decay2",         type=float, default=0.1,
                        help="LR decay for adversary")

    # === GNN-specific flags ===
    parser.add_argument("--use_gnn",        action="store_true",
                        help="Turn on GNN mode (build & use graphs)")
    parser.add_argument("--gnn_hidden",    type=int,   default=64,
                        help="Hidden size for GNN layers")
    parser.add_argument("--gnn_output",    type=int,   default=128,
                        help="Output size of GNN featurizer")
    parser.add_argument("--gnn_layers",    type=int,   default=2,
                        help="Number of GNN layers")
    parser.add_argument("--graph_threshold", type=float,
                        default=0.3,
                        help="Corr threshold for EMG→graph")

    args = parser.parse_args()

    # If the user asked for the GNN variant, force the flag on
    if args.algorithm.lower() == "gnn":
        args.use_gnn = True

    # (Optional) halve batch size in full GNN mode to fit GPU
    if args.use_gnn:
        args.batch_size = max(16, args.batch_size // 2)

    return args
