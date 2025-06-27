import time
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Batch

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
from gnn.temporal_gcn import TemporalGCN
from gnn.graph_builder import GraphBuilder  # if needed directly
from datautil.graph_utils import convert_to_graph
import numpy as np
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_y, all_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch).argmax(dim=1)
        all_y.extend(batch.y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
    acc = correct / total if total > 0 else 0
    f1 = f1_score(all_y, all_pred, average='macro')
    precision = precision_score(all_y, all_pred, average='macro')
    recall = recall_score(all_y, all_pred, average='macro')
    return acc, f1, precision, recall

def main():
    class Args:
        seed = 42
        max_epoch = 20
        local_epoch = 3
        output = "./data/train_output/"
        batch_size = 32
        N_WORKERS = 2
        model_type = "gnn"
        algorithm = "gnn"
        num_classes = 6
        data_dir = "./data/"
        task = "cross_people"
        dataset = "emg"
        act_people = {
            "emg": list(range(36))
        }
        latent_domain_num = 3

    args = Args()
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s = print_args(args, [])
    print_environ()
    print(s)
    print(f"[INFO] Using GNN Encoder: {args.model_type}")

    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, tr_subset, val_subset, target_subset = get_act_dataloader(args)

    model = TemporalGCN(
        input_dim=tr_subset[0].x.size(1),
        hidden_dim=64,
        output_dim=args.num_classes
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_valid_acc, best_target_acc = 0.0, 0.0

    history = {
        "train_acc": [], "valid_acc": [], "target_acc": [],
        "class_loss": [], "dis_loss": [],
        "f1": [], "precision": [], "recall": []
    }

    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss / num_batches:.4f}")

        train_acc, _, _, _ = evaluate(model, train_loader_noshuffle, device)
        valid_acc, _, _, _ = evaluate(model, valid_loader, device)
        target_acc, f1, precision, recall = evaluate(model, target_loader, device)

        print(f"Train Acc: {train_acc:.3f} | Valid Acc: {valid_acc:.3f} | Target Acc: {target_acc:.3f}")

        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["target_acc"].append(target_acc)
        history["f1"].append(f1)
        history["precision"].append(precision)
        history["recall"].append(recall)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_target_acc = target_acc

    print(f"Best Target Acc at Best Valid Acc: {best_target_acc:.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main()
