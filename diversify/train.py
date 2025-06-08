# -*- coding: utf-8 -*-
"""train.py"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader


def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    print(f"[INFO] Using GNN Encoder: {args.use_gnn}")

    if args.latent_domain_num < 6:
        args.batch_size = 32 * args.latent_domain_num
    else:
        args.batch_size = 16 * args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    best_valid_acc, target_acc = 0, 0
    history = {
        "train_acc": [],
        "valid_acc": [],
        "target_acc": [],
        "class_loss": [],
        "dis_loss": [],
        "f1": [],
        "precision": [],
        "recall": []
    }

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n======== ROUND {round} ========')

        print('==== Feature update ====')
        loss_list = ['class']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                inputs, labels, domains, cls_labels, pd_labels, index, edge_indices = data
                loss_result_dict = algorithm.update_a(
                    (inputs, labels, domains, cls_labels, pd_labels, index, edge_indices), opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15)

        print('==== Latent domain characterization ====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                inputs, labels, domains, cls_labels, pd_labels, index, edge_indices = data
                loss_result_dict = algorithm.update_d(
                    (inputs, labels, domains, cls_labels, pd_labels, index, edge_indices), optd)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('==== Domain-invariant feature learning ====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend(['f1', 'precision', 'recall'])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                inputs, labels, domains, cls_labels, pd_labels, index, edge_indices = data
                step_vals = algorithm.update(
                    (inputs, labels, domains, cls_labels, pd_labels, index, edge_indices), opt)

            results = {
                'epoch': step,
                'train_acc': modelopera.accuracy(algorithm, train_loader_noshuffle, None),
                'valid_acc': modelopera.accuracy(algorithm, valid_loader, None),
                'target_acc': modelopera.accuracy(algorithm, target_loader, None),
            }

            # Extra classification metrics on target set
            y_true = []
            y_pred = []
            algorithm.eval()
            with torch.no_grad():
                for batch in target_loader:
                    x = batch[0].cuda().float()
                    y = batch[1].cuda().long()
                    preds = algorithm.predict(x).argmax(dim=1)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')

            results['f1'] = f1
            results['precision'] = precision
            results['recall'] = recall

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]

            history["train_acc"].append(results["train_acc"])
            history["valid_acc"].append(results["valid_acc"])
            history["target_acc"].append(results["target_acc"])
            history["class_loss"].append(results["class_loss"])
            history["dis_loss"].append(results["dis_loss"])
            history["f1"].append(results["f1"])
            history["precision"].append(results["precision"])
            history["recall"].append(results["recall"])

            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']

            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')

    # Save history
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.pkl", "wb") as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    args = get_args()
    args.use_gnn = True
    args.layer = 'ln'
    args.bottleneck = 256
    args.dis_hidden = 128
    args.classifier = 'linear'

    if not hasattr(args, 'lr_decay2'):
        args.lr_decay2 = 0.1
    if not hasattr(args, 'lr_decay1'):
        args.lr_decay1 = 0.01

    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 0.0005
    if not hasattr(args, 'beta1'):
        args.beta1 = 0.9

    main(args)
