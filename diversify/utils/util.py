# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

# ==== Data utility ====

def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir + dataset + '/' + dataset + '_x1.npy')
        ty = np.load(root_dir + dataset + '/' + dataset + '_y1.npy')
    else:
        x = np.load(root_dir + dataset + '/' + dataset + '_x.npy')
        ty = np.load(root_dir + dataset + '/' + dataset + '_y.npy')
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy

class combindataset(Dataset):
    def __init__(self, args, x, c, p, s, transform=None):
        self.args = args
        self.x = x
        self.c = c
        self.p = p
        self.s = s
        self.transform = transform
        self.length = len(x)
        self.pdlabels = np.zeros(len(x), dtype=np.int64)
        self.indices = np.arange(len(x))

    def set_labels_by_index(self, labels, index, mode='pdlabel'):
        if mode == 'pdlabel':
            self.pdlabels[index] = labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x).float()
        x = x.unsqueeze(1)  # Shape: (C, 1, T)
        c = self.c[index]
        p = self.p[index]
        s = self.s[index]
        pdlabel = self.pdlabels[index]
        return x, c, p, s, pdlabel, index

class subdataset(Dataset):
    def __init__(self, args, dataset, indices, transform=None):
        self.args = args
        self.dataset = dataset
        self.indices = np.array(indices, dtype=np.int64)
        self.transform = transform
        self.pdlabels = dataset.pdlabels[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, c, p, s, pdlabel, _ = self.dataset[real_idx]
        return x, c, p, s, pdlabel, real_idx

# ==== General training utility ====

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_row(row, colwidth=10, latex=False):
    def format_val(val):
        return f"{val:{colwidth}}" if not isinstance(val, float) else f"{val:.4f}".ljust(colwidth)
    if latex:
        end = " \\\\\n"
    else:
        end = "\n"
    print("".join([format_val(val) for val in row]), end=end)

def print_environ():
    print("Environment:")
    try:
        import PIL
        print(f"\tPython: {'.'.join(map(str, list(map(int, list(torch.__version__.split('+')[0].split('.'))))))}")
        print(f"\tPyTorch: {torch.__version__}")
        print(f"\tTorchvision: {__import__('torchvision').__version__}")
        print(f"\tCUDA: {torch.version.cuda}")
        print(f"\tCUDNN: {torch.backends.cudnn.version()}")
        print(f"\tNumPy: {np.__version__}")
        print(f"\tPIL: {PIL.__version__}")
    except Exception as e:
        print(f"\tError fetching environment details: {e}")

def train_valid_target_eval_names(args):
    return {
        'train': 0,
        'valid': 1,
        'target': 2
    }

def alg_loss_dict(args):
    return ['class', 'dis']

def print_args(args, exclude_keys):
    s = ''
    for k, v in sorted(vars(args).items()):
        if k not in exclude_keys:
            s += f"{k}: {v}\n"
    return s

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default='cross_people')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--algorithm', type=str, default='diversify')
    parser.add_argument('--latent_domain_num', type=int, default=5)
    parser.add_argument('--alpha1', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='./data/train_output/')
    parser.add_argument('--layer', type=str, default='ln')  # default to LayerNorm for GNN
    parser.add_argument('--use_gnn', action='store_true')
    args = parser.parse_args()
    return args
