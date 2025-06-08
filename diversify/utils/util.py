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
        self.use_gnn = args.use_gnn  # ✅ GNN toggle

    def set_labels_by_index(self, labels, index, mode='pdlabel'):
        if mode == 'pdlabel':
            self.pdlabels[index] = labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x).float().unsqueeze(1)  # Shape: (C, 1, T)
        c = self.c[index]
        p = self.p[index]
        s = self.s[index]
        pdlabel = self.pdlabels[index]

        if self.use_gnn:
            edge_indices = self.generate_edge_index(x)
            return x, c, p, s, pdlabel, index, edge_indices  # ✅ 7 items
        else:
            return x, c, p, s, pdlabel, index  # ✅ 6 items

    def generate_edge_index(self, x):
        """
        Simple full-connected graph over channels (C nodes).
        Input shape: (C, 1, T)
        """
        num_nodes = x.shape[0]  # Usually 8 for EMG
        src = torch.arange(num_nodes).repeat_interleave(num_nodes)
        dst = torch.arange(num_nodes).repeat(num_nodes)
        return torch.stack([src, dst], dim=0)  # Shape: [2, num_edges]


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
        import torchvision
        print(f"\tPython: {'.'.join(map(str, list(map(int, list(torch.__version__.split('+')[0].split('.'))))))}")
        print(f"\tPyTorch: {torch.__version__}")
        print(f"\tTorchvision: {torchvision.__version__}")
        print(f"\tCUDA: {torch.version.cuda}")
        print(f"\tCUDNN: {torch.backends.cudnn.version()}")
        print(f"\tNumPy: {np.__version__}")
        print(f"\tPIL: {PIL.__version__}")
    except Exception as e:
        print(f"\tError fetching environment details: {e}")

def train_valid_target_eval_names(args):
    return {'train': 0, 'valid': 1, 'target': 2}

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
    parser.add_argument('--layer', type=str, default='ln')  # LayerNorm safe for GNN
    parser.add_argument('--use_gnn', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Add missing defaults that caused previous errors
    args.act_people = {
        'emg': [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31, 32, 33, 34, 35]
        ]
    }

    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.array([0, 1, 2, 3, 4, 5, 6, 7])}
    args.hz_list = {'emg': 1000}
    args.num_classes = 6
    args.input_shape = (8, 1, 200)
    args.grid_size = 10

    return args

Nmax = {
    'emg': 36,
    'pamap': 10,
    'dsads': 8
}

class mydataset:
    def __init__(self, args):
        self.args = args
        self.x, self.labels, self.dlabels, self.pclabels = loaddata_from_numpy(
            dataset=args.dataset,
            task=args.task,
            root_dir=args.data_dir
        )
        self.pdlabels = None
