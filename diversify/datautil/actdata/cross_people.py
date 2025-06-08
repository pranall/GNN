# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch


class ActList(mydataset):
    def __init__(
        self, args, dataset, root_dir, people_group, group_num,
        transform=None, target_transform=None, pclabels=None, pdlabels=None,
        shuffle_grid=True, use_gnn=False  # ✅ New flag
    ):
        super(ActList, self).__init__(args)
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        self.use_gnn = use_gnn  # ✅ Store GNN usage flag

        # Load raw data
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)

        self.p = py  # person IDs
        self.s = sy  # session IDs
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))

        self.comb_position(x, cy, py, sy)

        self.x = self.x[:, :, np.newaxis, :]
        self.x = torch.tensor(self.x).float()
        self.c = self.labels

        n_samples = len(self.labels)
        self.pclabels = pclabels if pclabels is not None else np.ones(n_samples) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(n_samples) * 0

        self.tdlabels = np.ones(n_samples) * group_num
        self.dlabels = np.ones(n_samples) * (group_num - Nmax[args.dataset])

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            idx_peo = np.where(py == peo)[0]
            tx, tcy, tsy = x[idx_peo], cy[idx_peo], sy[idx_peo]

            for j, sen in enumerate(self.position):
                idx_sen = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[idx_sen], tcy[idx_sen]
                else:
                    ttx = np.hstack((ttx, tx[idx_sen]))
                    ttcy = np.hstack((ttcy, tcy[idx_sen]))

            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def set_x(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        sample = self.x[index]
        label = self.c[index]
        person = self.pclabels[index]
        domain = self.dlabels[index]
        cls_label = self.c[index]
        pd_label = self.pdlabels[index]
        index_tensor = torch.tensor(index)  

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)

        if self.use_gnn:
            edge_indices = self.generate_edge_index(sample)
            return sample, label, domain, cls_label, pd_label, index_tensor, edge_indices
        else:
            return sample, label, domain, cls_label, pd_label, index_tensor


    def generate_edge_index(self, x):
        """
        Generate edge indices assuming linear 1D adjacency for 8-channel EMG.
        Each node connects to its immediate neighbor (bi-directional).
        """
        num_channels = x.shape[0] if x.dim() == 2 else x.shape[1]  # should be 8

        edge_list = []

        for i in range(num_channels - 1):
            edge_list.append((i, i + 1))  # i → i+1
            edge_list.append((i + 1, i))  # i+1 → i

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index  # shape: [2, num_edges]
