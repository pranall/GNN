# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch


class ActList(mydataset):
    def __init__(
        self, args, dataset, root_dir, people_group, group_num, 
        transform=None, target_transform=None, pclabels=None, pdlabels=None, shuffle_grid=True
    ):
        super(ActList, self).__init__(args)
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform

        # Load raw data: x = features, cy = class labels, py = person ids, sy = session ids
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)

        self.p = py  # person IDs
        self.s = sy  # session IDs

        self.people_group = people_group
        self.position = np.sort(np.unique(sy))

        # Combine data according to people group and position
        self.comb_position(x, cy, py, sy)

        # Reshape x to (samples, channels, 1, sequence_length)
        self.x = self.x[:, :, np.newaxis, :]

        # Convert to torch tensor for model input
        self.x = torch.tensor(self.x).float()

        # Set labels attribute c to combined labels for easier access
        self.c = self.labels

        # Initialize pclabels and pdlabels with defaults if None
        n_samples = len(self.labels)
        if pclabels is not None:
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(n_samples) * (-1)

        if pdlabels is not None:
            self.pdlabels = pdlabels
        else:
            self.pdlabels = np.ones(n_samples) * 0

        self.tdlabels = np.ones(n_samples) * group_num
        self.dlabels = np.ones(n_samples) * (group_num - Nmax[args.dataset])

    def comb_position(self, x, cy, py, sy):
        # Combine data samples based on people group and session position
        for i, peo in enumerate(self.people_group):
            # Select data for person 'peo'
            idx_peo = np.where(py == peo)[0]
            tx, tcy, tsy = x[idx_peo], cy[idx_peo], sy[idx_peo]

            # Combine across sessions
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
                # Stack vertically (rows) for data and horizontally (1D) for labels
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def set_x(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Allow dataset to be indexed: return (data, label)
        sample = self.x[index]
        label = self.c[index]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label
