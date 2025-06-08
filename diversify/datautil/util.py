# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset

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

class subdataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        indices = torch.LongTensor(indices)  # ✅ Fix: ensure indices are torch.LongTensor

        self.x = dataset.x[indices]
        self.labels = dataset.labels[indices]

        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None
        self.pclabels = dataset.pclabels[indices] if dataset.pclabels is not None else None
        self.pdlabels = dataset.pdlabels[indices] if dataset.pdlabels is not None else None

        self.index = indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.x[i]
        if self.transform:
            x = self.transform(x)
        label = self.labels[i]
        dlabel = self.dlabels[i] if self.dlabels is not None else -1
        pclabel = self.pclabels[i] if self.pclabels is not None else -1
        pdlabel = self.pdlabels[i] if self.pdlabels is not None else -1
        return x, label, dlabel, pclabel, pdlabel, self.index[i]

    def __len__(self):
        return len(self.labels)

    def set_labels_by_index(self, tlabels, tindex, type='pdlabel'):
        tindex = torch.LongTensor(tindex)
        if type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif type == 'dlabel':
            self.dlabels[tindex] = tlabels
