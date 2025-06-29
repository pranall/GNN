import numpy as np
import torch
from torch_geometric.data import Data

__all__ = [
    'Nmax',
    'basedataset',
    'mydataset',
    'subdataset',
    'combindataset'
]

def Nmax(args, d):
    for i in range(len(args.test_envs)):
        if d < args.test_envs[i]:
            return i
    return len(args.test_envs)

class basedataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

class mydataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.x = None
        self.labels = None
        self.dlabels = None
        self.pclabels = None
        self.pdlabels = None
        self.task = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args
        self.mean = None
        self.std = None
        self.class_distribution = None

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'pclabel':
            self.pclabels = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels = tlabels
        elif label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pclabel':
            self.pclabels[tindex] = tlabels
        elif label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels
        elif label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def compute_statistics(self):
        if isinstance(self.x, np.ndarray):
            self.mean = np.mean(self.x, axis=0)
            self.std = np.std(self.x, axis=0)
        elif torch.is_tensor(self.x):
            self.mean = torch.mean(self.x, dim=0)
            self.std = torch.std(self.x, dim=0)
        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            self.class_distribution = dict(zip(unique, counts))

    def normalize(self):
        if self.mean is None or self.std is None:
            self.compute_statistics()
        if isinstance(self.x, np.ndarray):
            self.x = (self.x - self.mean) / (self.std + 1e-8)
        elif torch.is_tensor(self.x):
            self.x = (self.x - self.mean) / (self.std + 1e-8)

    def __getitem__(self, index):
        if isinstance(index, np.integer):
            index = int(index)
        elif isinstance(index, np.ndarray):
            index = index.item()
        x = self.input_trans(self.x[index])
        ctarget = self.labels[index] if self.labels is not None else -1
        dtarget = self.dlabels[index] if self.dlabels is not None else -1
        pctarget = self.pclabels[index] if self.pclabels is not None else -1
        pdtarget = self.pdlabels[index] if self.pdlabels is not None else -1
        if isinstance(ctarget, np.generic):
            ctarget = ctarget.item()
        if isinstance(dtarget, np.generic):
            dtarget = dtarget.item()
        if isinstance(pctarget, np.generic):
            pctarget = pctarget.item()
        if isinstance(pdtarget, np.generic):
            pdtarget = pdtarget.item()
        if hasattr(self.args, 'use_gnn') and self.args.use_gnn and not isinstance(x, Data):
            x = Data(x=x, edge_index=torch.tensor([[], []], dtype=torch.long), edge_attr=torch.tensor([]))
        return x, ctarget, dtarget, pctarget, pdtarget, index

    def __len__(self):
        return len(self.x)

class subdataset(mydataset):
    def
