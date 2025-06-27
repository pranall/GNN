import numpy as np
import torch
import random
import collections
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Optional

import datautil.actdata.util as actutil
import datautil.actdata.cross_people as cross_people
from datautil.util import combindataset, subdataset
from datautil.graph_utils import convert_to_graph
from datautil.actdata import cross_people as pcross_act

# Task mapping
task_act = {'cross_people': cross_people}

class ConsistentFormatWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, tuple) and len(sample) >= 3:
            return sample[0], sample[1], sample[2]
        elif isinstance(sample, Data):
            return sample, getattr(sample, 'y', 0), getattr(sample, 'domain', 0)
        elif isinstance(sample, dict) and 'graph' in sample:
            return sample['graph'], sample.get('label', 0), sample.get('domain', 0)
        elif isinstance(sample, (tuple, list)):
            return sample[0], sample[1] if len(sample) > 1 else 0, sample[2] if len(sample) > 2 else 0
        else:
            return sample, 0, 0

    def __getattr__(self, name):
        if 'dataset' in self.__dict__:
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class SafeSubset(Subset):
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return self.convert_data(data)
    
    def convert_data(self, data):
        if isinstance(data, tuple):
            return tuple(self.convert_data(x) for x in data)
        elif isinstance(data, list):
            return [self.convert_data(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, Data):
            keys = data.keys() if callable(data.keys) else data.keys
            for key in keys:
                data[key] = self.convert_data(data[key])
            return data
        else:
            try:
                return torch.tensor(data)
            except:
                return data

def collate_gnn(batch):
    graphs, labels, domains = [], [], []
    for sample in batch:
        if isinstance(sample, tuple):
            g = sample[0]
            y = sample[1]
            d = sample[2] if len(sample) > 2 else 0
        elif isinstance(sample, Data):
            g = sample
            y = getattr(g, "y", 0)
            d = getattr(g, "domain", 0)
        elif isinstance(sample, dict) and 'graph' in sample:
            g = sample['graph']
            y = sample.get('label', 0)
            d = sample.get('domain', 0)
        else:
            if isinstance(sample, (tuple, list)):
                g = sample[0]
                y = sample[1] if len(sample) > 1 else 0
                d = sample[2] if len(sample) > 2 else 0
            else:
                g = sample
                y = d = 0
        graphs.append(g)
        labels.append(y)
        domains.append(d)
    batched_graph = Batch.from_data_list(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.long), torch.tensor(domains, dtype=torch.long)

def get_gnn_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      num_workers=num_workers, shuffle=shuffle,
                      drop_last=shuffle, collate_fn=collate_gnn)

def get_dataloader(args, tr, val, tar):
    is_graph_data = hasattr(args, 'model_type') and args.model_type == 'gnn'
    sample = tr[0] if len(tr) > 0 else None
    if isinstance(sample, tuple) and len(sample) >= 3 and isinstance(sample[0], Data):
        is_graph_data = True
    elif isinstance(sample, Data):
        is_graph_data = True
    elif isinstance(sample, dict) and 'graph' in sample:
        is_graph_data = True

    if is_graph_data:
        return (
            get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=True),
            get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=False),
            get_gnn_dataloader(val, args.batch_size, args.N_WORKERS, shuffle=False),
            get_gnn_dataloader(tar, args.batch_size, args.N_WORKERS, shuffle=False),
        )

    return (
        DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=True),
        DataLoader(tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False),
        DataLoader(val, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False),
        DataLoader(tar, batch_size=args.batch_size, num_workers=args.N_WORKERS, shuffle=False),
    )

def get_act_dataloader(args):
    task = task_act[args.task]
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    source_datasetlist = []
    target_datalist = []
    for i, person in enumerate(tmpp):
        transform = actutil.act_to_graph_transform(args) if getattr(args, 'use_gnn', False) else actutil.act_train()
        tdata = task.ActList(args, args.dataset, args.data_dir, tmpp, i, transform=transform)
        
        if i == 0 and getattr(args, 'use_gnn', False):
            try:
                sample_check = tdata[0]
                if hasattr(sample_check, 'x'):
                    print("✅ Shape going into GNN featurizer:", sample_check.x.shape)
                elif isinstance(sample_check, tuple) and hasattr(sample_check[0], 'x'):
                    print("✅ Shape going into GNN featurizer:", sample_check[0].x.shape)
                else:
                    print("⚠️ Could not detect .x in sample for debug shape check.")
            except Exception as e:
                print(f"❌ Failed to check sample shape: {e}")

        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)

    source_data = combindataset(args, source_datasetlist)
    source_data = ConsistentFormatWrapper(source_data)

    rate = 0.2
    l = len(source_data)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * rate)
    indextr, indexval = indexall[ted:], indexall[:ted]

    tr = SafeSubset(source_data, indextr)
    val = SafeSubset(source_data, indexval)

    targetdata = combindataset(args, target_datalist)
    targetdata = ConsistentFormatWrapper(targetdata)

    print(f"Train samples: {len(tr)}, Val samples: {len(val)}, Target samples: {len(targetdata)}")
    return (*get_dataloader(args, tr, val, targetdata), tr, val, targetdata)
