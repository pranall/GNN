import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from datautil.actdata import cross_people as pcross_act
from datautil import actdata as actutil
from datautil.graph_utils import convert_to_graph

# ==================== UTILS ====================

def collate_gnn(batch):
    """Collate list of Data into a torch_geometric Batch + labels + domains."""
    graphs, labels, domains = [], [], []
    for sample in batch:
        # Accept Data objects or (Data, label, domain) tuples
        if isinstance(sample, tuple):
            g = sample[0]; y = sample[1]; d = sample[2] if len(sample) > 2 else 0
        elif isinstance(sample, Data):
            g = sample; y = getattr(g, "y", 0); d = getattr(g, "domain", 0)
        else:
            raise ValueError(f"Unknown sample type {type(sample)}")
        graphs.append(g); labels.append(y); domains.append(d)

    batched_graph = Batch.from_data_list(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.long), torch.tensor(domains, dtype=torch.long)

def get_gnn_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=shuffle,
        collate_fn=collate_gnn,
    )

# ==================== MAIN DATALOADER ====================

def get_act_dataloader(args):
    """
    Main function to load data as GNN `Data` objects for Diversify or ERM.
    Returns train_loader, train_loader_noshuffle, valid_loader, target_loader, train_subset, val_subset, target_subset
    """
    # Get train/test people splits
    tmpp = args.act_people[args.dataset]

    source_datasets = []
    target_datasets = []
    for i, person_group in enumerate(tmpp):
        transform = actutil.act_to_graph_transform(args)
        tdata = pcross_act.ActList(args, args.dataset, args.data_dir, tmpp, i, transform=transform)
        if i in args.test_envs:
            target_datasets.append(tdata)
        else:
            source_datasets.append(tdata)

    # Combine source datasets
    train_data = actutil.combindataset(args, source_datasets)
    # Wrap into consistent format
    train_data = actutil.ConsistentFormatWrapper(train_data)

    # Train/val split
    l = len(train_data.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * 0.2)
    indextr, indexval = indexall[ted:], indexall[:ted]

    tr = SafeSubset(train_data, indextr)
    val = SafeSubset(train_data, indexval)

    # Combine target datasets
    target_data = actutil.combindataset(args, target_datasets)
    target_data = actutil.ConsistentFormatWrapper(target_data)

    train_loader = get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=True)
    train_loader_noshuffle = get_gnn_dataloader(tr, args.batch_size, args.N_WORKERS, shuffle=False)
    valid_loader = get_gnn_dataloader(val, args.batch_size, args.N_WORKERS, shuffle=False)
    target_loader = get_gnn_dataloader(target_data, args.batch_size, args.N_WORKERS, shuffle=False)

    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, target_data


# ==================== HELPER CLASSES ====================

class SafeSubset(Subset):
    """Subset that converts NumPy to torch tensors and handles Data objects."""
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return self._convert_data(data)

    def _convert_data(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            return tuple(self._convert_data(x) for x in data)
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, Data):
            # Traverse attributes
            for k in data.keys:
                data[k] = self._convert_data(data[k])
            return data
        else:
            return data
