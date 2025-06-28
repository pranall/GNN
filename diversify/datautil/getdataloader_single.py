import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import datautil.actdata.util as actutil
import datautil.actdata.cross_people as cross_people
from datautil.util import combindataset, subdataset

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
        elif isinstance(sample, (tuple, list)):
            return sample[0], sample[1] if len(sample) > 1 else 0, sample[2] if len(sample) > 2 else 0
        else:
            return sample, 0, 0

def collate_gnn(batch):
    graphs, labels, domains = zip(*batch)
    
    for graph, y, d in zip(graphs, labels, domains):
        # Modern tensor construction (avoids warnings)
        graph.y = y.clone().detach().long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
        graph.domain = d.clone().detach().long() if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.long)
    
    batched = Batch.from_data_list(graphs)
    
    # Handle empty edge_index (optional)
    if batched.edge_index.shape[1] == 0:
        num_nodes = batched.num_nodes
        batched.edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    
    return batched

def get_act_dataloader(args):
    task = task_act[args.task]
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    source_datasetlist = []
    target_datalist = []
    for i, person in enumerate(tmpp):
        transform = actutil.act_to_graph_transform(args) if args.use_gnn else actutil.act_train()
        tdata = task.ActList(args, args.dataset, args.data_dir, tmpp, i, transform=transform)
        
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)

    source_data = combindataset(args, source_datasetlist)
    source_data = ConsistentFormatWrapper(source_data)

    l = len(source_data)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * 0.2)
    indextr, indexval = indexall[ted:], indexall[:ted]

    tr = Subset(source_data, indextr)
    val = Subset(source_data, indexval)
    targetdata = combindataset(args, target_datalist)
    targetdata = ConsistentFormatWrapper(targetdata)

    train_loader = DataLoader(
        tr, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(4, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    train_ns_loader = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    test_loader = DataLoader(
        targetdata,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, args.N_WORKERS),
        collate_fn=collate_gnn if args.use_gnn else None
    )
    
    return train_loader, train_ns_loader, val_loader, test_loader, tr, val, targetdata
