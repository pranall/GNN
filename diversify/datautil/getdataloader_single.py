import numpy as np
from torch.utils.data import DataLoader
import torch
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}

def generate_emg_edge_index(num_sensors=8):
    """Generate fully-connected edge indices for EMG sensor graph"""
    edge_index = []
    for i in range(num_sensors):
        for j in range(num_sensors):
            if i != j:  # No self-connections
                edge_index.append([i, j])
    return torch.tensor(edge_index).t().contiguous()

def get_dataloader(args, tr, val, tar):
    """Modified to handle graph data structure"""
    def collate_fn(batch):
        if args.use_gnn:
            # For GNN: unpack and add edge indices
            inputs = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            domains = torch.stack([item[2] for item in batch])
            cls_labels = torch.stack([item[3] for item in batch])
            pd_labels = torch.stack([item[4] for item in batch])
            indices = torch.stack([item[5] for item in batch])
            
            # Generate edge indices (same for all samples in batch)
            edge_indices = generate_emg_edge_index()
            
            return (inputs, labels, domains, cls_labels, pd_labels, indices, edge_indices)
        else:
            # Original processing
            return torch.utils.data.default_collate(batch)

    train_loader = DataLoader(
        dataset=tr, 
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=True,
        collate_fn=collate_fn if args.use_gnn else None
    )
    
    train_loader_noshuffle = DataLoader(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn if args.use_gnn else None
    )
    
    valid_loader = DataLoader(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn if args.use_gnn else None
    )
    
    target_loader = DataLoader(
        dataset=tar,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn if args.use_gnn else None
    )
    
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args):
    """Main loader function with GNN support"""
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    # Set default GNN parameters if not specified
    if not hasattr(args, 'node_features'):
        args.node_features = 16  # Default features per sensor node
    if not hasattr(args, 'gnn_hidden'):
        args.gnn_hidden = 64     # Default GNN hidden dimension

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size
    
    rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch*(1-rate))
    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l*rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    
    tr = subdataset(args, tdata, indextr)
    val = subdataset(args, tdata, indexval)
    targetdata = combindataset(args, target_datalist)
    
    # Add graph structure info if using GNN
    if args.use_gnn:
        for dataset in [tr, val, targetdata]:
            dataset.edge_index = generate_emg_edge_index()
    
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, tr, val, targetdata)
    
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata
