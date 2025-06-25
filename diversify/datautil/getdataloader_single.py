import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from datautil.util import graph_collate_fn
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}

def get_dataloader(args, tr, val, tar):
    """GNN-compatible data loader with dynamic batch handling"""
    collate_fn = graph_collate_fn if args.use_gnn else None
    
    # Train loaders (shuffled/non-shuffled)
    train_loader = DataLoader(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    train_loader_noshuffle = DataLoader(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Validation and target loaders
    valid_loader = DataLoader(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    target_loader = DataLoader(
        dataset=tar,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, tar

def get_act_dataloader(args):
    """Main loader with GNN graph conversion"""
    source_datasets = []
    target_datasets = []
    pcross_act = task_act[args.task]
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    # Load and convert datasets
    for i, item in enumerate(tmpp):
        data_root = os.path.abspath(args.data_dir)
        dataset = pcross_act.ActList(
            args,
            args.dataset,
            data_root,
            item,
            i,
            transform=actutil.act_train()
        )
        
        if args.use_gnn:
            dataset.convert_to_graph(
                threshold=getattr(args, 'graph_threshold', 0.3),
                sensor_layout='myo'  # MYO armband specific
            )

        if i in args.test_envs:
            target_datasets.append(dataset)
        else:
            source_datasets.append(dataset)

    # Combine datasets with validation split
    combined_source = combindataset(args, source_datasets)
    indices = np.random.permutation(len(combined_source))
    val_size = int(0.2 * len(combined_source))
    
    train_data = subdataset(args, combined_source, indices[val_size:])
    val_data = subdataset(args, combined_source, indices[:val_size])
    target_data = combindataset(args, target_datasets)

    return get_dataloader(args, train_data, val_data, target_data)
