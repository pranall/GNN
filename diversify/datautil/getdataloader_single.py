import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from datautil.util import graph_collate_fn
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people
from datautil.util import combindataset, subdataset, graph_collate_fn

task_act = {
    'cross_people': cross_people,
}

def get_dataloader(args, tr, val, tar):
    """Return (train, train_noshuffle, valid, target) DataLoaders."""
    collate = graph_collate_fn if args.use_gnn else None

    train_loader = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.N_WORKERS, collate_fn=collate
    )
    train_loader_noshuffle = DataLoader(
        tr, batch_size=args.batch_size, shuffle=False,
        num_workers=args.N_WORKERS, collate_fn=collate
    )
    valid_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.N_WORKERS, collate_fn=collate
    )
    target_loader = DataLoader(
        tar, batch_size=args.batch_size, shuffle=False,
        num_workers=args.N_WORKERS, collate_fn=collate
    )

    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_act_dataloader(args):
    """Load EMG data (cross_people), wrap in ActList, combine & split."""
    # ensure N_WORKERS is set
    if not hasattr(args, 'N_WORKERS'):
        args.N_WORKERS = 4

    src_datasets = []
    tgt_datasets = []

    pcross = task_act[args.task]
    people_groups = args.act_people[args.dataset]
    args.domain_num = len(people_groups)

    # build one ActList per group
    for grp_idx, people in enumerate(people_groups):
        ds = pcross.ActList(
            args, args.dataset, args.data_dir,
            people_group=people,
            group_num=grp_idx,
            transform=actutil.act_train(),
        )
        if grp_idx in args.test_envs:
            tgt_datasets.append(ds)
        else:
            src_datasets.append(ds)

    # combine all source into one big dataset
    combined_src = combindataset(args, src_datasets)

    # train/val split
    total = len(combined_src)
    idx = np.arange(total)
    np.random.seed(args.seed)
    np.random.shuffle(idx)
    val_size = int(0.2 * total)

    val_idx   = idx[:val_size]
    train_idx = idx[val_size:]

    train_ds = subdataset(args, combined_src, train_idx)
    valid_ds = subdataset(args, combined_src, val_idx)

    # combine all target envs
    combined_tgt = combindataset(args, tgt_datasets)

    return get_dataloader(args, train_ds, valid_ds, combined_tgt)
