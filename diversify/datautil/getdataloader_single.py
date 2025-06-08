# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from torch.utils.data import DataLoader

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {
    'cross_people': cross_people,
}


def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=True
    )
    train_loader_noshuffle = DataLoader(
        dataset=tr,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False
    )
    valid_loader = DataLoader(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False
    )
    target_loader = DataLoader(
        dataset=tar,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False
    )
    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_act_dataloader(args):
    # Ensure N_WORKERS attribute exists (set to 4 if missing)
    if not hasattr(args, 'N_WORKERS'):
        args.N_WORKERS = 4

    # Initialize steps_per_epoch with a large number
    args.steps_per_epoch = float('inf')

    source_datasetlist = []
    target_datalist = []

    # Get the task-specific module, e.g. cross_people
    pcross_act = task_act[args.task]
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    # Iterate over domain splits (people groups)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args,
            args.dataset,
            args.data_dir,
            item,
            i,
            transform=actutil.act_train()
        )
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)

            # Calculate steps per epoch as the min over source datasets
            steps = len(tdata) / args.batch_size
            if steps < args.steps_per_epoch:
                args.steps_per_epoch = steps

    # Reserve a fraction of data for validation
    val_rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch * (1 - val_rate))

    # Merge all source datasets into one combined dataset
    x_list, c_list, p_list, s_list = [], [], [], []
    for ds in source_datasetlist:
        x_list.append(ds.x)
        c_list.append(ds.c)
        p_list.append(ds.p)
        s_list.append(ds.s)

    x = np.concatenate(x_list)
    c = np.concatenate(c_list)
    p = np.concatenate(p_list)
    s = np.concatenate(s_list)

    combined_source_dataset = combindataset(args, x, c, p, s)

    # Shuffle and split combined source dataset into train and validation subsets
    total_len = len(combined_source_dataset)
    indices = np.arange(total_len)
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    val_size = int(total_len * val_rate)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_subset = subdataset(args, combined_source_dataset, train_indices)
    val_subset = subdataset(args, combined_source_dataset, val_indices)

    # Merge target datasets into one combined dataset
    x_list, c_list, p_list, s_list = [], [], [], []
    for ds in target_datalist:
        x_list.append(ds.x)
        c_list.append(ds.c)
        p_list.append(ds.p)
        s_list.append(ds.s)

    x = np.concatenate(x_list)
    c = np.concatenate(c_list)
    p = np.concatenate(p_list)
    s = np.concatenate(s_list)

    combined_target_dataset = combindataset(args, x, c, p, s)

    # Create DataLoaders for train, val, and target datasets
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, train_subset, val_subset, combined_target_dataset
    )

    return train_loader, train_loader_noshuffle, valid_loader, target_loader, train_subset, val_subset, combined_target_dataset
