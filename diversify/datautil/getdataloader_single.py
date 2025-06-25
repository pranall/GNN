import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from datautil.util import graph_collate_fn
import datautil.actdata.cross_people as cross_people

def get_dataloader(args, train, val, test):
    collate = graph_collate_fn if hasattr(args, 'use_gnn') and args.use_gnn else None
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate),
        DataLoader(train, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    )

def get_act_dataloader(args):
    """Simplified loader that works with EMG data"""
    full_dataset = cross_people.ActList(args, args.dataset, args.data_dir)
    train_size = int(0.8 * len(full_dataset))
    train, val = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    return get_dataloader(args, train, val, full_dataset)
