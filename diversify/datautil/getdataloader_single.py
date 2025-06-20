# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from datautil.util import graph_collate_fn  # From updated util.py
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}

def get_dataloader(args, tr, val, tar):
    """Enhanced data loader with GNN support"""
    collate_fn = graph_collate_fn if args.algorithm.lower() == 'gnn' else None
    
    loaders = (
        DataLoader(
            dataset=tr,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_fn
        )
        for shuffle in [True, False]  # For train and train_noshuffle
    )
    
    train_loader, train_loader_noshuffle = loaders
    
    valid_loader = DataLoader(
        dataset=val,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    target_loader = DataLoader(
        dataset=tar,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return train_loader, train_loader_noshuffle, valid_loader, target_loader

def get_act_dataloader(args):
    """Main function to get EMG data loaders with GNN support"""
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)

    # Load and process datasets
    for i, item in enumerate(tmpp):
        data_root = os.path.abspath(args.data_dir)
        tdata = pcross_act.ActList(
            args, 
            args.dataset, 
            data_root, 
            item, 
            i, 
            transform=actutil.act_train()
        )
        
        # Convert to graph if in GNN mode
        if args.algorithm.lower() == 'gnn':
            tdata.convert_to_graph(threshold=args.graph_threshold)
            
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size

    # Prepare datasets
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
    
    return get_dataloader(args, tr, val, targetdata) + (tr, val, targetdata)

# Add this to your ActList class (in cross_people.py)
class ActList:
    # ... existing code ...
    
    def convert_to_graph(self, threshold=0.3):
        """Convert EMG data to graph format"""
        from datautil.actdata.util import emg_to_graph  # Avoid circular import
        if not hasattr(self, 'x'):
            return
            
        self.graphs = emg_to_graph(self.x, self.labels, threshold)
        self.edge_indices = [g.edge_index for g in self.graphs]
        self.batches = [g.batch if hasattr(g, 'batch') else None for g in self.graphs]
        
        # Update dataset attributes
        self.set_graph_attributes(self.edge_indices, self.batches)
