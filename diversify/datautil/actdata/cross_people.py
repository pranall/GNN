# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
from torch_geometric.data import Data
from datautil.actdata.util import *
from datautil.util import mydataset, Nmax, graph_collate_fn

class ActList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group, group_num, 
                 transform=None, target_transform=None, pclabels=None, 
                 pdlabels=None, shuffle_grid=True):
        super(ActList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        self.is_graph = args.algorithm.lower() == 'gnn'
        
        # Load raw data
        data_root = os.path.abspath(root_dir)
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, data_root)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        
        # Process and combine data
        self.comb_position(x, cy, py, sy)
        
        # Convert to appropriate format
        if not self.is_graph:
            self.x = self.x[:, :, np.newaxis, :]  # Original shape for CNN
            self.x = torch.tensor(self.x).float()
        else:
            # Initialize graph attributes
            self.edge_indices = []
            self.batches = []
        
        # Initialize labels
        self._init_labels(group_num, args, pclabels, pdlabels)
        
        # Convert to graphs if in GNN mode
        if self.is_graph:
            self.convert_to_graph(args.graph_threshold)

    def _init_labels(self, group_num, args, pclabels, pdlabels):
        """Initialize all label types"""
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape)*(-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape)*(0)
        self.tdlabels = np.ones(self.labels.shape)*group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num-Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        """Combine data from different positions"""
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def convert_to_graph(self, threshold=0.3):
        """Convert EMG data to graph format"""
        if not hasattr(self, 'x') or self.is_graph:
            return
            
        graphs = []
        for sample, label in zip(self.x, self.labels):
            # Convert each sample to graph
            g = self._emg_sample_to_graph(sample, label, threshold)
            graphs.append(g)
            
        # Store graph attributes
        self.graphs = graphs
        self.edge_indices = [g.edge_index for g in graphs]
        self.batches = [g.batch if hasattr(g, 'batch') else None for g in graphs]
        
        # Update dataset type
        self.is_graph = True

    def _emg_sample_to_graph(self, sample, label, threshold):
        """Convert single EMG sample to graph"""
        # Compute correlation matrix
        corr = np.corrcoef(sample.T)
        edges = np.argwhere(np.abs(corr) > threshold)
        edge_attr = np.abs(corr[edges[:,0], edges[:,1]])
        
        # Create PyG Data object
        return Data(
            x=torch.tensor(sample.T, dtype=torch.float32),
            edge_index=torch.tensor(edges.T, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=sample.shape[1]  # Number of EMG channels
        )

    def __getitem__(self, index):
        """Enhanced to handle both graph and non-graph cases"""
        if self.is_graph:
            return self.graphs[index]
            
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
            
        return (
            x,
            self.target_trans(self.labels[index]),
            self.target_trans(self.dlabels[index]),
            self.target_trans(self.pclabels[index]),
            self.target_trans(self.pdlabels[index]),
            index
        )

    def set_x(self, x):
        """Update data while preserving format"""
        if self.is_graph:
            raise NotImplementedError("set_x not supported for graph mode")
        self.x = x
