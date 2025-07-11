from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch
import time
from tqdm import tqdm
from torch_geometric.data import Data  
import os  # Needed for file existence check

class ActList(mydataset):
    """
    Dataset class for cross-person activity recognition
    Combines data from multiple people and positions into a unified dataset
    """
    def __init__(self, args, dataset, root_dir, people_group, group_num, 
                 transform=None, target_transform=None, pclabels=None, 
                 pdlabels=None, shuffle_grid=True):
        """
        Initialize dataset with enhanced graph precomputation
        """
        super(ActList, self).__init__(args)
        
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        
        # Load raw data
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        print("🔎 Raw data shapes:")
        print("  x:", x.shape)
        print("  cy:", cy.shape)
        print("  py:", py.shape)
        print("  sy:", sy.shape)
        
        # Flatten people group if nested
        self.people_group = [p for group in people_group for p in (group if isinstance(group, list) else [group])]

        self.position = np.sort(np.unique(sy))
        
        # Combine data from different people and positions
        self.comb_position(x, cy, py, sy)
        print("✅ After comb_position: self.x.shape =", self.x.shape, "self.labels.shape =", self.labels.shape)
        
        # Expand dims and convert to tensor
        self.x = self.x[:, :, np.newaxis, :]  # [samples, channels, 1, timesteps]
        self.x = torch.tensor(self.x).float()
        print(f"🎯 FINAL ActList sample count: {self.x.shape[0]}")
        
        # Pseudo-labels and domain labels
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape) * 0
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))

        # Memory check before graph computation
        print(f"💻 Memory Info - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB | "
              f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
                     
        # ----------- START OF CACHING LOGIC -----------
        self.graph_file = f"{args.output}/precomputed_graphs.pt"
        if os.path.exists(self.graph_file):
            print(f"🔍 Graph cache HIT ✅ — Loading from {self.graph_file}")
            self.graphs = torch.load(self.graph_file)
            print(f"✅ Loaded {len(self.graphs)} precomputed graphs")
        else:
            print("🔍 Graph cache MISS ❌")
            self.precompute_graphs(args)
        # ----------- END OF CACHING LOGIC -----------

        # Enhanced graph precomputation with progress tracking
        #self.precompute_graphs(args)

    def __getitem__(self, idx):
        return self.graphs[idx], int(self.labels[idx]), int(self.dlabels[idx]) if hasattr(self, "dlabels") else 0

    def comb_position(self, x, cy, py, sy):
        """
        Combine data from different people and positions
        """
        for i, person_id in enumerate(self.people_group):
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
            #print(f"  👤 Person {person_id} initial samples: {tx.shape[0]}")
            valid_idxs = np.isin(tsy, self.position)
            if not np.any(valid_idxs):
                print(f"Skipping person {person_id} due to no valid sensor positions.")
                continue
            ttx, ttcy = tx[valid_idxs], tcy[valid_idxs]
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))
            #print(f"    → After adding: self.x.shape = {self.x.shape}, self.labels.shape = {self.labels.shape}")

    def precompute_graphs(self, args):
        """
        Optimized graph precomputation with progress tracking and memory management
        """
        print("⏳ Precomputing graphs...")
        
        # Determine batch size based on available memory
        batch_size = min(256, len(self.x))  # Adjust based on your GPU memory
        
        if batch_size == len(self.x):
            # Process all at once if small dataset
            self.graphs = []
            for x_i in tqdm(self.x, desc="Building graphs", unit="sample"):
                #self.graphs.append(self.transform(x_i) if self.transform else x_i)
                x_in = x_i.squeeze(1)  # [8, 1, 200] -> [8, 200]
                self.graphs.append(self.transform(x_in) if self.transform else x_in)

        else:
            # Process in batches for large datasets
            self.graphs = []
            for i in tqdm(range(0, len(self.x), batch_size), desc="Processing batches"):
                batch = self.x[i:i+batch_size]
                self.graphs.extend([self.transform(x_i) for x_i in batch])
                
                # Optional: Clear memory if needed
                if i % (10*batch_size) == 0:  # Every 10 batches
                    torch.cuda.empty_cache()
        
        # Save precomputed graphs
        torch.save(self.graphs, f"{args.output}/precomputed_graphs.pt")
        print(f"✅ Saved {len(self.graphs)} precomputed graphs")
        
        # Final memory check
        print(f"💻 Memory After Graphs - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB | "
              f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")

    def set_x(self, x):
        """Update input features"""
        self.x = x
