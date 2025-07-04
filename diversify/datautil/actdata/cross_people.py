from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch
import time

def debug_timer(msg):
    print(f"[⏰] {msg} @ {time.strftime('%H:%M:%S')}")

class ActList(mydataset):
    """
    Dataset class for cross-person activity recognition
    Combines data from multiple people and positions into a unified dataset
    """
    def __init__(self, args, dataset, root_dir, people_group, group_num, 
                 transform=None, target_transform=None, pclabels=None, 
                 pdlabels=None, shuffle_grid=True):
        """
        Initialize dataset
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

        # For anti-spam printing
        self._print_count = 0

        # === GRAPH CACHE for speeding up transforms ===
        self._graph_cache = {}

    def __getitem__(self, idx):
        # ---- CACHING LOGIC ----
        if idx in self._graph_cache:
            return self._graph_cache[idx]
        # -----------------------

        debug_timer(f"__getitem__ START idx={idx}")
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
            # Print only for first 5 calls
            if self._print_count < 5:
                print("IN ActList __getitem__, x after transform:", x.shape if hasattr(x, "shape") else type(x))
                self._print_count += 1
        y = int(self.labels[idx])
        d = int(self.dlabels[idx]) if hasattr(self, "dlabels") else 0
        debug_timer(f"__getitem__ END idx={idx}")
        out = (x, y, d)
        # ---- STORE IN CACHE ----
        self._graph_cache[idx] = out
        return out

    def comb_position(self, x, cy, py, sy):
        """
        Combine data from different people and positions
        """
        for i, person_id in enumerate(self.people_group):
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
            print(f"  👤 Person {person_id} initial samples: {tx.shape[0]}")
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
            print(f"    → After adding: self.x.shape = {self.x.shape}, self.labels.shape = {self.labels.shape}")

    def set_x(self, x):
        """Update input features"""
        self.x = x
