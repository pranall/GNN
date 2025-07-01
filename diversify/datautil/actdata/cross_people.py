from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch

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
        
        # Flatten people group if nested
        self.people_group = [p for group in people_group for p in (group if isinstance(group, list) else [group])]

        self.position = np.sort(np.unique(sy))
        
        # Combine data from different people and positions
        self.comb_position(x, cy, py, sy)
        
        # Expand dims and convert to tensor
        self.x = self.x[:, :, np.newaxis, :]  # [samples, channels, 1, timesteps]
        self.x = torch.tensor(self.x).float()
        
        # Pseudo-labels and domain labels
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape) * 0
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))

        # For anti-spam printing
        self._print_count = 0

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
            # Print only for first 5 calls
            if self._print_count < 5:
                print("IN ActList __getitem__, x after transform:", x.shape if hasattr(x, "shape") else type(x))
                self._print_count += 1
        y = int(self.labels[idx])
        # Use actual domain label if needed
        d = int(self.dlabels[idx]) if hasattr(self, "dlabels") else 0
        return x, y, d

    def comb_position(self, x, cy, py, sy):
        """
        Combine data from different people and positions
        """
        for i, person_id in enumerate(self.people_group):
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
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

    def set_x(self, x):
        """Update input features"""
        self.x = x
