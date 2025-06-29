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
        Args:
            args: Configuration arguments
            dataset: Dataset name
            root_dir: Root directory of data
            people_group: List of people IDs to include
            group_num: Group identifier number
            transform: Input transformations
            target_transform: Target transformations
            pclabels: Precomputed class labels
            pdlabels: Precomputed domain labels
        """
        super(ActList, self).__init__(args)
        
        # Initialize dataset properties
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cross_people'
        self.transform = transform
        self.target_transform = target_transform
        
        # Load raw data
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        
        # Set up groups and positions
        self.people_group = people_group
        # Flatten people group if it's nested
        self.people_group = [p for group in people_group for p in (group if isinstance(group, list) else [group])]

        self.position = np.sort(np.unique(sy))
        
        # Combine data from different people and positions
        self.comb_position(x, cy, py, sy)
        
        # Format data as tensors
        # Preserve both dimension expansion approaches
        self.x = self.x[:, :, np.newaxis, :]  # From first version
        self.transform = None  # From first version
        self.x = torch.tensor(self.x).float()
        
        # Handle pseudo-labels
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape) * 0
        
        # Domain labels
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        """
        Combine data from different people and positions
        Args:
            x: Input features
            cy: Class labels
            py: Person IDs
            sy: Position/sensor IDs
        """
        # Preserve both implementation approaches
        # First version implementation
        for i, person_id in enumerate(self.people_group):
            # Get data for current person
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
            
            # Combine data from all positions for this person
            valid_idxs = np.isin(tsy, self.position)
            if not np.any(valid_idxs):
                print(f"Skipping person {person_id} due to no valid sensor positions.")
                continue

            ttx, ttcy = tx[valid_idxs], tcy[valid_idxs]
            # Add to dataset
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def set_x(self, x):
        """Update input features"""
        self.x = x

# THIS SHOULD BE AT MODULE LEVEL (NOT INDENTED)
def load_datasets(args):
    """Create train/val/test splits from ActList"""
    from datautil.actdata.util import loaddata_from_numpy

    # Load raw data
    x, cy, py, sy = loaddata_from_numpy(args.dataset, 'cross_people', args.data_dir)

    # Create splits (adjust ratios as needed)
    train_idx = int(0.7 * len(x))
    val_idx = int(0.85 * len(x))

    # Initialize datasets
    train_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir,
        people_group=args.act_people['emg'][:24],  # First 24 people for train
        group_num=0
    )

    val_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir,
        people_group=args.act_people['emg'][24:30],  # Next 6 for val
        group_num=1
    )

    test_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir,
        people_group=args.act_people['emg'][30:],  # Last 6 for test
        group_num=2
    )    

    return train_data, val_data, test_data
