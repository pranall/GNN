from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch

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
        
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = [p for group in people_group for p in (group if isinstance(group, list) else [group])]
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        self.pclabels = pclabels if pclabels is not None else np.ones(self.labels.shape) * (-1)
        self.pdlabels = pdlabels if pdlabels is not None else np.ones(self.labels.shape) * 0
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))

    def comb_position(self, x, cy, py, sy):
        for i, person_id in enumerate(self.people_group):
            person_idx = np.where(py == person_id)[0]
            tx, tcy, tsy = x[person_idx], cy[person_idx], sy[person_idx]
            valid_idxs = np.isin(tsy, self.position)
            if not np.any(valid_idxs):
                continue
            ttx, ttcy = tx[valid_idxs], tcy[valid_idxs]
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))

    def set_x(self, x):
        self.x = x

def load_datasets(args):
    x, cy, py, sy = loaddata_from_numpy(args.dataset, 'cross_people', args.data_dir)
    train_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir,
        people_group=args.act_people['emg'][:24],
        group_num=0
    )
    val_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir, 
        people_group=args.act_people['emg'][24:30],
        group_num=1
    )
    test_data = ActList(
        args,
        dataset=args.dataset,
        root_dir=args.data_dir,
        people_group=args.act_people['emg'][30:],
        group_num=2
    )
    return train_data, val_data, test_data
