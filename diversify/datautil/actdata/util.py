import os
import numpy as np
from torchvision import transforms

def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    dataset_dir = os.path.join(root_dir, dataset)

    if dataset == 'pamap' and task == 'cross_people':
        x_path = os.path.join(dataset_dir, dataset + '_x1.npy')
        y_path = os.path.join(dataset_dir, dataset + '_y1.npy')
    else:
        x_path = os.path.join(dataset_dir, dataset + '_x.npy')
        y_path = os.path.join(dataset_dir, dataset + '_y.npy')

    print(f"🔍 Loading X from: {x_path}")
    print(f"🔍 Loading Y from: {y_path}")

    x = np.load(x_path)
    ty = np.load(y_path)

    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy
