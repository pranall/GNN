import os
import numpy as np
from torchvision import transforms
from scipy.signal import resample

def act_train():
    """Default transforms for EMG data"""
    return transforms.Compose([
        transforms.ToTensor()
    ])

def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    """Enhanced with graph data support"""
    dataset_dir = os.path.join(root_dir, dataset)
    
    # Path handling
    suffix = '_x1.npy' if (dataset == 'pamap' and task == 'cross_people') else '_x.npy'
    x_path = os.path.join(dataset_dir, f"{dataset}{suffix}")
    y_path = os.path.join(dataset_dir, f"{dataset}_y.npy")

    print(f" Loading X from: {x_path}")
    print(f" Loading Y from: {y_path}")

    # Load and preprocess
    x = np.load(x_path)
    ty = np.load(y_path)
    
    # Standard EMG preprocessing
    if dataset == 'emg':
        x = np.array([resample(ch, 200) for ch in x])  # Downsample to 200Hz
    
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy

def emg_to_graph(x, y, threshold=0.3):
    """Convert EMG data to graph format"""
    from gnn.graph_builder import build_emg_graph  # Avoid circular import
    graphs = []
    for sample, label in zip(x, y):
        g = build_emg_graph(sample, threshold)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)
    return graphs
