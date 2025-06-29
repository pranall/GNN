from torchvision import transforms
import numpy as np
import torch
from datautil.graph_utils import convert_to_graph

class StandardScaler:
    """Normalize sensor data channel-wise"""
    def __call__(self, tensor):
        # tensor shape: [channels, timesteps, features]
        for c in range(tensor.size(0)):
            for f in range(tensor.size(2)):
                channel_data = tensor[c, :, f]
                mean = channel_data.mean()
                std = channel_data.std()
                if std > 0:
                    tensor[c, :, f] = (channel_data - mean) / std
                else:
                    tensor[c, :, f] = channel_data - mean
        return tensor

def act_train():
    """Original transformation for activity data"""
    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])

def _to_graph(x):
    print("ORIGINAL x.shape:", x.shape)
    if isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.shape == (8, 1, 200):
            x = x.squeeze(1).transpose(1, 0)
            print("AFTER FIX (8,1,200) -> (200,8):", x.shape)
        elif x.dim() == 3:
            x = x[..., 0]
            print("AFTER ...0:", x.shape)
        x = x.float()
        if x.shape == (8, 200):
            x = x.transpose(1, 0)
            print("AFTER (8,200) -> (200,8):", x.shape)
        elif x.shape == (200, 8):
            print("ALREADY CORRECT:", x.shape)
    # Convert to graph
    data = convert_to_graph(
        x.unsqueeze(-1),  # convert [200, 8] to [200, 8, 1] for compatibility
        adjacency_strategy=getattr(args, 'graph_method', 'correlation'),
        threshold=getattr(args, 'graph_threshold', 0.5),
        top_k=getattr(args, 'graph_top_k', 3)
    )
    return data

    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        lambda x: x.view(args.input_shape[0], args.input_shape[2]),
        _to_graph  # <------ THIS is what actually creates the PyG Data object!
    ])


def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
    else:
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy
