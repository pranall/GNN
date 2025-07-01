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

def act_to_graph_transform(args):
    def print_before_permute(x):
        if not hasattr(print_before_permute, "printed"):
            print("BEFORE SHAPE CHANGE, x.shape:", x.shape)
            print_before_permute.printed = True
        return x

    def print_after_permute(x):
        if not hasattr(print_after_permute, "printed"):
            print("AFTER SHAPE CHANGE, x.shape:", x.shape)
            print_after_permute.printed = True
        return x

    def _to_graph(x):
        if not hasattr(_to_graph, "printed"):
            print("IN _to_graph, input x.shape:", x.shape)
            _to_graph.printed = True
        if isinstance(x, torch.Tensor):
            x = x.float()
        data = convert_to_graph(
            x.unsqueeze(-1),
            adjacency_strategy=getattr(args, 'graph_method', 'correlation'),
            threshold=getattr(args, 'graph_threshold', 0.5),
            top_k=getattr(args, 'graph_top_k', 3)
        )
        return data

    return transforms.Compose([
        #transforms.ToTensor(),  # Omit if already a tensor
        StandardScaler(),
        print_before_permute,                        # <--- Print before permute (should be [8, 200])
        lambda x: x.view(args.input_shape[0], args.input_shape[2]),
        lambda x: x.permute(1, 0),                   # Now becomes [200, 8]
        print_after_permute,                         # <--- Print after permute ([200, 8])
        _to_graph
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
