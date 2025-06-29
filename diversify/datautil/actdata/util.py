from torchvision import transforms
import numpy as np
import torch
from datautil.graph_utils import convert_to_graph

__all__ = [
    'StandardScaler',
    'act_train',
    'act_to_graph_transform', 
    'loaddata_from_numpy'
]

class StandardScaler:
    def __call__(self, tensor):
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
    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])

def act_to_graph_transform(args):
    def _to_graph(x):
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x[..., 0]
            x = x.float()
        data = convert_to_graph(
            x.unsqueeze(-1),
            adjacency_strategy=getattr(args, 'graph_method', 'correlation'),
            threshold=getattr(args, 'graph_threshold', 0.5),
            top_k=getattr(args, 'graph_top_k', 3)
        )
        return data

    return transforms.Compose([
        transforms.ToTensor(),
        StandardScaler(),
        lambda x: x.view(args.input_shape[0], args.input_shape[2]),
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
