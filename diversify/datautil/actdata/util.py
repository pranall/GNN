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
    """Transformation pipeline for GNN models: Always produces (200,8) per sample."""
    def _to_graph(x):
        # --- x is expected to be a numpy array or tensor ---
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input x must be torch.Tensor or np.ndarray, got {type(x)}")

        # Handle known EMG shapes:
        # - (8, 1, 200): squeeze and permute to (200, 8)
        # - (8, 200): permute to (200, 8)
        # - (200, 8): already fine
        # - (200,): rare edge-case
        if x.dim() == 3 and x.shape == (8, 1, 200):
            x = x.squeeze(1).transpose(1, 0)  # (8, 200) -> (200, 8)
        elif x.dim() == 3 and x.shape[1] == 1 and x.shape[2] == 200:
            x = x.squeeze(1).transpose(1, 0)
        elif x.shape == (8, 200):
            x = x.transpose(1, 0)
        elif x.shape == (200, 8):
            pass
        elif x.dim() == 1 and x.shape[0] == 200:
            x = x.unsqueeze(-1).repeat(1, 8)
        else:
            raise ValueError(f"Unrecognized EMG sample shape for GNN: {x.shape}")

        # Final check: x should always be (200, 8) now
        if x.shape != (200, 8):
            raise ValueError(f"After processing, EMG sample not (200,8): got {x.shape}")

        # Convert to graph
        data = convert_to_graph(
            x.unsqueeze(-1),  # [200, 8] -> [200, 8, 1] if needed by convert_to_graph
            adjacency_strategy=getattr(args, 'graph_method', 'correlation'),
            threshold=getattr(args, 'graph_threshold', 0.5),
            top_k=getattr(args, 'graph_top_k', 3)
        )
        return data

    return transforms.Compose([
        StandardScaler(),
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
