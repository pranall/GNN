import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL
from collections import defaultdict

__all__ = [
    'disable_inplace_relu',
    'set_random_seed',
    'train_valid_target_eval_names',
    'alg_loss_dict',
    'print_args',
    'print_row',
    'print_environ',
    'Tee',
    'act_param_init',
    'get_args'
]

def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(f'eval{i}_in')
            eval_name_dict['valid'].append(f'eval{i}_out')
        else:
            eval_name_dict['target'].append(f'eval{i}_out')
    return eval_name_dict

def alg_loss_dict(args):
    loss_dict = {
        'diversify': ['class', 'dis', 'total']
    }
    return loss_dict.get(args.algorithm, ['total'])

def print_args(args, print_list=[]):
    s = "==========================================\n"
    if not print_list:
        print_list = args.__dict__.keys()
    for arg in print_list:
        if hasattr(args, arg):
            s += f"{arg}: {getattr(args, arg)}\n"
    return s

def print_row(row, colwidth=10, latex=False):
    sep = " & " if latex else "  "
    end_ = "\\\\" if latex else ""
    def format_val(x):
        if isinstance(x, float):
            return f"{x:.6f}".ljust(colwidth)[:colwidth]
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join(format_val(x) for x in row), end_)

def print_environ():
    print("Environment:")
    print(f"\tPython: {sys.version.split(' ')[0]}")
    print(f"\tPyTorch: {torch.__version__}")
    print(f"\tTorchvision: {torchvision.__version__}")
    print(f"\tCUDA: {torch.version.cuda}")
    print(f"\tCUDNN: {torch.backends.cudnn.version()}")
    print(f"\tNumPy: {np.__version__}")
    print(f"\tPIL: {PIL.__version__}")

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def act_param_init(args):
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9 + j for j in range(9)] for i in range(4)]}
    
    dataset_params = {
        'emg': ((8, 1, 200), 6, 10)
    }
    params = dataset_params.get(args.dataset, ((0, 0, 0), 0, 0))
    args.input_shape = params[0]
    args.num_classes = params[1]
    args.grid_size = params[2]
    return args

def get_args():
    parser = argparse.ArgumentParser(description='Domain Generalization for Activity Recognition')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'gnn'])
    parser.add_argument('--algorithm', type=str, default="diversify")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--alpha1', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay1', type=float, default=1.0)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--classifier', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--model_size', default='median', choices=['small', 'median', 'large', 'transformer'])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--latent_domain_num', type=int, default=None)
    parser.add_argument('--domain_num', type=int, default=0)
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--task', type=str, default="cross_people")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--automated_k', action='store_true')
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--CL_PHASE_EPOCHS', type=int, default=5)
    parser.add_argument('--enable_shap', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--use_gnn', action='store_true')
    parser.add_argument('--gnn_hidden_dim', type=int, default=32)
    parser.add_argument('--gnn_output_dim', type=int, default=128)
    parser.add_argument('--gnn_lr', type=float, default=0.001)
    parser.add_argument('--gnn_weight_decay', type=float, default=0.0001)
    parser.add_argument('--gnn_pretrain_epochs', type=int, default=5)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--old', action='store_true')

    args = parser.parse_args()
    args.steps_per_epoch = 10000000000
    args.data_dir = os.path.join(args.data_file, args.data_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args
