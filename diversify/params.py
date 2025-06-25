import argparse

def get_args():
    """Main argument parser for training"""
    parser = argparse.ArgumentParser()
    
    # Base parameters
    parser.add_argument('--algorithm', type=str, default='Diversify',
                      choices=['Diversify', 'GNN'],
                      help='Algorithm type')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=5)
    
    # Model architecture
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default='wn',
                      choices=['linear', 'wn', 'bn'])
    parser.add_argument('--classifier', type=str, default='bn',
                      choices=['wn', 'bn', 'linear'])
    
    # Domain adaptation
    parser.add_argument('--latent_domain_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--alpha1', type=float, default=0.1)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--dis_hidden', type=int, default=256)
    
    # GNN-specific parameters
    parser.add_argument('--use_gnn', type=int, default=0,
                      choices=[0, 1], help='Deprecated: Use --algorithm GNN instead')
    parser.add_argument('--gnn_hidden_dim', type=int, default=64)
    parser.add_argument('--gnn_output_dim', type=int, default=128)
    parser.add_argument('--gnn_heads', type=int, default=4)  # New parameter for GAT heads
    
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--num_classes', type=int, default=36)
    
    args = parser.parse_args()
    
    # Backward compatibility
    if args.use_gnn == 1:
        args.algorithm = 'GNN'
    
    # Auto-configure for GNN mode
    if args.algorithm == 'GNN':
        args.batch_size = max(16, args.batch_size // 2)  # Smaller batches for graphs
    
    return args

def get_params():
    """Parameter search configurations (preserves your original structure)"""
    paramname = {
        'diversify': [
            '--latent_domain_num',
            '--alpha1',
            '--alpha', 
            '--lam',
            '--algorithm',  # Replaces --use_gnn
            '--gnn_hidden_dim',
            '--gnn_output_dim',
            '--gnn_heads'  # New parameter for GAT heads
        ]
    }

    paramlist = {
        'diversify': [
            [2, 3, 5, 10, 20],          # latent_domain_num
            [0.1, 0.5, 1],               # alpha1
            [0.1, 1, 10],                # alpha
            [0],                         # lam
            ['Diversify', 'GNN'],        # algorithm (replaces use_gnn)
            [16, 32, 64],                # gnn_hidden_dim
            [64, 128, 256],              # gnn_output_dim
            [2, 4, 8]                    # gnn_heads
        ]
    }
    return paramname, paramlist
