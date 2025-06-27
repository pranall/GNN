import torch
from network import act_network
from gnn.temporal_gcn import TemporalGCN

def get_fea(args):
    """Initialize feature extractor network with GNN support"""
    if hasattr(args, 'model_type') and args.model_type == 'gnn':
        # Default values if not present
        input_dim = args.input_shape[2]
        hidden_dim = getattr(args, 'gnn_hidden_dim', 64)
        output_dim = getattr(args, 'gnn_output_dim', 256)
        
        net = TemporalGCN(input_dim, hidden_dim, output_dim)
        net.in_features = output_dim  # Needed for downstream bottleneck
        return net
    else:
        return act_network.ActNetwork(args.dataset)

# inside alg/modelopera.py
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            if hasattr(x, 'x'):
                # PyG Data or Batch
                data = x.to(device)
                out = model(data)              # your model.forward handles DataBatch
            else:
                # plain tensor
                xt = x.to(device).float()
                out = model(xt)
            y = y.to(device)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def predict_proba(network, x):
    """
    Predict class probabilities with safety checks
    
    Args:
        network: Model to use for prediction
        x: Input tensor
        
    Returns:
        Class probabilities tensor
    """
    network.eval()
    with torch.no_grad():
        x = x.to(device).float()
        logits = network.predict(x)
        
        # Handle multi-dimensional outputs
        if logits.dim() > 2:
            logits = logits.squeeze(1)
            
        probs = torch.nn.functional.softmax(logits, dim=1)
    network.train()
    return probs
