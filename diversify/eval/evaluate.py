import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from ..gnn.temporal_gcn import TemporalGCN
from ..datautil.graph_utils import build_graph
from ..utils.util import check_system

def evaluate_model(model, test_loader, device="cuda"):
    """
    Evaluate the GNN model on test data.
    Args:
        model: Trained GNN model (e.g., TemporalGCN).
        test_loader: DataLoader for test data.
        device: Device to run evaluation on.
    Returns:
        Dict with metrics: accuracy, F1, confusion matrix.
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in test_loader:
            graphs, labels = data
            graphs = graphs.to(device)
            labels = labels.to(device)
            outputs = model(graphs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return metrics

def plot_metrics(metrics, save_path="metrics_plot.png"):
    """Plot accuracy/F1 and confusion matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy/F1 plot
    ax1.bar(["Accuracy", "F1"], [metrics["accuracy"], metrics["f1"]])
    ax1.set_ylim(0, 1)
    ax1.set_title("Model Performance")
    
    # Confusion matrix
    cm = metrics["confusion_matrix"]
    ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGCN().to(device)
    model.load_state_dict(torch.load("../models/emg_gnn.pth"))
    
    # Replace with your test DataLoader
    test_loader = None  # TODO: Load your test data
    metrics = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}")
    plot_metrics(metrics)