import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    embeddings = defaultdict(list)
    
    with torch.no_grad():
        for data in test_loader:
            graphs, labels = data
            graphs = graphs.to(device)
            labels = labels.to(device)
            outputs, emb = model(graphs, return_embeddings=True)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            embeddings["features"].extend(emb.cpu().numpy())
            embeddings["labels"].extend(labels.cpu().numpy())
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "embeddings": embeddings
    }
    return metrics

def plot_metrics(metrics, save_path="metrics_plot.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.bar(["Accuracy", "F1"], [metrics["accuracy"], metrics["f1"]])
    ax1.set_ylim(0, 1)
    ax1.set_title("Model Performance")
    
    cm = metrics["confusion_matrix"]
    ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix")
    
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(np.array(metrics["embeddings"]["features"]))
    for label in np.unique(metrics["embeddings"]["labels"]):
        idx = np.where(np.array(metrics["embeddings"]["labels"]) == label)
        ax3.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Class {label}")
    ax3.set_title("t-SNE Embeddings")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def domain_adaptation_metrics(source_metrics, target_metrics):
    h_div = np.linalg.norm(
        np.mean(source_metrics["embeddings"]["features"], axis=0) - 
        np.mean(target_metrics["embeddings"]["features"], axis=0)
    )
    
    combined_embeddings = np.vstack([
        source_metrics["embeddings"]["features"],
        target_metrics["embeddings"]["features"]
    ])
    labels = np.hstack([
        np.zeros(len(source_metrics["embeddings"]["features"])),
        np.ones(len(target_metrics["embeddings"]["features"]))
    ])
    
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(combined_embeddings)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[labels == 0, 0], X_tsne[labels == 0, 1], label="Source")
    plt.scatter(X_tsne[labels == 1, 0], X_tsne[labels == 1, 1], label="Target")
    plt.title("Domain Shift Visualization")
    plt.legend()
    plt.savefig("domain_shift.png")
    plt.close()

    def evaluate_domain_adaptation(model, source_loader, target_loader, device="cuda"):
    """
    Comprehensive evaluation for domain adaptation scenarios
    Returns:
        {
            'classification': {accuracy, f1, ...},
            'domain': {
                'h_divergence': float,
                'silhouette': float,
                'confusion': ndarray
            }
        }
    """
    model.eval()
    source_features, source_labels = [], []
    target_features, target_labels = [], []
    
    with torch.no_grad():
        # Process source domain
        for data in source_loader:
            graphs, labels = data
            graphs = graphs.to(device)
            outputs = model(graphs)
            source_features.append(outputs.cpu())
            source_labels.append(labels.cpu())
        
        # Process target domain
        for data in target_loader:
            graphs, labels = data
            graphs = graphs.to(device)
            outputs = model(graphs)
            target_features.append(outputs.cpu())
            target_labels.append(labels.cpu())
    
    # Convert to tensors
    source_features = torch.cat(source_features)
    source_labels = torch.cat(source_labels)
    target_features = torch.cat(target_features)
    target_labels = torch.cat(target_labels)
    
    # Calculate metrics
    results = {
        'classification': calculate_metrics(source_labels, model.predict(source_features)),
        'domain': {
            'h_divergence': calculate_h_divergence(source_features, target_features),
            'silhouette': calculate_silhouette(
                torch.cat([source_features, target_features]),
                torch.cat([source_labels, target_labels])
            ),
            'confusion': confusion_matrix(source_labels, target_labels)
        }
    }
    
    return results



    
    return {
        "h_divergence": h_div,
        "domain_shift_plot": "domain_shift.png"
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGCN().to(device)
    model.load_state_dict(torch.load("../models/emg_gnn.pth"))
    
    test_loader = None  # TODO: Load your test data
    metrics = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}")
    plot_metrics(metrics)
