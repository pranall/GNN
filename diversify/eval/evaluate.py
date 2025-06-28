import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, silhouette_score)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def calculate_h_divergence(source_feats, target_feats):
    """Robust H-Divergence calculation"""
    X = np.vstack([source_feats, target_feats])
    y = np.hstack([np.zeros(len(source_feats)), 
                  np.ones(len(target_feats))])
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5)
    clf.fit(X, y)
    probas = clf.predict_proba(X)[:, 1]
    error = np.mean(np.where(y == 0, probas, 1 - probas))
    return max(0, 2 * (1 - error))  # Ensure non-negative

def evaluate_model(model, loaders, device="cuda"):
    """
    Unified evaluation for both source and target domains
    Returns:
        {
            'source': {accuracy, f1, embeddings...},
            'target': {...},
            'domain_metrics': {
                'h_divergence': float,
                'silhouette': float
            }
        }
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for domain in ['source', 'target']:
            features, labels = [], []
            for data in loaders[domain]:
                graphs, y = data
                graphs = graphs.to(device)
                outputs, emb = model(graphs, return_embeddings=True)
                features.append(emb.cpu().numpy())
                labels.append(y.cpu().numpy())
            
            features = np.concatenate(features)
            labels = np.concatenate(labels)
            preds = model.predict(torch.from_numpy(features).to(device)).cpu().numpy()
            
            results[domain] = {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted'),
                'embeddings': features,
                'labels': labels
            }
    
    # Domain adaptation metrics
    combined_features = np.vstack([results['source']['embeddings'], 
                                 results['target']['embeddings']])
    combined_labels = np.hstack([np.zeros(len(results['source']['embeddings'])), 
                               np.ones(len(results['target']['embeddings']))])
    
    results['domain_metrics'] = {
        'h_divergence': calculate_h_divergence(
            results['source']['embeddings'],
            results['target']['embeddings']
        ),
        'silhouette': silhouette_score(combined_features, combined_labels)
    }
    
    return results

def visualize_results(results, save_dir="results"):
    """Enhanced visualization with domain metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Performance metrics
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    metrics = ['accuracy', 'f1']
    ax[0].bar(['Source', 'Target'], 
              [results['source']['accuracy'], results['target']['accuracy']])
    ax[0].set_title('Accuracy Comparison')
    
    ax[1].bar(['H-Divergence', 'Silhouette'],
              [results['domain_metrics']['h_divergence'],
              results['domain_metrics']['silhouette']])
    ax[1].set_title('Domain Metrics')
    plt.savefig(f"{save_dir}/metrics.png")
    plt.close()
    
    # 2. t-SNE plots
    tsne = TSNE(n_components=2, random_state=42)
    for domain in ['source', 'target']:
        X_tsne = tsne.fit_transform(results[domain]['embeddings'])
        plt.scatter(X_tsne[:,0], X_tsne[:,1], 
                   c=results[domain]['labels'], 
                   label=domain, alpha=0.6)
    plt.legend()
    plt.savefig(f"{save_dir}/tsne.png")
    plt.close()
