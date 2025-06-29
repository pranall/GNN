import os
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, 
                           confusion_matrix, silhouette_score)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def calculate_h_divergence(source_feats, target_feats):
    X = np.vstack([source_feats, target_feats])
    y = np.hstack([np.zeros(len(source_feats)), 
                 np.ones(len(target_feats))])
    
    clf = RandomForestClassifier(n_estimators=100, 
                               max_depth=10,
                               class_weight='balanced')
    clf.fit(X, y)
    probas = clf.predict_proba(X)[:, 1]
    error = np.mean(np.where(y == 0, probas, 1 - probas))
    return np.clip(2 * (1 - error), 0, 1)

def evaluate_model(model, loaders, device="cuda"):
    model.eval()
    results = {'source': defaultdict(list), 'target': defaultdict(list)}
    
    with torch.no_grad():
        for domain in ['source', 'target']:
            for data in loaders[domain]:
                if isinstance(data, tuple):
                    graphs, y = data
                else:
                    graphs, y = data, data.y
                
                graphs = graphs.to(device)
                y = y.to(device)
                
                outputs, emb = model(graphs, return_embeddings=True)
                
                results[domain]['embeddings'].append(emb.cpu().numpy())
                results[domain]['labels'].append(y.cpu().numpy())
                results[domain]['preds'].append(
                    torch.argmax(outputs, dim=1).cpu().numpy()
                )
    
    for domain in results:
        results[domain]['embeddings'] = np.concatenate(results[domain]['embeddings'])
        results[domain]['labels'] = np.concatenate(results[domain]['labels'])
        results[domain]['preds'] = np.concatenate(results[domain]['preds'])
        
        results[domain].update({
            'accuracy': accuracy_score(results[domain]['labels'], results[domain]['preds']),
            'f1': f1_score(results[domain]['labels'], results[domain]['preds'], average='weighted'),
            'confusion': confusion_matrix(results[domain]['labels'], results[domain]['preds'])
        })
    
    results['domain_metrics'] = {
        'h_divergence': calculate_h_divergence(
            results['source']['embeddings'],
            results['target']['embeddings']
        ),
        'silhouette': silhouette_score(
            np.vstack([results['source']['embeddings'], 
                      results['target']['embeddings']]),
            np.hstack([np.zeros(len(results['source']['embeddings'])), 
                      np.ones(len(results['target']['embeddings']))])
        )
    }
    
    return results

def visualize_results(results, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1.bar(['Accuracy', 'F1-score'],
           [results['source']['accuracy'], results['source']['f1']],
           label='Source')
    ax1.bar(['Accuracy', 'F1-score'],
           [results['target']['accuracy'], results['target']['f1']],
           bottom=[results['source']['accuracy'], results['source']['f1']],
           label='Target')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title('Classification Performance')
    
    ax2.bar(['H-Divergence', 'Silhouette'],
           [results['domain_metrics']['h_divergence'],
            results['domain_metrics']['silhouette']],
           color=['red', 'green'])
    ax2.set_ylim(0, 1)
    ax2.set_title('Domain Adaptation Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    
    combined_emb = np.vstack([results['source']['embeddings'],
                            results['target']['embeddings']])
    tsne_results = tsne.fit_transform(combined_emb)
    
    for i, domain in enumerate(['source', 'target']):
        start = 0 if i == 0 else len(results['source']['embeddings'])
        end = start + len(results[domain]['embeddings'])
        
        scatter = plt.scatter(
            tsne_results[start:end, 0],
            tsne_results[start:end, 1],
            c=results[domain]['labels'],
            cmap='tab10',
            marker='o' if i == 0 else 's',
            alpha=0.6,
            label=f'{domain} domain'
        )
    
    plt.colorbar(scatter, label='Class Labels')
    plt.legend()
    plt.title('t-SNE of Embeddings (Shape=Domain, Color=Class)')
    plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
