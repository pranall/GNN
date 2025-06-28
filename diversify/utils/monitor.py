import matplotlib.pyplot as plt
import numpy as np
import torch

class TrainingMonitor:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.domain_metrics = {
            'h_divergence': [],
            'silhouette': []
        }
        self.gpu_mem = []

    def update(self, phase, loss, accuracy):
        if phase == 'train':
            self.train_loss.append(loss)
            self.train_acc.append(accuracy)
        elif phase == 'val':
            self.val_loss.append(loss)
            self.val_acc.append(accuracy)
        
        if torch.cuda.is_available():
            self.gpu_mem.append(torch.cuda.memory_allocated() / 1e9)

    def update_domain_metrics(self, h_div, silhouette):
        self.domain_metrics['h_divergence'].append(h_div)
        self.domain_metrics['silhouette'].append(silhouette)

    def plot(self, output_dir):
        plt.figure(figsize=(18, 12))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(self.train_loss, label='Train')
        plt.plot(self.val_loss, label='Validation')
        plt.title('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(2, 3, 2)
        plt.plot(self.train_acc, label='Train')
        plt.plot(self.val_acc, label='Validation')
        plt.title('Accuracy')
        plt.legend()
        
        # Domain metrics
        plt.subplot(2, 3, 3)
        plt.plot(self.domain_metrics['h_divergence'], label='H-Divergence')
        plt.plot(self.domain_metrics['silhouette'], label='Silhouette')
        plt.title('Domain Metrics')
        plt.legend()
        
        # GPU memory
        if self.gpu_mem:
            plt.subplot(2, 3, 4)
            plt.plot(self.gpu_mem)
            plt.title('GPU Memory Usage (GB)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_metrics.png')
        plt.close()
        
        # Save data
        np.savez(f'{output_dir}/metrics.npz',
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 train_acc=self.train_acc,
                 val_acc=self.val_acc,
                 h_divergence=self.domain_metrics['h_divergence'],
                 silhouette=self.domain_metrics['silhouette'],
                 gpu_mem=self.gpu_mem)
