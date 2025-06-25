import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from collections import Counter

class Diversify(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        
        # Network initialization
        self.featurizer = get_fea(args)  # Should return GNN model when args.use_gnn=True
        self.gnn_mode = args.use_gnn
        
        # Domain components
        self.dbottleneck = self._make_bottleneck(args)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        
        # Main classifier
        self.bottleneck = self._make_bottleneck(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        
        # Auxiliary networks
        self.abottleneck = self._make_bottleneck(args)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes * args.latent_domain_num, args.bottleneck, args.classifier)
        
        self.args = args

    def _make_bottleneck(self, args):
        return common_network.feat_bottleneck(
            self.featurizer.in_features if hasattr(self.featurizer, 'in_features') else 64,
            args.bottleneck,
            args.layer
        )

    def forward(self, x, edge_index=None, batch=None):
        """Unified forward pass for both GNN and non-GNN modes"""
        if self.gnn_mode:
            if edge_index is None:
                raise ValueError("edge_index must be provided in GNN mode")
            z = self.featurizer(x, edge_index, batch)
        else:
            z = self.featurizer(x)
        return self.classifier(self.bottleneck(z))

    def update(self, batch, optimizer):
        """Main training step with GNN support"""
        x = batch.x if isinstance(batch, Data) else batch[0]
        y = batch.y if isinstance(batch, Data) else batch[1]
        d = batch.domain if isinstance(batch, Data) else batch[4]
        
        # GNN-specific handling
        if self.gnn_mode:
            edge_index = batch.edge_index
            batch_idx = batch.batch
            z = self.bottleneck(self.featurizer(x, edge_index, batch_idx))
        else:
            z = self.bottleneck(self.featurizer(x))
        
        # Loss computation
        preds = self.classifier(z)
        cls_loss = F.cross_entropy(preds, y)
        
        disc_input = Adver_network.ReverseLayerF.apply(z, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        adv_loss = F.cross_entropy(disc_out, d)
        
        total_loss = cls_loss + adv_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {'total': total_loss.item(), 'cls': cls_loss.item(), 'adv': adv_loss.item()}

    def update_d(self, batch, optimizer):
        """Domain characterization step"""
        x = batch.x if isinstance(batch, Data) else batch[0]
        d = batch.domain if isinstance(batch, Data) else batch[4]
        
        if self.gnn_mode:
            z = self.dbottleneck(self.featurizer(x, batch.edge_index, batch.batch))
        else:
            z = self.dbottleneck(self.featurizer(x))
        
        # Domain adversarial loss
        disc_input = Adver_network.ReverseLayerF.apply(z, self.args.alpha1)
        disc_out = self.ddiscriminator(disc_input)
        disc_loss = F.cross_entropy(disc_out, d)
        
        # Update
        optimizer.zero_grad()
        disc_loss.backward()
        optimizer.step()
        
        return {'disc': disc_loss.item()}

    def set_dlabel(self, loader):
        """Cluster samples into latent domains"""
        self.featurizer.eval()
        features, indices = [], []
        
        with torch.no_grad():
            for batch in loader:
                if self.gnn_mode:
                    x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
                    feats = self.featurizer(x, edge_index, batch_idx)
                else:
                    feats = self.featurizer(batch[0])
                
                features.append(feats.cpu())
                indices.append(batch.idx if hasattr(batch, 'idx') else torch.arange(len(batch[0])))
        
        # K-means clustering
        features = torch.cat(features).numpy()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.args.latent_domain_num)
        pred_labels = kmeans.fit_predict(features)
        
        # Update dataset
        loader.dataset.set_labels_by_index(pred_labels, torch.cat(indices).numpy(), 'pdlabel')
        print(f"Domain distribution: {Counter(pred_labels)}")
        
        self.featurizer.train()

    def extract_features(self, x, edge_index=None, batch=None):
        """Feature extraction for metrics"""
        if self.gnn_mode:
            if edge_index is None:
                raise ValueError("edge_index required in GNN mode")
            return self.featurizer(x, edge_index, batch)
        return self.featurizer(x)
