from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch_geometric.nn import GCNConv  # New import for GNN

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class GNNEncoder(nn.Module):
    """GNN module to process graph-structured EMG data"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(args.node_features, args.gnn_hidden)
        self.conv2 = GCNConv(args.gnn_hidden, args.gnn_hidden)
        self.norm = nn.LayerNorm(args.gnn_hidden) if args.layer == 'ln' else None
        
    def forward(self, x, edge_index):
        # x shape: [batch_size, num_nodes, node_features]
        # edge_index shape: [2, num_edges]
        
        # Reshape for GNN processing
        batch_size, num_nodes, num_features = x.shape
        x = x.view(-1, num_features)  # [batch_size*num_nodes, node_features]
        
        # Replicate edge_index for batch processing
        edge_index = self._expand_edge_index(edge_index, batch_size, num_nodes)
        
        # GNN operations
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        if self.norm:
            x = self.norm(x)
            
        # Reshape back to [batch_size, num_nodes, hidden_dim]
        x = x.view(batch_size, num_nodes, -1)
        return x
    
    def _expand_edge_index(self, edge_index, batch_size, num_nodes):
        """Replicate edge_index for batch processing"""
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, 
                             device=edge_index.device).repeat_interleave(edge_index.size(1))
        edge_index_batch = edge_index.repeat(1, batch_size) + offsets.view(1, -1)
        return edge_index_batch

class DiversifyGNN(Algorithm):
    """GNN version of Diversify algorithm with domain adaptation"""
    def __init__(self, args):
        super().__init__(args)
        
        # GNN Components
        self.gnn_encoder = GNNEncoder(args)
        self.node_features = args.node_features
        self.num_nodes = 8  # Fixed for MYO armband (8 sensors)
        
        # Domain Adaptation Components (modified for GNN)
        self.dbottleneck = common_network.feat_bottleneck(
            args.gnn_hidden, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)
            
        self.bottleneck = common_network.feat_bottleneck(
            args.gnn_hidden, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
            
        self.abottleneck = common_network.feat_bottleneck(
            args.gnn_hidden, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args

    def _process_input(self, x, edge_index):
        """Convert raw input to graph representation"""
        # x shape: [batch_size, features]
        # Convert to [batch_size, num_nodes, node_features]
        x = x.view(-1, self.num_nodes, self.node_features)
        return x, edge_index

    def forward(self, x, edge_index):
        """Forward pass for GNN"""
        x, edge_index = self._process_input(x, edge_index)
        node_embeddings = self.gnn_encoder(x, edge_index)
        graph_embedding = node_embeddings.mean(dim=1)  # Global mean pooling
        return graph_embedding

    def update_d(self, minibatch, opt):
        """Domain discriminator update with GNN"""
        all_x1, edge_index = minibatch[0].cuda().float(), minibatch[-1]
        all_d1 = minibatch[1].cuda().long()
        all_c1 = minibatch[4].cuda().long()
        
        graph_embedding = self.forward(all_x1, edge_index)
        z1 = self.dbottleneck(graph_embedding)
        
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        
        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1)*self.args.lam + F.cross_entropy(cd1, all_c1)
        
        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def update(self, data, opt):
        """Main update with GNN"""
        all_x, edge_index = data[0].cuda().float(), data[-1]
        all_y = data[1].cuda().long()
        
        graph_embedding = self.forward(all_x, edge_index)
        all_z = self.bottleneck(graph_embedding)
        
        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = data[4].cuda().long()
        
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        loss = classifier_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        """Auxiliary classifier update with GNN"""
        all_x, edge_index = minibatches[0].cuda().float(), minibatches[-1]
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[4].cuda().long()
        all_y = all_d * self.args.num_classes + all_c
        
        graph_embedding = self.forward(all_x, edge_index)
        all_z = self.abottleneck(graph_embedding)
        all_preds = self.aclassifier(all_z)
        
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'class': classifier_loss.item()}

    def predict(self, x):
        """Prediction with GNN (assumes edge_index is handled externally)"""
        # For inference, use default edge_index (fully connected)
        edge_index = torch.tensor([[i, j] for i in range(8) for j in range(8) if i != j]).t().cuda()
        x = x.view(-1, 8, self.node_features)
        graph_embedding = self.forward(x, edge_index)
        return self.classifier(self.bottleneck(graph_embedding))

    # Keep original set_dlabel and predict1 methods
    set_dlabel = Algorithm.set_dlabel
    predict1 = Algorithm.predict1

class Diversify(DiversifyGNN):
    """Original non-GNN version of Diversify (kept for backward compatibility)"""
    def __init__(self, args):
        # Explicitly skip DiversifyGNN initialization
        super(Algorithm, self).__init__()  # This calls Algorithm.__init__ directly
        Algorithm.__init__(self, args)
        
        # Original non-GNN components
        self.featurizer = get_fea(args)
        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args

    def forward(self, x, edge_index=None):
        """Original non-GNN forward pass (ignores edge_index)"""
        return self.featurizer(x)
