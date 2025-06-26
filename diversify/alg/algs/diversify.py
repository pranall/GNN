from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch_geometric.nn import GCNConv

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(args.node_features, args.gnn_hidden)
        self.conv2 = GCNConv(args.gnn_hidden, args.gnn_hidden)
        self.norm = nn.LayerNorm(args.gnn_hidden) if args.layer == 'ln' else None
        
    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features = x.shape
        x = x.view(-1, num_features)
        edge_index = self._expand_edge_index(edge_index, batch_size, num_nodes)
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        if self.norm:
            x = self.norm(x)
            
        return x.view(batch_size, num_nodes, -1)
    
    def _expand_edge_index(self, edge_index, batch_size, num_nodes):
        offsets = torch.arange(0, batch_size * num_nodes, num_nodes, 
                             device=edge_index.device).repeat_interleave(edge_index.size(1))
        return edge_index.repeat(1, batch_size) + offsets.view(1, -1)

class DiversifyGNN(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        self.gnn_encoder = GNNEncoder(args)
        self.node_features = args.node_features
        self.num_nodes = 8
        
        # Domain adaptation components
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

    def forward(self, x, edge_index):
        x = x.view(-1, self.num_nodes, self.node_features)
        node_embeddings = self.gnn_encoder(x, edge_index)
        return node_embeddings.mean(dim=1)

    def update_d(self, minibatch, opt):
        x, edge_index = minibatch[0].cuda().float(), minibatch[-1]
        d = minibatch[1].cuda().long()
        c = minibatch[4].cuda().long()
        
        z = self.dbottleneck(self.forward(x, edge_index))
        disc_loss = F.cross_entropy(self.ddiscriminator(
            Adver_network.ReverseLayerF.apply(z, self.args.alpha1)), d)
        
        ent_loss = Entropylogits(self.dclassifier(z))*self.args.lam + \
                 F.cross_entropy(self.dclassifier(z), c)
        
        (ent_loss + disc_loss).backward()
        opt.step()
        return {'total': (ent_loss + disc_loss).item(), 
                'dis': disc_loss.item(), 
                'ent': ent_loss.item()}

    def update(self, data, opt):
        x, edge_index = data[0].cuda().float(), data[-1]
        y = data[1].cuda().long()
        
        z = self.bottleneck(self.forward(x, edge_index))
        disc_loss = F.cross_entropy(self.discriminator(
            Adver_network.ReverseLayerF.apply(z, self.args.alpha)), data[4].cuda().long())
        
        cls_loss = F.cross_entropy(self.classifier(z), y)
        
        (cls_loss + disc_loss).backward()
        opt.step()
        return {'total': (cls_loss + disc_loss).item(),
                'class': cls_loss.item(),
                'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        x, edge_index = minibatches[0].cuda().float(), minibatches[-1]
        c = minibatches[1].cuda().long()
        d = minibatches[4].cuda().long()
        
        preds = self.aclassifier(self.abottleneck(self.forward(x, edge_index)))
        loss = F.cross_entropy(preds, d * self.args.num_classes + c)
        
        loss.backward()
        opt.step()
        return {'class': loss.item()}

    def predict(self, x):
        edge_index = torch.tensor([[i, j] for i in range(8) for j in range(8) if i != j]).t().cuda()
        return self.classifier(self.bottleneck(self.forward(
            x.view(-1, 8, self.node_features), edge_index)))

    set_dlabel = Algorithm.set_dlabel
    predict1 = Algorithm.predict1

class Diversify(DiversifyGNN):
    def __init__(self, args):
        super(Algorithm, self).__init__()
        Algorithm.__init__(self, args)
        
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
        return self.featurizer(x)
