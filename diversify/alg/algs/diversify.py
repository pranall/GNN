# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class Diversify(Algorithm):
    def __init__(self, args):
        super(Diversify, self).__init__(args)
        
        # Initialize networks
        self.featurizer = get_fea(args)
        self.gnn_mode = args.use_gnn
        
        # Domain characterization networks
        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)
        
        # Main classification networks
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        
        # Auxiliary networks
        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        
        self.args = args

    def forward(self, x, edge_index=None, batch=None):
        if self.gnn_mode and edge_index is not None:
            z = self.featurizer(x, edge_index, batch)
        else:
            z = self.featurizer(x)
        return self.classifier(self.bottleneck(z))

    def update_d(self, minibatch, opt):
        if isinstance(minibatch, Data):  # Graph data
            all_x1 = minibatch.x.cuda().float()
            all_d1 = minibatch.domain.cuda().long()
            all_c1 = minibatch.y.cuda().long()
        else:  # Traditional batch
            all_x1 = minibatch[0].cuda().float()
            all_d1 = minibatch[1].cuda().long()
            all_c1 = minibatch[4].cuda().long()

        z1 = self.dbottleneck(self.featurizer(all_x1))
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1)
        
        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1)*self.args.lam + F.cross_entropy(cd1, all_c1)
        
        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        all_feas, all_outputs, all_indices = [], [], []
        with torch.no_grad():
            for data in loader:
                if hasattr(data, 'edge_index'):  # Graph data
                    inputs = data.x.cuda().float()
                    edge_index = data.edge_index.cuda()
                    batch = data.batch.cuda()
                    feas = self.dbottleneck(self.featurizer(inputs, edge_index, batch))
                else:  # Traditional batch
                    inputs = data[0].cuda().float()
                    feas = self.dbottleneck(self.featurizer(inputs))
                
                outputs = self.dclassifier(feas)
                all_feas.append(feas.cpu())
                all_outputs.append(outputs.cpu())
                all_indices.append(data[-1] if isinstance(data, (list, tuple)) else data.idx.cpu())

        all_fea = torch.cat(all_feas, dim=0)
        all_output = torch.cat(all_outputs, dim=0)
        all_index = np.concatenate(all_indices)

        # Clustering logic remains the same...
        all_output = nn.Softmax(dim=1)(all_output)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        print(Counter(pred_label))
        
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def update(self, data, opt):
        if isinstance(data, Data):  # Graph data
            all_x = data.x.cuda().float()
            all_y = data.y.cuda().long()
            all_z = self.bottleneck(self.featurizer(all_x, data.edge_index, data.batch))
            disc_labels = data.domain.cuda().long()
        else:  # Traditional batch
            all_x = data[0].cuda().float()
            all_y = data[1].cuda().long()
            all_z = self.bottleneck(self.featurizer(all_x))
            disc_labels = data[4].cuda().long()

        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        loss = classifier_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        if isinstance(minibatches, Data):  # Graph data
            all_x = minibatches.x.cuda().float()
            all_c = minibatches.y.cuda().long()
            all_d = minibatches.domain.cuda().long()
        else:  # Traditional batch
            all_x = minibatches[0].cuda().float()
            all_c = minibatches[1].cuda().long()
            all_d = minibatches[4].cuda().long()

        all_y = all_d * self.args.num_classes + all_c
        all_z = self.abottleneck(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        
        return {'class': classifier_loss.item()}

    def predict(self, x, edge_index=None, batch=None):
        if self.gnn_mode and edge_index is not None:
            z = self.featurizer(x, edge_index, batch)
        else:
            z = self.featurizer(x)
        return self.classifier(self.bottleneck(z))

    def extract_features(self, x, edge_index=None, batch=None):
        """For metric computation"""
        if self.gnn_mode and edge_index is not None:
            return self.featurizer(x, edge_index, batch)
        return self.featurizer(x)
