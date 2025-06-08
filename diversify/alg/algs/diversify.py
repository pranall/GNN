# diversify.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits
from gnn.graph_builder import build_correlation_graph


class Diversify(Algorithm):

    def __init__(self, args):
        super(Diversify, self).__init__(args)
        self.args = args
        self.use_gnn = hasattr(args, 'use_gnn') and args.use_gnn

        # Keep featurizer as nn.Module
        self.featurizer = get_fea(args)
        self._base_featurizer = self.featurizer  # Save original featurizer module

        fea_out_dim = 128 if self.use_gnn else self._base_featurizer.in_features

        self.dbottleneck = common_network.feat_bottleneck(fea_out_dim, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(fea_out_dim, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)

        self.abottleneck = common_network.feat_bottleneck(fea_out_dim, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes * args.latent_domain_num, args.bottleneck, args.classifier)

        self.dclassifier = common_network.feat_classifier(args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.latent_domain_num)

    def extract_features(self, x):
        if self.use_gnn:
            B, C, _, T = x.shape
            x = x.squeeze(2).permute(0, 2, 1)  # shape [B, T, C]
            out_list = []
            for i in range(B):
                sample = x[i]
                edge_index = build_correlation_graph(sample.cpu().numpy(), threshold=0.3).cuda()
                gnn_out = self._base_featurizer(sample.cuda(), edge_index, batch_size=1)
                out_list.append(gnn_out)
            return torch.stack(out_list)
        else:
            return self._base_featurizer(x)

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()
        all_d1 = minibatch[1].cuda().long()
        all_c1 = minibatch[4].cuda().long()

        z1 = self.dbottleneck(self.extract_features(all_x1))
        if len(z1.shape) == 3 and z1.shape[1] == 1:
            z1 = z1.squeeze(1)

        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)

        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')

        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_c1)

        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self._base_featurizer.eval()

        all_fea = []
        all_output = []
        all_index = []

        with torch.no_grad():
            for data in loader:
                inputs = data[0].cuda().float()
                index = data[5]  # correct scalar index tensor
                feas = self.dbottleneck(self.extract_features(inputs))

                if feas.dim() > 2:
                    feas = feas.view(feas.size(0), -1)

                all_fea.append(feas.cpu())
                outputs = self.dclassifier(feas)
                all_output.append(outputs.cpu())
                all_index.append(index.cpu())

        all_fea = torch.cat(all_fea, dim=0)        # shape: [N, feat_dim]
        all_output = torch.cat(all_output, dim=0)  # shape: [N, K]
        all_index = torch.cat(all_index, dim=0).numpy().flatten()

        all_output = nn.Softmax(dim=1)(all_output)

        # Add bias term (column of ones)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)

        # Normalize features (L2 norm) along feature dimension
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.numpy()

        K = all_output.size(1)
        aff = all_output.numpy()
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
        self._base_featurizer.train()

    def update(self, data, opt):
        all_x = data[0].cuda().float()
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.extract_features(all_x))

        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        if disc_input.dim() > 2:
            disc_input = disc_input.view(disc_input.size(0), -1)

        disc_out = self.discriminator(disc_input)

        disc_labels = data[4].cuda().long()
        disc_loss = F.cross_entropy(disc_out, disc_labels)

        all_preds = self.classifier(all_z)

        if all_y.ndim > 1:
            all_y = torch.argmax(all_y, dim=1)

        if all_preds.dim() == 3 and all_preds.shape[1] == 1:
            all_preds = all_preds.squeeze(1)

        classifier_loss = F.cross_entropy(all_preds, all_y)

        loss = classifier_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[4].cuda().long()
        all_y = all_d * self.args.num_classes + all_c
        all_z = self.abottleneck(self.extract_features(all_x))
        all_preds = self.aclassifier(all_z)
        all_preds = all_preds.squeeze(1)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.extract_features(x)))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.extract_features(x)))
