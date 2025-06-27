from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.nn import GCNConv, GATConv
from sklearn.cluster import KMeans

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type

        if gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.stack([x[batch == i].mean(0) for i in torch.unique(batch)])
        return self.classifier(x)

class Diversify(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        self.featurizer = get_fea(args)
        self.dbottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.domain_num)
        self.dclassifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.bottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.abottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(int(args.num_classes * args.latent_domain_num), args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, "label_smoothing", 0.0))
        self.lambda_cls = getattr(args, "lambda_cls", 1.0)
        self.lambda_dis = getattr(args, "lambda_dis", 0.1)
        self.explain_mode = False
        self.global_step = 0
        self.patch_skip_connection()

    def patch_skip_connection(self):
        device = next(self.featurizer.parameters()).device
        sample_input = torch.randn(1, *self.args.input_shape).to(device)
        with torch.no_grad():
            actual_features = self.featurizer(sample_input).shape[-1]
            print(f"Detected actual feature dimension: {actual_features}")
        for name, module in self.featurizer.named_modules():
            if isinstance(module, nn.Linear) and "skip" in name.lower():
                if module.in_features != actual_features:
                    print(f"Patching skip connection: {actual_features} features")
                    new_layer = nn.Linear(actual_features, module.out_features).to(device)
                    if module.in_features > actual_features:
                        new_layer.weight.data = module.weight.data[:, :actual_features].clone()
                        new_layer.bias.data = module.bias.data.clone()
                    elif actual_features > module.in_features:
                        new_weights = torch.randn(module.out_features, actual_features).to(device) * 0.01
                        new_weights[:, :module.in_features] = module.weight.data.clone()
                        new_layer.weight.data = new_weights
                        new_layer.bias.data = module.bias.data.clone()
                    if '.' in name:
                        parts = name.split('.')
                        parent = self.featurizer
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], new_layer)
                    else:
                        setattr(self.featurizer, name, new_layer)
                    print(f"Patched {name}: in_features={actual_features}, out_features={module.out_features}")
                return
        print("Warning: No skip connection layer found in featurizer")
        self.actual_features = actual_features

    def update_a(self, x, c, d, opt):
        x = x.cuda().float()
        x = self.ensure_correct_dimensions(x)
        c = c.cuda().long()
        d = d.cuda().long()
        n_domains = self.args.latent_domain_num
        d = torch.clamp(d, 0, n_domains - 1)
        y = d * self.args.num_classes + c
        max_class = self.aclassifier.fc.out_features
        y = torch.clamp(y, 0, max_class - 1)
        z = self.abottleneck(self.featurizer(x))
        if self.explain_mode:
            z = z.clone()
        preds = self.aclassifier(z)
        classifier_loss = F.cross_entropy(preds, y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}
