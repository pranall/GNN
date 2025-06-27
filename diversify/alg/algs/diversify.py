import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import Data, Batch

from alg.modelopera import get_fea
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits
from network import Adver_network, common_network
from sklearn.cluster import KMeans


# Utility: move tensors or Data objects to device

def to_device(batch, device):
    if hasattr(batch, 'to') and not isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device).float()
    else:
        raise ValueError(f"Unknown batch type: {type(batch)}")

# Reshape raw EMG tensors for GNN: output [B, C=8, T=200]

def transform_for_gnn(x):
    # If it's already a PyG Data or Batch, leave it alone
    if isinstance(x, (Data, Batch)):
        return x

    # otherwise assume it's a raw tensor [B, C, 1, T] or [B, T, C] etc
    if x.dim() == 4 and x.size(2) == 1:
        x = x.squeeze(2)
    if x.dim() == 3 and x.size(1) != 8:
        x = x.permute(0, 2, 1)
    B, C, T = x.shape
    if T < 200:
        x = F.pad(x, (0, 0, 0, 200 - T))
    elif T > 200:
        x = x[:, :, :200]
    return x


class Diversify(Algorithm):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.featurizer = get_fea(args)
        # adversarial and classification modules
        self.dbottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.domain_num)
        self.dclassifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.bottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes, args.bottleneck, args.classifier)
        self.abottleneck = common_network.feat_bottleneck(self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(int(args.num_classes * args.latent_domain_num), args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, 'label_smoothing', 0.0))
        self.lambda_cls = getattr(args, 'lambda_cls', 1.0)
        self.lambda_dis = getattr(args, 'lambda_dis', 0.1)
        self.explain_mode = False
        self.global_step = 0
        self.patch_skip_connection()

    def patch_skip_connection(self):
        device = next(self.featurizer.parameters()).device
        sample = torch.randn(1, *self.args.input_shape).to(device)
        with torch.no_grad():
            x = sample
            if x.dim() == 4 and x.size(2) == 1:
                x = x.squeeze(2)
            T = x.size(-1)
            idx = torch.arange(T, device=device)
            dummy_e = torch.stack([idx, idx], dim=0)
            actual = self.featurizer(x, dummy_e).shape[-1]
            print(f"Detected actual feature dimension: {actual}")
        for name, m in self.featurizer.named_modules():
            if isinstance(m, nn.Linear) and 'skip' in name.lower():
                if m.in_features != actual:
                    new = nn.Linear(actual, m.out_features).to(device)
                    old_w, old_b = m.weight.data, m.bias.data
                    if actual < m.in_features:
                        new.weight.data = old_w[:, :actual].clone()
                    else:
                        new.weight.data[:, :m.in_features] = old_w.clone()
                    new.bias.data = old_b.clone()
                    parent, attr = name.rsplit('.', 1)
                    target = self.featurizer.get_submodule(parent) if parent else self.featurizer
                    setattr(target, attr, new)
                break

    def ensure_correct_dimensions(self, x):
        # If it's a PyG Data/Batch, we assume it already has the right shape
        if isinstance(x, (Data, Batch)):
            return x

        # Otherwise force it into [B, 8, 1, 200]
        if x.dim() == 3:
            return x.unsqueeze(2)   # [B,8,200] -> [B,8,1,200]
        elif x.dim() == 4:
            expected = (self.args.input_shape[0], 1, self.args.input_shape[-1])
            if x.shape[1:] != expected:
                raise ValueError(f"Unexpected 4D shape: {x.shape}, expected [B{expected}]")
            return x
        else:
            raise ValueError(f"Unsupported x.dim(): {x.dim()}")


    def update_d(self, batch, opt):
        x, c, d = batch
        device = next(self.parameters()).device
        x = to_device(x, device)
        c = c.to(device).long()
        d = d.to(device).long().clamp(0, self.args.domain_num - 1)
        z = self.dbottleneck(self.featurizer(x))
        d_in = Adver_network.ReverseLayerF.apply(z, self.args.alpha1)
        d_out = self.ddiscriminator(d_in)
        c_out = self.dclassifier(z)
        loss = F.cross_entropy(d_out, d) * self.lambda_dis + (Entropylogits(c_out) * self.args.lam + self.criterion(c_out, c))
        opt.zero_grad(); loss.backward(); opt.step()
        return {'total': loss.item()}

    def set_dlabel(self, loader):
        self.featurizer.eval()
        feats, idxs = [], []
        device = next(self.parameters()).device
        base = 0
        with torch.no_grad():
            for batch in loader:
                x = to_device(batch[0], device)
                # --- Remove all reshaping for GNN! ---
                # x is already PyG Batch if using GNN, else tensor for non-GNN
                f = self.dbottleneck(self.featurizer(x)).cpu()
                feats.append(f.numpy())
                idxs.append(np.arange(base, base + f.size(0)))
                base += f.size(0)
        labels = Counter(KMeans(n_clusters=self.args.latent_domain_num, random_state=42).fit_predict(np.vstack(feats)))
        ds = loader.dataset
        ds.set_labels_by_index(torch.tensor(list(labels.keys())), torch.tensor(list(labels.values())), 'pdlabel')
        print(f"Pseudo-domain labels set: {labels}")
        self.featurizer.train()


    def update(self, batch, opt):
        x, y = batch
        x = to_device(x, next(self.parameters()).device)
        x = self.ensure_correct_dimensions(x) if not hasattr(x, 'x') else x
        out = self.classifier(self.bottleneck(self.featurizer(x)))
        loss = self.criterion(out, y.to(out.device).long())
        opt.zero_grad(); loss.backward(); opt.step()
        return {'class': loss.item()}

    def update_a(self, minibatches, opt):
        device = next(self.parameters()).device
        raw_x, y, d = minibatches[0], minibatches[1], minibatches[2]
        y = y.to(device).long()
        d = d.to(device).long().clamp(0, self.args.latent_domain_num - 1)

        # 1) reshape EMG -> [B,8,1,200]
        x = to_device(raw_x, device)                # whatever collate_gnn gave you
        x = transform_for_gnn(x)                    # -> [B,8,200]
        x = self.ensure_correct_dimensions(x)       # -> [B,8,1,200]

        # 2) forward through your TemporalGCN (will build edges under the hood)
        features = self.featurizer(x)               # -> [B, gnn_output_dim]

        # 3) bottleneck + classifier
        z = self.abottleneck(features)              # -> [B, bottleneck]
        preds = self.aclassifier(z)                 # -> [B*T?, num_classes*latent_domain]

        # 4) combine y+d labels
        maxc = self.aclassifier.fc.out_features
        yc = (d * self.args.num_classes + y).clamp(0, maxc - 1)

        # 5) if your a-classifier somehow spit out B*T rows, collapse:
        if preds.size(0) != yc.size(0):
            B = yc.size(0)
            T = preds.size(0)//B
            preds = preds.view(B, T, -1).mean(1)

        loss = F.cross_entropy(preds, yc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class': loss.item()}

    def predict(self, x):
        x = to_device(x, next(self.parameters()).device)
        x = self.ensure_correct_dimensions(x) if not hasattr(x, 'x') else x
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def forward(self, batch):
        return self.predict(batch[0])

    def explain(self, x):
        return self.predict(x)
