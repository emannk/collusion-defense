#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import types
import torch
from torch import nn
import torch.nn.functional as F
from opacus.layers import DPLSTM
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck


def patch_resnet_blocks(model: nn.Module):
    for m in model.modules():
        if isinstance(m, BasicBlock):
            def safe_forward(self, x):
                identity = x
                out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=False)
                out = self.conv2(out); out = self.bn2(out)
                if self.downsample is not None: identity = self.downsample(x)
                out = out + identity
                out = F.relu(out, inplace=False)
                return out
            m.forward = types.MethodType(safe_forward, m)
        elif isinstance(m, Bottleneck):
            def safe_forward(self, x):
                identity = x
                out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=False)
                out = self.conv2(out); out = self.bn2(out); out = F.relu(out, inplace=False)
                out = self.conv3(out); out = self.bn3(out)
                if self.downsample is not None: identity = self.downsample(x)
                out = out + identity
                out = F.relu(out, inplace=False)
                return out
            m.forward = types.MethodType(safe_forward, m)

def make_opacus_safe_resnet18(num_classes: int) -> nn.Module:
    resnet = models.resnet18(pretrained=False)
    # CIFAR stem
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    # BN→GN BEFORE patching
    convert_bn_to_gn(resnet, num_groups=32)
    # patch inplace ops
    patch_resnet_blocks(resnet)
    # classifier
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_feature_list(self, x):
        feature_list = []

        # conv1 + pool
        x1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(x1, 2)
        feature_list.append(x1)

        # conv2 + dropout + pool
        x2 = F.relu(self.conv2(p1))
        d2 = self.conv2_drop(x2)
        p2 = F.max_pool2d(d2, 2)
        feature_list.append(x2)

        # flatten → fc1
        f = p2.view(-1, p2.shape[1] * p2.shape[2] * p2.shape[3])  # 20*4*4 = 320
        x3 = F.relu(self.fc1(f))
        feature_list.append(x3)

        return feature_list


class CNNMnistHighAccuracy(nn.Module):
    """Higher-capacity, Opacus-friendly CNN for MNIST/Fashion-MNIST.

    The model keeps the final classifier name ``fc2`` so existing defenses such
    as DeepSight can continue to locate the output layer without modification.
    GroupNorm is used instead of BatchNorm because client batches can be small,
    non-IID, or wrapped by Opacus for per-sample gradients.
    """

    def __init__(self, args):
        super(CNNMnistHighAccuracy, self).__init__()
        in_channels = int(args.num_channels)
        num_classes = int(args.num_classes)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, 32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.gn4 = nn.GroupNorm(8, 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feature_dropout = nn.Dropout2d(p=0.10)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.classifier_dropout = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(128, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)), inplace=False)
        x = F.relu(self.gn2(self.conv2(x)), inplace=False)
        x = self.pool(x)
        x = self.feature_dropout(x)

        x = F.relu(self.gn3(self.conv3(x)), inplace=False)
        x = F.relu(self.gn4(self.conv4(x)), inplace=False)
        x = self.pool(x)
        x = self.feature_dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=False)
        x = self.classifier_dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_feature_list(self, x):
        feature_list = []

        x = F.relu(self.gn1(self.conv1(x)), inplace=False)
        x = F.relu(self.gn2(self.conv2(x)), inplace=False)
        feature_list.append(x)
        x = self.pool(x)
        x = self.feature_dropout(x)

        x = F.relu(self.gn3(self.conv3(x)), inplace=False)
        x = F.relu(self.gn4(self.conv4(x)), inplace=False)
        feature_list.append(x)
        x = self.pool(x)
        x = self.feature_dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=False)
        feature_list.append(x)
        return feature_list



def convert_bn_to_gn(module, num_groups=32):
    # Recursively replace ALL BatchNorm2d with GroupNorm
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            # Groups must divide channels
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(groups, num_channels, affine=True)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups=num_groups)
    return module


class CNNCifar_ResNet18(nn.Module):
    def __init__(self, args):
        super().__init__()
        # build a version that is Opacus-safe
        self.resnet = make_opacus_safe_resnet18(args.num_classes)

        # (optional) if you must convert BN->GN, do it BEFORE patching above
        # or re-run the patches after conversion. If you already have:
        #   self.resnet = convert_bn_to_gn(self.resnet)
        # then call again:
        #   patch_resnet_blocks(self.resnet)
        #   replace_inplace_activations(self.resnet)

        # If you still need BN->GN, uncomment these three lines:
        # self.resnet = convert_bn_to_gn(self.resnet)
        # patch_resnet_blocks(self.resnet)
        # replace_inplace_activations(self.resnet)

    def forward(self, x):
        return self.resnet(x)

    def get_feature_list(self, x):
        """
        Returns [feat_conv1, feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_pooled]
        """
        features = []

        # Explicit out-of-place activations to avoid surprises
        x1 = self.resnet.conv1(x)
        # bn1 may be GroupNorm if you converted; either way fine
        x1 = self.resnet.bn1(x1)
        x1 = F.relu(x1, inplace=False)
        features.append(x1)

        x2 = self.resnet.layer1(x1)   # safe: residual add already patched
        features.append(x2)
        x3 = self.resnet.layer2(x2)
        features.append(x3)
        x4 = self.resnet.layer3(x3)
        features.append(x4)
        x5 = self.resnet.layer4(x4)
        features.append(x5)

        p = self.resnet.avgpool(x5)
        p = torch.flatten(p, 1)
        features.append(p)

        return features


class FastTextBinary(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, pad_idx=0, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self, x):
        emb = self.embedding(x)                 # [B, T, E]
        mask = (x != self.pad_idx).unsqueeze(-1)           # pad mask
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths               # masked mean
        out = self.fc(self.dropout(pooled)).squeeze(1)  # logits [B]
        return out