"""B0 baseline and feature-fusion MLP with train-only normalization buffers."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from authenfy.fusion_features import FUSION_DIM


class EfficientBaseline(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))

    def features(self, pixels):
        return self.net.avgpool(self.net.features(pixels)).flatten(1)

    def forward(self, pixels):
        return self.net.classifier(self.features(pixels)).squeeze(1)


class FusionHead(nn.Module):
    def __init__(self, dimension=FUSION_DIM):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.layers = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def fit_normalizer(self, train_features):
        self.mean.copy_(train_features.mean(0))
        self.scale.copy_(train_features.std(0, unbiased=False).clamp_min(1e-5))

    def forward(self, features):
        return self.layers((features - self.mean) / self.scale).squeeze(1)


class FusionModel(nn.Module):
    def __init__(self, baseline=None, head=None):
        super().__init__()
        self.baseline = baseline or EfficientBaseline()
        self.head = head or FusionHead()

    def forward(self, pixels, handcrafted):
        return self.head(torch.cat([self.baseline.features(pixels), handcrafted], dim=1))
