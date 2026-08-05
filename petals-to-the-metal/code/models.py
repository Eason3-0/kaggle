"""
GoogLeNet (Inception v1) for 104-class flower classification.

Based on "Going Deeper with Convolutions" (Szegedy et al., 2014).

Usage::

    from models import GoogLeNet
    model = GoogLeNet(num_classes=104)

ResNet50 for 104-class flower classification.
    
Based on "Deep Residual Learning for Image Recognition" (He et al., 2015).

Uses torchvision's pre-trained weights by default.
    
    Usage::
    
        from models import ResNet50
        model = ResNet50(num_classes=104)
    
"""

import torch
from torch import nn
import torchvision.models as tv_models

# -- GoogLeNet (Scratch) ------------------------------------------------
class Inception(nn.Module):
    """Inception module — four parallel branches, channel-wise concatenation.

    Every convolution is followed by BatchNorm + ReLU.
    """

    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1), nn.ReLU(),
        )
        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1]), nn.ReLU(),
        )
        # Branch 3: 1x1 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1]), nn.ReLU(),
        )
        # Branch 4: 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4), nn.ReLU(),
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ], dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1) from scratch with BatchNorm.

    Args:
        num_classes: number of output classes (default 104 for flowers)
        dropout: dropout rate after the 1024-dim FC layer
    """

    def __init__(self, num_classes=104, dropout=0.4):
        super().__init__()

        # -- Stem --------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # -- Inception groups ---------------------------------------
        self.inception3a = Inception(192, 64, (96, 128), (16, 32), 32)
        self.inception3b = Inception(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, (96, 208), (16, 48), 64)
        self.inception4b = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4c = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4d = Inception(512, 112, (144, 288), (32, 64), 64)
        self.inception4e = Inception(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, (160, 320), (32, 128), 128)
        self.inception5b = Inception(832, 384, (192, 384), (48, 128), 128)

        # -- Classifier head ---------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x



# -- ResNet50 (torchvision) ------------------------------------------------
class ResNet50(nn.Module):
    def __init__(self, num_classes=104, pretrained=True, **kwargs):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)

# -- Model registry ------------------------------------------------

MODEL_REGISTRY = {
    'googlenet': lambda **kw: GoogLeNet(**kw),
    'resnet50': lambda **kw: ResNet50(**kw)
}

AVAILABLE_MODELS = {
    'googlenet': 'GoogLeNet (Inception v1) — 7.1M params',
    'resnet50': 'ResNet50 (torchvision pretrained) — 25.6M params'
}


def build_model(name, **kwargs):
    """Factory: return a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


# -- Quick sanity check ------------------------------------------
if __name__ == '__main__':
    model = GoogLeNet(num_classes=104)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input  {tuple(x.shape)}')
    print(f'Output {tuple(y.shape)}')
    print(f'Params {sum(p.numel() for p in model.parameters()):,}')
