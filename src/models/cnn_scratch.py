"""EmotionCNN: custom CNN trained from scratch (CLAUDE.md 2.1)."""
import torch.nn as nn


def _conv_bn_relu(cin: int, cout: int, dilation: int = 1) -> list[nn.Module]:
    return [
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]


def _block(cin: int, cout: int, drop: float, pool: bool = True, dilation: int = 1) -> nn.Sequential:
    """[Conv3x3 -> BN -> ReLU] x2 -> (MaxPool2) -> Dropout2d."""
    layers = _conv_bn_relu(cin, cout, dilation) + _conv_bn_relu(cout, cout, dilation)
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class EmotionCNN(nn.Module):
    """Four conv blocks (block 4 dilated, no pooling) and a global-average-pool head.

    ``width`` 32 gives channels 32-64-128-256 (~1.2 M params); 64 gives 64-128-256-512.
    Grad-CAM target: ``block4[3]`` (last conv layer of block 4).
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1, width: int = 32):
        super().__init__()
        w = width
        self.block1 = _block(in_channels, w, 0.10)                         # 96 -> 48
        self.block2 = _block(w, 2 * w, 0.15)                               # 48 -> 24
        self.block3 = _block(2 * w, 4 * w, 0.20)                           # 24 -> 12
        self.block4 = _block(4 * w, 8 * w, 0.25, pool=False, dilation=2)   # 12 x 12
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(8 * w, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.block4(self.block3(self.block2(self.block1(x)))))
