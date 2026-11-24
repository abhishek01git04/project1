"""TruthNet: our own CNN for telling real photos from AI-generated (GAN) faces.

Shared by training/image_cnn.py and the gateway, so a saved checkpoint loads in both
and images are preprocessed identically.
"""

import torch
from PIL import Image
from torch import nn

IMAGE_SIZE = 128


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> normalised CHW float tensor in [-1, 1]."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(IMAGE_SIZE, IMAGE_SIZE, 3)
    return (x.permute(2, 0, 1).float() / 255 - 0.5) / 0.5


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TruthNet(nn.Module):
    """5 conv blocks (128 -> 4 px), global average pooling, dropout, linear classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256), _block(256, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
