import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels: int, image_size: int = 512, layer_depths: list = None, *_args, **_kwargs):
        super(CNN, self).__init__()
        if layer_depths is None:
            layer_depths = [32, 64, 128]

        layers = []
        channels = in_channels

        for depth in layer_depths:
            layers += [
                nn.Conv2d(channels, depth, kernel_size=3, padding=1),
                nn.BatchNorm2d(depth),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            channels = depth

        self.layers = nn.Sequential(*layers)

        num_pools = len(layer_depths)
        final_spatial = image_size // (2 ** num_pools)
        feat_dim = channels * final_spatial * final_spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.classifier(x).squeeze(1)
        return x
