from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from torchvision import models as tvm


class CNN2D(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        chs = [32, 64, 128, 128, 256, 256]
        layers = []
        c = in_ch
        for k in chs:
            layers += [
                nn.Conv2d(c, k, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # /2
            ]
            c = k
        self.features = nn.Sequential(*layers)  # 6 conv blocks
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x  # logits


# ---------- 3D CNN ----------
class CNN3D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        chs = [32, 64, 128, 128, 256, 256]
        layers = []
        c = in_ch
        for k in chs:
            layers += [
                nn.Conv3d(c, k, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(k),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2)
            ]
            c = k
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------- Pretrained 2D (ResNet50, VGG16, GoogLeNet) ----------
def pretrained_2d(name: str, freeze_until: Optional[str] = "layer3") -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        # freeze earlier layers
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer3.parameters(): p.requires_grad = True
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        return m

    if name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        for p in m.features.parameters(): p.requires_grad = False
        # unfreeze last conv block
        for p in list(m.features.parameters())[-10:]: p.requires_grad = True
        m.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        return m

    if name in ("googlenet", "inception_v1", "inception"):
        m = tvm.googlenet(weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
        for p in m.parameters(): p.requires_grad = False
        # unfreeze last inception blocks
        for p in m.inception4d.parameters(): p.requires_grad = True
        for p in m.inception4e.parameters(): p.requires_grad = True
        for p in m.inception5a.parameters(): p.requires_grad = True
        for p in m.inception5b.parameters(): p.requires_grad = True
        feat_dim = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        return m

    raise ValueError(f"Unsupported pretrained model: {name}")


# ---------- Autoencoders ----------
class AE2D(nn.Module):
    def __init__(self, in_ch: int = 3, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc_enc = nn.Linear(256, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, in_ch, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size(0)
        z = self.encoder(x).view(b, -1)
        z = self.fc_enc(z)
        h = self.fc_dec(z).view(b, 256, 1, 1)
        xhat = self.decoder(h)
        return xhat, z


class AE3D(nn.Module):
    def __init__(self, in_ch: int = 1, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc_enc = nn.Linear(256, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="trilinear", align_corners=False),
            nn.Conv3d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv3d(32, in_ch, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size(0)
        z = self.encoder(x).view(b, -1)
        z = self.fc_enc(z)
        h = self.fc_dec(z).view(b, 256, 1, 1, 1)
        xhat = self.decoder(h)
        return xhat, z
