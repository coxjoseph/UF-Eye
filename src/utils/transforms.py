from torchvision.transforms import v2 as transforms
import torch


IMG_MEAN = [0.21301243, 0.15193095, 0.00540677]
IMG_STD = [0.21383261, 0.154583, 0.01645251]

train_transforms = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.Resize((512, 512)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

val_transforms = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((512, 512)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])