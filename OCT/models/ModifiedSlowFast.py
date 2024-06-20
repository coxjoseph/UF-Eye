import torch
import torchvision


class ModifiedSlowFast(torch.nn):
    def __init__(self):
        super().__init__()
        model = torchvision.models.video.r3d_18(weights=)

