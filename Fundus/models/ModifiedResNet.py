from torch import nn
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedResNet, self).__init__()

        # Load the pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all layers except the final layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

        # Unfreeze the final fully connected layers
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
