import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class FundusMLP(nn.Module):
    def __init__(self, num_features=100):
        super(FundusMLP, self).__init__()
        self.num_features = num_features

        self.pca = PCA(n_components=self.num_features)

        self.fc1 = nn.Linear(self.num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Apply PCA
        x = self.apply_pca(x)

        # Pass through the MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def apply_pca(self, x):
        # Apply PCA transformation (convert tensor to numpy for PCA, then back to tensor)
        if not hasattr(self.pca, 'components_'):  # Check if PCA is fitted
            x_pca = self.pca.fit_transform(x.detach().cpu().numpy())
        else:
            x_pca = self.pca.transform(x.detach().cpu().numpy())
        return torch.tensor(x_pca, device=x.device, dtype=x.dtype)

