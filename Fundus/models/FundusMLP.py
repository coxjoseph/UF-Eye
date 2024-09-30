import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FundusMLP(nn.Module):
    def __init__(self, input_shape, n_components, hidden_dims):
        super(FundusMLP, self).__init__()

        self.input_shape = input_shape
        self.n_components = n_components

        # Standardize the input
        self.scaler = StandardScaler()

        # Initialize PCA (sklearn PCA for dimensionality reduction)
        self.pca = PCA(n_components=n_components)

        # Initialize a placeholder for PCA components (will be learned once)
        self.pca_components_ = None
        self.pca_mean_ = None

        # Define MLP layers
        layers = []
        current_dim = n_components  # Input dimension after PCA
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Output layer for binary classification
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.model = nn.Sequential(*layers)

    def fit_pca(self, X_train):
        """
        Fit PCA on the training data.
        """
        # Reshape training data to 2D: (num_samples, height * width * channels)
        num_samples = X_train.shape[0]
        X_flattened = X_train.reshape(num_samples, -1)

        # Standardize the flattened data
        X_standardized = self.scaler.fit_transform(X_flattened)

        # Fit PCA and store components and mean for later use
        self.pca.fit(X_standardized)
        self.pca_components_ = torch.tensor(self.pca.components_, dtype=torch.float32)
        self.pca_mean_ = torch.tensor(self.pca.mean_, dtype=torch.float32)

    def apply_pca(self, x):
        """
        Apply PCA to the input batch x.
        """
        # Flatten the input
        x_flattened = x.view(x.size(0), -1)

        # Standardize the data using the fitted scaler mean and std
        x_standardized = (x_flattened - torch.tensor(self.scaler.mean_, dtype=torch.float32,
                                                     device=torch.device('cuda'))) / torch.tensor(self.scaler.scale_,
                                                                                                  dtype=torch.float32,
                                                                                                  device=torch.device('cuda'))
        x_standardized.to(torch.device('cuda'))

        # Apply PCA transformation: (X - mean) * components.T
        x_pca = torch.matmul(x_standardized - self.pca_mean_, self.pca_components_.T)
        return x_pca

    def forward(self, x):
        # Apply PCA to reduce dimensionality
        x_pca = self.apply_pca(x)

        # Pass through MLP layers
        out = self.model(x_pca)
        return out
