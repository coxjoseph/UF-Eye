import os
from PIL import Image
from torch.utils.data import Dataset


class FundusDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Base directory containing 'train' and 'test' subfolders.
            split (str): One of 'train' or 'test'.
            transform (callable, optional): Transform to apply on PIL images.
        """
        self.transform = transform
        self.samples = []  # list of (image_path, label)
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")

        # Define class labels
        classes = {'controls': 0, 'cases': 1}

        # Walk through each class subdirectory
        for cls_name, label in classes.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir} with classes {list(classes.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
