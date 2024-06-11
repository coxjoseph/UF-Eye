from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class FundusDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        super().__init__()
        self.paths_labels = [(path, label) for path, label in zip(paths, labels)]
        self.transform = transform

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        path, label = self.paths_labels[idx]
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        return image, label


def ids_to_path(ids: list[str], directories: list[Path], dir_labels: list[int]) -> tuple[list[Path], list[int]]:
    files = []
    labels = []
    for directory, label in zip(directories, dir_labels):
        for file in directory.iterdir():
            if (file.suffix in ['.jpeg', '.jpg']) and any(file.name.startswith(uid) for uid in ids):
                files.append(file)
                labels.append(label)
    return files, labels
