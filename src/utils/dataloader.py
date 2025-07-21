import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import FundusDataset
from utils.transforms import train_transforms, val_transforms


def get_fundus_dataloaders(
        root_dir: str,
        batch_size: int,
        val_fraction: float = 0.2,
        seed: int = 1972,
        num_workers: int = 4,
        pin_memory: bool = True
):
    full_dataset = FundusDataset(root_dir, transform=None)

    # Determine split sizes
    total = len(full_dataset)
    val_size = int(total * val_fraction)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataset = FundusDataset(root_dir, split='test', transform=None)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    test_loader.dataset.transform = val_transforms

    print(f"Training on {len(train_loader)} batches "
          f"validation on {len(val_loader)} batches "
          f"test on {len(test_loader)} samples")

    return train_loader, val_loader, test_loader


def get_kfolds_datasets(
        root_dir: str
):
    train_dataset = FundusDataset(root_dir, split='train', transform=None)
    test_dataset = FundusDataset(root_dir, split='test', transform=None)

    return train_dataset, test_dataset
