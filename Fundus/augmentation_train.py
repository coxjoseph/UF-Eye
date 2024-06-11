import torch.cuda
from models.SimpleCNN import SimpleCNN
import torchvision.transforms.v2 as transforms
from itertools import product
import json
from pathlib import Path
from data.FundusDataset import ids_to_path, FundusDataset
from torch import nn, optim
import concurrent.futures
import torch.utils.data


def train_and_evaluate(train_loader, val_loader, device, combination_id=0):
    num_epochs = 500
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = 1e5
    time_since = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        if epoch % 10 == 9:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                time_since = 0
            else:
                time_since += 1
                if time_since > 15:
                    print('Patience reached')
                    break

    print(f'{combination_id}: Best Validation Loss: {best_val_loss:.4f}')
    return best_val_loss


if __name__ == "__main__":
    rotation_angles = [5 * i for i in range(5)]  # No rotation
    crops = [transforms.RandomResizedCrop(224), transforms.CenterCrop(224)]
    flips = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    jitter_brightness = [0.1 * i for i in range(6)]
    jitter_hue = [0.1 * i for i in range(6)]

    combinations = product(rotation_angles, crops, flips, jitter_brightness, jitter_hue)

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(Path("./split_data.json"), 'r') as json_file:
        fundus_data = json.load(json_file)

    split_data = fundus_data['train_data'][f'fold_0']
    train_ids, val_ids = split_data['train_data'], split_data['val_data']

    train_paths, train_labels = ids_to_path(train_ids, directories=[Path("./data/healthy"), Path("./data/diseased")],
                                            dir_labels=[0, 1])

    val_paths, val_labels = ids_to_path(val_ids, directories=[Path("./data/healthy"), Path("./data/diseased")],
                                        dir_labels=[0, 1])

    train_dataset = FundusDataset(paths=train_paths, labels=train_labels)
    val_dataset = FundusDataset(paths=train_paths, labels=train_labels, transform=val_transforms)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for i, (angle, crop, flip, brightness, hue) in enumerate(combinations):
            print(f'Testing combination {i + 1}: Rotation={angle}, Crop={crop}, Flip={flip}, Jitter={brightness}, {hue}')

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(angle),
                crop,
                flip,
                transforms.ColorJitter(brightness=brightness, hue=hue),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset.transform = transform

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

            futures.append(executor.submit(train_and_evaluate, train_loader, val_loader, device, i))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
