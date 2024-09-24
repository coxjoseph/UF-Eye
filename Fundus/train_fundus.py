import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models import DeepCNN, FundusMLP, ModifiedResNet, SimpleCNN
from data.FundusDataset import FundusDataset, ids_to_path
from pathlib import Path
import json
from torchvision.transforms.v2 import Compose, ToTensor, RandomRotation, ColorJitter, Normalize
import torch.multiprocessing as mp


def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                num_epochs: int, criterion: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                model_id: str) -> float:
    best_val_loss = 1e5
    epochs_since_improve = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device, type=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 9:
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device, type=torch.float)
                    preds = model(val_inputs)
                    val_loss = criterion(preds)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_since_improve = 0
                        torch.save(model.state_dict(), f'{model_id}.pt')
                    else:
                        epochs_since_improve += 1
                        if epochs_since_improve > 15:
                            print(f'Patience reached: best validation loss - {best_val_loss}')
                            return best_val_loss
    print(f'Finished training model: best validation loss - {best_val_loss}')
    return best_val_loss


def train_fold(fold_index: int, json_path: Path, device: torch.device, batch_size: int, num_epochs):
    torch.cuda.set_device(device)
    device = torch.device(f'cuda:{device}')
    print(f'Device {device} assigned fold {fold_index}')
    with open(json_path, 'r') as json_file:
        fundus_data = json.load(json_file)

    split_data = fundus_data['train_data'][f'fold_{fold_index}']
    train_ids, val_ids = split_data['train_data'], split_data['val_data']

    train_paths, train_labels = ids_to_path(train_ids, directories=[Path("./data/healthy"), Path("./data/diseased")],
                                            dir_labels=[0, 1])

    val_paths, val_labels = ids_to_path(val_ids, directories=[Path("./data/healthy"), Path("./data/diseased")],
                                        dir_labels=[0, 1])

    train_transforms = Compose([
        ToTensor(),
        RandomRotation(15),
        ColorJitter(brightness=.5, hue=.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusDataset(train_paths, train_labels, train_transforms)
    val_dataset = FundusDataset(val_paths, val_labels, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    val_losses = []
    print(f'Device {device} training SimpleCNN...')
    model = SimpleCNN.SimpleCNN().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    val_loss = train_model(model, optimizer, device, num_epochs, criterion, train_loader, val_loader,
                           model_id=f'SimpleCNN-fold_{fold_index}')

    val_losses.append(val_loss)

    print(f'Device {device} training DeepCNN...')
    model = DeepCNN.DeepCNN().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    val_loss = train_model(model, optimizer, device, num_epochs, criterion, train_loader, val_loader,
                           model_id=f'DeepCNN-fold_{fold_index}')

    val_losses.append(val_loss)

    print(f'Device {device} training MLP...')
    model = FundusMLP.FundusMLP().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    val_loss = train_model(model, optimizer, device, num_epochs, criterion, train_loader, val_loader,
                           model_id=f'DeepCNN-fold_{fold_index}')

    val_losses.append(val_loss)

    print(f'Device {device} training ModifiedResNet...')
    model = ModifiedResNet.ModifiedResNet().to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    val_loss = train_model(model, optimizer, device, num_epochs, criterion, train_loader, val_loader,
                           model_id=f'DeepCNN-fold_{fold_index}')

    val_losses.append(val_loss)
    print(f'All models trained!')


def main(args: argparse.Namespace):
    num_gpus = torch.cuda.device_count()
    start_fold = args.start_fold
    batch_size = args.batch_size
    num_epochs = args.epochs

    json_path = Path("./split_data.json")
    processes = []
    for fold_idx in range(start_fold, start_fold + num_gpus):
        p = mp.Process(target=train_fold, args=(fold_idx, json_path, fold_idx - start_fold, batch_size, num_epochs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_fold', type=int, help='which fold to give GPU 0')
    parser.add_argument('--batch_size', type=int, help='batch size for models')
    parser.add_argument('--epochs', type=int, help='maximum number of training epochs')
    arguments = parser.parse_args()
    mp.set_start_method('spawn')
    main(arguments)
