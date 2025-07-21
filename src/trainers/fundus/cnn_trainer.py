import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from models.cnn import CNN
from utils.dataloader import get_kfolds_datasets, get_fundus_dataloaders
import numpy as np
from sklearn.model_selection import KFold

from utils.transforms import val_transforms, train_transforms
import logging

logger = logging.getLogger(__name__)


class CNNTrainer:
    def __init__(self, config, device):
        self.best_state_dict = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.config = config
        self.device = device

        self.model = CNN(**config['model']).to(device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = Adam(self.model.parameters(), lr=config['training']['lr'])

        self.train_dataset, self.test_dataset = get_kfolds_datasets(config['data']['path'])
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=config['training']['num_workers'],
                                                       pin_memory=config['training']['pin_memory'])
        self.test_loader.dataset.transform = val_transforms

    def train(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.config['training']['seed'])

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_dataset)):
            logger.info(f"Fold {fold + 1}\n--")
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                num_workers=self.config['training']['num_workers'],
                pin_memory=self.config['training']['pin_memory']
            )

            self.val_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=torch.utils.data.SubsetRandomSampler(val_idx),
                num_workers=self.config['training']['num_workers'],
                pin_memory=self.config['training']['pin_memory'],

            )
            logger.debug('DataLoaders initialized')
            self.val_loader.dataset.transform = val_transforms
            self.train_loader.dataset.transform = train_transforms
            logger.debug('Transforms initialized')

            train_dataset = self.train_loader.dataset
            labels = []
            for _, lbl in train_dataset:
                labels.append(int(lbl))
            labels = np.array(labels)

            pos = (labels == 1).sum()
            neg = (labels == 0).sum()
            pos_weight = neg / pos if pos > 0 else torch.tensor(1.0)
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float, device=self.device)
            self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

            epochs = self.config['training']['epochs']
            patience = self.config['training']['patience']

            logger.info(f'Training for {epochs} epochs, with patience {patience}')
            logger.debug(f'--- with hyperparamters {self.config["training"]}')

            best_val_loss = np.inf
            time_since_improvement = 0
            for epoch in range(epochs):
                total_train, correct_train = 0, 0
                self.model.train()
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device).float()
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    total_train += labels.size(0)
                    correct_train += (preds == labels).sum().item()
                    self.optimizer.step()
                train_acc = correct_train / total_train

                self.model.eval()
                total_val_loss = 0.0
                num_batches = 0
                correct, total = 0, 0

                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)

                        outputs = self.model(images)
                        if outputs.dim() > 1 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(1)

                        if isinstance(self.criterion, BCEWithLogitsLoss):
                            loss = self.criterion(outputs, labels.float())
                        else:
                            loss = self.criterion(outputs, labels.long())

                        total_val_loss += loss.item()
                        num_batches += 1

                        if outputs.dim() == 1:
                            probs = torch.sigmoid(outputs)
                            preds = (probs > 0.5).long()
                        else:
                            preds = outputs.argmax(dim=1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()

                avg_val_loss = total_val_loss / num_batches
                val_acc = correct / total

                logger.debug(f"Epoch {epoch + 1}/{epochs} – avg_val_loss: {avg_val_loss:.4f}, val_acc: {val_acc:.4f}")

                logger.debug(f'Epoch {epoch + 1}/{epochs} // training accuracy: {train_acc:.4f}')
                logger.debug(f'Epoch {epoch + 1}/{epochs} // validation accuracy: {val_acc:.4f}')
                if avg_val_loss < best_val_loss:
                    logger.info(f'New best validation loss: {avg_val_loss:.4f}')
                    time_since_improvement = 0
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), f'fold_{fold+1}-best_val.pt')
                    self.best_state_dict = self.model.state_dict()
                else:
                    time_since_improvement += 1
                    if time_since_improvement > patience:
                        logger.info(f'No improvement in {time_since_improvement} epochs - stopping training')
                        break
            torch.save(self.model.state_dict(), f"fold_{fold + 1}-final.pt")
            logger.info(f'Done training - best validation loss: {best_val_loss:.4f}')
            logger.info(f'Starting testing')
            self.test()

    def test(self):
        self.model.load_state_dict(self.best_state_dict)
        logger.debug('Loaded best state dict')
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float()
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                total += labels.size(0)
                correct += (preds == labels).sum().item()
                logger.debug(f'{probs=}, {labels=}, {preds=}, {correct=}')
        test_acc = correct / total
        logger.info(f'Test accuracy: {test_acc:.4f}')
