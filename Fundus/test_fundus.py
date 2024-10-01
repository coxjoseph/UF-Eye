import torch.nn
import torch.utils.data
from pathlib import Path
import json

from models.SimpleCNN import SimpleCNN
from models.DeepCNN import DeepCNN
from models.FundusMLP import FundusMLP
from models.ModifiedResNet import ModifiedResNet

from data.FundusDataset import ids_to_path, FundusDataset
import numpy as np
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize


MODELS = {
    'SimpleCNN': SimpleCNN(),
    'DeepCNN': DeepCNN(),
    'ResNet2D': ModifiedResNet(),
    'MLP': FundusMLP(hidden_dims=[512, 256], input_shape=(224, 224, 3))
}


def eval_model(trained_model: torch.nn.Module, test_data: torch.utils.data.DataLoader,
               device: torch.device) -> tuple[list[int], list[int]]:
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label in test_data:
            image = image.to(device)
            prediction = trained_model(image).tolist()
            if not isinstance(prediction, list):
                predictions.append(round(prediction))
                labels.append(round(label.item()))
            else:
                if isinstance(prediction[0], list):
                    prediction = prediction[0]
                predictions.extend([round(pred) for pred in prediction])
                labels.extend([round(lab.item()) for lab in labels])

    return predictions, labels


def load_model(architecture: str, fold: int, device: torch.device,
               base_dir: Path = Path('./models/trained')) -> torch.nn.Module:
    checkpoint = base_dir / architecture / f'{architecture}-fold_{fold}.pt'
    m = MODELS[architecture].to(device)
    m.load_state_dict(torch.load(checkpoint, weights_only=True))
    return m


def get_test_loader(json_path: Path = Path('./split_data.json')) -> torch.utils.data.DataLoader:
    with open(json_path) as json_file:
        fundus_data = json.load(json_file)
    test_ids = fundus_data['test_data']
    test_paths, test_labels = ids_to_path(test_ids, directories=[Path("./data/healthy"), Path("./data/diseased")],
                                          dir_labels=[0, 1])

    test_dataset = FundusDataset(test_paths, test_labels, transform=Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataloader


if __name__ == '__main__':
    architectures = ['SimpleCNN', 'DeepCNN', 'ResNet2D', 'MLP']
    num_folds = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader = get_test_loader()
    labels = None

    votes = []

    for model_name in architectures:
        architecture_predictions = []
        for i in range(num_folds):
            print(f'Model {model_name} predicticting...')
            model = load_model(model_name, i, device)
            model_predictions, true_labels = eval_model(model, dataloader, device)
            architecture_predictions.append(model_predictions)
            if labels is None:
                labels = true_labels

        architecture_predictions = np.array(architecture_predictions)
        total_votes = np.sum(architecture_predictions, axis=1)
        majority_vote = np.round(total_votes/num_folds)
        votes.append(list(majority_vote))

    # Could probably be more efficient with concatenation but this works
    votes = np.array(votes)  # 4 x num_labels
    weights = np.array([[0.25, 0.25, 0.25, 0.25]]).T  # 1 x 4

    final_predictions = np.matmul(weights, votes)
    print(final_predictions)
    print(labels)

