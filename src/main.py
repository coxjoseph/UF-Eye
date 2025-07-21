import argparse
from pathlib import Path

import yaml
import torch
from trainers.fundus.cnn_trainer import CNNTrainer as FundusCNNTrainer
import os
import logging

VALID_MODELS = ['cnn', 'resnet50', 'vgg16', 'googlenet', 'conv_autoencoder', 'mlp', 'random_forest', 'xgboost']
ALL_MODELS = 'all'
VALID_MODALITIES = ['oct', 'fundus']
ALL_MODALITIES = 'all'

TRAINERS = {
    'cnn': {
        'fundus': FundusCNNTrainer,
        # 'oct': OctCNNTrainer
    },
    'resnet50': {
        # 'fundus': FundusResnetTrainer,
        # 'oct': FundusOctTrainer
    }
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GatorADEye")
    parser.add_argument('--config_dir', type=str, required=True, help="Path to YAML config directory")
    parser.add_argument('--models', type=str, default=ALL_MODELS, choices=VALID_MODELS + [ALL_MODELS])
    parser.add_argument('--folds', nargs='+', type=int, default=[i for i in range(1, 6)])
    parser.add_argument('--modalities', type=str, default=ALL_MODALITIES,
                        choices=VALID_MODALITIES + [ALL_MODALITIES])
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--logfile', type=str, default='train.log')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s',
                        filename=args.logfile)
    logger.info('Beginning GatorADEye training')
    logger.debug(f'models: {args.models}, modalities: {args.modalities}')

    if args.models == ALL_MODELS:
        models_to_run = VALID_MODELS
    else:
        models_to_run = [args.models]

    if args.modalities == ALL_MODALITIES:
        modalities_to_run = VALID_MODALITIES
    else:
        modalities_to_run = [args.modalities]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f'Using device: {device}')
    # Todo - add toggle for folds (currently runs all no matter what)
    # folds_to_run = args.folds
    config_dir = Path(args.config_dir)

    for modality in modalities_to_run:
        logger.info(f'Starting training for modality: {modality}')
        for model_name in models_to_run:
            logger.info(f'-- {model_name} ...')
            config_path = config_dir / modality / f'{model_name}.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f'Loaded config: {config_path}')
            trainer = TRAINERS[model_name][modality](config, device)
            logger.debug(f'Trainer: {trainer}')
            trainer.train()



def save(config, state_dict):
    output_path = config["model"]["path"]
    name = config["model"]["name"]
    os.makedirs(output_path, exist_ok=True)
    output = os.path.join(output_path, name)
    torch.save(state_dict, output)
    print(f"Saved model to {output}")


if __name__ == '__main__':
    main()
