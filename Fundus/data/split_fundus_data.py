# This code is used to generate the training and splits folds. While this can be accomplished with train-test split
# and using a torch Sampler or Subset, we create a .json file here for 3 reasons:
#   (1) Avoid ambiguity: rather than leave it up to you, dear reader, to run the code with our seeds and figure out
#   which subject is in which fold we find creating a file allows for much more clarity
#   (2) Ensure proper stratification: subjects have a variable number of images. Splitting at the subject level first
#   and loading the IDs later ensures stratification occurs at the disease level properly
#   (i.e., the ratio of diseased / healthy subjects is the same in each fold)
#   (3) Avoid data leakage: Similarly, splitting at the subject level first ensures that we don't have a subject's
#   images appearing in both the training data and the testing data.
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


HEALTHY_LABEL = 0
DISEASED_LABEL = 1
TRAIN_FRACTION = 0.8
NUM_FOLDS = 5


def get_subjects_and_labels(base_dir: Path, label: int) -> tuple[list, list]:
    subs = []
    for file in base_dir.iterdir():
        uid = re.match(r'^(\d+)', file.name).group(0)
        if uid is None:
            raise ValueError(f'UID is None: {file}, {label}')
        subs.append(uid)

    # Preserving order is not necessary, but useful here for readability
    subs = list(dict.fromkeys(subs))
    labs = [label] * len(subs)
    return subs, labs


if __name__ == '__main__':
    healthy_directory = Path("./healthy")
    diseased_directory = Path("./diseased")

    subjects, labels = get_subjects_and_labels(healthy_directory, HEALTHY_LABEL)
    diseased_subjects, diseased_labels = get_subjects_and_labels(diseased_directory, DISEASED_LABEL)

    subjects.extend(diseased_subjects)
    labels.extend(diseased_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(subjects, labels, train_size=TRAIN_FRACTION,
                                                                        stratify=labels, random_state=720118)

    folds = {}
    for fold in range(NUM_FOLDS):
        seed = 720118 + fold
        fold_train_data, fold_val_data, fold_train_labels, fold_val_labels = train_test_split(train_data, train_labels,
                                                                                              train_size=TRAIN_FRACTION,
                                                                                              stratify=train_labels,
                                                                                              random_state=seed)

        folds[f'fold_{fold}'] = {
            'seed': seed,
            'train_data': fold_train_data,
            'train_labels': fold_train_labels,
            'val_data': fold_val_data,
            'val_labels': fold_val_labels
        }

    data = {
        'image_type': 'fundus',
        'seed': 720118,
        'test_data': test_data,
        'test_labels': test_labels,
        'train_data': folds
    }

    with open("../split_data.json", 'w') as f:
        json.dump(data, f)
        print('Saved split information to split_data.json')
