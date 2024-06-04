# Script for preprocessing the raw AI Fundus data (found in ./raw) into a healthy and diseased folder in the style of
# Pytorch's ImageFolder, which we will be using to load the dataset. The data in ./raw is already structured this way,
# but contains improperly sized images. Most of this script is cropping the black border and resizing to 224x224.
# All images modified using this script were inspected after processing for clear errors.
import re

import torchvision
from functools import partial, reduce
from pathlib import Path
from PIL import Image
import os


def process_folder(in_path: Path, out_path: Path, processing_functions: list[callable]) -> None:
    for image in in_path.iterdir():
        try:
            im = Image.open(image)
        except IOError:
            # not image
            continue

        processed_image = reduce(lambda res, f: f(res), processing_functions, im)
        new_name = create_name(out_path, image.name)
        processed_image.save(os.path.join(out_path, new_name))
        print(f"Saved {new_name}")


def create_name(out_path: Path, original_name: str) -> str:
    direction = 'L' if 'OS' in original_name else 'R'
    uid = re.match(r'^(\d+)', original_name)
    time = 0
    filename = None
    if uid is not None:
        filename = f'{uid.group(0)}-{direction}-{time}.jpg'
        filepath = os.path.join(out_path, filename)
        while os.path.isfile(filepath):
            time += 1
            filename = f'{uid.group(0)}-{direction}-{time}.jpg'
            filepath = os.path.join(out_path, filename)
    return filename


if __name__ == '__main__':
    IMAGE_SIZE = (826, 1920)
    IN_PATHS = [Path('./raw/Fundus photos-controls'), Path('./raw/Fundus photo- cases (Alzheimer\'s only)')]
    OUT_PATHS = ['./healthy', './diseased']

    transforms = [partial(torchvision.transforms.functional.center_crop, output_size=[IMAGE_SIZE[0], IMAGE_SIZE[0]]),
                  partial(torchvision.transforms.functional.resize, size=[224, 224])]

    for i, o in zip(IN_PATHS, OUT_PATHS):
        os.makedirs(o, exist_ok=True)
        o = Path(o)
        process_folder(i, o, transforms)
