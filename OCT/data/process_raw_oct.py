import os
import numpy as np
import cv2
import glob
import re
from collections import defaultdict


def parse_filename(filename: str) -> tuple[str, ...] | None:
    pattern = r"(\d+)\s(OS|OD)\s(?:(?:O|o)nly(?:\()?)?(\d{4})?(?:-|\)|\s?\(only\))?(\D+)?(?:\w+)"
    match = re.match(pattern, filename)
    if match:
        uid, eye, year, month = match.groups()
        if month is None:
            month = 'NON'
        # Hacky but not modifying any more regex
        if len(month) > 3:
            month = month[:3]
        if year is None:
            year = 'None'

        return uid, eye, year, month
    print(f'No matches for {filename}')
    return None


def process_images(input_dir: str, output_dir: str) -> None:
    grouped_images = defaultdict(list)

    for filepath in glob.glob(os.path.join(input_dir, '*.jpg')):
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        if parsed:
            grouped_images[parsed].append(filepath)

    os.makedirs(output_dir, exist_ok=True)

    for group_key, filepaths in grouped_images.items():
        images = [cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) for filepath in filepaths]
        if not all(image is not None for image in images):
            print(f"{group_key} had unreadable image(s)")
            continue

        stacked_image = np.stack(images, axis=-1)
        uid, eye, year, month = group_key
        output_filename = f'{uid}-{year}-{month}-{eye}.npy'
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, stacked_image)


if __name__ == '__main__':
    input_dirs = ['raw/OCT- cases', 'raw/OCT- controls']
    output_dirs = ['diseased', 'healthy']

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        process_images(input_dir, output_dir)
