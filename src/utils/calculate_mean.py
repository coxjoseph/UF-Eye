from pathlib import Path
from PIL import Image
import numpy as np

if __name__ == "__main__":
    total_sum = np.zeros(3, dtype=np.float64)
    total_sqsum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    data_dir = Path("/Users/cox.j/PycharmProjects/UF-Eye/data/processed/fundus/train")
    for img_path in data_dir.rglob("*.jpg"):
        with Image.open(img_path) as img:
            arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)

        total_sum += pixels.sum(axis=0)
        total_sqsum += (pixels ** 2).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = total_sum / total_pixels
    var = total_sqsum / total_pixels - mean ** 2
    std = np.sqrt(var)

    print("Mean per channel:", mean / 255)
    print("Std  per channel:", std / 255)
