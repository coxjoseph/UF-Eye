import os
from PIL import Image

def crop_center(image: Image.Image, size: tuple) -> Image.Image:
    w, h = image.size
    left = (w - size[0]) // 2
    top = (h - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


if __name__ == "__main__":
    root = "/Users/cox.j/PycharmProjects/UF-Eye/data/processed/fundus"
    for dirpath, dirnames, filenames, in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jpg"):
                continue

            path = os.path.join(dirpath, filename)
            try:
                with Image.open(path) as img:
                    cropped = crop_center(img, (826, 826))
                    cropped.save(path)
            except Exception as e:
                print(e)
