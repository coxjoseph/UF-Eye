import csv
import shutil
import random
from pathlib import Path

SRC_ROOT = Path("../../data/oct")
DST_ROOT = Path("../../data/processed/oct")
TRAIN_FRAC = 0.8
RANDOM_SEED = 1972
LOG_CSV = DST_ROOT / "id_splits.csv"

random.seed(RANDOM_SEED)

files_by_label = {"cases": [], "controls": []}
for label in files_by_label:
    for img_path in (SRC_ROOT / label).glob("*.nii.gz"):
        patient_id = img_path.stem.split('_')[0]
        files_by_label[label].append((patient_id, img_path))

ids_by_label = {
    label: sorted({pid for pid, _ in files})
    for label, files in files_by_label.items()
}

train_ids = set()
test_ids = set()

for label, id_list in ids_by_label.items():
    n_train = int(len(id_list) * TRAIN_FRAC)
    shuffled = id_list.copy()
    random.shuffle(shuffled)
    train_part = shuffled[:n_train]
    test_part = shuffled[n_train:]
    train_ids.update(train_part)
    test_ids.update(test_part)

DST_ROOT.mkdir(parents=True, exist_ok=True)
with open(LOG_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["patient_id", "label", "split"])
    for label, id_list in ids_by_label.items():
        for pid in id_list:
            split = "train" if pid in train_ids else "test"
            writer.writerow([pid, label, split])

for split in ("train", "test"):
    for label in ("cases", "controls"):
        d = DST_ROOT / split / label
        d.mkdir(parents=True, exist_ok=True)

for label, files in files_by_label.items():
    for pid, src in files:
        split = "train" if pid in train_ids else "test"
        dst_dir = DST_ROOT / split / label
        shutil.copy2(src, dst_dir / src.name)

print("Sorted into ")
print(f"  Train IDs: {len(train_ids)},  Test IDs: {len(test_ids)}")
for split in ("train", "test"):
    cnt = sum(1 for _ in (DST_ROOT / split / "cases").iterdir()) + \
          sum(1 for _ in (DST_ROOT / split / "controls").iterdir())
    print(f"  {split.capitalize()} images: {cnt}")
