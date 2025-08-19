from __future__ import annotations

import glob
import os
import re
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import nibabel as nib

    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

IMG_EXT = (".jpg", ".jpeg", ".png")


def _label_from_path(p: str) -> int:
    # cases => 1, controls => 0
    lowered = p.lower()
    if "/cases/" in lowered or os.sep + "cases" + os.sep in lowered:
        return 1
    if "/controls/" in lowered or os.sep + "controls" + os.sep in lowered:
        return 0
    raise ValueError(f"Could not infer label from path: {p}")


class Fundus2DDataset(Dataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            image_size: Tuple[int, int] = (224, 224),
    ):
        self.root = root
        self.transform = transform
        self.image_size = image_size
        self.paths = []
        for cls in ("cases", "controls"):
            self.paths += sum([glob.glob(os.path.join(root, cls, f"*{ext}")) for ext in IMG_EXT], [])
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        y = _label_from_path(p)
        img = Image.open(p).convert("RGB")
        if self.transform:
            x = self.transform(img)
        else:
            # resize + to tensor [0,1] then normalize like ImageNet
            img = img.resize(self.image_size, Image.BILINEAR)
            x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.).permute(2, 0, 1)
            x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1,
                                                                                                                   1)
        return x, torch.tensor(y, dtype=torch.float32)


def _stack_slices(paths: List[str]) -> np.ndarray:
    vol = []
    for p in sorted(paths):
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        vol.append(arr)
    return np.stack(vol, axis=0)


class OCT3DDataset(Dataset):
    """
    Loads 3D volumes from either:
    - NIfTI files in /{cases,controls}/*.nii(.gz)
    - Or stacks of 2D slices named like ID_slice-###.jpg in the same folder.
    Returns x: [1,D,H,W] normalized to [0,1], y: scalar {0,1}
    """

    def __init__(
            self,
            root: str,
            volume_size: Tuple[int, int, int] = (64, 128, 128),  # D,H,W target
            transform: Optional[Callable] = None,
    ):
        self.root = root
        self.volume_size = volume_size
        self.transform = transform

        self.vol_specs: List[Tuple[str, Optional[List[str]]]] = []

        nii_paths = sum(
            [glob.glob(os.path.join(root, c, "*.nii")) + glob.glob(os.path.join(root, c, "*.nii.gz")) for c in
             ("cases", "controls")], [])
        for p in nii_paths:
            self.vol_specs.append((p, None))

        # collect slice stacks by prefix up to "_slice-"
        for cls in ("cases", "controls"):
            slice_files = sum([glob.glob(os.path.join(root, cls, f"*{ext}")) for ext in IMG_EXT], [])
            prefix_map: Dict[str, List[str]] = {}
            for p in slice_files:
                m = re.search(r"(.*)_slice-\d+", os.path.basename(p))
                if m:
                    key = os.path.join(os.path.dirname(p), m.group(1))
                    prefix_map.setdefault(key, []).append(p)
            for key, group in prefix_map.items():
                self.vol_specs.append((key, group))

        if not self.vol_specs:
            raise RuntimeError(f"No 3D volumes found under {root}")

        self.vol_specs.sort()

    def __len__(self):
        return len(self.vol_specs)

    def _load_volume(self, spec: Tuple[str, Optional[List[str]]]) -> np.ndarray:
        if spec[1] is None:
            if not _HAS_NIB:
                raise RuntimeError("nibabel not installed but NIfTI files were found.")
            vol = np.asarray(nib.load(spec[0]).get_fdata(), dtype=np.float32)
            # normalize to [0,1]
            vol = (vol - vol.min()) / (vol.ptp() + 1e-8)
        else:
            vol = _stack_slices(spec[1])  # [D,H,W] in [0,1]
        return vol  # [D,H,W]

    def __getitem__(self, idx: int):
        spec = self.vol_specs[idx]
        y = _label_from_path(spec[0])
        vol = self._load_volume(spec)  # [D,H,W]
        D, H, W = self.volume_size
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        x = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)  # [1,D,H,W]
        if self.transform:
            x = self.transform(x)  # expect transform to keep shape [1,D,H,W]
        return x, torch.tensor(y, dtype=torch.float32)
