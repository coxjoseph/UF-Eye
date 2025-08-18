from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import os, re, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from typing import Iterable, Tuple, Optional


def make_loaders(dataset, batch_size=16, val_frac=0.2, seed=1972) -> Tuple[DataLoader, DataLoader]:
    n = len(dataset)
    nv = int(n * val_frac)
    nt = n - nv
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [nt, nv], generator=g)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def _extract_labels(dataset) -> np.ndarray:
    # Try to grab precomputed labels if present
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=int)
    # Fallback: read one sample at a time (slower)
    ys = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        ys.append(int(y))
    return np.asarray(ys, dtype=int)


def _infer_groups_from_paths(dataset) -> np.ndarray:
    """
    Optional grouping (patient-level) to prevent leakage when multiple images/volumes
    belong to the same participant. Heuristic: take a leading ID token from filename.
    """
    paths = None
    if hasattr(dataset, "paths"):
        paths = dataset.paths
    elif hasattr(dataset, "vol_specs"):
        # use nifti path or the first slice path as representative
        paths = [spec[0] if spec[1] is None else spec[1][0] for spec in dataset.vol_specs]
    if paths is None:
        # no paths available; default each sample to its own group
        return np.arange(len(dataset))
    gids = []
    for p in paths:
        bn = os.path.basename(p)
        m = re.match(r"^(\d{4,})", bn) or re.match(r"^([A-Za-z0-9]+)", bn)
        gids.append(m.group(1) if m else bn.split("_")[0])
    # encode as integers for sklearn
    _, enc = np.unique(gids, return_inverse=True)
    return enc


def iter_kfold_loaders(
        dataset,
        k: int = 5,
        batch_size: int = 16,
        stratified: bool = True,
        group_by_patient: bool = True,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Iterable[Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]]:
    """
    Yields (train_loader, val_loader, train_idx, val_idx) for each fold.
    - stratified=True keeps class balance per fold (recommended).
    - group_by_patient=True uses patient/ID grouping if filenames allow.
    """
    y = _extract_labels(dataset)
    groups = _infer_groups_from_paths(dataset) if group_by_patient else None

    # Choose splitter
    splits = None
    try:
        # sklearn >= 1.1
        from sklearn.model_selection import StratifiedGroupKFold
        if stratified and group_by_patient:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            splits = sgkf.split(np.zeros(len(y)), y, groups)
    except Exception:
        pass

    if splits is None:
        from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
        if stratified and not group_by_patient:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            splits = skf.split(np.zeros(len(y)), y)
        elif (not stratified) and group_by_patient:
            gkf = GroupKFold(n_splits=k)
            splits = gkf.split(np.zeros(len(y)), y, groups)
        else:
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            splits = kf.split(np.zeros(len(y)))

    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_loader = DataLoader(
            Subset(dataset, tr_idx), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_seed_worker
        )
        va_loader = DataLoader(
            Subset(dataset, va_idx), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_seed_worker
        )
        yield tr_loader, va_loader, tr_idx, va_idx
