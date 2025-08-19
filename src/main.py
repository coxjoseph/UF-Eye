#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

try:
    import yaml  # optional

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from sklearn.metrics import roc_auc_score
import joblib

from data import Fundus2DDataset, OCT3DDataset
from models import CNN2D, CNN3D, pretrained_2d, AE2D, AE3D
from trainers import (
    SupervisedTrainer, AETrainer, PCAPipeline, evaluate_auc
)
from utils import iter_kfold_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 1972):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]):
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def pos_weight_from_indices(labels: np.ndarray, idx: np.ndarray) -> float | None:
    y = labels[idx]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return None
    return float(neg / pos)


def collect_targets_from_loader(loader) -> np.ndarray:
    ys = []
    for _, y in loader:
        ys.append(y.numpy())
    return np.concatenate(ys)


def only_controls(dl):
    for x, y in dl:
        m = (y == 0)
        if m.any():
            yield x[m], y[m]


def cv_fundus_supervised(
        model_name: str,
        data_dir: str,
        k: int,
        batch_size: int,
        epochs: int,
        early_stop: int,
        lr_supervised: float,
        weight_decay: float,
        out_root: Path,
        seed: int
) -> Dict[str, Any]:
    ds = Fundus2DDataset(data_dir, image_size=(224, 224))
    labels = np.array(ds.labels if hasattr(ds, "labels") else [int(ds[i][1].item()) for i in range(len(ds))])

    model_tag = f"fundus_{model_name}"
    run_dir = out_root / model_tag
    ensure_dir(run_dir)

    aucs = []
    fold_infos = []

    for fold, (tr, va, tr_idx, va_idx) in enumerate(
            iter_kfold_loaders(
                ds, k=k, batch_size=batch_size,
                stratified=True, group_by_patient=True, seed=seed
            ),
            start=1
    ):
        if model_name == "cnn2d":
            model = CNN2D(in_ch=3)
            lr = lr_supervised
        elif model_name in ("resnet50", "vgg16", "googlenet"):
            model = pretrained_2d(model_name)
            lr = 1e-5  # per spec for fine-tuning
        else:
            raise ValueError(f"Unknown 2D supervised model: {model_name}")

        pw = pos_weight_from_indices(labels, tr_idx)
        trainer = SupervisedTrainer(
            model, lr=lr, weight_decay=weight_decay, pos_weight=pw, device=DEVICE
        )
        trainer.fit(tr, va, epochs=epochs, early_stop=early_stop)
        auc = evaluate_auc(trainer.model, va, device=DEVICE)
        aucs.append(auc)

        # save checkpoint
        ckpt_path = run_dir / f"fold{fold:02d}.pt"
        torch.save(
            {
                "model_name": model_name,
                "fold": fold,
                "state_dict": trainer.model.state_dict(),
                "class": trainer.model.__class__.__name__,
                "config": {
                    "lr": lr, "weight_decay": weight_decay, "pos_weight": pw,
                    "epochs": epochs, "early_stop": early_stop, "seed": seed
                }
            },
            ckpt_path
        )

        fold_infos.append({"fold": fold, "auc": float(auc), "checkpoint": str(ckpt_path)})

        print(f"[Fundus {model_name}] Fold {fold}/{k}: AUC={auc:.4f}")

    result = {
        "model": model_tag,
        "k": k,
        "folds": fold_infos,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    save_json(run_dir / "metrics.json", result)
    print(f"[Fundus {model_name}] AUC mean={result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


def cv_oct_supervised(
        data_dir: str,
        k: int,
        batch_size: int,
        epochs: int,
        early_stop: int,
        lr_supervised: float,
        weight_decay: float,
        volume_size: Tuple[int, int, int],
        out_root: Path,
        seed: int
) -> Dict[str, Any]:
    ds = OCT3DDataset(data_dir, volume_size=volume_size)
    labels = np.array(ds.labels if hasattr(ds, "labels") else [int(ds[i][1].item()) for i in range(len(ds))])

    model_tag = "oct_cnn3d"
    run_dir = out_root / model_tag
    ensure_dir(run_dir)

    aucs = []
    fold_infos = []

    for fold, (tr, va, tr_idx, va_idx) in enumerate(
            iter_kfold_loaders(
                ds, k=k, batch_size=batch_size,
                stratified=True, group_by_patient=True, seed=seed
            ),
            start=1
    ):
        model = CNN3D(in_ch=1)
        pw = pos_weight_from_indices(labels, tr_idx)
        trainer = SupervisedTrainer(
            model, lr=lr_supervised, weight_decay=weight_decay, pos_weight=pw, device=DEVICE
        )
        trainer.fit(tr, va, epochs=epochs, early_stop=early_stop)
        auc = evaluate_auc(trainer.model, va, device=DEVICE)
        aucs.append(auc)

        ckpt_path = run_dir / f"fold{fold:02d}.pt"
        torch.save(
            {
                "model_name": "cnn3d",
                "fold": fold,
                "state_dict": trainer.model.state_dict(),
                "class": trainer.model.__class__.__name__,
                "config": {
                    "lr": lr_supervised, "weight_decay": weight_decay, "pos_weight": pw,
                    "epochs": epochs, "early_stop": early_stop, "seed": seed
                }
            },
            ckpt_path
        )

        fold_infos.append({"fold": fold, "auc": float(auc), "checkpoint": str(ckpt_path)})
        print(f"[OCT CNN3D] Fold {fold}/{k}: AUC={auc:.4f}")

    result = {
        "model": model_tag,
        "k": k,
        "folds": fold_infos,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    save_json(run_dir / "metrics.json", result)
    print(f"[OCT CNN3D] AUC mean={result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


def cv_fundus_autoencoder(
        data_dir: str,
        k: int,
        batch_size: int,
        epochs: int,
        quantile: float,
        out_root: Path,
        seed: int
) -> Dict[str, Any]:
    ds = Fundus2DDataset(data_dir, image_size=(128, 128))
    model_tag = "fundus_ae2d"
    run_dir = out_root / model_tag
    ensure_dir(run_dir)

    aucs = []
    fold_infos = []

    for fold, (tr, va, *_) in enumerate(
            iter_kfold_loaders(ds, k=k, batch_size=batch_size, stratified=True, group_by_patient=True, seed=seed),
            start=1
    ):
        ae = AE2D(in_ch=3, latent_dim=512)
        t = AETrainer(ae, lr=1e-3, device=DEVICE)
        t.fit_controls(only_controls(tr), epochs=epochs)
        t.calibrate_threshold(only_controls(va), quantile=quantile)

        scores = t.predict_scores(va)
        ys = collect_targets_from_loader(va)
        auc = roc_auc_score(ys, scores)
        aucs.append(auc)

        ckpt_path = run_dir / f"fold{fold:02d}.pt"
        torch.save(
            {"model_name": "ae2d", "fold": fold, "state_dict": t.ae.state_dict(), "threshold": t.threshold_},
            ckpt_path
        )
        fold_infos.append(
            {"fold": fold, "auc": float(auc), "threshold": float(t.threshold_), "checkpoint": str(ckpt_path)})
        print(f"[Fundus AE2D] Fold {fold}/{k}: AUC={auc:.4f}, thr={t.threshold_:.6f}")

    result = {
        "model": model_tag,
        "k": k,
        "folds": fold_infos,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    save_json(run_dir / "metrics.json", result)
    print(f"[Fundus AE2D] AUC mean={result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


def cv_oct_autoencoder(
        data_dir: str,
        k: int,
        batch_size: int,
        epochs: int,
        quantile: float,
        volume_size: Tuple[int, int, int],
        out_root: Path,
        seed: int
) -> Dict[str, Any]:
    ds = OCT3DDataset(data_dir, volume_size=volume_size)
    model_tag = "oct_ae3d"
    run_dir = out_root / model_tag
    ensure_dir(run_dir)

    aucs = []
    fold_infos = []

    for fold, (tr, va, *_) in enumerate(
            iter_kfold_loaders(ds, k=k, batch_size=batch_size, stratified=True, group_by_patient=True, seed=seed),
            start=1
    ):
        ae = AE3D(in_ch=1, latent_dim=512)
        t = AETrainer(ae, lr=1e-3, device=DEVICE)
        t.fit_controls(only_controls(tr), epochs=epochs)
        t.calibrate_threshold(only_controls(va), quantile=quantile)

        scores = t.predict_scores(va)
        ys = collect_targets_from_loader(va)
        auc = roc_auc_score(ys, scores)
        aucs.append(auc)

        ckpt_path = run_dir / f"fold{fold:02d}.pt"
        torch.save(
            {"model_name": "ae3d", "fold": fold, "state_dict": t.ae.state_dict(), "threshold": t.threshold_},
            ckpt_path
        )
        fold_infos.append(
            {"fold": fold, "auc": float(auc), "threshold": float(t.threshold_), "checkpoint": str(ckpt_path)})
        print(f"[OCT AE3D] Fold {fold}/{k}: AUC={auc:.4f}, thr={t.threshold_:.6f}")

    result = {
        "model": model_tag,
        "k": k,
        "folds": fold_infos,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    save_json(run_dir / "metrics.json", result)
    print(f"[OCT AE3D] AUC mean={result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


def cv_pca_branch(
        domain: str,  # "fundus" or "oct"
        clf: str,  # "mlp" | "rf" | "xgb"
        data_dir: str,
        k: int,
        batch_size: int,
        epochs_mlp: int,
        volume_size: Tuple[int, int, int] | None,
        out_root: Path,
        seed: int
) -> Dict[str, Any]:
    is_2d = (domain == "fundus")
    if is_2d:
        ds = Fundus2DDataset(data_dir, image_size=(128, 128))
    else:
        assert volume_size is not None
        ds = OCT3DDataset(data_dir, volume_size=(32, 64, 64))

    model_tag = f"{domain}_pca_{clf}"
    run_dir = out_root / model_tag
    ensure_dir(run_dir)

    aucs, fold_infos = [], []

    for fold, (tr, va, *_) in enumerate(
            iter_kfold_loaders(ds, k=k, batch_size=batch_size, stratified=True, group_by_patient=True, seed=seed),
            start=1
    ):
        try:
            pipe = PCAPipeline(n_components=100, clf=clf, device=DEVICE)
            pipe.fit(tr, va, epochs=(epochs_mlp if clf == "mlp" else 0), lr=1e-3)
            probs = pipe.predict_proba(va)
            ys = collect_targets_from_loader(va)
            auc = roc_auc_score(ys, probs)
            aucs.append(auc)
        except RuntimeError as e:
            # xgboost missing, etc.
            print(f"[{model_tag}] Fold {fold}: ERROR {e}")
            auc = float("nan")

        # Save PCA + classifier
        fold_dir = run_dir / f"fold{fold:02d}"
        ensure_dir(fold_dir)
        joblib.dump(pipe.ipca, fold_dir / "ipca.joblib")
        if pipe.clf_name == "mlp":
            torch.save({"state_dict": pipe.mlp.state_dict(), "in_dim": pipe.mlp.net[0].in_features},
                       fold_dir / "mlp.pt")
        elif pipe.clf_name == "rf":
            joblib.dump(pipe.rf, fold_dir / "rf.joblib")
        else:
            if pipe.xgb is not None:
                joblib.dump(pipe.xgb, fold_dir / "xgb.joblib")

        fold_infos.append({"fold": fold, "auc": float(auc)})
        print(f"[{model_tag}] Fold {fold}/{k}: AUC={auc:.4f}")

    result = {
        "model": model_tag,
        "k": k,
        "folds": fold_infos,
        "mean_auc": float(np.nanmean(aucs)),
        "std_auc": float(np.nanstd(aucs)),
    }
    save_json(run_dir / "metrics.json", result)
    print(f"[{model_tag}] AUC mean={result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


# ------------------------- CLI -------------------------

ALL_MODELS = [
    # 2D supervised
    "cnn2d", "resnet50", "vgg16", "googlenet",
    # 2D AE
    "ae2d",
    # 2D PCA
    "pca_mlp_2d", "pca_rf_2d", "pca_xgb_2d",
    # 3D supervised
    "cnn3d",
    # 3D AE
    "ae3d",
    # 3D PCA
    "pca_mlp_3d", "pca_rf_3d", "pca_xgb_3d"
]


def parse_args():
    p = argparse.ArgumentParser(description="UF Eye: Train all models with k-fold CV")
    p.add_argument("--config", type=str, default=None, help="Optional YAML to override CLI args")
    p.add_argument("--data2d", type=str, default="/data/processed/fundus")
    p.add_argument("--data3d", type=str, default="/data/processed/oct")
    p.add_argument("--outdir", type=str, default="runs")

    p.add_argument("--models", type=str, nargs="+",
                   choices=["all"] + ALL_MODELS,
                   default=["all"],
                   help="Which models to train. Use 'all' to run everything.")

    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # training hyperparams
    p.add_argument("--epochs_supervised", type=int, default=20)
    p.add_argument("--epochs_ae", type=int, default=20)
    p.add_argument("--epochs_pca_mlp", type=int, default=30)
    p.add_argument("--early_stop", type=int, default=5)
    p.add_argument("--lr_supervised", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ae_quantile", type=float, default=0.95)

    p.add_argument("--bs2d", type=int, default=16)
    p.add_argument("--bs3d", type=int, default=2)

    p.add_argument("--volD", type=int, default=64)
    p.add_argument("--volH", type=int, default=128)
    p.add_argument("--volW", type=int, default=128)

    return p.parse_args()


def maybe_load_yaml(args):
    if args.config is None:
        return args
    if not _HAS_YAML:
        print("Warning: yaml not installed; ignoring --config")
        return args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


# ------------------------- Main -------------------------

def main():
    args = parse_args()
    args = maybe_load_yaml(args)

    seed_everything(args.seed)

    out_root = Path(args.outdir) / timestamp()
    ensure_dir(out_root)

    # Expand model list
    models = ALL_MODELS if ("all" in args.models) else args.models

    vol_size = (args.volD, args.volH, args.volW)

    summary: Dict[str, Any] = {"runs_dir": str(out_root), "device": DEVICE, "results": []}

    for m in models:
        if m in ("cnn2d", "resnet50", "vgg16", "googlenet"):
            res = cv_fundus_supervised(
                model_name=m,
                data_dir=args.data2d,
                k=args.kfolds,
                batch_size=args.bs2d,
                epochs=args.epochs_supervised,
                early_stop=args.early_stop,
                lr_supervised=args.lr_supervised,
                weight_decay=args.weight_decay,
                out_root=out_root,
                seed=args.seed
            )
        elif m == "ae2d":
            res = cv_fundus_autoencoder(
                data_dir=args.data2d,
                k=args.kfolds,
                batch_size=args.bs2d,
                epochs=args.epochs_ae,
                quantile=args.ae_quantile,
                out_root=out_root,
                seed=args.seed
            )
        elif m == "cnn3d":
            res = cv_oct_supervised(
                data_dir=args.data3d,
                k=args.kfolds,
                batch_size=args.bs3d,
                epochs=args.epochs_supervised,
                early_stop=args.early_stop,
                lr_supervised=args.lr_supervised,
                weight_decay=args.weight_decay,
                volume_size=vol_size,
                out_root=out_root,
                seed=args.seed
            )
        elif m == "ae3d":
            res = cv_oct_autoencoder(
                data_dir=args.data3d,
                k=args.kfolds,
                batch_size=args.bs3d,
                epochs=args.epochs_ae,
                quantile=args.ae_quantile,
                volume_size=vol_size,
                out_root=out_root,
                seed=args.seed
            )
        elif m in ("pca_mlp_2d", "pca_rf_2d", "pca_xgb_2d"):
            clf = m.split("_")[1]  # mlp/rf/xgb
            res = cv_pca_branch(
                domain="fundus",
                clf=clf,
                data_dir=args.data2d,
                k=args.kfolds,
                batch_size=max(args.bs2d, 32),
                epochs_mlp=args.epochs_pca_mlp,
                volume_size=None,
                out_root=out_root,
                seed=args.seed
            )
        elif m in ("pca_mlp_3d", "pca_rf_3d", "pca_xgb_3d"):
            clf = m.split("_")[1]
            res = cv_pca_branch(
                domain="oct",
                clf=clf,
                data_dir=args.data3d,
                k=args.kfolds,
                batch_size=max(args.bs3d, 4),
                epochs_mlp=args.epochs_pca_mlp,
                volume_size=vol_size,
                out_root=out_root,
                seed=args.seed
            )
        else:
            raise ValueError(f"Unknown model key: {m}")

        summary["results"].append(res)

    save_json(out_root / "summary.json", summary)
    print("\n=== DONE ===")
    print(json.dumps(
        {r["model"]: {"mean_auc": r["mean_auc"], "std_auc": r["std_auc"]} for r in summary["results"]},
        indent=2
    ))


if __name__ == "__main__":
    main()
