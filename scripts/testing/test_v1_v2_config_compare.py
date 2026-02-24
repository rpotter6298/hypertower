from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from classes.frontend import Multifold
from classes.bridge import Bridge, VoteBridge
from classes.dataset import ClinicalDataset
from classes.hypertower import _ClinicalView
from classes.image_tower import ImageTower
from classes.md_tower import MDTower
from classes.papila_builders import build_papila_clinical
from classes.v2 import (
    PatientSplit,
    SlotLoaderFactory,
    SlotDataset,
    slot_collate,
    assemble_config,
    build_model_bundle,
    build_papila_profile,
    resolve_imports,
)
from classes.v2.split_manager import PatientFirstSplitManager


def build_v1_defaults() -> Dict[str, Any]:
    parser = Multifold.build_parser()
    args = parser.parse_args([])

    clinical = build_papila_clinical(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=list(args.cat_cols),
        n_splits=args.n_splits,
        random_seed=args.fold_seed,
    )

    return {
        "args": args,
        "clinical": clinical,
        "image_dir": args.image_dir,
        "clinical_dir": args.clinical_dir,
        "label_col": args.label_col,
        "cat_cols": list(args.cat_cols),
        "image_transform": {
            "resize": 256,
            "center_crop": 224,
            "hflip": True,
            "vflip": True,
            "rotation": 15,
            "color_jitter": (0.1, 0.1, 0.1, 0.05),
        },
        "image_tower": {
            "backbone": args.backbone,
            "freeze_ratio": args.freeze_ratio,
            "augment": args.img_augment,
            "geometry_dim": 0,
            "use_se": False,  # se_where default is bridge
            "se_reduction": args.se_reduction_tower,
            "se_pre_norm": args.se_pre_norm_tower,
        },
        "md_tower": {
            "hidden_dim": 128,
            "dropout": 0.1,
            "use_se": False,
            "se_reduction": args.se_reduction_tower,
            "se_pre_norm": args.se_pre_norm_tower,
            "freeze_ratio": 0.0,
        },
        "bridge": {
            "method": "fusion" if args.fusion_mode == "fused" else "consensus",
            "fusion_dim": 256,
            "use_se": args.use_se,
            "se_reduction": args.se_reduction,
            "se_pre_norm": args.se_pre_norm,
        },
    }


def build_v2_from_config(path: Path) -> Dict[str, Any]:
    assembly = assemble_config(path)
    imports = resolve_imports(assembly)
    if not imports:
        raise ValueError("Config did not include any imports.")
    clinical = next(iter(imports.values()))

    image_loader = _find_loader(assembly, input_type="image")
    matrix_loader = _find_loader(assembly, input_type="matrix")

    return {
        "assembly": assembly,
        "clinical": clinical,
        "image_loader": image_loader,
        "matrix_loader": matrix_loader,
        "image_transform_chain": [t.transform_type for t in image_loader.transforms],
    }


def _find_loader(assembly, input_type: str):
    matches = [
        loader for loader in assembly.loaders.values() if loader.input_type == input_type
    ]
    if not matches:
        raise ValueError(f"No loader with input_type={input_type!r} found in config.")
    if len(matches) > 1:
        raise ValueError(f"Multiple loaders with input_type={input_type!r} found.")
    return matches[0]


def compare_configs(v1: Dict[str, Any], v2: Dict[str, Any]) -> List[str]:
    diffs: List[str] = []

    # data sources
    v1_rows, v1_cols = v1["clinical"].df.shape
    v2_rows, v2_cols = v2["clinical"].df.shape
    if v1_rows != v2_rows or v1_cols != v2_cols:
        diffs.append(
            f"Clinical DF shape mismatch: v1={v1_rows}x{v1_cols}, v2={v2_rows}x{v2_cols}"
        )

    # loader presence
    if not v2.get("image_loader"):
        diffs.append("Missing image loader in v2 config.")
    if not v2.get("matrix_loader"):
        diffs.append("Missing metadata loader in v2 config.")

    # transform chain expectations
    expected_chain = ["resize", "center_crop", "jitter_bundle"]
    if v2.get("image_transform_chain") != expected_chain:
        diffs.append(
            f"Image transform chain mismatch: v1 expects {expected_chain}, v2 has {v2.get('image_transform_chain')}"
        )

    # image tower settings
    v1_img = v1["image_tower"]
    v2_img = _extract_tower(assembly=v2["assembly"], tower_type="image")
    _compare_dict(diffs, "ImageTower", v1_img, v2_img)

    # metadata tower settings
    v1_md = v1["md_tower"]
    v2_md = _extract_tower(assembly=v2["assembly"], tower_type="metadata")
    _compare_dict(diffs, "MDTower", v1_md, v2_md)

    # bridge settings
    v2_bridge = _extract_bridge(v2["assembly"])
    _compare_dict(diffs, "Bridge", v1["bridge"], v2_bridge)

    if not v2["assembly"].classifiers:
        diffs.append("Missing classifier node in v2 config.")

    # splits
    v1_train, v1_val = v1["clinical"].get_split_dfs(0)
    sm = PatientFirstSplitManager()
    args = SimpleNamespace(
        n_splits=v1["args"].n_splits,
        fold_seed=v1["args"].fold_seed,
        holdout_per_class=v1["args"].holdout_per_class,
        holdout_seed=v1["args"].holdout_seed,
        eval_mode=v1["args"].eval_mode,
    )
    splits = sm.build_plans(clinical=v2["clinical"], args=args, profile=None)
    v2_train = splits[0].train
    v2_val = splits[0].val
    if len(v1_train) != len(v2_train) or len(v1_val) != len(v2_val):
        diffs.append(
            f"Split sizes mismatch: v1 train/val={len(v1_train)}/{len(v1_val)}, "
            f"v2 train/val={len(v2_train)}/{len(v2_val)}"
        )

    return diffs


def _extract_tower(*, assembly, tower_type: str) -> Dict[str, Any]:
    towers = [
        tower for tower in assembly.towers.values() if tower.tower_type == tower_type
    ]
    if not towers:
        raise ValueError(f"No {tower_type} tower found in v2 config.")
    if len(towers) > 1:
        raise ValueError(f"Multiple {tower_type} towers found in v2 config.")
    return towers[0].params


def _extract_bridge(assembly) -> Dict[str, Any]:
    if not assembly.bridges:
        raise ValueError("No bridge node found in v2 config.")
    if len(assembly.bridges) > 1:
        raise ValueError("Multiple bridge nodes found in v2 config.")
    bridge = next(iter(assembly.bridges.values()))
    payload = dict(bridge.params)
    payload["method"] = bridge.method
    return payload


def _compare_dict(diffs: List[str], label: str, v1: Dict[str, Any], v2: Dict[str, Any]) -> None:
    for key, v1_val in v1.items():
        v2_val = v2.get(key)
        if isinstance(v1_val, tuple):
            v1_val = list(v1_val)
        if isinstance(v2_val, tuple):
            v2_val = list(v2_val)
        if v1_val != v2_val:
            diffs.append(f"{label} mismatch for {key}: v1={v1_val} v2={v2_val}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hypertower_v2_config.json"),
        help="Path to v2 config JSON",
    )
    parser.add_argument("--samples", type=int, default=8, help="Number of samples to compare")
    parser.add_argument("--seed", type=int, default=1234, help="Seed used for deterministic comparisons")
    parser.add_argument(
        "--image-compare",
        choices=["shape", "value"],
        default="value",
        help="Compare image tensors by shape only or by value",
    )
    parser.add_argument(
        "--no-data-compare",
        action="store_true",
        help="Skip the data/loader comparison step",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["eye", "patient"],
        default="eye",
        help="Sample mode for V2 loaders (eye-level or patient-level).",
    )
    parser.add_argument("--train-epochs", type=int, default=2, help="Epochs to run in train comparison.")
    parser.add_argument("--train-folds", type=int, default=2, help="Folds to run in train comparison.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Batch size for train comparison.")
    parser.add_argument("--max-batches", type=int, default=10, help="Max batches per epoch (train/val).")
    parser.add_argument("--loss-tol", type=float, default=0.5, help="Tolerance for loss diffs.")
    parser.add_argument("--acc-tol", type=float, default=0.15, help="Tolerance for accuracy diffs.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device to use for training comparison.",
    )
    parser.add_argument(
        "--no-train-compare",
        action="store_true",
        help="Skip the training comparison step.",
    )
    args = parser.parse_args()
    v1 = build_v1_defaults()
    v2 = build_v2_from_config(args.config)
    diffs = compare_configs(v1, v2)
    if diffs:
        print("Differences detected:")
        for diff in diffs:
            print(f"- {diff}")
        return 1

    print("V1 vs V2 config comparison: OK (settings and loaders match).")

    if not args.no_data_compare:
        data_diffs = compare_initial_data(
            v1,
            v2,
            samples=args.samples,
            seed=args.seed,
            compare_mode=args.image_compare,
        )
        if data_diffs:
            print("Differences detected in initial data:")
            for diff in data_diffs:
                print(f"- {diff}")
            return 1
        print("Initial data comparison: OK (image/meta/label inputs match).")

    if not args.no_train_compare:
        train_diffs = compare_training_runs(
            v1,
            v2,
            epochs=args.train_epochs,
            folds=args.train_folds,
            batch_size=args.train_batch_size,
            max_batches=args.max_batches,
            seed=args.seed,
            loss_tol=args.loss_tol,
            acc_tol=args.acc_tol,
            sample_mode=args.sample_mode,
            device=args.device,
        )
        if train_diffs:
            print("Differences detected in training comparison:")
            for diff in train_diffs:
                print(f"- {diff}")
            return 1
        print("Training comparison: OK (metrics within tolerance).")

    return 0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_v1_modules(v1: Dict[str, Any]) -> Dict[str, Any]:
    args = v1["args"]
    clinical = v1["clinical"]

    img_tower = ImageTower(
        backbone=args.backbone,
        freeze_ratio=args.freeze_ratio,
        use_se=False,
        se_reduction=args.se_reduction_tower,
        se_pre_norm=args.se_pre_norm_tower,
        augment=args.img_augment,
        geometry_dim=0,
    )
    md_tower = MDTower(
        clinical,
        hidden_dim=128,
        dropout=0.1,
        use_se=False,
        se_reduction=args.se_reduction_tower,
        se_pre_norm=args.se_pre_norm_tower,
    )
    bridge = None
    if args.fusion_mode == "vote":
        bridge = VoteBridge(num_classes=args.num_classes)
    else:
        bridge = Bridge(
            img_dim=img_tower.out_dim,
            meta_dim=md_tower.out_dim,
            num_classes=args.num_classes,
            fusion_dim=256,
            mode="fused",
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            se_pre_norm=args.se_pre_norm,
        )

    return {"image_tower": img_tower, "metadata_tower": md_tower, "bridge": bridge}


def compare_initial_data(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
    *,
    samples: int = 8,
    seed: int = 1234,
    compare_mode: str = "value",
) -> List[str]:
    diffs: List[str] = []

    v1_modules = _build_v1_modules(v1)
    v2_modules = build_model_bundle(v2["assembly"], v2["clinical"])

    # Build consistent train split for both datasets
    v1_train, _ = v1["clinical"].get_split_dfs(0)
    split = PatientSplit(train=v1_train, val=v1_train.iloc[:0], holdout=None)

    # V1 dataset
    v1_view = _ClinicalView(v1["clinical"], v1_train)
    v1_ds = ClinicalDataset(v1_view, v1_modules["image_tower"].transform)

    # V2 dataset
    if v2_modules.image_transform is None:
        diffs.append("V2 image transform could not be built from config.")
        return diffs
    loader_factory = SlotLoaderFactory(image_transform=v2_modules.image_transform)
    v2_loaders = loader_factory.build(
        clinical=v2["clinical"],
        split=split,
        args=SimpleNamespace(batch_size=1),
        fold=0,
        profile=None,
    )
    v2_ds = v2_loaders.train.dataset

    total = min(samples, len(v1_ds), len(v2_ds))
    for idx in range(total):
        _seed_all(seed + idx)
        v1_item = v1_ds[idx]
        _seed_all(seed + idx)
        v2_item = v2_ds[idx]

        if len(v1_item) == 4:
            v1_img, v1_meta, _, v1_label = v1_item
        else:
            v1_img, v1_meta, v1_label = v1_item

        v2_img = v2_item.get("image_1")
        v2_meta = v2_item.get("matrix_1")
        v2_label = v2_item.get("label_1")

        if v2_label is None or int(v2_label) != int(v1_label):
            diffs.append(f"Label mismatch at idx {idx}: v1={int(v1_label)} v2={v2_label}")

        if v2_meta is None:
            diffs.append(f"Missing v2 metadata at idx {idx}")
        else:
            if not torch.allclose(v1_meta, v2_meta, atol=1e-6, rtol=0.0):
                max_diff = float((v1_meta - v2_meta).abs().max().item())
                diffs.append(f"Metadata mismatch at idx {idx}: max_abs_diff={max_diff:.6f}")

        if v2_img is None:
            diffs.append(f"Missing v2 image at idx {idx}")
        else:
            if tuple(v1_img.shape) != tuple(v2_img.shape):
                diffs.append(
                    f"Image shape mismatch at idx {idx}: v1={tuple(v1_img.shape)} v2={tuple(v2_img.shape)}"
                )
            elif compare_mode == "value":
                max_diff = float((v1_img - v2_img).abs().max().item())
                if max_diff > 1e-5:
                    diffs.append(f"Image tensor mismatch at idx {idx}: max_abs_diff={max_diff:.6f}")

    return diffs


def compare_training_runs(
    v1: Dict[str, Any],
    v2: Dict[str, Any],
    *,
    epochs: int,
    folds: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    loss_tol: float,
    acc_tol: float,
    sample_mode: str,
    device: str,
) -> List[str]:
    diffs: List[str] = []
    if sample_mode != "eye":
        diffs.append("Training compare only supports sample_mode='eye' for parity with v1.")
        return diffs

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _seed_all(seed)

    # Build profile for V2 dataset
    profile = build_papila_profile(
        patient_col="Patient ID",
        label_col=v1["args"].label_col,
        sample_mode=sample_mode,
    )

    # Build splits
    sm = PatientFirstSplitManager()
    split_args = SimpleNamespace(
        n_splits=v1["args"].n_splits,
        fold_seed=v1["args"].fold_seed,
        holdout_per_class=v1["args"].holdout_per_class,
        holdout_seed=v1["args"].holdout_seed,
        eval_mode=v1["args"].eval_mode,
    )
    plans = sm.build_plans(clinical=v2["clinical"], args=split_args, profile=profile)

    folds = min(folds, len(plans))

    for fold in range(folds):
        # Build V1 modules per fold (seeded)
        _seed_all(seed + fold * 1000 + 1)
        v1_modules = _build_v1_modules(v1)
        _move_modules(v1_modules, device)

        # Build V2 modules per fold (seeded to match V1 init)
        _seed_all(seed + fold * 1000 + 1)
        v2_bundle = build_model_bundle(v2["assembly"], v2["clinical"])
        if v2_bundle.bridge is None:
            diffs.append("V2 model bundle missing bridge.")
            return diffs
        if isinstance(v2_bundle.bridge, VoteBridge):
            diffs.append("V2 bridge is VoteBridge; training compare only supports fusion bridge.")
            return diffs
        if v2_bundle.image_transform is None:
            diffs.append("V2 image transform missing; cannot run training compare.")
            return diffs
        v2_modules = {
            "image_tower": v2_bundle.image_tower,
            "metadata_tower": v2_bundle.metadata_tower,
            "bridge": v2_bundle.bridge,
        }
        _move_modules(v2_modules, device)

        split = plans[fold]
        v1_train = split.train
        v1_val = split.val

        v1_train_ds = _build_v1_dataset(v1, v1_train, v1_modules["image_tower"].transform)
        v1_val_ds = _build_v1_dataset(v1, v1_val, v1_modules["image_tower"].transform)

        v2_train_ds = _build_v2_dataset(v2, v1_train, profile, v2_bundle.image_transform)
        v2_val_ds = _build_v2_dataset(v2, v1_val, profile, v2_bundle.image_transform)

        # Optimizers
        v1_opt = torch.optim.Adam(
            list(v1_modules["image_tower"].parameters())
            + list(v1_modules["metadata_tower"].parameters())
            + list(v1_modules["bridge"].parameters()),
            lr=float(v1["args"].lr),
        )
        v2_opt = torch.optim.Adam(
            list(v2_modules["image_tower"].parameters())
            + list(v2_modules["metadata_tower"].parameters())
            + list(v2_modules["bridge"].parameters()),
            lr=float(v1["args"].lr),
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            _seed_all(seed + fold * 100 + epoch)
            v1_train_metrics = _run_epoch_v1(
                v1_modules,
                v1_train_ds,
                v1_opt,
                criterion,
                device,
                batch_size=batch_size,
                max_batches=max_batches,
                train=True,
                seed=seed + fold * 100 + epoch,
            )
            v2_train_metrics = _run_epoch_v2(
                v2_modules,
                v2_train_ds,
                v2_opt,
                criterion,
                device,
                batch_size=batch_size,
                max_batches=max_batches,
                train=True,
                seed=seed + fold * 100 + epoch,
            )

            v1_val_metrics = _run_epoch_v1(
                v1_modules,
                v1_val_ds,
                None,
                criterion,
                device,
                batch_size=batch_size,
                max_batches=max_batches,
                train=False,
                seed=seed + fold * 100 + epoch + 777,
            )
            v2_val_metrics = _run_epoch_v2(
                v2_modules,
                v2_val_ds,
                None,
                criterion,
                device,
                batch_size=batch_size,
                max_batches=max_batches,
                train=False,
                seed=seed + fold * 100 + epoch + 777,
            )

            print(
                f"[fold {fold} epoch {epoch}] "
                f"v1 train loss={v1_train_metrics['loss']:.4f} acc={v1_train_metrics['acc']:.4f} | "
                f"v2 train loss={v2_train_metrics['loss']:.4f} acc={v2_train_metrics['acc']:.4f}"
            )
            print(
                f"[fold {fold} epoch {epoch}] "
                f"v1 val   loss={v1_val_metrics['loss']:.4f} acc={v1_val_metrics['acc']:.4f} | "
                f"v2 val   loss={v2_val_metrics['loss']:.4f} acc={v2_val_metrics['acc']:.4f}"
            )

            diffs.extend(
                _compare_epoch_metrics(
                    fold,
                    epoch,
                    v1_train_metrics,
                    v2_train_metrics,
                    v1_val_metrics,
                    v2_val_metrics,
                    loss_tol,
                    acc_tol,
                )
            )

    return diffs


def _compare_epoch_metrics(
    fold: int,
    epoch: int,
    v1_train: Dict[str, float],
    v2_train: Dict[str, float],
    v1_val: Dict[str, float],
    v2_val: Dict[str, float],
    loss_tol: float,
    acc_tol: float,
) -> List[str]:
    diffs: List[str] = []
    for split_name, a, b in (
        ("train", v1_train, v2_train),
        ("val", v1_val, v2_val),
    ):
        loss_diff = abs(a["loss"] - b["loss"])
        acc_diff = abs(a["acc"] - b["acc"])
        if loss_diff > loss_tol:
            diffs.append(
                f"Fold {fold} epoch {epoch} {split_name} loss diff {loss_diff:.4f} (v1={a['loss']:.4f} v2={b['loss']:.4f})"
            )
        if acc_diff > acc_tol:
            diffs.append(
                f"Fold {fold} epoch {epoch} {split_name} acc diff {acc_diff:.4f} (v1={a['acc']:.4f} v2={b['acc']:.4f})"
            )
    return diffs


def _move_modules(modules: Dict[str, Any], device: str) -> None:
    for module in modules.values():
        if module is not None and hasattr(module, "to"):
            module.to(device)


def _build_v1_dataset(v1: Dict[str, Any], df, image_transform) -> ClinicalDataset:
    view = _ClinicalView(v1["clinical"], df)
    return ClinicalDataset(view, image_transform)


def _build_v2_dataset(v2: Dict[str, Any], df, profile, image_transform) -> SlotDataset:
    samples = profile.build_samples(df=df, clinical=v2["clinical"])
    return SlotDataset(
        samples,
        profile.slot_descriptors(),
        image_transform=image_transform,
    )


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    collate_fn=None,
) -> torch.utils.data.DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        collate_fn=collate_fn,
    )


def _run_epoch_v1(
    modules: Dict[str, Any],
    dataset: ClinicalDataset,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: str,
    *,
    batch_size: int,
    max_batches: int,
    train: bool,
    seed: int,
) -> Dict[str, float]:
    _seed_all(seed)
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=train, seed=seed)
    image_tower = modules["image_tower"]
    md_tower = modules["metadata_tower"]
    bridge = modules["bridge"]

    image_tower.train(train)
    md_tower.train(train)
    bridge.train(train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            if len(batch) == 4:
                imgs, metas, _, labels = batch
            else:
                imgs, metas, labels = batch
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            img_feats = image_tower(imgs)
            md_feats = md_tower(metas)
            out_fused, _, _ = bridge(img_feats, md_feats)
            loss = criterion(out_fused, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().item()) * labels.size(0)
            total_correct += (out_fused.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    if total_count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


def _run_epoch_v2(
    modules: Dict[str, Any],
    dataset: SlotDataset,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: str,
    *,
    batch_size: int,
    max_batches: int,
    train: bool,
    seed: int,
) -> Dict[str, float]:
    _seed_all(seed)
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        seed=seed,
        collate_fn=slot_collate,
    )
    image_tower = modules["image_tower"]
    md_tower = modules["metadata_tower"]
    bridge = modules["bridge"]

    image_tower.train(train)
    md_tower.train(train)
    bridge.train(train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            imgs = batch.get("image_1")
            metas = batch.get("matrix_1")
            labels = batch.get("label_1")
            if imgs is None or metas is None or labels is None:
                continue
            if not torch.is_tensor(imgs) or not torch.is_tensor(metas):
                continue
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = torch.as_tensor(labels, device=device)

            if optimizer is not None:
                optimizer.zero_grad()

            img_feats = image_tower(imgs)
            md_feats = md_tower(metas)
            out_fused, _, _ = bridge(img_feats, md_feats)
            loss = criterion(out_fused, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().item()) * labels.size(0)
            total_correct += (out_fused.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    if total_count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


if __name__ == "__main__":
    raise SystemExit(main())
