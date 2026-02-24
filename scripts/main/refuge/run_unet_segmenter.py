#!/usr/bin/env python3
"""Train and evaluate the U-Net optic disc/cup segmenter."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.unet_segmenter import UNetSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNet segmenter runner")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV")
    parser.add_argument("--train", action="store_true", help="Train the segmenter")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on holdout set")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--disc-weight", type=float, default=1.0)
    parser.add_argument("--cup-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/unet_segmenter"))
    parser.add_argument("--eval-output", type=Path, default=Path("analysis_data/segmenter_eval"))
    parser.add_argument(
        "--normalize",
        choices=["none", "imagenet", "per_image"],
        default="none",
        help="Image normalization mode for train/eval",
    )
    parser.add_argument(
        "--strong-aug",
        action="store_true",
        help="Enable stronger train-time augmentations (flips/rotations)",
    )
    parser.add_argument("--train-datasets", nargs="+", help="Datasets to use for training/validation (default: all)")
    parser.add_argument("--val-datasets", nargs="+", help="Datasets eligible for validation sampling (default: match training)")
    parser.add_argument("--holdout-datasets", nargs="+", help="Restrict holdout entries to these datasets (default: all)")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training data reserved for validation (default: 0.1)",
    )
    parser.add_argument("--eval-datasets", nargs="+", help="Datasets to evaluate (default: holdout split only)")
    parser.add_argument("--eval-splits", nargs="+", help="Splits to evaluate (default: holdout or all when --eval-datasets is set)")
    parser.add_argument("--eval-metrics-path", type=Path, help="Optional CSV path for evaluation metrics output")
    parser.add_argument("--no-eval-overlays", action="store_true", help="Skip writing overlay images during evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarizing predictions")
    parser.add_argument("--tta", action="store_true", help="Enable simple test-time augmentation (H/V flips) during evaluation")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional model weights (.pt) for eval-only runs; defaults to <checkpoint-dir>/best.pt",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device for UNet (default: auto).")
    parser.add_argument("--loader-workers", type=int, default=0, help="DataLoader workers for train/eval.")
    parser.add_argument("--mask-cache-dir", type=Path, default=None, help="Optional cache dir for parsed/resized disc+cup masks.")
    parser.add_argument("--image-cache-dir", type=Path, default=None, help="Optional cache dir for resized RGB images before augmentation.")
    parser.add_argument("--in-memory-cache", action="store_true", help="Cache preprocessed images and masks in RAM (per DataLoader worker process).")
    parser.add_argument("--cache-workers", type=int, default=0, help="Worker threads for prebuilding in-memory cache before training/eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    print(
        f"[UNet] device={selected_device} "
        f"(cuda_available={torch.cuda.is_available()}, workers={args.loader_workers})"
    )
    if selected_device == "cuda":
        idx = torch.cuda.current_device()
        print(f"[UNet] gpu={torch.cuda.get_device_name(idx)}")

    mask_cache_dir = None if args.in_memory_cache else args.mask_cache_dir
    image_cache_dir = None if args.in_memory_cache else args.image_cache_dir
    if args.in_memory_cache and (args.mask_cache_dir or args.image_cache_dir):
        print("[UNet] in_memory_cache enabled: disk caches disabled for this run.")

    segmenter = UNetSegmenter(
        manifest_path=args.manifest,
        device=selected_device,
        cup_weight=args.cup_weight,
        disc_weight=args.disc_weight,
        val_ratio=args.val_ratio,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        holdout_datasets=args.holdout_datasets,
        normalize=args.normalize,
        use_stronger_aug=args.strong_aug,
        mask_cache_dir=mask_cache_dir,
        image_cache_dir=image_cache_dir,
        in_memory_cache=args.in_memory_cache,
        loader_workers=args.loader_workers,
    )
    if mask_cache_dir:
        print(f"[UNet] mask_cache_dir={mask_cache_dir}")
    if image_cache_dir:
        print(f"[UNet] image_cache_dir={image_cache_dir}")
    if args.in_memory_cache:
        print("[UNet] in_memory_cache=enabled (note: memory use scales with loader workers)")
        segmenter.prebuild_in_memory_cache(
            cache_workers=max(0, int(args.cache_workers)),
            include_train=bool(args.train),
            include_val=bool(args.train),
            include_holdout=bool(args.evaluate),
        )

    if args.train:
        segmenter.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
        )

    if args.evaluate:
        if not args.train:
            ckpt = args.weights or (args.checkpoint_dir / "best.pt")
            if ckpt and ckpt.exists():
                state = torch.load(ckpt, map_location=segmenter.device)
                state_dict = state.get("model", state)
                segmenter.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {ckpt}")
            else:
                print(f"[warn] No checkpoint found at {ckpt}. Evaluating untrained weights.")

        split_filter = {"holdout"} if args.eval_splits is None else args.eval_splits
        segmenter.evaluate_dataset(
            dataset_filter=args.eval_datasets,
            split_filter=split_filter,
            output_dir=args.eval_output,
            save_overlays=not args.no_eval_overlays,
            metrics_path=args.eval_metrics_path,
            threshold=args.threshold,
            tta=args.tta,
        )


if __name__ == "__main__":
    main()
