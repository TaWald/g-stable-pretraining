"""Paper-matching final linear segmentation eval (DINOv2/v3) on ADE20k.

Frozen backbone at higher resolution -> linear head trained with full-resolution
(bilinear-upsampled logit) cross-entropy on the full train split, swept over a
small LR grid -> best val mIoU. This is the number to report; the
``--probe linear`` runner in ``mae_vit_large.py`` is a coarse 224 monitor, not a
publishable figure.

Example::

    python benchmarks/ade20k/final_linear_eval.py --model mae \
        --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
        --image-size 512 --epochs 20 --lrs 1e-3,1e-2,1e-1 --batch-size 8

    python benchmarks/ade20k/final_linear_eval.py --model ijepa \
        --checkpoint /dkfz/.../gijepa/vith14.../<ckpt>.pth.tar \
        --gijepa-root ../gijepa --image-size 518 --batch-size 4
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from _common import (  # noqa: E402
    load_ijepa_vit_huge,
    load_mae_vit_large,
)
from _final_eval import (  # noqa: E402
    build_final_ade20k_loaders,
    train_and_eval_linear_seg,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mae", "ijepa"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--gijepa-root",
        default=str(Path(__file__).resolve().parents[3] / "gijepa"),
        help="Path to the gijepa repo (I-JEPA only)",
    )
    p.add_argument("--image-size", type=int, default=512, help="eval resolution (mult. of patch)")
    p.add_argument("--epochs", type=int, default=20, help="linear-head epochs per LR")
    p.add_argument("--lrs", default="1e-3,1e-2,1e-1", help="comma-separated LR grid")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-aug", action="store_true", help="disable train-time augmentation")
    p.add_argument("--no-batchnorm", action="store_true", help="plain Linear head")
    p.add_argument("--data-cache", default=None)
    args = p.parse_args()

    if args.model == "mae":
        backbone, grid_size, embed_dim = load_mae_vit_large(
            args.checkpoint, image_size=args.image_size
        )
    else:
        backbone, grid_size, embed_dim = load_ijepa_vit_huge(
            args.checkpoint, args.gijepa_root, image_size=args.image_size
        )

    train_loader, val_loader = build_final_ade20k_loaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_cache=args.data_cache,
        augment=not args.no_aug,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrs = [float(x) for x in args.lrs.split(",")]
    best = train_and_eval_linear_seg(
        backbone=backbone,
        grid_size=grid_size,
        embed_dim=embed_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lrs=lrs,
        epochs=args.epochs,
        use_batchnorm=not args.no_batchnorm,
    )
    print(
        f"\n=== FINAL linear seg @ {args.image_size}px (grid {grid_size}) ===\n"
        f"best miou={best['miou']:.4f}  pixel_acc={best['pixel_acc']:.4f}  "
        f"(lr={best['lr']:g})"
    )


if __name__ == "__main__":
    main()
