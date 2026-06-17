"""ADE20k dense-kNN segmentation eval for an MAE ViT-L/16 pretrained encoder.

Frozen backbone -> patch tokens -> weighted kNN over a support queue built from
the train split -> per-pixel mIoU on the val split. See ``_common.py`` and
``README.md`` for details.

Example::

    python benchmarks/ade20k/mae_vit_large.py \
        --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
        --batch-size 32 --queue-length 1000000 --k 20
"""

import argparse
import sys
from pathlib import Path

import lightning as pl

sys.path.append(str(Path(__file__).parent))
from _common import (  # noqa: E402
    build_ade20k_datamodule,
    build_knn_segmentation_callback,
    build_linear_segmentation_callback,
    build_module,
    load_mae_vit_large,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="MAE ViT-L pretrain .pth")
    p.add_argument(
        "--probe",
        choices=["knn", "linear"],
        default="knn",
        help="knn = CAPI weighted-kNN seg; linear = DINOv3 linear seg head",
    )
    p.add_argument("--data-cache", default=None, help="HF datasets cache_dir")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    # kNN knobs
    p.add_argument("--queue-length", type=int, default=1_000_000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=4096)
    # linear-probe knobs
    p.add_argument("--epochs", type=int, default=20, help="linear-probe epochs")
    p.add_argument("--lr", type=float, default=1e-2, help="linear-probe LR")
    p.add_argument(
        "--no-batchnorm",
        action="store_true",
        help="use a plain Linear head (default: BatchNorm1d + Linear)",
    )
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args()

    backbone, grid_size, embed_dim = load_mae_vit_large(args.checkpoint)

    data = build_ade20k_datamodule(
        image_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.data_cache,
    )
    module = build_module(backbone, grid_size)

    if args.probe == "knn":
        callback = build_knn_segmentation_callback(
            embed_dim=embed_dim,
            grid_size=grid_size,
            queue_length=args.queue_length,
            k=args.k,
            chunk_size=args.chunk_size,
            name="mae_vitl_ade20k_knn_seg",
        )
        max_epochs = 1  # one pass just fills the support queue
    else:
        callback = build_linear_segmentation_callback(
            module=module,
            embed_dim=embed_dim,
            grid_size=grid_size,
            use_batchnorm=not args.no_batchnorm,
            lr=args.lr,
            epochs=args.epochs,
            name="mae_vitl_ade20k_linear_seg",
        )
        max_epochs = args.epochs

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=args.devices,
        num_sanity_val_steps=0,
        callbacks=[callback],
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(module, datamodule=data)


if __name__ == "__main__":
    main()
