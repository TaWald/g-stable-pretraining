"""Example: monitor ADE20k segmentation inline during an SSL pretraining run.

This is a *skeleton* showing how to attach ``PeriodicSegmentationEval`` to your
own SSL ``Module`` + ``Trainer``. It does not define an SSL method — drop the
callback into your existing pretraining script's ``callbacks=[...]`` list.

Key idea: the callback runs its own ADE20k passes every ``eval_every_n_epochs``
on the frozen backbone, so all you must provide is a ``feature_fn`` that maps a
batch of images to patch tokens ``(B, N, D)`` (CLS dropped) for *your* backbone.

Example::

    python benchmarks/ade20k/periodic_eval_example.py --data-cache /path/to/hf
"""

import argparse
import sys
from pathlib import Path

import lightning as pl

sys.path.append(str(Path(__file__).parent))
from _common import build_periodic_ade20k_seg_callback  # noqa: E402


def make_feature_fn(num_prefix_tokens: int = 1):
    """Return a ``feature_fn`` for a timm-style ViT backbone (drops CLS).

    Adapt this to your backbone. The only contract is:
    ``feature_fn(pl_module, images) -> (B, N, D)`` patch tokens at the eval grid.
    """

    def feature_fn(pl_module, images):
        # e.g. a timm ViT exposing forward_features -> (B, 1 + N, D)
        tokens = pl_module.backbone.forward_features(images)
        return tokens[:, num_prefix_tokens:]

    return feature_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-cache", default=None, help="HF datasets cache_dir")
    p.add_argument("--every", type=int, default=25, help="eval cadence (epochs)")
    p.add_argument("--support-images", type=int, default=2048)
    p.add_argument("--no-linear", action="store_true", help="kNN probe only")
    args = p.parse_args()

    # --- Build YOUR SSL module + datamodule here ---------------------------
    #   module = MySSLModule(...)
    #   data = MySSLDataModule(...)
    # For a ViT-L/16 @224 the patch grid is (14, 14) with embed dim 1024.
    grid_size, embed_dim = (14, 14), 1024  # noqa: F841

    seg_monitor = build_periodic_ade20k_seg_callback(
        grid_size=grid_size,
        feature_fn=make_feature_fn(num_prefix_tokens=1),
        support_images=args.support_images,
        eval_every_n_epochs=args.every,
        run_linear=not args.no_linear,
        data_cache=args.data_cache,
        name="ade20k_periodic_seg",
    )

    # Attach alongside your other callbacks; metrics log under
    #   eval/ade20k_periodic_seg_knn_miou, eval/ade20k_periodic_seg_linear_miou, ...
    trainer = pl.Trainer(  # noqa: F841
        max_epochs=300,
        accelerator="gpu",
        devices=1,
        callbacks=[seg_monitor],
    )
    # trainer.fit(module, datamodule=data)
    print(
        "Skeleton only: wire `module`/`data` to your SSL run, then "
        "`trainer.fit(module, datamodule=data)`.\n"
        f"Attached callback: {seg_monitor.state_key}"
    )


if __name__ == "__main__":
    main()
