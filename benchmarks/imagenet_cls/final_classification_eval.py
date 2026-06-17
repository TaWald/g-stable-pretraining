"""Paper-matching final classification eval of a frozen pretrained ViT.

Produces the standard frozen-backbone ImageNet numbers — **linear probe**,
**attentive-pooling probe**, and **kNN** — on the full train/val splits. This is
the figure to report; the scripts in ``benchmarks/imagenet1k/`` are SSL
pretraining runs with *online* probes, not a standalone eval.

Example::

    # open dataset, kNN only (cheapest sanity check)
    python benchmarks/imagenet_cls/final_classification_eval.py --model mae \
        --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
        --dataset imagenet-100 --probe knn

    # full ImageNet-1k, all three probes (heavy: augmented re-forward)
    python benchmarks/imagenet_cls/final_classification_eval.py --model mae \
        --checkpoint .../checkpoint-799.pth --dataset imagenet-1k --probe all \
        --epochs 20 --lrs 1e-2,1e-1,1 --batch-size 256

    python benchmarks/imagenet_cls/final_classification_eval.py --model ijepa \
        --checkpoint .../vith14.../<ckpt>.pth.tar --gijepa-root ../gijepa \
        --probe attentive --batch-size 128
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))  # benchmarks/ (for _backbones)
sys.path.append(str(Path(__file__).parent))
from _backbones import load_ijepa_vit_huge, load_mae_vit_large  # noqa: E402
from _common import build_imagenet_loaders  # noqa: E402
from _final_eval import eval_knn, train_and_eval_probe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mae", "ijepa"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--gijepa-root",
        default=str(Path(__file__).resolve().parents[3] / "gijepa"),
        help="Path to the gijepa repo (I-JEPA only)",
    )
    p.add_argument(
        "--probe",
        choices=["linear", "knn", "attentive", "all"],
        default="all",
    )
    p.add_argument("--dataset", choices=["imagenet-1k", "imagenet-100"], default="imagenet-1k")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20, help="probe epochs per LR")
    p.add_argument("--lrs", default="1e-2,1e-1,1", help="comma-separated LR grid")
    p.add_argument("--knn-ks", default="10,20,200", help="comma-separated k grid")
    p.add_argument("--num-heads", type=int, default=4, help="attentive-probe heads")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--data-cache", default=None)
    args = p.parse_args()

    if args.model == "mae":
        backbone, _, embed_dim = load_mae_vit_large(args.checkpoint, image_size=args.image_size)
    else:
        backbone, _, embed_dim = load_ijepa_vit_huge(
            args.checkpoint, args.gijepa_root, image_size=args.image_size
        )

    train_aug, train_noaug, val, num_classes = build_imagenet_loaders(
        dataset=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_cache=args.data_cache,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrs = [float(x) for x in args.lrs.split(",")]
    ks = [int(x) for x in args.knn_ks.split(",")]
    want = {"linear", "knn", "attentive"} if args.probe == "all" else {args.probe}

    results = {}
    if "knn" in want:
        results["knn"] = eval_knn(
            backbone, train_noaug, val, device, num_classes, ks=ks
        )
    for kind in ("linear", "attentive"):
        if kind in want:
            results[kind] = train_and_eval_probe(
                backbone, embed_dim, kind, train_aug, val, device, num_classes,
                lrs=lrs, epochs=args.epochs, num_heads=args.num_heads,
            )

    print(f"\n=== FINAL classification eval ({args.dataset}, {num_classes} classes) ===")
    for kind, r in results.items():
        knob = f"lr={r['lr']:g}" if "lr" in r else f"k={r['k']}"
        print(f"  {kind:10s}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  ({knob})")


if __name__ == "__main__":
    main()
