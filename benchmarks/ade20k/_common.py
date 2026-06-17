"""Shared pieces for the ADE20k dense-kNN segmentation eval (MAE + I-JEPA).

This is a *frozen-backbone* protocol: we ``fit`` for a single epoch only to
stream the training split's patch tokens into the support queue, then the
``OnlineKNNSegmentation`` callback computes per-pixel mIoU on the validation
split. No representation learning happens — the backbone has
``requires_grad=False`` and a throw-away ``anchor`` parameter satisfies
Lightning's optimizer requirement.

Dataset: ``scene_parse_150`` (the HuggingFace mirror of ADE20K /
ADEChallengeData2016). Classes are 1..150; 0 is "unlabeled" and is treated as
the ignore index in the mIoU metric.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import stable_pretraining as spt
from stable_pretraining.data import transforms

# ADE20k: 150 semantic classes + label 0 = unlabeled/ignore -> 151 indices.
ADE20K_NUM_CLASSES = 151
ADE20K_IGNORE_INDEX = 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class MaskToLabels(transforms.Transform):
    """Convert a segmentation mask (PIL/ndarray/tensor) to an integer label map.

    Unlike :class:`~stable_pretraining.data.transforms.ToImage`, this does **no**
    scaling or normalization — pixel values are class indices and must be
    preserved exactly. Produces a ``(H, W)`` ``torch.long`` tensor.
    """

    def __init__(self, source: str = "mask", target: str = "mask"):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, x):
        m = self.nested_get(x, self.source)
        if isinstance(m, Image.Image):
            arr = np.array(m, copy=True)
            t = torch.from_numpy(arr)
        elif torch.is_tensor(m):
            t = m
        else:
            t = torch.as_tensor(np.array(m))
        if t.dim() == 3:  # (H, W, C) or (C, H, W) -> single channel
            t = t[..., 0] if t.shape[-1] <= 4 else t.squeeze(0)
        self.nested_set(x, t.long(), self.target)
        return x


def _build_transform(image_size: int):
    norm = spt.data.static.ADE20K
    return transforms.Compose(
        transforms.RGB(source="image", target="image"),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            source="image",
            target="image",
        ),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
            source="mask",
            target="mask",
        ),
        transforms.ToImage(
            mean=norm["mean"], std=norm["std"], source="image", target="image"
        ),
        MaskToLabels(source="mask", target="mask"),
    )


def build_ade20k_datamodule(
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    cache_dir=None,
):
    """ADE20k train/val DataModule yielding ``{"image", "mask", ...}`` samples."""

    def _loader(split, shuffle):
        ds = spt.data.HFDataset(
            "scene_parse_150",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
            rename_columns={"annotation": "mask"},
            transform=_build_transform(image_size),
        )
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    return spt.data.DataModule(
        train=_loader("train", shuffle=True),
        val=_loader("validation", shuffle=False),
    )


# ---------------------------------------------------------------------------
# Frozen dense forward + module
# ---------------------------------------------------------------------------
def make_dense_forward(grid_size):
    """Build a forward fn that emits flattened patch tokens + patch labels.

    The backbone (``self.backbone``) must return patch tokens of shape
    ``(B, H_g * W_g, D)`` with the CLS token already removed.
    """
    h_g, w_g = grid_size
    n_patches = h_g * w_g

    def forward(self, batch, stage):
        self.backbone.eval()  # keep features deterministic even in train mode
        with torch.no_grad():
            tokens = self.backbone(batch["image"])  # (B, N, D)
        b, n, d = tokens.shape
        if n != n_patches:
            raise ValueError(
                f"backbone returned {n} patch tokens but grid_size={grid_size} "
                f"implies {n_patches}; check image_size / patch_size"
            )
        patch_feats = tokens.reshape(b * n, d)

        mask = batch["mask"]  # (B, H, W) long
        patch_labels = (
            F.interpolate(mask[:, None].float(), size=(h_g, w_g), mode="nearest")
            .long()
            .reshape(b * n)
        )

        # Zero loss with a live grad path to the throw-away anchor param so
        # Lightning's optimizer has something to step (backbone stays frozen).
        loss = self.anchor(patch_feats.new_zeros(1, 1)).sum() * 0.0
        return {
            "loss": loss,
            "patch_embedding": patch_feats,
            "patch_label": patch_labels,
            "mask": mask,
        }

    return forward


def build_module(backbone, grid_size):
    backbone.requires_grad_(False)
    return spt.Module(
        forward=make_dense_forward(grid_size),
        backbone=backbone,
        anchor=nn.Linear(1, 1),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.0},
            "scheduler": {"type": "ConstantLR"},
            "interval": "epoch",
        },
    )


def build_knn_segmentation_callback(
    embed_dim: int,
    grid_size,
    queue_length: int = 1_000_000,
    k: int = 20,
    chunk_size: int = 4096,
    name: str = "ade20k_knn_seg",
):
    return spt.OnlineKNNSegmentation(
        name=name,
        input="patch_embedding",
        target="patch_label",
        grid_size=grid_size,
        mask_key="mask",
        num_classes=ADE20K_NUM_CLASSES,
        metrics=_ade20k_seg_metrics(),
        queue_length=queue_length,
        input_dim=embed_dim,
        k=k,
        temperature=0.07,
        chunk_size=chunk_size,
        distance_metric="cosine",
    )


def _ade20k_seg_metrics():
    """Build mIoU + pixel accuracy with label 0 (unlabeled) as ignore index."""
    return {
        "miou": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=ADE20K_NUM_CLASSES,
            ignore_index=ADE20K_IGNORE_INDEX,
            average="macro",
        ),
        "pixel_acc": torchmetrics.classification.MulticlassAccuracy(
            num_classes=ADE20K_NUM_CLASSES,
            ignore_index=ADE20K_IGNORE_INDEX,
            average="micro",
            multidim_average="global",
        ),
    }


def build_linear_segmentation_callback(
    module,
    embed_dim: int,
    grid_size,
    use_batchnorm: bool = True,
    lr: float = 1e-2,
    epochs: int = 20,
    name: str = "ade20k_linear_seg",
):
    """DINOv3-style linear segmentation probe over frozen patch tokens.

    A ``BatchNorm1d`` + ``Linear`` head (or plain ``Linear`` when
    ``use_batchnorm=False``) is trained with SGD + cosine schedule on the patch
    labels, then scored as per-pixel mIoU at the full mask resolution. ``module``
    is the ``spt.Module`` being trained (required to register the head's
    optimizer). ``epochs`` should match ``Trainer(max_epochs=...)`` so the cosine
    schedule spans the full run.
    """
    if use_batchnorm:
        probe = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, ADE20K_NUM_CLASSES),
        )
    else:
        probe = nn.Linear(embed_dim, ADE20K_NUM_CLASSES)

    return spt.OnlineProbeSegmentation(
        module=module,
        name=name,
        input="patch_embedding",
        target="patch_label",
        grid_size=grid_size,
        mask_key="mask",
        num_classes=ADE20K_NUM_CLASSES,
        probe=probe,
        metrics=_ade20k_seg_metrics(),
        loss=nn.CrossEntropyLoss(ignore_index=ADE20K_IGNORE_INDEX),
        optimizer={"type": "SGD", "lr": lr, "momentum": 0.9, "weight_decay": 0.0},
        scheduler={"type": "CosineAnnealingLR", "T_max": epochs},
    )


def _ade20k_loader(split, shuffle, batch_size, num_workers, cache_dir, image_size, subset_n=None):
    ds = spt.data.HFDataset(
        "scene_parse_150",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
        rename_columns={"annotation": "mask"},
        transform=_build_transform(image_size),
    )
    if subset_n is not None and subset_n < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:subset_n]
        ds = torch.utils.data.Subset(ds, idx.tolist())
    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )


def build_periodic_ade20k_seg_callback(
    grid_size,
    num_classes: int = ADE20K_NUM_CLASSES,
    feature_fn=None,
    support_images: int = 2048,
    eval_every_n_epochs: int = 25,
    warmup_epochs: int = 0,
    run_knn: bool = True,
    run_linear: bool = True,
    linear_every_n_epochs=None,
    linear_use_batchnorm: bool = True,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 8,
    data_cache=None,
    name: str = "ade20k_periodic_seg",
):
    """Inline ADE20k seg monitor for an SSL run (see ``PeriodicSegmentationEval``).

    Builds a subsampled ADE20k support loader (``support_images`` images) + the
    full val loader and returns the callback. Pass ``feature_fn(pl_module,
    images) -> (B, N, D)`` returning the backbone's patch tokens (CLS dropped);
    if ``None``, ``pl_module.backbone(images)`` is used directly.
    """
    train_loader = _ade20k_loader(
        "train", True, batch_size, num_workers, data_cache, image_size, subset_n=support_images
    )
    val_loader = _ade20k_loader(
        "validation", False, batch_size, num_workers, data_cache, image_size
    )
    return spt.PeriodicSegmentationEval(
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        grid_size=grid_size,
        num_classes=num_classes,
        feature_fn=feature_fn,
        mask_key="mask",
        ignore_index=ADE20K_IGNORE_INDEX,
        eval_every_n_epochs=eval_every_n_epochs,
        warmup_epochs=warmup_epochs,
        support_images=support_images,
        metrics=_ade20k_seg_metrics(),
        run_knn=run_knn,
        run_linear=run_linear,
        linear_every_n_epochs=linear_every_n_epochs,
        linear_use_batchnorm=linear_use_batchnorm,
    )


# ---------------------------------------------------------------------------
# Checkpoint loaders (shared across benchmarks)
# ---------------------------------------------------------------------------
# The model-generic loaders live in ``benchmarks/_backbones.py`` so the
# classification eval can reuse them too. Add the benchmarks root to sys.path
# (this module is imported with ``benchmarks/ade20k`` on the path).
sys.path.append(str(Path(__file__).resolve().parents[1]))
from _backbones import (  # noqa: E402
    _DropClsTokens,  # noqa: F401  (re-exported for backwards compat)
    _IJEPATokens,  # noqa: F401
    _grid_from,  # noqa: F401
    load_ijepa_vit_base,
    load_ijepa_vit_huge,
    load_mae_vit_base,
    load_mae_vit_large,
)

__all__ = [
    "load_mae_vit_large",
    "load_mae_vit_base",
    "load_ijepa_vit_huge",
    "load_ijepa_vit_base",
]
