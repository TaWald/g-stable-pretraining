"""Shared pieces for the standalone ImageNet classification eval.

Frozen-backbone, paper-protocol image-level classification: linear probe,
attentive-pooling probe, and weighted kNN. Datasets are loaded from the HF hub
(``ILSVRC/imagenet-1k`` is gated — needs ``huggingface-cli login``;
``clane9/imagenet-100`` is open).
"""

from torchvision.transforms.functional import InterpolationMode

import torch
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms

# HF dataset id -> (split names, num_classes, label column).
_DATASETS = {
    "imagenet-1k": ("ILSVRC/imagenet-1k", 1000),
    "imagenet-100": ("clane9/imagenet-100", 100),
}


def _train_transform(image_size):
    norm = spt.data.static.ImageNet
    return transforms.Compose(
        transforms.RGB(source="image", target="image"),
        transforms.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
            source="image",
            target="image",
        ),
        transforms.RandomHorizontalFlip(p=0.5, source="image", target="image"),
        transforms.ToImage(
            mean=norm["mean"], std=norm["std"], source="image", target="image"
        ),
    )


def _eval_transform(image_size):
    norm = spt.data.static.ImageNet
    resize = int(round(image_size * 256 / 224))  # 224 -> 256 short-side then crop
    return transforms.Compose(
        transforms.RGB(source="image", target="image"),
        transforms.Resize(
            (resize, resize),
            interpolation=InterpolationMode.BICUBIC,
            source="image",
            target="image",
        ),
        transforms.CenterCrop((image_size, image_size), source="image", target="image"),
        transforms.ToImage(
            mean=norm["mean"], std=norm["std"], source="image", target="image"
        ),
    )


def build_imagenet_loaders(
    dataset: str = "imagenet-1k",
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    data_cache=None,
):
    """Build ``(train_aug, train_noaug, val, num_classes)`` loaders.

    ``train_aug`` is for the trained probes (RandomResizedCrop + flip);
    ``train_noaug`` uses the eval transform and feeds the kNN feature bank.
    Both train loaders are the same split; the val loader is the validation
    split (no augmentation).
    """
    if dataset not in _DATASETS:
        raise ValueError(f"dataset must be one of {list(_DATASETS)}, got {dataset}")
    hf_id, num_classes = _DATASETS[dataset]

    def _loader(split, transform, shuffle, drop_last):
        ds = spt.data.HFDataset(
            hf_id,
            split=split,
            cache_dir=data_cache,
            transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    train_aug = _loader("train", _train_transform(image_size), True, True)
    train_noaug = _loader("train", _eval_transform(image_size), False, False)
    val = _loader("validation", _eval_transform(image_size), False, False)
    return train_aug, train_noaug, val, num_classes


def cls_metrics(num_classes: int):
    """Top-1 + top-5 accuracy."""
    return {
        "top1": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro"
        ),
        "top5": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=5, average="micro"
        ),
    }
