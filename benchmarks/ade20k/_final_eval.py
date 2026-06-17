"""Paper-matching DINOv2/v3-style linear segmentation eval on ADE20k.

Unlike the coarse 224 / nearest-upsample probes, this:

* runs the backbone at higher resolution (``image_size``, default 512), via the
  positional-embedding interpolation enabled in the checkpoint loaders;
* trains a linear head with cross-entropy on **bilinearly-upsampled logits** vs
  the full ``(S, S)`` mask (not grid-downsampled labels);
* uses the **full** train/val splits with light joint image+mask augmentation;
* sweeps a small LR grid and reports the best val mIoU.

It is a standalone evaluation (run once on a checkpoint) — no Lightning trainer,
no training-loop coupling. See ``final_linear_eval.py`` for the runner.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torchvision.transforms import RandomResizedCrop as _RRC
from torchvision.transforms.functional import InterpolationMode

import stable_pretraining as spt
from stable_pretraining.callbacks.knn_segmentation import upsample_logits_to

from _common import (  # noqa: E402
    ADE20K_IGNORE_INDEX,
    ADE20K_NUM_CLASSES,
    _ade20k_seg_metrics,
    _build_transform,
)


class FinalTrainTransform:
    """Joint image+mask augmentation: random-resized-crop + h-flip + normalize.

    Self-contained (does not rely on the repo's per-key transforms) so the crop
    is applied with the *same* params to both image (bilinear) and mask
    (nearest). Produces a normalized image tensor ``(3, S, S)`` and a long mask
    ``(S, S)``.
    """

    def __init__(self, image_size, mean, std, scale=(0.5, 1.0), ratio=(0.75, 1.333), hflip=0.5):
        self.size = (image_size, image_size)
        self.scale = scale
        self.ratio = ratio
        self.hflip = hflip
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, x):
        img = x["image"]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        img_t = TF.to_image(img)  # uint8 (3, H, W)

        mask = x["mask"]
        m = torch.as_tensor(np.array(mask)) if not torch.is_tensor(mask) else mask
        if m.dim() == 3:
            m = m[..., 0] if m.shape[-1] <= 4 else m.squeeze(0)
        mask_t = m.long().unsqueeze(0)  # (1, H, W)

        i, j, h, w = _RRC.get_params(img_t, self.scale, self.ratio)
        img_c = TF.resized_crop(
            img_t, i, j, h, w, self.size, InterpolationMode.BILINEAR, antialias=True
        )
        mask_c = TF.resized_crop(
            mask_t, i, j, h, w, self.size, InterpolationMode.NEAREST
        )
        if random.random() < self.hflip:
            img_c = TF.horizontal_flip(img_c)
            mask_c = TF.horizontal_flip(mask_c)

        img_c = img_c.float().div_(255.0).sub_(self.mean).div_(self.std)
        x["image"] = img_c
        x["mask"] = mask_c.squeeze(0).long()
        return x


def build_final_ade20k_loaders(
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 8,
    data_cache=None,
    augment: bool = True,
):
    """Full ADE20k train/val loaders at ``image_size`` (no subsampling)."""
    norm = spt.data.static.ADE20K
    train_tf = (
        FinalTrainTransform(image_size, norm["mean"], norm["std"])
        if augment
        else _build_transform(image_size)
    )

    def _loader(split, transform, shuffle):
        ds = spt.data.HFDataset(
            "scene_parse_150",
            split=split,
            cache_dir=data_cache,
            trust_remote_code=True,
            rename_columns={"annotation": "mask"},
            transform=transform,
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

    return (
        _loader("train", train_tf, True),
        _loader("validation", _build_transform(image_size), False),
    )


def build_linear_seg_head(embed_dim, num_classes, use_batchnorm=True):
    if use_batchnorm:
        return nn.Sequential(nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, num_classes))
    return nn.Linear(embed_dim, num_classes)


def _grid_logits(head, backbone, images, grid_size, num_classes):
    """``images`` -> per-patch logits grid ``(B, C, H_g, W_g)``."""
    with torch.no_grad():
        tokens = backbone(images)  # (B, N, D)
    feats = tokens.reshape(-1, tokens.size(-1))
    logits = head(feats)  # (B*N, C)
    h_g, w_g = grid_size
    return logits.view(images.size(0), h_g, w_g, num_classes).permute(0, 3, 1, 2)


def train_and_eval_linear_seg(
    backbone,
    grid_size,
    embed_dim,
    train_loader,
    val_loader,
    device,
    num_classes: int = ADE20K_NUM_CLASSES,
    ignore_index: int = ADE20K_IGNORE_INDEX,
    lrs=(1e-3, 1e-2, 1e-1),
    epochs: int = 20,
    use_batchnorm: bool = True,
):
    """Train a linear head (LR sweep) and return the best val mIoU result.

    Loss and scoring both use full-resolution bilinear logit upsampling. Returns
    a dict ``{"miou", "pixel_acc", "lr"}`` for the best LR (by val mIoU).
    """
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    best = None
    for lr in lrs:
        head = build_linear_seg_head(embed_dim, num_classes, use_batchnorm).to(device)
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        for _ in range(epochs):
            head.train()
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True).long()
                grid = _grid_logits(head, backbone, images, grid_size, num_classes)
                up = upsample_logits_to(grid, mask.shape[-2:])  # (B, C, S, S)
                loss = loss_fn(up, mask)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        metrics = {k: v.to(device) for k, v in _ade20k_seg_metrics().items()}
        head.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True).long()
                grid = _grid_logits(head, backbone, images, grid_size, num_classes)
                pix = upsample_logits_to(grid, mask.shape[-2:]).argmax(dim=1).long()
                for m in metrics.values():
                    m.update(pix, mask)
        res = {k: m.compute().item() for k, m in metrics.items()}
        res["lr"] = lr
        print(f"[final-linear] lr={lr:g}  miou={res['miou']:.4f}  pixel_acc={res['pixel_acc']:.4f}")
        if best is None or res["miou"] > best["miou"]:
            best = res
    return best
