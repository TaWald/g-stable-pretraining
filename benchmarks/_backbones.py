"""Shared frozen-backbone checkpoint loaders for the eval benchmarks.

These return a backbone whose ``forward(x)`` yields patch tokens ``(B, N, D)``
(CLS dropped) plus the ``(H_g, W_g)`` grid and ``embed_dim``. The grid is used by
the dense (segmentation) evals and ignored by the image-level (classification)
evals. ``dynamic_img_size`` / positional-embedding interpolation lets the same
encoder run at higher resolution (e.g. 512) for the paper-matching seg eval.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


class _DropClsTokens(nn.Module):
    """Wrap a timm ViT so ``forward`` returns patch tokens (CLS dropped)."""

    def __init__(self, vit, num_prefix_tokens: int = 1):
        super().__init__()
        self.vit = vit
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # (B, 1 + N, D), final-norm applied
        return tokens[:, self.num_prefix_tokens :]


class _IJEPATokens(nn.Module):
    """Wrap an I-JEPA VisionTransformer (no CLS token) to return patch tokens."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x)  # (B, N, D), final-norm applied, no CLS


def _grid_from(image_size: int, patch: int):
    if image_size % patch != 0:
        raise ValueError(
            f"image_size={image_size} not divisible by patch_size={patch}; "
            f"pick a multiple (e.g. {patch * round(image_size / patch)})"
        )
    g = image_size // patch
    return (g, g)


def load_mae_vit_large(ckpt_path, image_size: int = 224):
    """Load a FB-MAE ViT-L/16 pretrain checkpoint into a timm encoder.

    Returns ``(backbone, grid_size, embed_dim)`` where ``backbone(x)`` yields
    patch tokens ``(B, (image_size//16)**2, 1024)``. ``dynamic_img_size=True``
    interpolates the positional embedding, so any ``image_size`` that is a
    multiple of 16 works (e.g. 512 -> 32x32 grid).
    """
    import timm

    vit = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
        dynamic_img_size=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    # Keep only encoder weights: drop decoder, mask token, and any classifier head.
    sd = {
        k: v
        for k, v in sd.items()
        if not k.startswith(("decoder", "mask_token"))
        and k not in ("head.weight", "head.bias")
    }
    msg = vit.load_state_dict(sd, strict=False)
    _report_load("MAE ViT-L", msg)
    return _DropClsTokens(vit, num_prefix_tokens=1), _grid_from(image_size, 16), 1024


def load_mae_vit_base(ckpt_path, image_size: int = 224):
    """Load a FB-MAE ViT-B/16 pretrain checkpoint into a timm encoder.

    Returns ``(backbone, grid_size, embed_dim)`` where ``backbone(x)`` yields
    patch tokens ``(B, (image_size//16)**2, 768)``. ``dynamic_img_size=True``
    interpolates the positional embedding, so any ``image_size`` that is a
    multiple of 16 works (e.g. 512 -> 32x32 grid).
    """
    import timm

    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=image_size,
        dynamic_img_size=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    # Keep only encoder weights: drop decoder, mask token, and any classifier head.
    sd = {
        k: v
        for k, v in sd.items()
        if not k.startswith(("decoder", "mask_token"))
        and k not in ("head.weight", "head.bias")
    }
    msg = vit.load_state_dict(sd, strict=False)
    _report_load("MAE ViT-B", msg)
    return _DropClsTokens(vit, num_prefix_tokens=1), _grid_from(image_size, 16), 768


def load_ijepa_vit_base(ckpt_path, gijepa_root, image_size: int = 224):
    """Load an I-JEPA ViT-B/16 ``target_encoder`` checkpoint.

    Returns ``(backbone, grid_size, embed_dim)`` where ``backbone(x)`` yields
    patch tokens ``(B, (image_size//16)**2, 768)``. ``image_size`` must be a
    multiple of 16 (e.g. 512 -> 32x32). The gijepa VisionTransformer
    interpolates its positional embedding for non-224 inputs in ``forward``.

    ``gijepa_root`` must point at the I-JEPA repo (the ``gijepa/`` dir) so its
    ``src.models.vision_transformer`` is importable.
    """
    gijepa_root = str(Path(gijepa_root).resolve())
    if gijepa_root not in sys.path:
        sys.path.insert(0, gijepa_root)
    from src.models.vision_transformer import vit_base

    vit = vit_base(patch_size=16, img_size=[image_size])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("target_encoder", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = vit.load_state_dict(sd, strict=False)
    _report_load("I-JEPA ViT-B", msg)
    return _IJEPATokens(vit), _grid_from(image_size, 16), 768


def load_ijepa_vit_huge(ckpt_path, gijepa_root, image_size: int = 224):
    """Load an I-JEPA ViT-H/14 ``target_encoder`` checkpoint.

    Returns ``(backbone, grid_size, embed_dim)`` where ``backbone(x)`` yields
    patch tokens ``(B, (image_size//14)**2, 1280)``. ``image_size`` must be a
    multiple of 14 (e.g. 518 -> 37x37). The gijepa VisionTransformer
    interpolates its positional embedding for non-224 inputs in ``forward``.

    ``gijepa_root`` must point at the I-JEPA repo (the ``gijepa/`` dir) so its
    ``src.models.vision_transformer`` is importable.
    """
    gijepa_root = str(Path(gijepa_root).resolve())
    if gijepa_root not in sys.path:
        sys.path.insert(0, gijepa_root)
    from src.models.vision_transformer import vit_huge

    vit = vit_huge(patch_size=14, img_size=[image_size])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("target_encoder", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = vit.load_state_dict(sd, strict=False)
    _report_load("I-JEPA ViT-H", msg)
    return _IJEPATokens(vit), _grid_from(image_size, 14), 1280


def _report_load(tag, msg):
    missing = [k for k in getattr(msg, "missing_keys", [])]
    unexpected = [k for k in getattr(msg, "unexpected_keys", [])]
    print(f"[{tag}] loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"[{tag}] missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"[{tag}] unexpected (first 10): {unexpected[:10]}")
