"""Utility functions for benchmarks."""

import math
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch


def ijepa_teacher_features(pl_module, images):
    """Full-grid patch tokens from the I-JEPA *target* (teacher/EMA) encoder.

    Mirrors the ``teacher_full`` path inside ``IJEPA.forward`` — all patches, no
    masking, no CLS — i.e. the representation actually distilled during training
    and the one the post-hoc eval restores from the ``target_encoder`` checkpoint
    (see ``benchmarks/_backbones.py``).

    We deliberately do *not* route through ``IJEPA.forward(..., embedding_source=
    "teacher")``: in eval mode that method ignores ``embedding_source`` and always
    encodes through the student (context) encoder, so it can never return teacher
    features. Returns ``(B, N, D)`` with the teacher's final norm applied.
    """
    teacher = pl_module.encoder.teacher
    b = images.shape[0]
    grid_h, grid_w = teacher._get_grid_size(images)
    patches = teacher.patch_embed(images)
    all_idx = (
        torch.arange(grid_h * grid_w, device=images.device)
        .unsqueeze(0)
        .expand(b, -1)
    )
    return pl_module._encode(patches, all_idx, grid_h, grid_w, teacher)


def maybe_build_seg_eval(grid_size, feature_fn, *, name, data_cache=None):
    """Build an inline ADE20k kNN-segmentation monitor for an SSL pretraining run.

    Wraps :class:`stable_pretraining.PeriodicSegmentationEval` via the ADE20k
    ``build_periodic_ade20k_seg_callback`` helper: every ``SEG_EVAL_EVERY``
    epochs it snapshots the frozen backbone, extracts patch features over a
    subsampled ADE20k support set, and scores a weighted kNN segmentation probe
    (per-pixel mIoU / pixel-acc) on the val split. It owns its own dataloaders
    and never touches the SSL train/val loop.

    Gated by env so it is opt-out without code edits (and so the ADE20k dataset
    is only loaded when wanted):

    * ``SEG_EVAL`` (default ``"1"``): set ``"0"`` to disable entirely (returns None).
    * ``SEG_EVAL_EVERY`` (default 25): epoch cadence.
    * ``SEG_EVAL_SUPPORT`` (default 2048): support-image budget per trigger.
    * ``SEG_EVAL_LINEAR`` (default ``"0"``): also train a linear seg head (costlier).

    Args:
        grid_size: ``(H_g, W_g)`` patch grid the backbone produces at 224 (e.g.
            ``(14, 14)`` for /16, ``(16, 16)`` for ViT-H/14).
        feature_fn: ``(pl_module, images) -> (B, N, D)`` patch tokens (CLS/prefix
            dropped) for *this* backbone.
        name: Log-key prefix (``eval/{name}_knn_miou`` etc.).
        data_cache: Optional HF datasets ``cache_dir`` for ADE20k.

    Returns:
        The callback, or ``None`` when ``SEG_EVAL=0``.
    """
    if os.environ.get("SEG_EVAL", "1") == "0":
        return None
    # ``build_periodic_ade20k_seg_callback`` lives in benchmarks/ade20k/_common.py.
    sys.path.append(str(Path(__file__).parent / "ade20k"))
    from _common import build_periodic_ade20k_seg_callback

    return build_periodic_ade20k_seg_callback(
        grid_size=grid_size,
        feature_fn=feature_fn,
        eval_every_n_epochs=int(os.environ.get("SEG_EVAL_EVERY", 25)),
        support_images=int(os.environ.get("SEG_EVAL_SUPPORT", 2048)),
        run_knn=True,
        run_linear=os.environ.get("SEG_EVAL_LINEAR", "0") == "1",
        data_cache=data_cache,
        name=name,
    )


def get_data_dir(dataset_name: str = None) -> Path:
    """Get the data directory for storing datasets.

    The directory is determined in the following order:
    1. Environment variable STABLE_PRETRAINING_DATA_DIR if set
    2. Default to ~/.cache/stable-pretraining/data

    Args:
        dataset_name: Optional name of the dataset to create a subdirectory

    Returns:
        Path object pointing to the data directory

    Examples:
        >>> # Get general data directory
        >>> data_dir = get_data_dir()

        >>> # Get CIFAR-10 specific directory
        >>> cifar10_dir = get_data_dir("cifar10")

        >>> # Set custom directory via environment variable
        >>> # export STABLE_PRETRAINING_DATA_DIR=/path/to/my/data
    """
    # Check for environment variable
    if "STABLE_PRETRAINING_DATA_DIR" in os.environ:
        base_dir = Path(os.environ["STABLE_PRETRAINING_DATA_DIR"])
    else:
        # Use default location in user's cache directory
        base_dir = Path.home() / ".cache" / "stable-pretraining" / "data"

    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    # Add dataset subdirectory if specified
    if dataset_name:
        data_dir = base_dir / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = base_dir

    return data_dir


class CosineWDSchedule(pl.Callback):
    """Cosine-ramp weight decay from a start value to a final value over training.

    Replicates the I-JEPA reference (gijepa) weight-decay schedule, which ramps
    weight decay from ``start_weight_decay`` up to ``final_weight_decay`` along a
    cosine curve over the full run::

        wd(t) = final + (start - final) * 0.5 * (1 + cos(pi * t / total_steps))

    (At ``t=0`` this is ``start``; at ``t=total_steps`` it is ``final``.)

    Only the optimizer param groups whose weight decay *starts* at
    ``start_weight_decay`` are updated. This deliberately excludes both the
    no-decay groups (bias/norm, wd=0) and any online-probe optimizer (which uses
    a different wd), so the ramp applies to the encoder/predictor groups only —
    matching gijepa, which schedules wd for the regularized group alone.

    Args:
        start_weight_decay: Initial wd (must match the value set on the main
            optimizer, e.g. 0.04). Used both as the ramp start and to identify
            which param groups to schedule.
        final_weight_decay: Target wd at the end of training (e.g. 0.4).
        total_steps: Total optimizer steps over the run — use the same value
            passed to the LR scheduler's ``total_steps``.

    Note:
        Steps on ``on_train_batch_start`` using ``trainer.global_step``, so the
        cadence matches a per-step LR scheduler.
    """

    def __init__(
        self,
        start_weight_decay: float,
        final_weight_decay: float,
        total_steps: int,
    ):
        super().__init__()
        self.start_wd = start_weight_decay
        self.final_wd = final_weight_decay
        self.total_steps = max(1, int(total_steps))
        self._target_group_ids = None  # resolved on first batch

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._target_group_ids is None:
            self._target_group_ids = set()
            for opt in trainer.optimizers:
                for group in opt.param_groups:
                    if abs(group.get("weight_decay", 0.0) - self.start_wd) < 1e-12:
                        self._target_group_ids.add(id(group))

        progress = min(1.0, trainer.global_step / self.total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        new_wd = self.final_wd + (self.start_wd - self.final_wd) * cosine

        for opt in trainer.optimizers:
            for group in opt.param_groups:
                if id(group) in self._target_group_ids:
                    group["weight_decay"] = new_wd
