"""Dense (patch-level) linear-probe evaluator for semantic segmentation.

This is the dense analogue of :class:`OnlineProbe`, mirroring how
:class:`OnlineKNNSegmentation` is the dense analogue of :class:`OnlineKNN`. It
implements the DINOv3 linear-segmentation protocol: a lightweight classifier
(typically ``BatchNorm1d`` + ``Linear``, or a plain ``Linear``) is trained with
gradient descent on top of **frozen patch tokens**, and at validation time the
per-patch logits are folded back to the ViT patch grid, upsampled to the
ground-truth mask resolution, and scored with a per-pixel segmentation metric
(typically mIoU).

The training half is inherited verbatim from :class:`OnlineProbe`: with
``input`` pointing at flattened patch features ``(B * H_g * W_g, D)`` and
``target`` at flattened patch labels ``(B * H_g * W_g,)``, the parent's wrapped
forward already trains the head with cross-entropy using a callback-owned
optimizer/scheduler. This subclass only adds the dense, full-resolution
validation scoring, reusing the reshape/upsample helpers shared with
:class:`OnlineKNNSegmentation`.

Note:
    This is a *frozen-backbone* protocol. Run it by ``fit``-ing a module whose
    backbone has ``requires_grad=False`` (the only trainable parameters are the
    probe head's). Train for several epochs so the head converges. See
    ``benchmarks/ade20k/`` for ready-to-run MAE and I-JEPA examples.
"""

from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging

from .knn_segmentation import grid_labels_to_pixels, squeeze_mask
from .probe import OnlineProbe
from .utils import (
    format_metrics_as_dict,
    get_data_from_batch_or_outputs,
    log_header,
)


class OnlineProbeSegmentation(OnlineProbe):
    """Patch-level linear-probe evaluator producing dense segmentation metrics.

    Args:
        module: The ``spt.Module`` to probe.
        name: Unique identifier for this callback instance (used for logging and
            metric storage).
        input: Batch/outputs key holding the **flattened** patch features of
            shape ``(B * H_g * W_g, D)`` where ``(H_g, W_g) == grid_size``. The
            head is trained on these features and they are also scored at
            validation.
        target: Batch/outputs key holding the **flattened** patch labels of
            shape ``(B * H_g * W_g,)`` used as the cross-entropy targets when
            training the head.
        grid_size: ``(H_g, W_g)`` patch grid of the backbone for the eval
            resolution (e.g. ``(14, 14)`` for ViT-L/16 @224, ``(16, 16)`` for
            ViT-H/14 @224). Used to fold the flattened predictions back into a
            spatial map.
        mask_key: Batch key holding the full-resolution ground-truth mask of
            shape ``(B, H, W)`` (or ``(B, 1, H, W)``) used to score mIoU.
        num_classes: Number of classes including any ignore class. For ADE20k
            this is 151 (0 = ignore/unlabeled, 1..150 = classes).
        probe: The probe module to train (an ``nn.Module`` instance or a callable
            returning one). Typically ``nn.Sequential(nn.BatchNorm1d(D),
            nn.Linear(D, num_classes))`` or a plain ``nn.Linear(D, num_classes)``.
        metrics: Dict of torchmetrics keyed by name, computed on the dense
            **full-resolution** predictions ``(B, H, W)`` vs the GT mask — e.g.
            ``{"miou": MulticlassJaccardIndex(151, ignore_index=0)}``. These are
            stored separately from the parent probe metrics (under
            ``callbacks_metrics[f"{name}_dense"]``).
        loss: Cross-entropy-style loss for training the head. ``None`` defaults
            to ``nn.CrossEntropyLoss()``; pass one with ``ignore_index`` set to
            skip the unlabeled class (e.g. ``ignore_index=0`` for ADE20k).
        train_metrics: Optional list/tuple of torchmetrics evaluated at the
            **flat patch level** during training, for cheap convergence
            monitoring (logged as ``train/{name}_*``). Defaults to none.
        optimizer, scheduler, accumulate_grad_batches, gradient_clip_val,
        gradient_clip_algorithm, verbose, log_on_step: Forwarded to
            :class:`OnlineProbe`.

    Note:
        * The probe head is stored in ``pl_module.callbacks_modules[name]``.
        * Dense val metrics are stored in
          ``pl_module.callbacks_metrics[f"{name}_dense"]`` and logged as
          ``eval/{name}_{metric_name}``.
        * The parent's flat patch-level validation metric is disabled (empty
          ``_val``) so only the full-resolution score is reported.
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        input: str,
        target: str,
        grid_size: Tuple[int, int],
        mask_key: str,
        num_classes: int,
        probe: torch.nn.Module,
        metrics: Optional[Dict] = None,
        loss: callable = None,
        train_metrics: Optional[Union[list, tuple]] = None,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = "norm",
        verbose: bool = None,
        log_on_step: bool = True,
    ) -> None:
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be (H_g, W_g), got {grid_size}")
        self.grid_size = tuple(int(s) for s in grid_size)
        self.mask_key = mask_key
        self.num_classes = num_classes
        # Dense, full-resolution segmentation metrics, kept separate from the
        # parent probe's (flat patch-level) metrics.
        self._seg_metrics = metrics

        if loss is None:
            loss = nn.CrossEntropyLoss()

        # The parent computes a flat patch-level metric in its validate branch;
        # we want the full-resolution score instead, so hand it an empty ``val``
        # set (its fit branch still needs a ``_train`` entry to exist, hence the
        # train/val dict form). ``train_metrics`` enables optional flat
        # patch-level monitoring during training.
        super().__init__(
            module=module,
            name=name,
            input=input,
            target=target,
            probe=probe,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            metrics={"train": list(train_metrics) if train_metrics else [], "val": []},
            verbose=verbose,
            log_on_step=log_on_step,
        )

    @property
    def state_key(self) -> str:
        return f"OnlineProbeSegmentation[name={self.name}]"

    @property
    def _dense_metrics_key(self) -> str:
        return f"{self.name}_dense"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        key = self._dense_metrics_key
        if key not in pl_module.callbacks_metrics:
            pl_module.callbacks_metrics[key] = format_metrics_as_dict(self._seg_metrics)
        log_header("OnlineProbeSegmentation")
        logging.info(f"  name: {self.name}")
        logging.info(f"  grid_size: {self.grid_size}")
        logging.info(f"  mask_key: {self.mask_key}")
        logging.info(f"  num_classes: {self.num_classes}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Dense linear probe over patch tokens, scored at the mask resolution."""
        features = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        if features is None:
            return

        mask = get_data_from_batch_or_outputs(
            self.mask_key, batch, outputs, caller_name=self.name
        )
        if mask is None:
            logging.warning(f"! {self.name}: mask key '{self.mask_key}' not found")
            return

        pixel_preds = self._compute_segmentation(pl_module, features, mask)
        if pixel_preds is None:
            return

        self._log_metrics(pl_module, pixel_preds, squeeze_mask(mask).long())

    @torch.no_grad()
    def _compute_segmentation(
        self,
        pl_module: LightningModule,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        h_g, w_g = self.grid_size
        n_patches = h_g * w_g
        if features.dim() == 3:  # (B, N, D) -> flatten to (B*N, D)
            features = features.reshape(-1, features.size(-1))
        if features.size(0) % n_patches != 0:
            logging.warning(
                f"! {self.name}: {features.size(0)} patch rows not divisible by "
                f"grid {self.grid_size} ({n_patches}); skipping"
            )
            return None

        # The trained head (in eval mode with the rest of the module) maps each
        # patch to class logits (B*N, num_classes).
        logits = self.module(features)
        # (B*N,) -> (B, H_g, W_g) -> upsample (nearest) to mask resolution.
        return grid_labels_to_pixels(logits.argmax(dim=1), self.grid_size, mask)

    def _log_metrics(
        self, pl_module: LightningModule, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        logs = {}
        for metric_name, metric in pl_module.callbacks_metrics[self._dense_metrics_key][
            "_val"
        ].items():
            metric(preds, targets)
            logs[f"eval/{self.name}_{metric_name}"] = metric
        pl_module.log_dict(logs, on_step=False, on_epoch=True)
