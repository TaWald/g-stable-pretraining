"""Dense (patch-level) weighted-kNN evaluator for semantic segmentation.

This is the dense analogue of :class:`OnlineKNN`. Where ``OnlineKNN`` treats
one pooled embedding per image as a sample and predicts an image-level class,
``OnlineKNNSegmentation`` treats **every patch token as a sample**: the support
queue is filled with patch-level features + patch-level labels coming from the
training split, and at validation time each query patch is classified by
weighted kNN against that queue. The per-patch predictions are reshaped back to
the ViT patch grid, upsampled to the ground-truth mask resolution, and scored
with the segmentation metric(s) passed in (typically a per-pixel mIoU).

Because each patch is an independent row, the whole machinery of
:class:`OnlineKNN` (queue discovery, chunked distance computation, distance
weighting, num-class resolution) is reused verbatim — this subclass only adds
the spatial reshape/upsample/score step on top.

Note:
    This is a *frozen-backbone* evaluation protocol. Run it by ``fit``-ing a
    module whose backbone has ``requires_grad=False``: the train loader only
    serves to populate the support queue (no representation learning happens),
    and the kNN segmentation metric is computed on the val loader. See
    ``benchmarks/ade20k/`` for ready-to-run MAE and I-JEPA examples.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging

from .knn import OnlineKNN
from .utils import get_data_from_batch_or_outputs, log_header


def squeeze_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize a GT mask to ``(B, H, W)``, accepting ``(B, 1, H, W)`` too."""
    if mask.dim() == 4 and mask.size(1) == 1:
        return mask.squeeze(1)
    return mask


def grid_labels_to_pixels(
    grid_labels: torch.Tensor,
    grid_size: Tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fold flat per-patch labels onto the ViT grid and upsample to mask res.

    Args:
        grid_labels: Flat per-patch integer labels of shape ``(B * H_g * W_g,)``.
        grid_size: ``(H_g, W_g)`` patch grid the labels were produced on.
        mask: Ground-truth mask used only to read the target ``(H, W)``; may be
            ``(B, H, W)`` or ``(B, 1, H, W)``.

    Returns:
        Integer pixel predictions of shape ``(B, H, W)`` (nearest upsampled).
    """
    h_g, w_g = grid_size
    batch_size = grid_labels.numel() // (h_g * w_g)
    target_hw = squeeze_mask(mask).shape[-2:]
    grid = grid_labels.view(batch_size, h_g, w_g).float()
    pixel_preds = F.interpolate(
        grid.unsqueeze(1), size=target_hw, mode="nearest"
    ).squeeze(1)
    return pixel_preds.long()


def logits_grid_to_pixels(
    logits_flat: torch.Tensor,
    grid_size: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Bilinearly upsample per-patch class logits, then argmax to pixel labels.

    This is the paper-correct dense decoding (DINOv2/v3, Segmenter): the
    per-class scores are interpolated to the target resolution *before* the
    argmax, unlike :func:`grid_labels_to_pixels` which nearest-upsamples already
    argmaxed grid labels. Use it for both the training loss (keep the upsampled
    logits, before argmax) and validation scoring.

    Args:
        logits_flat: Flat per-patch logits of shape ``(B * H_g * W_g, C)``.
        grid_size: ``(H_g, W_g)`` patch grid the logits were produced on.
        target_hw: ``(H, W)`` resolution to upsample to (the scoring resolution).

    Returns:
        Integer pixel predictions of shape ``(B, H, W)``.
    """
    h_g, w_g = grid_size
    c = logits_flat.size(-1)
    batch_size = logits_flat.numel() // (h_g * w_g * c)
    # (B*N, C) -> (B, H_g, W_g, C) -> (B, C, H_g, W_g)
    grid = logits_flat.view(batch_size, h_g, w_g, c).permute(0, 3, 1, 2)
    up = upsample_logits_to(grid, target_hw)
    return up.argmax(dim=1).long()


def upsample_logits_to(
    logits_grid: torch.Tensor, target_hw: Tuple[int, int]
) -> torch.Tensor:
    """Bilinearly upsample a ``(B, C, H_g, W_g)`` logit map to ``target_hw``.

    Shared by the training loss (cross-entropy on the upsampled logits vs the
    full-resolution mask) and :func:`logits_grid_to_pixels` (which argmaxes the
    result for scoring).
    """
    return F.interpolate(
        logits_grid.float(), size=tuple(target_hw), mode="bilinear", align_corners=False
    )


class OnlineKNNSegmentation(OnlineKNN):
    """Patch-level weighted-kNN evaluator producing dense segmentation metrics.

    Args:
        name: Unique identifier for this callback instance (used for logging and
            metric storage).
        input: Batch/outputs key holding the **flattened** patch features of
            shape ``(B * H_g * W_g, D)`` where ``(H_g, W_g) == grid_size``. The
            same key feeds both the support queue (during training) and the
            queries (during validation).
        target: Batch/outputs key holding the **flattened** patch labels of
            shape ``(B * H_g * W_g,)`` used to fill the support label queue.
        grid_size: ``(H_g, W_g)`` patch grid of the backbone for the eval
            resolution (e.g. ``(14, 14)`` for ViT-L/16 @224, ``(16, 16)`` for
            ViT-H/14 @224). Used to fold the flattened predictions back into a
            spatial map.
        mask_key: Batch key holding the full-resolution ground-truth mask of
            shape ``(B, H, W)`` (or ``(B, 1, H, W)``) used to score mIoU.
        num_classes: Number of classes including any ignore class. For ADE20k
            this is 151 (0 = ignore/unlabeled, 1..150 = classes).
        metrics: Dict of torchmetrics keyed by name. Each receives integer
            pixel predictions ``(B, H, W)`` and the integer GT mask ``(B, H, W)``
            — e.g. ``{"miou": MulticlassJaccardIndex(151, ignore_index=0)}``.
        queue_length: Number of *patch* rows to keep in the support queue.
            Memory ≈ ``queue_length * D * 4`` bytes for the features.
        input_dim: Patch feature dimensionality ``D`` (pre-allocates the queue).
        k, temperature, chunk_size, distance_metric, verbose: Forwarded to
            :class:`OnlineKNN`.

    Note:
        ``chunk_size`` matters here: a single image contributes ``H_g * W_g``
        query rows, so the query batch passed to the distance computation is
        ``B * H_g * W_g`` wide. Keep ``chunk_size`` modest (e.g. 4096) to bound
        the ``(queue_length, chunk_size)`` distance matrix.
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        grid_size: Tuple[int, int],
        mask_key: str,
        num_classes: int,
        metrics: Dict,
        queue_length: int,
        input_dim: Optional[Union[Tuple[int, ...], List[int], int]] = None,
        k: int = 20,
        temperature: float = 0.07,
        chunk_size: int = 4096,
        distance_metric: str = "cosine",
        verbose: bool = None,
    ) -> None:
        super().__init__(
            name=name,
            input=input,
            target=target,
            queue_length=queue_length,
            metrics=metrics,
            input_dim=input_dim,
            target_dim=None,
            num_classes=num_classes,
            k=k,
            temperature=temperature,
            chunk_size=chunk_size,
            distance_metric=distance_metric,
            verbose=verbose,
        )
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be (H_g, W_g), got {grid_size}")
        self.grid_size = tuple(int(s) for s in grid_size)
        self.mask_key = mask_key

    @property
    def state_key(self) -> str:
        return f"OnlineKNNSegmentation[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        log_header("OnlineKNNSegmentation")
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
        """Dense kNN over patch tokens, scored at the mask resolution."""
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

        cached_features = self._input_queue.data
        cached_labels = self._target_queue.data
        if cached_features is None or cached_labels is None:
            logging.warning(f"! {self.name}: queue data not available")
            return
        if cached_features.numel() == 0 or cached_labels.numel() == 0:
            logging.warning(f"! {self.name}: queue empty, skipping")
            return

        pixel_preds = self._compute_segmentation(features, cached_features, cached_labels, mask)
        if pixel_preds is None:
            return

        self._log_metrics(pl_module, pixel_preds, squeeze_mask(mask).long())

    @torch.no_grad()
    def _compute_segmentation(
        self,
        features: torch.Tensor,
        cached_features: torch.Tensor,
        cached_labels: torch.Tensor,
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

        # Reuse OnlineKNN's weighted-vote machinery: soft preds (B*N, num_classes).
        soft = self._compute_knn_predictions(features, cached_features, cached_labels)
        if soft is None:
            return None

        # (B*N,) -> (B, H_g, W_g) -> upsample (nearest) to mask resolution.
        return grid_labels_to_pixels(soft.argmax(dim=1), self.grid_size, mask)

    def _log_metrics(
        self, pl_module: LightningModule, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        logs = {}
        for metric_name, metric in pl_module.callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(preds, targets)
            logs[f"eval/{self.name}_{metric_name}"] = metric
        pl_module.log_dict(logs, on_step=False, on_epoch=True)
