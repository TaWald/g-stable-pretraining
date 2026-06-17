"""Periodic dense-segmentation evaluation of a frozen backbone during training.

The dense segmentation probes (:class:`OnlineKNNSegmentation`,
:class:`OnlineProbeSegmentation`) are *post-hoc* — they run on a checkpoint via
their own ``trainer.fit``. This callback instead lets you **monitor segmentation
quality inline during SSL pretraining**: every ``eval_every_n_epochs`` it
snapshots the (frozen-for-eval) backbone, extracts patch features over a
subsampled support set *once*, and from that single extraction runs a weighted
k-NN probe (CAPI) and/or a freshly-trained linear head (DINOv3), scoring
per-pixel mIoU on a validation split and logging it as a trend line.

It is self-contained — it owns its own dataloaders and never touches the SSL
train/val loop, the kNN support queue, or the module's optimizers (the linear
head is a throwaway). This is necessary because the module ``forward`` is not
given a ``dataloader_idx`` (so it can't tell an eval batch from an SSL batch)
and the shared kNN queue only fills from SSL training batches.

Note:
    SSL backbones often return a pooled embedding, not patch tokens. Pass a
    ``feature_fn(pl_module, images) -> (B, N, D)`` that returns the patch tokens
    (CLS dropped) for your backbone. The ``_DropClsTokens`` / ``_IJEPATokens``
    wrappers in ``benchmarks/ade20k/_common.py`` are worked examples.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .knn import weighted_knn_predict
from .knn_segmentation import grid_labels_to_pixels, squeeze_mask
from .utils import log_header


def _default_seg_metrics(num_classes: int, ignore_index: int) -> Dict[str, object]:
    return {
        "miou": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"
        ),
        "pixel_acc": torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average="micro",
            multidim_average="global",
        ),
    }


class PeriodicSegmentationEval(Callback):
    """Periodically evaluate a frozen backbone on dense segmentation.

    Every ``eval_every_n_epochs`` (and optionally at fit end), this runs a
    self-contained ADE20k-style eval on ``trainer.is_global_zero``: extract
    patch features over a subsampled support set, then score a weighted k-NN
    probe and/or a freshly trained linear head as per-pixel mIoU on the
    validation loader.

    Args:
        name: Identifier used in log keys (``eval/{name}_knn_miou`` etc.).
        train_loader: DataLoader over the support split, yielding dicts with
            ``image_key`` and ``mask_key``. Subsample it (or rely on
            ``support_images``) to bound cost.
        val_loader: DataLoader over the validation split (same dict schema).
        grid_size: ``(H_g, W_g)`` patch grid the backbone produces at the eval
            resolution.
        num_classes: Number of classes including the ignore class.
        feature_fn: ``(pl_module, images) -> (B, N, D)`` patch tokens. Defaults
            to ``pl_module.backbone(images)`` with a shape check against
            ``grid_size``.
        image_key, mask_key: Batch dict keys for the image tensor ``(B, 3, H, W)``
            and the integer mask ``(B, H, W)``.
        ignore_index: Label treated as unlabeled (excluded from mIoU and from the
            linear-probe cross-entropy). Default 0 (ADE20k).
        eval_every_n_epochs: Run cadence (in epochs). Default 25.
        warmup_epochs: Skip triggers before this epoch. Default 0.
        run_on_fit_end: Also run once when training finishes. Default True.
        support_images: Max number of support images to extract per trigger
            (bounds time/memory). Default 2048.
        metrics: Optional dict of torchmetrics (keyed by name) scoring dense
            ``(B, H, W)`` preds vs the GT mask; defaults to mIoU + pixel acc. A
            fresh clone is used per probe.
        run_knn: Whether to run the k-NN probe each trigger. Default True.
        knn_k, knn_temperature, knn_distance_metric, knn_chunk_size: k-NN knobs
            forwarded to :func:`weighted_knn_predict`.
        run_linear: Whether to run the linear probe. Default True.
        linear_every_n_epochs: Run the linear probe only every this many epochs
            (must be a multiple of ``eval_every_n_epochs``). ``None`` = same
            cadence as k-NN.
        linear_use_batchnorm: Use ``BatchNorm1d + Linear`` (default) vs plain
            ``Linear`` for the head.
        linear_lr, linear_epochs, linear_batch_size: Linear-head training knobs
            (SGD + cosine schedule, trained on the cached support features).
    """

    def __init__(
        self,
        name: str,
        train_loader,
        val_loader,
        grid_size: Tuple[int, int],
        num_classes: int,
        feature_fn: Optional[Callable[[LightningModule, torch.Tensor], torch.Tensor]] = None,
        image_key: str = "image",
        mask_key: str = "mask",
        ignore_index: int = 0,
        eval_every_n_epochs: int = 25,
        warmup_epochs: int = 0,
        run_on_fit_end: bool = True,
        support_images: int = 2048,
        metrics: Optional[Dict] = None,
        run_knn: bool = True,
        knn_k: int = 20,
        knn_temperature: float = 0.07,
        knn_distance_metric: str = "cosine",
        knn_chunk_size: int = 4096,
        run_linear: bool = True,
        linear_every_n_epochs: Optional[int] = None,
        linear_use_batchnorm: bool = True,
        linear_lr: float = 1e-2,
        linear_epochs: int = 20,
        linear_batch_size: int = 4096,
    ) -> None:
        super().__init__()
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be (H_g, W_g), got {grid_size}")
        if not (run_knn or run_linear):
            raise ValueError("at least one of run_knn / run_linear must be True")
        self.name = name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grid_size = tuple(int(s) for s in grid_size)
        self.num_classes = num_classes
        self.feature_fn = feature_fn or self._default_feature_fn
        self.image_key = image_key
        self.mask_key = mask_key
        self.ignore_index = ignore_index
        self.eval_every_n_epochs = eval_every_n_epochs
        self.warmup_epochs = warmup_epochs
        self.run_on_fit_end = run_on_fit_end
        self.support_images = support_images
        self._metrics_spec = metrics or _default_seg_metrics(num_classes, ignore_index)

        self.run_knn = run_knn
        self.knn_k = knn_k
        self.knn_temperature = knn_temperature
        self.knn_distance_metric = knn_distance_metric
        self.knn_chunk_size = knn_chunk_size

        self.run_linear = run_linear
        self.linear_every_n_epochs = linear_every_n_epochs or eval_every_n_epochs
        self.linear_use_batchnorm = linear_use_batchnorm
        self.linear_lr = linear_lr
        self.linear_epochs = linear_epochs
        self.linear_batch_size = linear_batch_size

        self._last_eval_epoch = -1

    @property
    def state_key(self) -> str:
        return f"PeriodicSegmentationEval[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        log_header("PeriodicSegmentationEval")
        logging.info(f"  name: {self.name}")
        logging.info(f"  grid_size: {self.grid_size}  num_classes: {self.num_classes}")
        logging.info(
            f"  cadence: every {self.eval_every_n_epochs} epochs "
            f"(warmup {self.warmup_epochs}, fit_end={self.run_on_fit_end})"
        )
        logging.info(
            f"  probes: knn={self.run_knn} linear={self.run_linear} "
            f"(linear every {self.linear_every_n_epochs}); support≤{self.support_images} imgs"
        )

    # ------------------------------------------------------------------ hooks
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if epoch < self.warmup_epochs or epoch % self.eval_every_n_epochs != 0:
            return
        run_linear = self.run_linear and (epoch % self.linear_every_n_epochs == 0)
        self._maybe_run(trainer, pl_module, run_linear=run_linear)
        self._last_eval_epoch = epoch

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.run_on_fit_end:
            return
        if trainer.current_epoch == self._last_eval_epoch:
            return  # already evaluated this epoch
        self._maybe_run(trainer, pl_module, run_linear=self.run_linear)

    def _maybe_run(self, trainer, pl_module, run_linear: bool) -> None:
        # Only rank 0 evaluates; no collectives are used, so other ranks
        # returning early cannot deadlock.
        if not trainer.is_global_zero:
            return
        try:
            self._run_eval(pl_module, run_linear=run_linear)
        except Exception as e:  # eval must never crash the training run
            logging.warning(f"! {self.name}: periodic eval failed ({e!r}); skipping")

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _run_eval(self, pl_module: LightningModule, run_linear: bool) -> None:
        device = pl_module.device
        # Freeze for eval. Methods that build their own encoder (e.g. MAE/I-JEPA)
        # expose ``.encoder`` rather than ``.backbone``; fall back to the module
        # itself so the toggle never silently fails (and is swallowed upstream).
        eval_mod = getattr(pl_module, "backbone", pl_module)
        was_training = eval_mod.training
        eval_mod.eval()
        try:
            bank_feats, bank_labels = self._extract_support(pl_module, device)
            if bank_feats.numel() == 0:
                logging.warning(f"! {self.name}: empty support set, skipping")
                return
            logs = {}
            if self.run_knn:
                logs.update(self._eval_knn(pl_module, device, bank_feats, bank_labels))
            if run_linear:
                logs.update(
                    self._eval_linear(pl_module, device, bank_feats, bank_labels)
                )
            if logs:
                pl_module.log_dict(logs, rank_zero_only=True, sync_dist=False)
        finally:
            eval_mod.train(was_training)

    def _extract_support(self, pl_module, device):
        feats_chunks, label_chunks, n_imgs = [], [], 0
        for batch in self.train_loader:
            images = batch[self.image_key].to(device, non_blocking=True)
            mask = batch[self.mask_key].to(device, non_blocking=True)
            feats_chunks.append(self._patch_features(pl_module, images))
            label_chunks.append(self._grid_labels(mask))
            n_imgs += images.size(0)
            if n_imgs >= self.support_images:
                break
        bank_feats = torch.cat(feats_chunks, dim=0)
        bank_labels = torch.cat(label_chunks, dim=0)
        return bank_feats, bank_labels

    def _eval_knn(self, pl_module, device, bank_feats, bank_labels) -> Dict[str, float]:
        metrics = self._fresh_metrics(device)
        for batch in self.val_loader:
            images = batch[self.image_key].to(device, non_blocking=True)
            mask = batch[self.mask_key].to(device, non_blocking=True)
            qfeats = self._patch_features(pl_module, images)
            soft = weighted_knn_predict(
                qfeats,
                bank_feats,
                bank_labels,
                num_classes=self.num_classes,
                k=self.knn_k,
                temperature=self.knn_temperature,
                distance_metric=self.knn_distance_metric,
                chunk_size=self.knn_chunk_size,
            )
            pix = grid_labels_to_pixels(soft.argmax(dim=1), self.grid_size, mask)
            target = squeeze_mask(mask).long()
            for m in metrics.values():
                m.update(pix, target)
        return {f"eval/{self.name}_knn_{k}": m.compute().item() for k, m in metrics.items()}

    def _eval_linear(self, pl_module, device, bank_feats, bank_labels) -> Dict[str, float]:
        embed_dim = bank_feats.size(1)
        if self.linear_use_batchnorm:
            head = nn.Sequential(
                nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, self.num_classes)
            )
        else:
            head = nn.Linear(embed_dim, self.num_classes)
        head = head.to(device).train()
        opt = torch.optim.SGD(head.parameters(), lr=self.linear_lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.linear_epochs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        n = bank_feats.size(0)
        bs = self.linear_batch_size
        with torch.enable_grad():
            for _ in range(self.linear_epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, bs):
                    idx = perm[i : i + bs]
                    opt.zero_grad()
                    loss = loss_fn(head(bank_feats[idx]), bank_labels[idx])
                    loss.backward()
                    opt.step()
                sched.step()

        head.eval()
        metrics = self._fresh_metrics(device)
        for batch in self.val_loader:
            images = batch[self.image_key].to(device, non_blocking=True)
            mask = batch[self.mask_key].to(device, non_blocking=True)
            logits = head(self._patch_features(pl_module, images))
            pix = grid_labels_to_pixels(logits.argmax(dim=1), self.grid_size, mask)
            target = squeeze_mask(mask).long()
            for m in metrics.values():
                m.update(pix, target)
        return {
            f"eval/{self.name}_linear_{k}": m.compute().item() for k, m in metrics.items()
        }

    # --------------------------------------------------------------- helpers
    def _fresh_metrics(self, device):
        return {k: v.clone().to(device) for k, v in self._metrics_spec.items()}

    def _patch_features(self, pl_module, images) -> torch.Tensor:
        tokens = self.feature_fn(pl_module, images)
        h_g, w_g = self.grid_size
        n_patches = h_g * w_g
        if tokens.dim() != 3:
            raise ValueError(
                f"{self.name}: feature_fn must return (B, N, D) patch tokens, "
                f"got shape {tuple(tokens.shape)}"
            )
        if tokens.size(1) != n_patches:
            raise ValueError(
                f"{self.name}: feature_fn returned {tokens.size(1)} tokens but "
                f"grid_size={self.grid_size} implies {n_patches}; check the "
                f"backbone / eval resolution / CLS handling"
            )
        return tokens.reshape(-1, tokens.size(-1))

    def _grid_labels(self, mask) -> torch.Tensor:
        h_g, w_g = self.grid_size
        m = squeeze_mask(mask)
        return (
            F.interpolate(m[:, None].float(), size=(h_g, w_g), mode="nearest")
            .long()
            .reshape(-1)
        )

    @staticmethod
    def _default_feature_fn(pl_module, images):
        return pl_module.backbone(images)
