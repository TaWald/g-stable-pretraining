"""Utility functions for benchmarks."""

import math
import os
from pathlib import Path

import lightning.pytorch as pl


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
