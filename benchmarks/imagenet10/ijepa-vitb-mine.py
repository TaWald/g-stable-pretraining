"""I-JEPA ViT-B/16 pretraining on Imagenette (10-class) — personal sanity run.

Copy of ``ijepa-vit-base.py`` with the WandB destination parametrized so it logs to
*your* account instead of the hardcoded ``stable-ssl`` entity. Set, before running::

    wandb login
    export WANDB_ENTITY=<your-wandb-entity>      # e.g. your username or team
    export WANDB_PROJECT=ijepa-mine              # optional, defaults below

Quick smoke test: lower ``MAX_EPOCHS`` (env var) to e.g. 5; the scheduler ``total_steps``
is derived from it automatically. Use this to validate the pipeline + the kill/resume
behaviour before scaling to ViT-H / ImageNet-1k.
"""

import os
import sys
import types
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 64
    max_epochs = int(os.environ.get("MAX_EPOCHS", 600))
    # Where checkpoints are written. Defaults next to this script; override with
    # CHECKPOINT_DIR (e.g. point it at scratch to keep them out of $HOME).
    ckpt_dir = os.environ.get(
        "CHECKPOINT_DIR", str(Path(__file__).parent / "checkpoints")
    )

    def ijepa_forward(self, batch, stage):
        output = IJEPA.forward(self, batch["image"], embedding_source="student")
        embedding = output.embedding.mean(dim=1)
        if self.training:
            embedding = embedding.detach()

        self.log(
            f"{stage}/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["image"].shape[0],
        )

        return {
            "loss": output.loss,
            "embedding": embedding,
            **({"label": batch["label"].long()} if "label" in batch else {}),
        }

    data_dir = str(get_data_dir("imagenet10"))

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 16),
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 16),
            persistent_workers=num_workers > 0,
        ),
    )

    module = IJEPA(
        model_or_model_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        num_targets=4,
        target_scale=(0.15, 0.2),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(0.85, 1.0),
        ema_decay_start=0.996,
        ema_decay_end=1.0,
        pretrained=False,
    )

    module.forward = types.MethodType(ijepa_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": 6e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 40 / 300,
            "start_factor": 0.01,
            "end_lr": 6e-4 / 10,
            "total_steps": (len(data.train) // num_gpus) * max_epochs,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.TeacherStudentCallback(
                update_frequency=1,
                update_after_backward=True,
            ),
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(module.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
                log_on_step=False,
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=module.embed_dim,
                num_classes=10,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=module.embed_dim,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(ckpt_dir) / "ijepa-vitb-mine"),
                filename="ijepa-vitb-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=300,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "ijepa-mine"),
            name="ijepa-vitb-inet10",
            # Keep wandb's run files out of the code dir. Lightning passes
            # save_dir -> wandb.init(dir=...) explicitly, which overrides the
            # WANDB_DIR env var, so read it here for WANDB_DIR to take effect.
            save_dir=os.environ.get("WANDB_DIR", "."),
            log_model=False,
        ),
        precision="16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
