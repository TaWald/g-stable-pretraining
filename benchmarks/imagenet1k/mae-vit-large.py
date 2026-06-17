"""MAE ViT-L/16 pretraining on ImageNet-1k — matches the gmae reference recipe.

Reference: ``gmae/run_pretrain_h200.sh`` / ``gmae/PRETRAIN.md``
  model mae_vit_large_patch16, --epochs 800 --warmup_epochs 40,
  --mask_ratio 0.75 --norm_pix_loss, --blr 1.5e-4 --weight_decay 0.05.
  Actual lr = blr * effective_batch / 256 (linear scaling rule).

ImageNet-1k is gated on the Hub; accept the license once and authenticate
(`huggingface-cli login` or `export HF_TOKEN=...`). On offline compute nodes set
``HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`` and ``WANDB_MODE=offline`` (+ WANDB_ENTITY).

Env knobs: MAX_EPOCHS (default 800), BATCH_SIZE (per-GPU, default 128),
CHECKPOINT_DIR, WANDB_DIR, WANDB_ENTITY, WANDB_PROJECT.
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
from stable_pretraining.methods.mae import MAE

NUM_CLASSES = 1000


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = int(os.environ.get("BATCH_SIZE", 128))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 800))  # gmae reference
    warmup_epochs = 40
    # MAE linear scaling rule: lr = blr * effective_batch / 256.
    base_lr = 1.5e-4
    lr = base_lr * (batch_size * num_gpus) / 256
    ckpt_dir = os.environ.get(
        "CHECKPOINT_DIR", str(Path(__file__).parent / "checkpoints")
    )

    def mae_forward(self, batch, stage):
        output = MAE.forward(self, batch["image"])
        with torch.no_grad():
            features = self.encoder.forward_features(batch["image"])

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
            "embedding": features[:, 1:].mean(dim=1).detach(),  # skip cls
            **({"label": batch["label"].long()} if "label" in batch else {}),
        }

    data_dir = str(get_data_dir("imagenet1k"))

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "ILSVRC/imagenet-1k",
                split="train",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
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
                "ILSVRC/imagenet-1k",
                split="validation",
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

    module = MAE(
        model_or_model_name="vit_large_patch16_224",
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        block_size=1,  # random masking
        norm_pix_loss=True,  # normalize pixel targets per patch
        loss_type="mse",
        pretrained=False,
    )
    embed_dim = module.encoder.embed_dim

    module.forward = types.MethodType(mae_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": warmup_epochs / max_epochs,
            "start_factor": 0.0,  # MAE warms up from 0
            "end_lr": 0.0,  # MAE min_lr = 0
            "total_steps": (len(data.train) // num_gpus) * max_epochs,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(embed_dim, NUM_CLASSES),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        NUM_CLASSES, top_k=5
                    ),
                },
                optimizer={"type": "AdamW", "lr": 3e-3, "weight_decay": 1e-4},
                log_on_step=False,
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=50000,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES)
                },
                input_dim=embed_dim,
                num_classes=NUM_CLASSES,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=embed_dim,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(ckpt_dir) / "mae-vitl-inet1k"),
                filename="mae-vitl-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=50,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity=os.environ.get("WANDB_ENTITY", "tawald"),
            project=os.environ.get("WANDB_PROJECT", "imagenet1k-mae-ijepa"),
            name="mae-vitl-inet1k",
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
