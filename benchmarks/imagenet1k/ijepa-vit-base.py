"""I-JEPA ViT-B/16 pretraining on ImageNet-1k — gijepa recipe, ViT-Base backbone.

Same reference recipe as ``ijepa-vit-huge.py`` (gijepa: epochs 300, warmup 40,
ema [0.996, 1.0], predictor depth 12 / emb 384, masking 4 targets
pred_mask_scale [0.15, 0.2] aspect [0.75, 1.5] enc_mask_scale [0.85, 1.0],
lr 2.5e-4 / start 5e-5 / final 1e-6, wd 0.04 at effective batch 2048), but on
ViT-B/16. lr is linearly scaled from the reference batch of 2048.

The gijepa weight-decay ramp 0.04 -> 0.4 is replicated via ``CosineWDSchedule``
(benchmarks/utils.py): the optimizer starts at wd=0.04 and the callback cosine-ramps
it to 0.4 over ``total_steps``.

ImageNet-1k is gated on the Hub; accept the license once and authenticate
(`huggingface-cli login` or `export HF_TOKEN=...`). On offline compute nodes set
``HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`` and ``WANDB_MODE=offline`` (+ WANDB_ENTITY).

Env knobs: MAX_EPOCHS (default 300), BATCH_SIZE (per-GPU, default 256),
ACCUM (grad accumulation, default 1 -> effective batch 256*8*1=2048 on 8 GPUs,
the gijepa reference), CHECKPOINT_DIR, WANDB_DIR, WANDB_ENTITY, WANDB_PROJECT.
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

NUM_CLASSES = 1000


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir, CosineWDSchedule

    num_gpus = torch.cuda.device_count() or 1
    batch_size = int(os.environ.get("BATCH_SIZE", 256))
    accum = int(os.environ.get("ACCUM", 1))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 300))  # gijepa reference
    warmup_epochs = 40
    # gijepa reference lr 2.5e-4 at effective batch 2048; scale linearly.
    # effective_batch = per_gpu_batch * num_gpus * accum (reference 2048).
    ref_lr = 2.5e-4
    effective_batch = batch_size * num_gpus * accum
    lr = ref_lr * effective_batch / 2048
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

    data_dir = str(get_data_dir("imagenet1k"))

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "ILSVRC/imagenet-1k",
                split="train",
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

    # global_step counts optimizer steps (post-accumulation); divide by accum.
    total_steps = (len(data.train) // num_gpus // accum) * max_epochs
    module.forward = types.MethodType(ijepa_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.04,  # gijepa start wd; ramped to 0.4 by CosineWDSchedule
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": warmup_epochs / max_epochs,
            "start_factor": 0.2,  # start_lr 5e-5 / lr 2.5e-4
            "end_lr": 1e-6,  # gijepa final_lr
            "total_steps": total_steps,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=accum,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.TeacherStudentCallback(
                update_frequency=1,
                # False => EMA updates once per OPTIMIZER step (correct under
                # gradient accumulation), matching gijepa's per-iteration EMA.
                update_after_backward=False,
            ),
            CosineWDSchedule(
                start_weight_decay=0.04,
                final_weight_decay=0.4,
                total_steps=total_steps,
            ),
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(module.embed_dim, NUM_CLASSES),
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
                input_dim=module.embed_dim,
                num_classes=NUM_CLASSES,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=module.embed_dim,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(ckpt_dir) / "ijepa-vitb-inet1k"),
                filename="ijepa-vitb-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=50,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity=os.environ.get("WANDB_ENTITY", "tawald"),
            project=os.environ.get("WANDB_PROJECT", "imagenet1k-mae-ijepa"),
            name="ijepa-vitb-inet1k",
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
