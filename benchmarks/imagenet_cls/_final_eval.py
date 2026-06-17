"""Standalone frozen-backbone classification eval: linear / attentive / kNN.

The trained probes use the faithful **augmented re-forward** protocol (the frozen
backbone is run on freshly-augmented crops every step). kNN extracts a single
no-augmentation mean-pooled feature bank from the train split. All consume the
backbone's ``(B, N, D)`` patch-token output directly.
"""

import torch
import torch.nn as nn

from stable_pretraining.backbone.probe import LinearProbe, MultiHeadAttentiveProbe
from stable_pretraining.callbacks.knn import weighted_knn_predict

from _common import cls_metrics  # noqa: E402


def make_head(kind, embed_dim, num_classes, num_heads=4):
    """Build a probe head consuming ``(B, N, D)`` tokens -> ``(B, num_classes)``."""
    if kind == "linear":
        return LinearProbe(
            embed_dim, num_classes, pooling="mean", norm_layer=nn.BatchNorm1d
        )
    if kind == "attentive":
        return MultiHeadAttentiveProbe(embed_dim, num_classes, num_heads=num_heads)
    raise ValueError(f"unknown probe kind {kind!r} (expected 'linear'/'attentive')")


def _freeze(backbone, device):
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def train_and_eval_probe(
    backbone,
    embed_dim,
    kind,
    train_loader,
    val_loader,
    device,
    num_classes,
    lrs=(1e-3, 1e-2, 1e-1),
    epochs: int = 20,
    num_heads: int = 4,
    weight_decay: float = 0.0,
    image_key: str = "image",
    label_key: str = "label",
):
    """Train a probe head (LR sweep) and return the best val result.

    Returns ``{"top1", "top5", "lr"}`` for the best LR (by top-1). Used for both
    ``kind="linear"`` and ``kind="attentive"``.
    """
    backbone = _freeze(backbone, device)
    best = None
    for lr in lrs:
        head = make_head(kind, embed_dim, num_classes, num_heads).to(device)
        opt = torch.optim.SGD(
            head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(epochs):
            head.train()
            for batch in train_loader:
                images = batch[image_key].to(device, non_blocking=True)
                y = batch[label_key].to(device, non_blocking=True)
                with torch.no_grad():
                    tokens = backbone(images)  # (B, N, D)
                logits = head(tokens)
                loss = loss_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        metrics = {k: v.to(device) for k, v in cls_metrics(num_classes).items()}
        head.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch[image_key].to(device, non_blocking=True)
                y = batch[label_key].to(device, non_blocking=True)
                logits = head(backbone(images))
                for m in metrics.values():
                    m.update(logits, y)
        res = {k: m.compute().item() for k, m in metrics.items()}
        res["lr"] = lr
        print(f"[{kind}] lr={lr:g}  top1={res['top1']:.4f}  top5={res['top5']:.4f}")
        if best is None or res["top1"] > best["top1"]:
            best = res
    return best


@torch.no_grad()
def _extract_pooled(backbone, loader, device, image_key, label_key):
    """Mean-pooled features ``(M, D)`` (on CPU) + labels ``(M,)`` over a loader."""
    feats, labels = [], []
    for batch in loader:
        images = batch[image_key].to(device, non_blocking=True)
        tokens = backbone(images)  # (B, N, D)
        feats.append(tokens.mean(dim=1).cpu())
        labels.append(batch[label_key])
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def eval_knn(
    backbone,
    train_noaug_loader,
    val_loader,
    device,
    num_classes,
    ks=(10, 20, 200),
    temperature: float = 0.07,
    query_chunk: int = 512,
    image_key: str = "image",
    label_key: str = "label",
):
    """Weighted-kNN top-1/top-5 over a mean-pooled train feature bank.

    Returns ``{"top1", "top5", "k"}`` for the best k (by top-1). The val queries
    are processed in chunks of ``query_chunk`` so the ``(bank, chunk)`` distance
    matrix stays bounded for a large (e.g. 1.28M) bank.
    """
    backbone = _freeze(backbone, device)
    bank_f, bank_y = _extract_pooled(backbone, train_noaug_loader, device, image_key, label_key)
    val_f, val_y = _extract_pooled(backbone, val_loader, device, image_key, label_key)
    bank_f = bank_f.to(device)
    bank_y = bank_y.to(device)

    best = None
    for k in ks:
        metrics = {kk: v.to(device) for kk, v in cls_metrics(num_classes).items()}
        for i in range(0, val_f.size(0), query_chunk):
            qf = val_f[i : i + query_chunk].to(device)
            qy = val_y[i : i + query_chunk].to(device)
            soft = weighted_knn_predict(
                qf,
                bank_f,
                bank_y,
                num_classes=num_classes,
                k=k,
                temperature=temperature,
                distance_metric="cosine",
                chunk_size=-1,
            )
            for m in metrics.values():
                m.update(soft, qy)
        res = {kk: m.compute().item() for kk, m in metrics.items()}
        res["k"] = k
        print(f"[knn] k={k}  top1={res['top1']:.4f}  top5={res['top5']:.4f}")
        if best is None or res["top1"] > best["top1"]:
            best = res
    return best
