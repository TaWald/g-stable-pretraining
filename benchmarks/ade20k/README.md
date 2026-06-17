# ADE20k dense segmentation eval

Frozen-backbone semantic-segmentation eval for pretrained MAE / I-JEPA encoders,
with two interchangeable probes selected via `--probe`:

- `--probe knn` — **patch-level weighted kNN** (CAPI-style),
  `OnlineKNNSegmentation`.
- `--probe linear` — **linear head trained with gradient descent** on frozen
  patch tokens (DINOv3-style), `OnlineProbeSegmentation`.

Both score per-pixel **mIoU** (and pixel accuracy) at the full mask resolution
and log under `eval/<name>_miou` / `eval/<name>_pixel_acc`.

## Protocols

A frozen ViT turns each image into patch tokens `(B, H_g·W_g, D)` (last-layer
tokens, CLS dropped). The GT mask is nearest-downsampled to the patch grid for
patch labels; predictions are folded back to the `H_g×W_g` grid, upsampled
(nearest) to the mask resolution, and scored. Label `0` (unlabeled) is the
ignore index.

**kNN (`--probe knn`)** — during one "training" epoch the train split's patch
tokens + patch labels stream into a support queue (no weights updated). On the
val split each query patch is classified by weighted kNN against the queue.

**Linear (`--probe linear`)** — a `BatchNorm1d + Linear` head (or plain `Linear`
with `--no-batchnorm`) is trained for several epochs (`--epochs`) with SGD +
cosine schedule on the patch labels; the backbone stays frozen. The trained head
is applied densely on the val split. BatchNorm standardizes the frozen patch
features per channel across the dataset — complementary to the encoder's final
per-token LayerNorm, and the convention used by the repo's `multi_layer_probe`
example and DINOv2/v3 linear evals.

> **Resolution note:** eval runs at 224×224 (MAE → 14×14 grid, I-JEPA → 16×16),
> so this is a coarse, patch-grid segmentation score — ideal for *relative*
> comparison between backbones and between probes, not for matching published
> dense-eval numbers that use higher-resolution sliding-window inference.

## Run

From the `g-stable-pretraining/` root, in its venv:

```bash
# MAE ViT-L/16 — kNN seg (default)
python benchmarks/ade20k/mae_vit_large.py \
    --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
    --probe knn --batch-size 32 --queue-length 1000000 --k 20

# MAE ViT-L/16 — linear seg
python benchmarks/ade20k/mae_vit_large.py \
    --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
    --probe linear --batch-size 32 --epochs 20 --lr 1e-2

# I-JEPA ViT-H/14 (target encoder) — kNN or linear
python benchmarks/ade20k/ijepa_vit_huge.py \
    --checkpoint /dkfz/cluster/gpu/checkpoints/OE0441/t006d/generalized_mim/gijepa/vith14.224-bs.2048-ep.300/<ckpt>.pth.tar \
    --gijepa-root ../gijepa \
    --probe linear --batch-size 16 --epochs 20 --lr 1e-2
```

`scene_parse_150` (the HF mirror of ADE20K) is downloaded on first run; set
`--data-cache <dir>` to control where. It is fetched via a hub loading script,
so `datasets` must allow `trust_remote_code=True` (already passed for you).

## Online monitoring during pretraining

The two runners above are *post-hoc* (frozen checkpoint). To watch segmentation
mIoU **as a trend line during an SSL run**, attach `PeriodicSegmentationEval`
(`spt.PeriodicSegmentationEval`) to your pretraining `Trainer`. It is
self-contained: every `eval_every_n_epochs` it snapshots the frozen backbone,
extracts patch features over a **subsampled** ADE20k support set *once*, and
from that single extraction scores a weighted kNN probe and/or a freshly trained
linear head on the val split — without touching the SSL train/val loop, the kNN
queue, or the module optimizers.

```python
from _common import build_periodic_ade20k_seg_callback

seg_monitor = build_periodic_ade20k_seg_callback(
    grid_size=(14, 14),                  # your backbone's patch grid @224
    feature_fn=lambda m, x: m.backbone.forward_features(x)[:, 1:],  # -> (B, N, D), CLS dropped
    support_images=2048,                 # subsample budget per trigger
    eval_every_n_epochs=25,              # + once at fit end
    run_linear=True,                     # kNN always; linear toggle/coarser
)
trainer = pl.Trainer(max_epochs=300, callbacks=[seg_monitor, ...])
```

Logs: `eval/<name>_knn_miou`, `eval/<name>_knn_pixel_acc`,
`eval/<name>_linear_miou`, `eval/<name>_linear_pixel_acc`. See
`periodic_eval_example.py` for a full skeleton.

> **`feature_fn` is required for SSL backbones** — they usually return a pooled
> embedding, not patch tokens. Pass `feature_fn(pl_module, images) -> (B, N, D)`
> returning patch tokens with the CLS/prefix dropped (the `_DropClsTokens` /
> `_IJEPATokens` wrappers in `_common.py` are worked examples). If omitted,
> `pl_module.backbone(images)` is used and must already return `(B, N, D)`.

**Cost / cadence:** kNN is cheap (~minutes) — fine every trigger. The linear
probe re-extracts + re-trains each trigger (the backbone keeps changing), so use
the subsampled support set and a coarse cadence (every 25–50 epochs), or set
`linear_every_n_epochs` higher than `eval_every_n_epochs` to run kNN often and
linear rarely. The eval runs on rank 0 only.

## Paper-matching final linear eval

The runners above (and the online monitor) are *coarse*: 224×224, square resize,
and `argmax`-then-nearest upsampling — fine for relative comparison, **not a
publishable number**. For the figure you'd report, use `final_linear_eval.py`
(DINOv2/v3 linear protocol):

- **Higher resolution** (`--image-size 512`, MAE → 32×32 grid; `518` for I-JEPA
  /14 → 37×37) via positional-embedding interpolation (`dynamic_img_size` for the
  timm MAE encoder; the gijepa ViT interpolates pos-embed in its forward).
- **Bilinear logit upsampling**: per-class logits are upsampled to the scoring
  resolution and *then* argmaxed (`logits_grid_to_pixels`); the training loss is
  cross-entropy on those upsampled logits vs the full `(S, S)` mask.
- **Full** train/val splits (no subsampling), light joint image+mask
  augmentation (random-resized-crop + h-flip), and an **LR sweep** reporting the
  best val mIoU.

```bash
python benchmarks/ade20k/final_linear_eval.py --model mae \
    --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
    --image-size 512 --epochs 20 --lrs 1e-3,1e-2,1e-1 --batch-size 8

python benchmarks/ade20k/final_linear_eval.py --model ijepa \
    --checkpoint /dkfz/.../gijepa/vith14.../<ckpt>.pth.tar \
    --gijepa-root ../gijepa --image-size 518 --batch-size 4
```

> **Remaining gap to exact mmseg-style numbers:** this is *whole-image* inference
> at a single scale. Sliding-window crops + multi-scale TTA + multi-layer feature
> concatenation are not implemented yet (a future add-on); expect a small offset
> from published sliding-window results.

## Memory / speed knobs

kNN:

- `--queue-length`: number of **support patches** kept (≈ `length·D·4` bytes for
  features; 1e6 × 1024 floats ≈ 4 GB). The queue keeps the most-recent patches,
  and the train loader is shuffled, so it's a random ~5k-image support set.
- `--k`: neighbors per patch (20 is a good default for dense kNN).
- `--chunk-size`: query patches processed per distance-matrix chunk. A batch
  contributes `B·H_g·W_g` query rows, so keep this modest to bound the
  `(queue_length, chunk_size)` distance matrix.

Linear:

- `--epochs`: head training epochs (also sets the cosine schedule horizon).
- `--lr`: head learning rate (SGD, momentum 0.9, no weight decay).
- `--no-batchnorm`: drop the BatchNorm, train a plain `Linear` head.

## Files

- `_common.py` — ADE20k DataModule + paired image/mask transforms, the frozen
  dense forward, the kNN-seg and linear-seg callback factories, and the MAE /
  I-JEPA checkpoint loaders.
- `mae_vit_large.py`, `ijepa_vit_huge.py` — thin runners (pick the probe with
  `--probe {knn,linear}`).

The callbacks live at
`stable_pretraining/callbacks/knn_segmentation.py` (`OnlineKNNSegmentation`) and
`stable_pretraining/callbacks/probe_segmentation.py`
(`OnlineProbeSegmentation`), also exported as `spt.OnlineKNNSegmentation` and
`spt.OnlineProbeSegmentation`.
