# ImageNet classification eval (frozen backbone)

Standalone, paper-protocol **image-level classification** of a frozen pretrained
ViT (MAE ViT-L/16, I-JEPA ViT-H/14). This produces the number you'd report —
unlike `benchmarks/imagenet1k/`, which are *SSL pretraining* runs with online
probes.

Three probes (select with `--probe {linear,knn,attentive,all}`):

- **linear** — mean-pool the patch tokens → `BatchNorm1d + Linear`, trained with
  an LR sweep. The canonical MAE/DINO linear-probe top-1.
- **attentive** — `MultiHeadAttentiveProbe` (attention pooling over tokens) +
  Linear. The I-JEPA / DINOv2 headline number; usually beats plain linear.
- **knn** — weighted kNN (cosine, k swept) over the full mean-pooled train
  feature bank. DINO protocol, no training.

The trained probes (linear/attentive) use the faithful **augmented re-forward**
protocol: the frozen backbone is run on freshly `RandomResizedCrop`-augmented
crops every step. Val uses `Resize(256) + CenterCrop(224)`. Reported numbers are
the best over the LR (or k) sweep, top-1 and top-5.

## Run

```bash
# Open dataset, kNN only — cheapest sanity check
python benchmarks/imagenet_cls/final_classification_eval.py --model mae \
    --checkpoint ../gmae/output_dir/mae_vitl_800e/checkpoint-799.pth \
    --dataset imagenet-100 --probe knn

# Full ImageNet-1k, all three probes
python benchmarks/imagenet_cls/final_classification_eval.py --model mae \
    --checkpoint .../checkpoint-799.pth --dataset imagenet-1k --probe all \
    --epochs 20 --lrs 1e-2,1e-1,1 --batch-size 256

# I-JEPA ViT-H, attentive probe
python benchmarks/imagenet_cls/final_classification_eval.py --model ijepa \
    --checkpoint .../vith14.../<ckpt>.pth.tar --gijepa-root ../gijepa \
    --probe attentive --batch-size 128
```

## Notes / caveats

- **Datasets:** `--dataset imagenet-1k` uses `ILSVRC/imagenet-1k`, which is
  **gated** on the HF hub — run `huggingface-cli login` first.
  `--dataset imagenet-100` uses the open `clane9/imagenet-100` (100 classes) as a
  cheap proxy.
- **Cost:** the augmented re-forward protocol re-runs the frozen backbone over
  the full train split every epoch (1.28M images × `--epochs`), so the
  linear/attentive probes are heavy. kNN is comparatively cheap (one no-aug pass
  to build the bank). For quick iteration use `imagenet-100` and/or `--probe knn`.
  Papers train the linear head ~90 epochs; `--epochs` defaults to 20 for
  practicality — raise it (and widen `--lrs`) to close the last gap.
- **Pooling:** linear/kNN mean-pool the patch tokens (CLS dropped by the
  loaders), matching the MAE/I-JEPA convention; the attentive probe pools via
  learned attention. There is no CLS-token path (MAE's CLS is dropped, I-JEPA has
  none).
- The model checkpoint loaders are shared with the segmentation benchmark
  (`benchmarks/_backbones.py`).
