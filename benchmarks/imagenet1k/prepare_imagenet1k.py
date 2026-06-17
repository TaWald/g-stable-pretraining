"""Pre-download + build the ImageNet-1k cache so offline training nodes can read it.

Run this ONCE on a node WITH internet (e.g. a login node). It calls
``load_dataset`` with the *exact same* dataset id and ``cache_dir`` the imagenet1k
training scripts use (via ``get_data_dir``), so the Arrow cache it builds is
precisely what ``HF_HUB_OFFLINE=1`` / ``HF_DATASETS_OFFLINE=1`` will resolve
against on the compute nodes — no network calls, no hangs.

Prereqs (ImageNet-1k is a *gated* dataset):
  1. Accept the license once: https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. Authenticate:   huggingface-cli login        # or: export HF_TOKEN=hf_xxx
  3. Point at the SAME cache the training will use, on shared/scratch storage:
       export STABLE_PRETRAINING_DATA_DIR=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/data
       export HF_HOME=/dkfz/cluster/gpu/checkpoints/OE0441/t006d/hf
  4. (optional, ~150GB downloads much faster):
       pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1

Then:   python benchmarks/imagenet1k/prepare_imagenet1k.py
Verify offline:  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
                   python benchmarks/imagenet1k/prepare_imagenet1k.py
"""

import sys
from pathlib import Path

from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402


def main():
    cache_dir = str(get_data_dir("imagenet1k"))
    print(f"Using cache_dir = {cache_dir}")
    for split in ["train", "validation"]:
        ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=cache_dir)
        print(f"  {split}: {len(ds)} examples ready")
    print("ImageNet-1k cache is built; offline training nodes can now read it.")


if __name__ == "__main__":
    main()
