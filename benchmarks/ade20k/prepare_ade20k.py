"""Pre-download the ADE20k (``scene_parse_150``) parquet cache for offline nodes.

Run this ONCE on a node WITH internet (e.g. a login node). It lands the data
under ``$STABLE_PRETRAINING_DATA_DIR`` (via ``get_data_dir``), co-located with
the other dataset caches, so ``HF_DATASETS_OFFLINE=1`` compute nodes can read it.

Why this isn't a plain ``load_dataset`` like ``prepare_imagenet1k.py``:
  ``scene_parse_150`` is a legacy *bare* repo id (no ``namespace/name``) shipped
  as a loading *script*. huggingface_hub 1.x's ``hf://`` URI parser rejects bare
  ids (``HfUriError: Repository id must be 'namespace/name'``), and ``datasets``
  4.x dropped script support entirely — so every ``load_dataset("scene_parse_150",
  ...)`` route fails (the convert-parquet revision goes through the same parser).
  Instead we ``snapshot_download`` the auto-converted parquet files over plain
  HTTPS (which accepts the bare id) and load them with the local ``parquet``
  builder — no ``hf://`` URI, no script.

Prereqs (point at the SAME cache the training will use, on shared/scratch storage):
    export STABLE_PRETRAINING_DATA_DIR=/dkfz/cluster/gpu/data/OE0441/t006d/Natural_Datasets

Then:   python benchmarks/ade20k/prepare_ade20k.py
Verify offline:  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
                   python benchmarks/ade20k/prepare_ade20k.py
"""

import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

REPO_ID = "scene_parse_150"
CONFIG = "scene_parsing"  # semantic seg (image + annotation); not instance_segmentation
SPLITS = ["train", "validation"]  # test has no labels; skip


def main():
    root = Path(get_data_dir(REPO_ID))  # $STABLE_PRETRAINING_DATA_DIR/scene_parse_150
    print(f"Downloading {REPO_ID}/{CONFIG} parquet -> {root}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="refs/convert/parquet",  # bare-id-safe over HTTPS
        allow_patterns=[f"{CONFIG}/{s}/*" for s in SPLITS],
        local_dir=str(root),
    )
    # Verify offline-loadable via the local parquet builder (no hf:// URI, no script).
    cfg = root / CONFIG
    for split in SPLITS:
        ds = load_dataset(
            "parquet",
            data_files={split: str(cfg / split / "*.parquet")},
            split=split,
        )
        print(f"  {split}: {ds.num_rows} examples, columns={ds.column_names}")
    print("ADE20k (scene_parse_150) parquet is ready for offline nodes.")


if __name__ == "__main__":
    main()
