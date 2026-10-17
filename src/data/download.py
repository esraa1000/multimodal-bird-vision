"""Download CUB-200-2011 (with captions) once and cache it to disk.

On Colab, point ``paths.data_dir_colab`` to a Google Drive folder so the
download happens only once; later sessions just load the cached copy.
"""

from __future__ import annotations

from pathlib import Path

HF_SPLITS = {
    "train": ("Multimodal-Fatima/CUB_train", "train"),
    "test": ("Multimodal-Fatima/CUB_test", "test"),
}


def prepare_cub(data_dir: Path, force: bool = False) -> dict[str, Path]:
    """Save each split with ``save_to_disk`` under ``data_dir/cub/<split>``."""
    from datasets import load_dataset  # imported lazily: heavy import

    out: dict[str, Path] = {}
    for split, (repo, hf_split) in HF_SPLITS.items():
        target = data_dir / "cub" / split
        if target.exists() and not force:
            print(f"[prepare-data] {split}: cached at {target}")
        else:
            print(f"[prepare-data] {split}: downloading {repo} ...")
            ds = load_dataset(repo, split=hf_split)
            ds.save_to_disk(str(target))
            print(f"[prepare-data] {split}: {len(ds)} records -> {target}")
        out[split] = target
    return out


def load_cub(data_dir: Path, split: str):
    """Load a cached split (run ``prepare-data`` first)."""
    from datasets import load_from_disk

    path = data_dir / "cub" / split
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.cli prepare-data")
    return load_from_disk(str(path))
