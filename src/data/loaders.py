"""Build the train / val / test DataLoaders from a config.

The one function the training code calls::

    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)   # {"train": ..., "val": ..., "test": ...}

Config keys used: seed, paths.*, data.split_file, data.img_size, data.batch_size,
data.num_workers, data.augment, data.normalize.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.data.download import load_cub
from src.data.transforms import build_transforms
from src.utils.paths import get_data_dir, resolve_path
from src.utils.seed import seed_worker


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    d = cfg.data

    # which images go where (excluded duplicates are already left out of this file)
    with open(resolve_path(d.get("split_file", "configs/data/split_seed42.json")), encoding="utf-8") as f:
        split = json.load(f)

    img_size = d.get("img_size", 224)
    normalize = d.get("normalize", "dataset")
    train_tf = build_transforms(True, img_size, d.get("augment", "none"), normalize) # training transforms: resize, crop, augmentation, normalization
    eval_tf = build_transforms(False, img_size, "none", normalize) # validation/test transforms: resize, crop, normalization

    data_dir = get_data_dir(cfg.get("paths", {}))
    train_hf = load_cub(data_dir, "train")
    test_hf = load_cub(data_dir, "test")

    datasets = {
        "train": CUBDataset(train_hf, split["train"], train_tf),
        "val": CUBDataset(train_hf, split["val"], eval_tf),
        "test": CUBDataset(test_hf, None, eval_tf),   # the whole official test set
    }

    num_workers = d.get("num_workers", 2)
    generator = torch.Generator().manual_seed(cfg.get("seed", 42))   # same shuffle order every run
    return {
        name: DataLoader(
            ds,
            batch_size=d.get("batch_size", 64),
            shuffle=(name == "train"),
            num_workers=num_workers,
            worker_init_fn=seed_worker,          # reproducible augmentation in worker processes
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
        for name, ds in datasets.items()
    }
