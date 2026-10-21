"""Datasets, transforms and augmentation (owner: M2).

    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)   # {"train", "val", "test"} -> batches of (images, labels)
"""

from src.data.loaders import build_dataloaders

__all__ = ["build_dataloaders"]
