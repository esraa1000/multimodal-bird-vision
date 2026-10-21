"""PyTorch dataset over the cached Hugging Face CUB-200-2011 split."""

from __future__ import annotations

from typing import Callable

from torch.utils.data import Dataset

from src.registry import DATASETS


@DATASETS.register("cub")
class CUBDataset(Dataset):
    """Returns ``(image_tensor, label_int)``.

    Args:
        hf_dataset: one split loaded with ``src.data.download.load_cub``.
        file_names: which images to use (e.g. the train or val list from
            configs/data/split_seed42.json). ``None`` uses every image (the test set).
        transform: built with ``src.data.transforms.build_transforms``.
    """

    def __init__(self, hf_dataset, file_names: list[str] | None = None,
                 transform: Callable | None = None):
        all_names = list(hf_dataset["file_name"]) 
        if file_names is None: # with no list given, it uses all images. That’s the test set case.
            file_names = all_names

        row_of = {name: row for row, name in enumerate(all_names)} # A lookup table from file name to row number
        missing = [name for name in file_names if name not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} file names are not in this split, e.g. {missing[:3]}")

        self.ds = hf_dataset
        self.file_names = list(file_names)
        self.rows = [row_of[name] for name in self.file_names]
        self.transform = transform
        self.class_names = hf_dataset.features["label"].names

    def __len__(self) -> int:
        return len(self.rows) # returns how many images this dataset has: 5,390 for train, 599 for val, 5,794 for test

    def __getitem__(self, i: int):
        record = self.ds[self.rows[i]]
        image = record["image"].convert("RGB")   # 4 grayscale images per split -> 3 channels
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["label"])
