"""Image transforms: resize/crop, augmentation and normalization.

Chosen from the config with::

    data:
      img_size: 224
      augment: none | light        # training only; val/test never get augmentation
      normalize: dataset | imagenet

* ``dataset``: our CUB train-split mean/std (configs/data/channel_stats.json),
  for the model trained from scratch.
* ``imagenet``: ImageNet mean/std, for pretrained models (they were trained with these).
"""

from __future__ import annotations

import json

from torchvision import transforms as T

from src.utils.paths import PROJECT_ROOT

STATS_FILE = PROJECT_ROOT / "configs" / "data" / "channel_stats.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_mean_std(normalize: str) -> tuple[list[float], list[float]]:
    if normalize == "imagenet":
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    if normalize == "dataset":
        with open(STATS_FILE, encoding="utf-8") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    raise ValueError(f"Unknown normalize '{normalize}'. Use 'dataset' or 'imagenet'.")


def build_transforms(train: bool, img_size: int = 224, augment: str = "none",
                     normalize: str = "dataset") -> T.Compose:
    """PIL image -> normalized tensor of shape (3, img_size, img_size)."""
    resize_to = round(img_size * 256 / 224)   # 256 for 224: the standard resize-then-crop ratio

    if not train or augment == "none": # for test/val or no augmentation, just resize and center crop
        steps = [T.Resize(resize_to), T.CenterCrop(img_size)]
    elif augment == "light": # for training with light augmentation, resize and random crop/flip
        steps = [T.RandomResizedCrop(img_size, scale=(0.5, 1.0)), T.RandomHorizontalFlip()]
    else:
        raise ValueError(f"Unknown augment '{augment}'. Use 'none' or 'light'.")

    mean, std = get_mean_std(normalize)
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)]) # turns PIL image into numbers from 0 to 1, then normalizes 
