"""Shared test fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

CLASSES = ["Albatross", "Cardinal", "Jay", "Wren"]


@pytest.fixture
def fake_cub(tmp_path):
    """A tiny CUB in the same Hugging Face format (image / label / file_name).

    4 classes, 6 train images each (2 of them go to val) and 3 test images
    each; every class has its own colour so a model can learn something.
    Returns the folder, the class names, CLI overrides that point a config at
    it (CPU-sized model and images), and a test image.
    """
    from datasets import ClassLabel, Dataset, Features, Value
    from datasets import Image as HFImage

    rng = np.random.default_rng(0)
    features = Features({"image": HFImage(), "label": ClassLabel(names=CLASSES),
                         "file_name": Value("string")})

    def make(split, per_class):
        rows = {"image": [], "label": [], "file_name": []}
        for c, name in enumerate(CLASSES):
            for i in range(per_class):
                colour = np.array([c * 60, 255 - c * 60, 128])
                pixels = colour + rng.integers(-20, 20, size=(40, 48, 3))
                rows["image"].append(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))
                rows["label"].append(c)
                rows["file_name"].append(f"{name}_{split}_{i:03d}.jpg")
        Dataset.from_dict(rows, features=features).save_to_disk(str(tmp_path / "data" / "cub" / split))
        return rows["file_name"]

    train_names = make("train", 6)
    make("test", 3)
    is_val = lambda n: n.endswith(("_004.jpg", "_005.jpg"))  # noqa: E731
    (tmp_path / "split.json").write_text(json.dumps({
        "train": [n for n in train_names if not is_val(n)], "val": [n for n in train_names if is_val(n)]}))
    image = tmp_path / "bird.jpg"
    Image.fromarray(np.full((40, 48, 3), (0, 255, 128), dtype=np.uint8)).save(image)

    overrides = [
        f"paths.data_dir={(tmp_path / 'data').as_posix()}",
        f"paths.output_dir={(tmp_path / 'outputs').as_posix()}",
        f"data.split_file={(tmp_path / 'split.json').as_posix()}",
        "tracker.name=local", "model.num_classes=4", "model.params.depth=2", "model.params.width=8",
        "data.img_size=32", "data.batch_size=4", "data.num_workers=0",
        "train.epochs=2", "train.warmup_epochs=1", "train.progress=false",
    ]
    return SimpleNamespace(root=tmp_path, classes=CLASSES, overrides=overrides, image=image)
