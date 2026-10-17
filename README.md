# multimodal-bird-vision

Multimodal deep learning on **CUB-200-2011** with captions: one CNN encoder, three tasks.

1. **Classification**: 200 bird species (encoder trained from scratch and with transfer learning), Milestone 1
2. **Colorization**: grayscale to color using the encoder as backbone, Milestone 2
3. **Captioning**: frozen encoder plus a sequence decoder, Milestone 2

CSE488 course project.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, follow https://pytorch.org/get-started/locally/ first, then install the rest.

Check that everything works with a few-second run on random data:

```bash
python -m src.cli train --config configs/debug.yaml
```

## Data

```bash
python -m src.cli prepare-data
```

This downloads `Multimodal-Fatima/CUB_train` and `CUB_test` from Hugging Face once and caches them under `data/cub/`. The data folder is chosen per environment from `paths` in `configs/base.yaml`, which has separate settings for local, Colab, and Kaggle.

## Usage

```bash
python -m src.cli list                                                     # registered tasks / models / trackers
python -m src.cli train    --config configs/cls_scratch.yaml
python -m src.cli train    --config configs/cls_scratch.yaml train.lr=1e-3 train.epochs=5   # override any key
python -m src.cli evaluate --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --split test
python -m src.cli predict  --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --image bird.jpg --gradcam
```

Each run writes to `outputs/<run_name>_<timestamp>/`:
`config.yaml` (the resolved config plus git commit and hardware), `metrics.jsonl`, `summary.json`, and checkpoints.

## Project structure

```
configs/            YAML configs; each inherits from base.yaml via `defaults:`
notebooks/          experiments only + colab_runner.ipynb (not part of the final code)
src/
  cli.py            command line entry point
  config.py         YAML loading, inheritance, key=value overrides
  registry.py       name -> class registries (MODELS, DATASETS, TASKS, TRACKERS)
  tasks/            one Task class per project task (train / evaluate / predict)
  data/             datasets, transforms, augmentation
  models/           encoders, heads, colorizer, caption decoder
  training/         training loops
  evaluation/       metrics, benchmarking
  visualization/    plots, Grad-CAM
  tracking/         Tracker interface + implementations
  utils/            seeding, device, paths, checkpoints
outputs/            run outputs (not committed)
```

### Design

- **Registry / Factory**: models, tasks, and trackers register themselves by name (`@MODELS.register("scratch_cnn")`), and configs select them by that name. Adding an encoder means adding one file. The trainer and CLI stay unchanged.
- **Strategy**: trackers, and later losses and augmentations, are interchangeable objects behind one interface.
- **Dependency inversion**: training code depends on the abstract `Tracker`, not on a specific tool.
- **Task interface**: the CLI calls `train / evaluate / predict` on whichever `Task` is selected, so Milestone 2 tasks plug in without changing it.

## How to add a model

```python
# src/models/scratch_cnn.py
from torch import nn
from src.registry import MODELS

@MODELS.register("scratch_cnn")
class ScratchCNN(nn.Module):
    def __init__(self, depth=4, width=64, dropout=0.3, num_classes=200): ...
```

Import it in `src/models/__init__.py`, then set `model.name: scratch_cnn` in a config.

## Team workflow

- Create one branch per task (`feat/eda-captions`, `feat/gradcam`). Open a PR into `main` and get one review.
- Commit from **your own** GitHub account, in small and frequent commits.
- Notebooks are for experiments only. Final code lives in `src/` and runs through the CLI.
- On Colab or Kaggle, use `notebooks/colab_runner.ipynb`: clone, install, run the CLI.
