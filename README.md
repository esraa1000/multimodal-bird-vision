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

## Classification (Task 1.2)

First check the whole pipeline on the real data (a minute or two, CPU is fine, nothing uploaded):

```bash
python -m src.cli train --config configs/cls_smoke.yaml
```

Then train the from-scratch baseline (on a GPU: Colab/Kaggle):

```bash
python -m src.cli train --config configs/cls_scratch.yaml run_name=scratch_v1
```

A training run writes to its folder:

| File | What it is |
|---|---|
| `best.pt` | weights with the best val accuracy (no optimizer state); uploaded to W&B at the end |
| `last.pt` | the latest epoch including optimizer/scheduler state, to resume |
| `training_curves.png` | train vs. val loss and accuracy |
| `metrics.jsonl`, `summary.json` | per-epoch metrics, best val accuracy and epoch |

On Colab, write outputs to Google Drive so they survive a disconnect, and resume from `last.pt`:

```bash
python -m src.cli train --config configs/cls_scratch.yaml paths.output_dir=/content/drive/MyDrive/bird-outputs
python -m src.cli train --config configs/cls_scratch.yaml paths.output_dir=/content/drive/MyDrive/bird-outputs --resume /content/drive/MyDrive/bird-outputs/<run>/last.pt
```

Experiments (each one is a separate tracked run):

```bash
python -m src.cli train --config configs/cls_scratch.yaml data.augment=none   run_name=scratch_aug-none
python -m src.cli train --config configs/cls_scratch.yaml data.augment=strong run_name=scratch_aug-strong
python -m src.cli train --config configs/cls_scratch.yaml model.params.blocks_per_stage=4 run_name=scratch_plain4
python -m src.cli train --config configs/cls_scratch_residual.yaml
```

Evaluate a trained model. Use `--split val` while developing, and the test set only for final numbers:

```bash
python -m src.cli evaluate --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --split test
```

This writes `metrics.json` (accuracy, top-5, macro precision/recall/F1), `predictions.csv` (one row per image),
`per_class.csv`, `confusion_matrix.png` and `top_confused.png`. Predict one image:

```bash
python -m src.cli predict --config configs/cls_scratch.yaml --ckpt outputs/<run>/best.pt --image bird.jpg
```

## Hyperparameter tuning (grid vs. random search)

Both searches tune `train.lr`, `train.weight_decay` and `model.dropout` on the scratch CNN with the
same budget: 18 trials of 15 epochs each, same data, seed and model (`configs/tune_base.yaml`).
Set `data.augment` there to the winner of the augmentation ablation first.

The trial lists are fixed in advance, so the team splits each search with `--shard k/n`.
With 4 members, each runs their slice of both searches (about 5 trials, roughly 40 minutes on a T4, per search):

```bash
python -m src.cli tune --config configs/tune_grid.yaml   --list            # show the planned trials
python -m src.cli tune --config configs/tune_grid.yaml   --shard 1/4 paths.output_dir=/content/drive/MyDrive/bird-outputs
python -m src.cli tune --config configs/tune_random.yaml --shard 1/4 paths.output_dir=/content/drive/MyDrive/bird-outputs
```

Member 2 uses `--shard 2/4`, and so on. Each trial is a normal W&B run named `tune_grid_v1_t03`, in the group
`tune_grid_v1`. Results are also appended to `<output_dir>/<tune.name>/results.jsonl`. If a session
disconnects, run the same command again: finished trials are skipped (`--rerun` forces them).
A trial that diverges (for example, a too-high learning rate) is recorded as failed and the search continues.

When all shards are done, compare the methods from W&B (any machine that is logged in):

```bash
python -m src.cli tune-report --config configs/tune_grid.yaml --config configs/tune_random.yaml
```

This writes to `reports/tuning/`:
- `tuning_trials.csv`: every trial,
- `tuning_comparison.json`: best score and values, trials to reach the best and to get within
  2 points of the overall best, failures, and total time per method,
- `tuning_comparison.png`: best-so-far curves and score vs. learning rate.

Finally, retrain the best values for the full 40 epochs, for example:
`python -m src.cli train --config configs/cls_scratch.yaml train.lr=... train.weight_decay=... model.dropout=... run_name=scratch_tuned`.

## Weights & Biases tracking

The W&B tracker records configuration, metrics, summaries, checkpoint artifacts,
and figures. It also keeps the local JSON/YAML records described above. The
default tracker remains `local`; W&B is loaded only when selected.

First try the debug task offline (no login or upload):

```bash
python -m src.cli train --config configs/debug_wandb.yaml
```

For online tracking, authenticate in your terminal, then select online mode:

```bash
wandb login
python -m src.cli train --config configs/debug_wandb.yaml tracker.mode=online
```

The team workspace (`deep_learning_birds`) is set once in `configs/tracker_wandb.yaml`.
The classification configs include it, so those runs go to the team project once you are logged in.
Use `tracker.mode=offline` or `tracker.name=local` to run without uploading.
`WANDB_API_KEY` can be supplied through environment variables or Colab/Kaggle
secrets instead of interactive login. Never put API keys in configs or Git.

To enable tracking for another task, add the following to its YAML config:

```yaml
tracker:
  name: wandb
  project: multimodal-bird-vision
  mode: online              # online | offline | disabled
  # entity: YOUR_TEAM
  # group: scratch-baseline
  # tags: [classification, scratch]
  # notes: First baseline experiment
```

Classification, colorization, and captioning are still placeholders; use the
debug task to check tracking until those tasks are implemented. Task code uses
the existing `Tracker` interface: `log_metrics`, `log_summary`, `log_artifact`,
and `log_figure`. Figures appear as W&B images and downloadable artifacts.
The `step` argument is recorded as the `trainer/step` chart axis, independently
of W&B's internal counter, so logging figures between epochs does not skip
metrics. Exceptions inside the tracker context mark the W&B run as failed.

Offline W&B runs are stored under `outputs/<run>/wandb/`. To upload one later:

```bash
wandb sync outputs/<run>/wandb/offline-run-<timestamp>-<id>
```

Run tracking checks without downloading data or installing PyTorch:

```bash
pip install wandb pyyaml pytest
python -m pytest tests/test_wandb_tracker.py
```

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
