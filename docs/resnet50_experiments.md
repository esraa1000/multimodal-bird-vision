# ResNet-50 transfer-learning experiments

Trial 1 is the selected ResNet-50 model: an ImageNet-pretrained, frozen encoder
with a trainable linear classifier. Its epoch-16 checkpoint was selected using
validation results before evaluating the full test set. Final test accuracy is
**69.64%**, macro-F1 **69.53%**, and top-5 accuracy **91.70%**.

Final test run on W&B:
[cls_resnet50_trial1_final_test](https://forge.coreweave.com/wandb/deep_learning_birds/multimodal-bird-vision/runs/shvozh2k).
Only aggregate test metrics were published in that run. Per-image predictions,
figures, and the checkpoint remain in the local output folders. Small evaluation
tables and figures are also saved in this repository under `reports/resnet50/`.

## Data and preprocessing

- Dataset: CUB-200-2011, 200 bird species, using the cached Hugging Face splits.
- Training: 5,390 images; validation: 599; final test: 5,794.
- The fixed, stratified seed-42 train/validation split is stored in
  `configs/data/split_seed42.json`, with the project's duplicate exclusions.
- Images are converted to RGB. Training transforms produce 224 x 224 tensors;
  augmentation is applied when loading images, rather than creating a second
  saved dataset. All four trials use ImageNet normalization:
  mean `(0.485, 0.456, 0.406)`, standard deviation `(0.229, 0.224, 0.225)`.
- Light training augmentation: random resized crop (scale 0.5 to 1.0), then
  random horizontal flip (probability 0.5).
- Strong training augmentation: random resized crop (scale 0.35 to 1.0),
  horizontal flip (0.5), rotation up to 15 degrees, color jitter
  (brightness/contrast 0.3, saturation 0.2), and random erasing
  (probability 0.25, area 0.02 to 0.15).
- Validation and test use resize to 256, center crop to 224, and normalization;
  no random training augmentation is applied to either split.

## Model and shared settings

The `timm` ResNet-50 encoder starts with ImageNet weights
(`resnet50.a1_in1k`). Global average pooling produces 2,048 features, followed
by a linear layer with 200 outputs. The classifier has 409,800 parameters;
the full model has 23,917,832. Trials 1 and 2 train only the classifier and keep
the entire encoder, including BatchNorm running statistics, fixed.

All trials use batch size 16, seed 42, AdamW, weight decay 0.05, a cosine
learning-rate schedule, no warm-up, no label smoothing, and CUDA mixed precision.
Early stopping patience is 8 epochs, using validation accuracy. GPU execution
uses `deterministic: false`, so a repeated run need not be bit-for-bit identical.
Runs were performed on an NVIDIA GeForce RTX 5060 Laptop GPU.

## Trials and validation results

| Trial | Trainable encoder weights | Augmentation | Initial head / encoder LR | Epoch limit / completed | Best epoch | Best validation accuracy |
|---|---|---|---|---|---|---|
| 1 | None | Light | 1e-3 / frozen | 25 / 24 | 16 | **68.11%** |
| 2 | None | Strong | 1e-3 / frozen | 25 / 25 | 20 | 66.78% |
| 3 | `layer4` | Light | 1e-4 / 1e-5 | 15 / 15 | 12 | 65.44% |
| 4 | `layer4`; encoder BN statistics fixed | Light | 1e-4 / 1e-5 | 15 / 15 | 13 | 67.45% |

Trials 3 and 4 both start from **trial 1's epoch-16 best checkpoint**, loading
the trained classifier as well as the encoder weights and buffers. Each uses
a fresh optimizer, scheduler, epoch counter, and early-stopping history. Earlier
encoder stages remain fixed. In trial 3, BatchNorm statistics within `layer4`
update during training. In trial 4, all encoder BatchNorm running statistics
stay fixed; trainable `layer4` BatchNorm scale and bias can still learn.

Trial 4 measures validation performance immediately after initialization,
before any optimizer updates: **68.11%**, matching the source checkpoint. Its
first training epoch gives 66.44%, compared with 52.59% in trial 3. Keeping BN
statistics fixed reduced the initial degradation in these experiments, but
neither fine-tuning trial improved on trial 1. This comparison does not prove
that BN changes alone caused the entire drop in trial 3.

Trials 1 and 4 were also evaluated on the full validation set using their best
checkpoints to obtain macro metrics and per-class errors:

| Trial | Validation accuracy | Validation macro precision | Validation macro recall | Validation macro-F1 |
|---|---|---|---|---|
| 1 | **68.11%** | 70.42% | 68.17% | **66.94%** |
| 4 | 67.45% | 69.75% | 67.50% | 65.86% |

Macro-F1 averages the 200 per-species F1 scores. It is not the average of
macro precision and macro recall. Most species have only three validation
images, so individual per-class validation scores are coarse. These are
single-seed experiments; small differences should not be treated as definitive
rankings across repeated runs. Macro-F1 was not separately evaluated for trials
2 and 3.

Trial 1 was retained because it had the best validation accuracy across the four
trials, with higher validation macro-F1 than the closest fine-tuning follow-up.
The stronger augmentation and partial fine-tuning attempts did not improve the
selection metrics in these runs. No automated hyperparameter search was run
for these ResNet experiments. Trials 3 and 4 also use the compute already spent
training trial 1, so their additional training times are not standalone totals.

Training times from the saved summaries: trial 1, 2,307.8 seconds; trial 2,
2,671.1 seconds; trial 3, 1,560.3 seconds; trial 4, 1,603.3 seconds. These are
training-run timings, not inference benchmarks.

## Final test evaluation

The selected checkpoint is
`outputs/cls_resnet50_frozen_20261029-034949/best.pt` (epoch 16).
It was evaluated without weight updates on all 5,794 test images:

| Metric | Result |
|---|---|
| Accuracy | 69.64% |
| Top-5 accuracy | 91.70% |
| Macro precision | 70.48% |
| Macro recall | 69.92% |
| Macro-F1 | 69.53% |

The test set was used after selecting trial 1 on validation data. It should
remain the final report of that selected model, rather than guiding another
round of model selection. No new training was needed to compute macro-F1 or
per-class metrics. The existing evaluation code calculates them from predictions.

Local evaluation output:
`outputs/cls_resnet50_selected_test_eval-test_20261029-173732/`.
It contains `metrics.json`, `predictions.csv`, `per_class.csv`, the confusion
matrix, and the most confused class pairs. A separate upload published the
aggregate metrics to W&B without uploading predictions or figures.

The `predict` CLI was also checked on one cached validation image. It returned
a species prediction, confidence, and five ranked classes. This is a functional
inference check, not another accuracy evaluation or training experiment.

## Commands and saved evidence

Run commands from the repository root with the project environment active.
On Windows, `python` below can be replaced with `.\.venv\Scripts\python.exe`.

```powershell
# Trial 1: frozen encoder, light augmentation
python -m src.cli train --config configs/cls_resnet50.yaml

# Trial 2: change only augmentation and run name
python -m src.cli train --config configs/cls_resnet50.yaml data.augment=strong run_name=cls_resnet50_frozen_strong

# Trial 3: initialize from trial 1 and train layer4 plus the head
python -m src.cli train --config configs/cls_resnet50_finetune.yaml

# Trial 4: same initialization, fixed BN statistics, pre-training validation
python -m src.cli train --config configs/cls_resnet50_finetune_bn_frozen.yaml

# Final evaluation, performed locally; aggregate metrics were then logged to W&B
python -m src.cli evaluate --config configs/cls_resnet50.yaml --ckpt outputs/cls_resnet50_frozen_20261029-034949/best.pt --split test tracker.name=local data.num_workers=0 run_name=cls_resnet50_selected_test

# Single-image inference, after supplying a local image
python -m src.cli predict --config configs/cls_resnet50.yaml --ckpt outputs/cls_resnet50_frozen_20261029-034949/best.pt --image bird.jpg
```

For fine-tuning on another machine, copy the selected checkpoint and override
`train.init_checkpoint=PATH_TO_BASELINE/best.pt`. `train.init_checkpoint` starts
a new experiment from weights; `--resume PATH_TO_SAME_TRIAL/last.pt` continues
an interrupted experiment including its training state. Resume takes precedence.

Saved experiment records in this branch:

- [Machine-readable results, source run names, and checkpoint SHA-256](../reports/resnet50/results.json).
- Actual resolved configurations: [trial 1](../reports/resnet50/trial1_config.yaml),
  [trial 2](../reports/resnet50/trial2_config.yaml),
  [trial 3](../reports/resnet50/trial3_config.yaml),
  [trial 4](../reports/resnet50/trial4_config.yaml). Historical metadata is preserved.
- Training curves: [trial 1](../reports/resnet50/trial1_curves.png),
  [trial 2](../reports/resnet50/trial2_curves.png),
  [trial 3](../reports/resnet50/trial3_curves.png),
  [trial 4](../reports/resnet50/trial4_curves.png).
- Validation per-class metrics: [trial 1](../reports/resnet50/trial1_validation_per_class.csv),
  [trial 4](../reports/resnet50/trial4_validation_per_class.csv).
- Final test [per-class metrics](../reports/resnet50/trial1_test_per_class.csv),
  [confusion matrix](../reports/resnet50/trial1_test_confusion_matrix.png), and
  [most confused pairs](../reports/resnet50/trial1_test_top_confused.png).

Checkpoints, datasets, and full output folders are excluded from Git. The
selected `best.pt` must be preserved or shared separately to reproduce inference.
Grad-CAM, hardware benchmarking, the from-scratch CNN experiments, and other
transfer models remain separate work; this document covers the ResNet-50 trials.

## Implementation checks

`tests/test_resnet_finetune.py` checks partial unfreezing, fixed BN statistics,
fresh initialization versus resume, class-label compatibility, learning-rate
groups, and validation before training without changing the training RNG state.

```powershell
python -m pytest tests -q -p no:cacheprovider --basetemp=outputs/resnet-final-tests
```

Result before commit: **31 passed**. The final full-test evaluation and the
single-image prediction check additionally exercised the real saved checkpoint
and cached dataset on the GPU.
