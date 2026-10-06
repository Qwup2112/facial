# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this whole file before writing any code. If any item in Section 13 is still unconfirmed, run `src/data/inspect.py` and ask the user instead of guessing.

---

## 1. Project overview

- **Topic:** Facial Emotion Recognition (FER) from still images, using the team's own dataset in `dataset/` (7 emotion classes, 53,109 images of 48×48 grayscale faces; see Section 3).
- **Course:** Advanced Deep Learning, Faculty of Electrical and Electronic Engineering (EEE), Phenikaa University.
- **Team:** 2 members. The project has two phases:
  - **Midterm:** train a model **from scratch**; no real-time requirement.
  - **Final:** improve accuracy, compare several models covered in the course, and **run as a real-time application** (webcam).
- **Hardware:** everything runs on a personal computer, no GPU cluster. Code must run on a consumer GPU and still run (more slowly) on CPU.
- **Principle:** prefer simple, well-established methods that are easy to explain in the report over complex custom architectures.

---

## 2. The three models

| ID | Model | Training | Phase | Syllabus link |
|---|---|---|---|---|
| `cnn_scratch` | EmotionCNN (custom) | From scratch | Midterm | 2.1 (Dropout, BatchNorm, dilated conv), 2.3–2.5 |
| `resnet18` | ResNet18, ImageNet-pretrained | Two-stage fine-tuning | Final | 2.2 (ResNet), transfer learning |
| `mobilenet_v2` | MobileNetV2, ImageNet-pretrained | Two-stage fine-tuning | Final, real-time demo model | 2.1 (depthwise separable conv) |

Rationale: with ~3,400 unique training images per class (23,856 in total, 47,712 with the offline-augmented copies), training from scratch is viable, so `cnn_scratch` is a real competitor rather than a weak baseline. ImageNet-pretrained models still usually converge faster and reach higher accuracy, which makes "custom vs. transfer learning" a meaningful comparison for the report. ResNet50 and larger are excluded because of training time on a personal computer, not because of dataset size.

### 2.1 `cnn_scratch` — EmotionCNN

Input: 1-channel grayscale, 96×96 (upsampled from the 48×48 source images).

```
Block 1: [Conv3x3(32)  -> BN -> ReLU] x2 -> MaxPool2 -> Dropout2d(0.10)   # 96 -> 48
Block 2: [Conv3x3(64)  -> BN -> ReLU] x2 -> MaxPool2 -> Dropout2d(0.15)   # 48 -> 24
Block 3: [Conv3x3(128) -> BN -> ReLU] x2 -> MaxPool2 -> Dropout2d(0.20)   # 24 -> 12
Block 4: [Conv3x3(256, dilation=2, padding=2) -> BN -> ReLU] x2 -> Dropout2d(0.25)  # 12 x 12
Head:    GlobalAvgPool -> Dropout(0.5) -> Linear(256, 7)
```

- Channels shown are for `width: 32` (32-64-128-256), about 1.2 M parameters. If the model underfits (train and val accuracy plateau at a similar, low level), set `width: 64` (64-128-256-512, about 4.7 M parameters); the head then becomes `Linear(512, 7)`.
- Kaiming (He) initialization for conv layers.
- Block 4 uses dilated convolutions to enlarge the receptive field without reducing resolution (syllabus 2.1).
- Conv layers use `bias=False` when followed by BatchNorm.

### 2.2 `resnet18`

- `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.
- Replace `fc` with `Dropout(0.5) -> Linear(512, 7)`.
- **Stage A (head warm-up):** freeze the whole backbone, train only `fc`; 3 epochs, lr 1e-3.
- **Stage B (fine-tuning):** unfreeze `layer3`, `layer4`, `fc` with layer-wise learning rates: `layer3` 5e-5, `layer4` 1e-4, `fc` 5e-4; 20 epochs. Stem, `layer1`, `layer2` stay frozen.
- **Tuned on val (seed 42, split before the de-duplication in Section 3.1):** the first version (dropout 0.3, weight decay 1e-4) overfit (train macro-F1 0.95 vs. val 0.69). AdamW's decoupled weight decay is scaled by the learning rate, so 1e-4 barely regularized; the final config uses weight decay 5e-2 and dropout 0.5. Val macro-F1 stayed the same (0.6925 → 0.6932), and v2 was kept as the better-regularized recipe shared with `mobilenet_v2`. See `reports/tuning_log.md`.
- Keep BatchNorm layers in frozen blocks in `eval()` mode so their ImageNet running statistics are preserved.
- Grad-CAM target layer: `model.layer4[-1]`.

### 2.3 `mobilenet_v2`

- `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)`.
- Replace `classifier` with `Dropout(0.5) -> Linear(1280, 7)`.
- **Stage A:** freeze `features`, train only `classifier`; 3 epochs, lr 1e-3.
- **Stage B:** unfreeze `features[7:]` (the 14×14 and 7×7 stages, including the final 1×1 conv) with layer-wise learning rates: `features[7:14]` 5e-5, `features[14:]` 1e-4, `classifier` 5e-4; 20 epochs. `features[:7]` stays frozen.
- **Tuned on val (seed 42, split before the de-duplication in Section 3.1):** the first version (only `features[14:]`, dropout 0.3, weight decay 1e-4) reached val macro-F1 0.665, below `cnn_scratch` (0.674). The final config (above, dropout 0.5, weight decay 5e-2) reaches 0.679. See `reports/tuning_log.md`.
- Grad-CAM target layer: `model.features[-1]`.
- Default model for the webcam demo (the lighter of the two pretrained models).

---

## 3. Data

### 3.1 Dataset facts (verified 2026-10-04, after de-duplication)

- **7 emotion classes:** angry, disgust, fear, happy, neutral, sad, surprise. Always read class names from the folder names (sorted alphabetically) and assert there are exactly 7.
- **53,109 images, all 48×48 grayscale PNG, faces already cropped** (FER2013-style).
- **Already split**, and every split is exactly balanced:

| Split | Images | Per class | Content |
|---|---|---|---|
| train | 47,712 | 6,816 | 23,856 originals + 23,856 offline-augmented copies (`source == augmented`, exactly one per original) |
| val | 4,039 | 577 | originals only |
| test | 1,358 | 194 | originals only |

- Over the 29,253 original images the split is about 82 / 14 / 5. The augmented copies exist only in train.
- Sources (`source` column of `manifest.csv`): local 16,136, RAF-DB 9,984 and AffectNet 3,133 originals (AffectNet only for angry, disgust and fear). Per split: `reports/data_report.md`.
- **De-duplication (2026-10-04, approved by the user):** `src/data/inspect.py` found val/test images that are near copies of train images (same photo, other crop or brightness, so the MD5 differs); on the old split the seed-42 `cnn_scratch` checkpoint scored macro-F1 0.673 on val vs. 0.663 on the cleaned val split. A pair is a copy when the dHash Hamming distance is ≤ 1, or ≤ 4 with a pixel correlation (NCC, best over ±3 px shifts and a flip) ≥ 0.84. `src/data/dedup_split.py` **moved** (never deleted) the val/test member of each pair (the val image of a test–val pair) to `dataset_removed/<split>/<class>/`, then trimmed val and test at random (seed 0) so every class keeps the same count: 567 files (test 58 copies + 75 trimmed, val 218 copies + 216 trimmed). Each moved file and its reason is in `dataset_removed/removed.csv`; the original manifest is `dataset_removed/manifest_before_dedup.csv`. Since then there are no exact, mirrored or near copies across splits. To undo: move the files back to `dataset/<split>/<class>/`, restore `manifest.csv` from the backup and re-run `src.data.index_splits`.
- Runs trained before the de-duplication are kept in `runs/_leaky_split/` and are not used for results.

The raw data lives in `dataset/` (not `data/raw/`). Apart from the de-duplication above, never modify it:

```
dataset/
├── manifest.csv     # file,label,source,origin,md5,dhash,status,group,split
├── train/
│   ├── angry/
│   └── ...          # 7 class folders
├── val/
│   └── ...          # same 7 class folders
└── test/
    └── ...          # same 7 class folders
```

### 3.2 Use the existing split — do not re-split

- `src/data/index_splits.py` walks `dataset/{train,val,test}/` and writes `data/splits/{train,val,test}.csv` with columns `path,label,label_idx,source` (`source` from `manifest.csv`). All other scripts read only these CSVs.
- `data.use_offline_aug: false` drops the `augmented` rows from train (originals only).
- Never re-shuffle or re-split the data during training.
- **Model selection uses val only.** Hyperparameters, epoch count and checkpoint choice are decided on val (4,039 images, 577 per class). The test set is touched once per final model, in `evaluate.py`.

**Test set size:** 1,358 images (194 per class). At ~70% accuracy the 95% CI of the accuracy is about ±2.4 percentage points, and per-class F1 is noisier. Report mean ± std over 3 seeds and a bootstrap 95% confidence interval for accuracy and macro-F1 (see Section 5), because the gap between models may be only 1–2 percentage points.

### 3.3 Data inspection

`src/data/inspect.py` prints and saves to `reports/data_report.md`:
- Images per class and per source **for each split**, and the imbalance ratio (max/min) on train.
- Image sizes (min / median / max), number of color vs. grayscale images.
- Corrupt or unreadable files.
- **Leakage check:** exact (pixel MD5), mirrored (flip-invariant dHash) and near copies (dHash distance + NCC) across splits → `reports/near_duplicates.csv`, `reports/figures/near_duplicates.png`.
- Class distribution chart per split: `reports/figures/class_distribution.png`.

If duplicates are found across splits, list them and ask the user how to proceed. Do not move or delete files automatically.

### 3.4 Face cropping

Not needed for training: every image is already a cropped 48×48 face, so there is no `crop_faces.py`. The webcam demo (Section 7) must reproduce this look: a square crop around the detected face, converted to grayscale and resized to 48×48, then the model's val transform and `resize_batch` (Section 3.5). Tune the crop margin by saving a few demo crops and comparing them with training images.

### 3.5 Preprocessing and augmentation

Use `torchvision.transforms.v2`.

| | `cnn_scratch` | `resnet18`, `mobilenet_v2` |
|---|---|---|
| Channels | 1 (grayscale) | 3 (grayscale replicated to 3 channels) |
| Input size | 96×96 | 224×224 (112×112 if CPU-only) |
| Normalization | mean 0.5, std 0.5 | ImageNet mean/std |

The transforms below run on the CPU at the native 48×48 (`data.source_size`), about 5× cheaper than at 224×224; `resize_batch` (`src/data/transforms.py`) then upsamples each batch to the model input size on the GPU. Evaluation, Grad-CAM and the demo must use the same two steps. All augmentation parameters live in the `augment:` block of `configs/base.yaml` (`augment.enabled: false` for the no-augmentation ablation).

All images are converted to grayscale (`grayscale: true` in config) to reduce sensitivity to lighting and skin tone; this can be disabled as an experiment.

**Train:**
1. `Grayscale`
2. `RandomResizedCrop(source_size, scale=(0.85, 1.0), ratio=(0.9, 1.1))`
3. `RandomHorizontalFlip(p=0.5)`
4. `RandomRotation(degrees=10)`
5. `ColorJitter(brightness=0.3, contrast=0.3)`
6. `ToImage` → `ToDtype(float32, scale=True)` → `Normalize`
7. `RandomErasing(p=0.25, scale=(0.02, 0.1))`

**Val / test:** `Grayscale` → `Resize((source_size, source_size))` → `ToImage` → `ToDtype` → `Normalize`, then `resize_batch` on the GPU.

**Do not use:** vertical flips, rotations above 15°, CutMix (it can cut out the eyes or mouth and make the label wrong).

The images are small (48×48 grayscale, ~125 MB decoded for all 53,109), so RAM caching is on (`cache_in_memory: true`).

**Data loading on Windows:** keep `num_workers: 0`. Every DataLoader worker process loads the CUDA DLLs of torch (~1.4 GB of commit memory each); with several workers this exhausted the paging file of the training laptop (`WinError 1455`) and the DataLoader hung. Instead, `PrefetchLoader` (`src/data/dataset.py`) prepares the next batches in one background thread while the GPU trains. Raise `num_workers` only on a machine with plenty of free commit memory.

---

## 4. Training

A single training loop (`src/train.py`) serves all three models and is driven by config files. Pretrained models run the stages declared in their config in order.

| Component | Default | Notes |
|---|---|---|
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` | Add class weights when the train imbalance ratio > 3 |
| Class weights | w_c ∝ 1/√n_c (train counts), normalized to mean 1 | Train is exactly balanced (6,816 per class), so `auto` leaves this off. Never combine with `WeightedRandomSampler` |
| Focal loss | Off (γ = 2 when enabled) | Try only if minority-class F1 stays low |
| Optimizer | AdamW (fused kernel on CUDA) | No weight decay on biases and BatchNorm parameters |
| Weight decay | 5e-4 (scratch), 5e-2 (pretrained) | AdamW decay is scaled by the lr; 1e-4 barely regularized the pretrained models (tuned on val, `reports/tuning_log.md`). `cnn_scratch` kept 5e-4: no gain needed, and its seeds 123/2024 were already scheduled |
| Learning rate | 1e-3 (scratch); per stage (Sections 2.2, 2.3) | |
| Scheduler | 1-epoch linear warmup → cosine decay to 1e-6 | Stepped per iteration, reset at each stage |
| Batch size | 64 (scratch), 32 (pretrained) | Raise to 64 if VRAM allows; reduce if out of memory |
| Epochs | 40 (scratch); 3 + 20 (pretrained) | One epoch is ~1,490 iterations at batch 32 (47,712 images), so fewer epochs are needed |
| Early stopping | On val macro-F1, patience 6 | Applies to scratch and Stage B |
| Checkpoints | `best.pt` (best val macro-F1) and `last.pt` | |
| Mixed precision | On when CUDA is available (`torch.amp`) | Disabled automatically on CPU |
| Gradient clipping | `max_norm = 5.0` | |
| Seeds | 42, 123, 2024 | Report mean ± std |

Models are selected by **macro-F1** so that no single emotion is neglected, even though the dataset is roughly balanced.

**Compute budget.** One epoch covers 47,712 training images (23,856 with `use_offline_aug: false`), so training time matters on a personal computer:
- **With a CUDA GPU:** use the defaults (224×224 for pretrained models, AMP on). Benchmarked on the RTX 3050 Laptop: about 45 / 65 / 75 s per epoch for `cnn_scratch` / `resnet18` / `mobilenet_v2`, i.e. ~25–30 min per run and ~4 h for 3 seeds × 3 models (early stopping can shorten this).
- **CPU only:** set `img_size: 112` for the pretrained models (about 4× cheaper than 224), keep `cnn_scratch` at 96×96, and run 1 seed per model first; add the remaining seeds only if time allows.
- Log seconds per epoch on the first run. Before launching 3 seeds × 3 models, estimate the total training time from that number and tell the user.

**Per-run directory:** `runs/{model}_s{seed}_{YYYYmmdd-HHMM}/` containing:
- `config.yaml` (full merged config)
- `log.csv` (per epoch: train/val loss, accuracy, macro-F1, lr, epoch time in seconds)
- `best.pt`, `last.pt`
- `curves.png` (loss and macro-F1 vs. epoch)
- `test_metrics.json` (written by `evaluate.py`)

**Sanity checks before a full run:**
- `--dry-run`: 2 train batches and 2 val batches to verify the pipeline end to end.
- `--overfit-batch`: train repeatedly on one batch; train accuracy must approach 100%. Re-run whenever the model or loss changes.

---

## 5. Evaluation

`src/evaluate.py` runs on the test split with `best.pt` and reports:
- Accuracy, macro-F1, weighted-F1.
- Per-class precision / recall / F1 (`sklearn.metrics.classification_report`).
- **Bootstrap 95% CI** for accuracy and macro-F1: 1,000 resamples of the test set with replacement, report the 2.5th and 97.5th percentiles.
- Row-normalized confusion matrix → `confusion_matrix.png` in the run directory.
- Parameter count and checkpoint file size.
- Mean inference time (ms/image, batch size 1) on CPU, and GPU if available: 20 warm-up passes, 200 timed passes.
- Horizontal-flip TTA is optional and reported separately, never mixed into the main results.

`src/aggregate.py` collects every `test_metrics.json` into `reports/results_summary.csv` and a markdown table:

| Model | Params | Accuracy (%) | Macro-F1 | ms/image (CPU) |
|---|---|---|---|---|
| cnn_scratch | | mean ± std | mean ± std | |
| resnet18 | | | | |
| mobilenet_v2 | | | | |

**Optional ablations** (only if time allows; one seed each is enough):
- With vs. without augmentation.
- Stage A only vs. full two-stage fine-tuning.
- **Data-size learning curve:** train on a stratified 10% / 25% / 50% / 100% of the train split, evaluate on the same val/test, and plot macro-F1 vs. training-set size for each model. This shows how much each model benefits from more data, and is only possible because the dataset is large.

---

## 6. Explainable AI — Grad-CAM

- Implemented from scratch in `src/xai/gradcam.py` using forward and backward hooks (no external Grad-CAM library), so the method can be explained in the report (syllabus Chapter 3).
- Formulation: for class c and feature maps A^k of the target layer,
  - α_k^c = (1/Z) · Σ_i Σ_j ∂y^c / ∂A^k_ij
  - L^c = ReLU(Σ_k α_k^c · A^k), upsampled to the input size and normalized to [0, 1].
- Target layers:

| Model | Target layer |
|---|---|
| `cnn_scratch` | Last conv layer of Block 4 |
| `resnet18` | `model.layer4[-1]` |
| `mobilenet_v2` | `model.features[-1]` |

- Output: a grid per class with 4 correct and 4 misclassified test images, heatmap overlaid → `reports/figures/gradcam_{model}.png`.
- Qualitative check: does the model attend to eyes, eyebrows and mouth, or to background, hair or watermarks (a sign of shortcut learning)?

---

## 7. Real-time demo (Final, required)

`src/demo/webcam.py`:
1. Read frames from the webcam with OpenCV (`--video path.mp4` runs on a video file, useful for recording the presentation).
2. Face detection: default is the Haar cascade bundled with OpenCV (`cv2.data.haarcascades`). Optional upgrade to YuNet (`cv2.FaceDetectorYN`, requires downloading its ONNX model) if Haar misses too many faces.
3. Crop as described in Section 3.4 (square face crop → grayscale → 48×48), apply the model's val transform, then `resize_batch`.
4. Classify with `mobilenet_v2` (default) or any model selected via `--run`.
5. Smooth class probabilities over time with an EMA (α = 0.6) so the label does not flicker.
6. Draw the bounding box, label, confidence and FPS. Target ≥ 15 FPS on CPU.
7. Press `q` to quit, `s` to save a screenshot to `reports/demo/`.

---

## 8. Repository layout

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── cnn_scratch.yaml
│   ├── resnet18.yaml
│   └── mobilenet_v2.yaml
├── dataset/              # original data (train/val/test + manifest.csv), NEVER modify
├── dataset_removed/      # val/test copies moved out by dedup_split.py, removed.csv, manifest backup
├── data/
│   └── splits/           # train.csv, val.csv, test.csv
├── src/
│   ├── data/
│   │   ├── inspect.py
│   │   ├── dedup_split.py    # moves cross-split copies to dataset_removed/
│   │   ├── index_splits.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py       # build_model(name, num_classes, cfg)
│   │   ├── cnn_scratch.py
│   │   └── pretrained.py     # resnet18, mobilenet_v2, freeze/unfreeze helpers
│   ├── utils/
│   │   ├── config.py         # load YAML, merge with base.yaml
│   │   ├── seed.py
│   │   ├── metrics.py
│   │   └── logger.py
│   ├── train.py
│   ├── evaluate.py
│   ├── aggregate.py
│   ├── predict.py        # labeled image grids + per-image predictions CSV
│   ├── xai/
│   │   └── gradcam.py
│   └── demo/
│       └── webcam.py
├── runs/                 # checkpoints and logs (add to .gitignore)
└── reports/
    ├── figures/          # incl. predictions_{model}_s{seed}_{split}.png
    ├── tuning_log.md     # val-only tuning history (v1 -> v2)
    └── results_summary.csv
```

---

## 9. Configs

The current files in `configs/`:

```yaml
# configs/base.yaml
runs_dir: runs
data:
  root: dataset                 # real data folder (train/ val/ test/ manifest.csv); never modify
  splits_dir: data/splits       # written by src.data.index_splits
  num_classes: 7
  source_size: 48               # images are 48x48; CPU transforms run at this size, the GPU upsamples to img_size
  grayscale: true
  use_offline_aug: true         # false: drop train rows with source == augmented
  num_workers: 0                # 0 = one background thread; each worker process costs ~1.4 GB commit on Windows
  cache_in_memory: true         # images are 48x48 grayscale (~125 MB), see CLAUDE.md 3.5
augment:                        # train-time augmentation, CLAUDE.md 3.5
  enabled: true                 # false: train with the val transform (ablation)
  crop_scale: [0.85, 1.0]       # RandomResizedCrop
  crop_ratio: [0.9, 1.1]
  hflip: 0.5
  rotation: 10                  # degrees, never above 15
  brightness: 0.3
  contrast: 0.3
  erasing_p: 0.25
  erasing_scale: [0.02, 0.1]
train:
  optimizer: adamw
  label_smoothing: 0.1
  class_weight: auto            # auto | none — auto: on when train imbalance ratio > class_weight_ratio
  class_weight_ratio: 3.0
  focal_loss: false
  focal_gamma: 2.0
  warmup_epochs: 1
  min_lr: 1.0e-6
  grad_clip: 5.0
  amp: true
  early_stop_patience: 6
  monitor: val_macro_f1
eval:                           # src.evaluate reads this block from this file, not from the checkpoint
  bootstrap_samples: 1000
  bootstrap_seed: 0             # same resamples for every model, so the CIs compare like with like
  timing_warmup: 20             # inference time (batch size 1): warm-up passes ...
  timing_runs: 200              # ... then timed passes
seeds: [42, 123, 2024]
```

```yaml
# configs/cnn_scratch.yaml
_base_: base.yaml
model:
  name: cnn_scratch
  width: 32                     # 64 if underfitting
data:
  img_size: 96                  # upsampled from 48x48
  in_channels: 1
  normalize: half               # mean 0.5, std 0.5
train:
  batch_size: 64
  weight_decay: 5.0e-4
  stages:
    - name: scratch
      epochs: 40
      trainable: [all]
      lr: {all: 1.0e-3}
```

```yaml
# configs/resnet18.yaml
_base_: base.yaml
model:
  name: resnet18
  pretrained: true
  dropout: 0.5                  # was 0.3; run resnet18_s42_20261004-1344 overfit (train F1 0.95 vs val 0.69)
data:
  img_size: 224                 # upsampled from 48x48; 112 if CPU-only
  in_channels: 3
  normalize: imagenet
train:
  batch_size: 32                # 64 if VRAM allows (RTX 3050 Laptop: 4 GB)
  weight_decay: 5.0e-2          # was 1e-4: AdamW decay is scaled by lr, so 1e-4 barely regularized
  stages:
    - name: head
      epochs: 3
      trainable: [fc]
      lr: {fc: 1.0e-3}
    - name: finetune
      epochs: 20
      trainable: [layer3, layer4, fc]
      lr: {layer3: 5.0e-5, layer4: 1.0e-4, fc: 5.0e-4}
```

```yaml
# configs/mobilenet_v2.yaml
_base_: base.yaml
model:
  name: mobilenet_v2
  pretrained: true
  dropout: 0.5                  # was 0.3; v1 run mobilenet_v2_s42_20261004-1409 overfit (train F1 0.80 vs val 0.66)
data:
  img_size: 224                 # upsampled from 48x48; 112 if CPU-only
  in_channels: 3
  normalize: imagenet
train:
  batch_size: 32                # 64 if VRAM allows (RTX 3050 Laptop: 4 GB)
  weight_decay: 5.0e-2          # was 1e-4: AdamW decay is scaled by lr, so 1e-4 barely regularized
  stages:
    - name: head
      epochs: 3
      trainable: [classifier]
      lr: {classifier: 1.0e-3}
    - name: finetune
      epochs: 20
      # was features.14: only (v1 val F1 0.665 < cnn_scratch); features[7:] = the 14x14 and 7x7 stages
      trainable: ["features.7:", classifier]
      lr: {"features.14:": 1.0e-4, "features.7:": 5.0e-5, classifier: 5.0e-4}   # first matching prefix wins
```

`trainable` and `lr` refer to module-name prefixes of the model; `all` means every parameter. Any module not listed in `trainable` is frozen during that stage.

---

## 10. Commands

```bash
pip install -r requirements.txt

# Data
python -m src.data.inspect                     # data report + cross-split copy check
python -m src.data.dedup_split --dry-run       # done once (2026-10-04); without --dry-run it moves the copies
python -m src.data.index_splits                # dataset/ -> data/splits/*.csv (paths from configs/base.yaml)

# Training
python -m src.train --config configs/cnn_scratch.yaml --seed 42 --dry-run
python -m src.train --config configs/cnn_scratch.yaml --seed 42 --overfit-batch
python -m src.train --config configs/cnn_scratch.yaml --seed 42
python -m src.train --config configs/resnet18.yaml --seed 42
python -m src.train --config configs/mobilenet_v2.yaml --seed 42

# Evaluation and aggregation
python -m src.evaluate --run runs/resnet18_s42_<time>
python -m src.aggregate --runs runs --out reports/results_summary.csv
python -m src.predict --run runs/resnet18_s42_<time> --split test   # labeled grid + predictions CSV (reporting only)

# Explainability and demo
python -m src.xai.gradcam --run runs/resnet18_s42_<time> --split test
python -m src.demo.webcam --run runs/mobilenet_v2_s42_<time>
```

`requirements.txt`: `torch`, `torchvision>=0.16`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `opencv-python`, `pyyaml`, `tqdm`, `pillow`. Optional: `imagehash`.

---

## 11. Roadmap and task split

**Midterm**
- [ ] `inspect.py` (including the cross-split leakage check)
- [x] `index_splits.py` (no `crop_faces.py` needed: faces are already cropped)
- [x] `dataset.py`, `transforms.py`, shared training loop (`--dry-run` and `--overfit-batch` pass for all 3 models)
- [x] `predict.py`: labeled test-image grids for seed 42 (`reports/figures/predictions_{model}_s42_test.png`)
- [ ] `cnn_scratch`: train 3 seeds, evaluate, training curves, confusion matrix
- [ ] (Optional) Grad-CAM for `cnn_scratch`

**Final**
- [ ] `resnet18`, `mobilenet_v2`: two-stage fine-tuning, 3 seeds each (configs tuned on val → v2; seed 42 done)
- [ ] Comparison table of the 3 models (accuracy, macro-F1, params, ms/image, 95% CI)
- [ ] Grad-CAM for all 3 models
- [ ] Real-time webcam demo and recorded presentation
- [ ] (Optional) Ablations

**Suggested split** (two parts with minimal dependencies):
- **Member A:** data (`inspect`, `index_splits`, `dataset`, `transforms`), `cnn_scratch`, `evaluate`, `aggregate`.
- **Member B:** `train.py` (stage mechanism), `pretrained.py` (`resnet18`, `mobilenet_v2`), Grad-CAM, webcam demo.
- **Shared interfaces to agree on early:** the `data/splits/*.csv` format and `build_model(name, num_classes, cfg)`.

---

## 12. Rules for Claude in this repository

- Never delete anything under `dataset/` (the raw data) or `dataset.zip`. The only change made to `dataset/` is the de-duplication by `src/data/dedup_split.py` (Section 3.1: files moved to `dataset_removed/`, logged); any further change needs the user's explicit approval.
- Never re-split the data; use the existing 80 / 15 / 5 split via `data/splits/*.csv`.
- No hard-coded paths or hyperparameters; everything lives in `configs/`. Read class names from folders and assert `num_classes == 7`.
- Never use the test set to choose hyperparameters, epochs or checkpoints.
- Always seed `random`, `numpy`, `torch` and `torch.cuda`, and save the full config into the run directory.
- Run `--dry-run` before every full training run; run `--overfit-batch` whenever the model or loss changes.
- Code must run on Windows (wrap scripts that use DataLoaders in `if __name__ == "__main__":`; keep `num_workers: 0`, see Section 3.5) and on CPU when CUDA is unavailable.
- Comments in English, short; docstrings for public functions.
- Do not add heavy dependencies (`timm`, `pytorch-lightning`, `mmcv`, ...) without asking the user.
- When reporting results, always include mean ± std across seeds, macro-F1, the bootstrap CI and the confusion matrix, not accuracy alone.
- Report figures go to `reports/figures/` as 300 dpi PNG, English axis labels, Times New Roman font (serif fallback) to match the report format.

---

## 13. Confirmed facts

| Question | Answer (checked 2026-10-04) | Affects |
|---|---|---|
| Number of classes | 7: angry, disgust, fear, happy, neutral, sad, surprise | everything |
| Train/val/test split | Fixed in `dataset/`: 47,712 / 4,039 / 1,358 after de-duplication (was 47,712 / 4,473 / 1,491) | `index_splits.py` |
| Cross-split copies | None after `dedup_split.py`: 567 val/test files moved to `dataset_removed/` | leakage; runs before it are in `runs/_leaky_split/` |
| Folder layout | `dataset/<split>/<class>/*.png` + `dataset/manifest.csv` | `index_splits.py` |
| Images per class in train | 6,816 (3,408 originals + 3,408 offline-augmented), exactly balanced | class weights and focal loss stay off |
| Faces already cropped? | Yes | no `crop_faces.py` |
| Image size and color | 48×48 grayscale | `source_size: 48`, `grayscale: true`, `cache_in_memory: true` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM, CUDA available; 16 GB RAM | batch 64 / 32, AMP on, `num_workers: 0` |

Update this table and the related values in `configs/` if the data or the machine changes.
