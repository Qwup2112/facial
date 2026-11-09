# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this whole file before writing any code. If any item in Section 13 is still unconfirmed, run `src/data/inspect.py` and ask the user instead of guessing.

---

## 1. Project overview

- **Topic:** Facial Emotion Recognition (FER) from still images, using the team's own dataset in `dataset/` (7 emotion classes, 53,088 images of 48×48 grayscale faces; see Section 3).
- **Current plan (2026-10-21):** train each model on CK+ first, then transfer it and fine-tune on the target dataset (Section 3.6). The target dataset is being built from Vietnamese TV dramas (VTV, Section 3.7, 2026-11-07); until it is ready the transfer configs point at `data/splits` (the current `dataset/`). The midterm runs and result tables were deleted on 2026-10-21 at the user's request; they are still in git commit `03f3b7c` and in `D:\Project\facial emotion.zip`.
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

### 3.1 Dataset facts (verified 2026-10-21, after de-duplication and the CK+ merge)

- **7 emotion classes:** angry, disgust, fear, happy, neutral, sad, surprise. Always read class names from the folder names (sorted alphabetically) and assert there are exactly 7.
- **53,088 images, all 48×48 grayscale PNG, faces already cropped** (FER2013-style).
- **Already split**, and every split is exactly balanced:

| Split | Images | Per class | Content |
|---|---|---|---|
| train | 47,712 | 6,816 | 23,856 originals + 23,856 offline-augmented copies (`source == augmented`, exactly one per original) |
| val | 4,025 | 575 | originals only |
| test | 1,351 | 193 | originals only |

- Over the 29,232 original images the split is about 82 / 14 / 5. The augmented copies exist only in train.
- Sources (`source` column of `manifest.csv`): local 15,198, RAF-DB 9,979, AffectNet 3,133 (only angry, disgust and fear) and CK+ 922 originals (no neutral). CK+ was merged on 2026-10-04 by `src/data/add_external.py`, split by person (train 756 / val 121 / test 45 images); each CK+ image replaced a random `local` image of the same class and split, which was moved to `dataset_removed/`. Per split: `reports/data_report.md`.
- **De-duplication (2026-10-04, approved by the user):** `src/data/inspect.py` found val/test images that are near copies of train images (same photo, other crop or brightness, so the MD5 differs); on the old split the seed-42 `cnn_scratch` checkpoint scored macro-F1 0.673 on val vs. 0.663 on the cleaned val split. A pair is a copy when the dHash Hamming distance is ≤ 1, or ≤ 4 with a pixel correlation (NCC, best over ±3 px shifts and a flip) ≥ 0.84. `src/data/dedup_split.py` **moved** (never deleted) the val/test member of each pair (the val image of a test–val pair) to `dataset_removed/<split>/<class>/`, then trimmed val and test at random (seed 0) so every class keeps the same count: 567 files (test 58 copies + 75 trimmed, val 218 copies + 216 trimmed). Each moved file and its reason is in `dataset_removed/removed.csv`; the original manifest is `dataset_removed/manifest_before_dedup.csv`. Since then there are no exact, mirrored or near copies across splits. To undo: move the files back to `dataset/<split>/<class>/`, restore `manifest.csv` from the backup and re-run `src.data.index_splits`.

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
- **Model selection uses val only.** Hyperparameters, epoch count and checkpoint choice are decided on val (4,025 images, 575 per class). The test set is touched once per final model, in `evaluate.py`.

**Test set size:** 1,351 images (193 per class). At ~70% accuracy the 95% CI of the accuracy is about ±2.4 percentage points, and per-class F1 is noisier. Report mean ± std over 3 seeds and a bootstrap 95% confidence interval for accuracy and macro-F1 (see Section 5), because the gap between models may be only 1–2 percentage points.

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

The images are small (48×48 grayscale, ~125 MB decoded for all 53,088), so RAM caching is on (`cache_in_memory: true`).

**Data loading on Windows:** keep `num_workers: 0`. Every DataLoader worker process loads the CUDA DLLs of torch (~1.4 GB of commit memory each); with several workers this exhausted the paging file of the training laptop (`WinError 1455`) and the DataLoader hung. Instead, `PrefetchLoader` (`src/data/dataset.py`) prepares the next batches in one background thread while the GPU trains. Raise `num_workers` only on a machine with plenty of free commit memory.

### 3.6 CK+ stage and transfer (Final plan, 2026-10-21)

Each model goes (ImageNet →) **CK+** → **target dataset**.

- **Where it runs:** on **Google Colab** (user's choice, 2026-10-21), with `colab/ckplus_colab.ipynb`. `python colab/make_bundle.py` zips `src/`, `configs/`, `requirements.txt` and `data/ckplus/` into `colab/upload/facial_emotion_colab.zip` (gitignored: it holds the CK+ images); upload it to `MyDrive/facial_emotion/`. The notebook unzips it to `/content/facial`, runs sections 1–4 (sanity checks, transfer checkpoints, CV, summary) and copies `runs/` and `reports/` to `MyDrive/facial_emotion/` after every block (without `last.pt` and the CV fold checkpoints); a rerun skips finished work. Section 5 (transfer to the target dataset) is off until that dataset is uploaded. Rebuild the bundle after every code or config change. A few CK+ runs were trained on the laptop before the switch (`runs/ckplus/`, `runs/ckplus_cv/` on the PC); the reported results come from Colab.

- **CK+ data:** `src/data/ckplus_folds.py` reads `data/external/ckplus/*.parquet` (48×48 grayscale, the last 3 frames of the labelled sequences): 927 images, 106 people, 309 sequences, **6 classes** (CK+ has no neutral; contempt is dropped): angry 135, disgust 177, fear 75, happy 207, sad 84, surprise 249. It writes `data/ckplus/images/`, `data/ckplus/index.csv` and the split CSVs; `dataset/` is only read.
- **Evidence of accuracy — subject-independent 10-fold CV** (`src/ckplus_cv.py`): folds from StratifiedGroupKFold over people (seed 0). Fold k: test = fold k, val = fold k+1 (best.pt), train = the other 8 folds. Each fold model labels its test fold once; pooled over the folds every image is tested once by a model that never saw that person. 3 models × 3 seeds × 10 folds = 90 runs in `runs/ckplus_cv/` (~3 h on the RTX 3050). Output: `reports/ckplus_cv.md`, `ckplus_cv_summary.csv`, `ckplus_cv_runs.csv`, `reports/figures/ckplus_cv_confusion_<model>.png`.
- **Transfer checkpoints** (`data/ckplus/splits/transfer/`): CK+ people keep the split they have in `dataset/` (train 84 people / 756 images, val 16 / 126, test 6 / 45), so a CK+ model never saw a val or test person of `dataset/`. Runs: `runs/ckplus/{model}_s{seed}_*`, each evaluated once on its 45 test images.
- **CK+ configs** (`configs/ckplus/`): `num_classes: 6`, batch 32 (~24 steps per epoch), `class_weight_ratio: 2.0` (CK+ max/min ≈ 3.3, so class weights are on in every fold). `cnn_scratch`: 100 epochs at lr 1e-3. `resnet18`: head 5 + finetune 40 epochs with the layer-wise lrs of `configs/resnet18.yaml`. `mobilenet_v2`: head 5 + finetune 40 epochs with 5× the lrs of `configs/mobilenet_v2.yaml` (CK+ gives ~35× fewer steps; val macro-F1 0.872 → 0.905 over 2 seeds, `reports/tuning_log.md`). The val people of the transfer split are test people of other CV folds, so this one choice makes the `mobilenet_v2` CV estimate slightly optimistic.
- **Transfer** (`configs/transfer/`, `python -m src.train --init <ckplus run>`): `load_init` copies every tensor whose name and shape match; only the classifier head (6 → 7 classes) starts fresh. `resnet18` and `mobilenet_v2` keep the two stages of their main configs, so only the starting weights differ from ImageNet-only training. `cnn_scratch` gets a 3-epoch head stage, then the whole network at lr 5e-4 (untuned; tune on val). `data.splits_dir` is the target dataset: point it at the new dataset's split CSVs once it exists.
- **Leakage:** when the target dataset contains CK+ people (as `dataset/` does), start from the transfer checkpoints, never from the CV fold models. If the new dataset has no CK+ images, re-run `src.data.ckplus_folds`: the transfer split then becomes val = fold 0, train = the other folds. Note that some `local` images are near copies of CK+ frames (`dataset_removed/removed.csv`: 4 CK+ val frames were copies of `local` train images).
- **Status (2026-11-07):** CV and the 9 transfer checkpoints finished on Colab on 2026-11-03 (pooled CV accuracy / macro-F1: `cnn_scratch` 90.36 ± 1.18 / 0.867, `resnet18` 94.28 ± 0.39 / 0.926, `mobilenet_v2` 93.38 ± 0.53 / 0.903). The results are in `MyDrive/facial_emotion/` and in `D:\Download\drive-download-20261103T085056Z-1-001.zip`; the local `reports/ckplus_cv.md` is still an old empty version.

### 3.7 VTV target dataset (2026-11-07, in progress)

The target dataset is built from **Vietnamese TV dramas** (VTV): Vietnamese faces, spontaneous expressions, TV lighting, unlike CK+ (posed peak expressions, lab lighting, mostly European/American people). All outputs go to `data/vtv/` (gitignored: TV frames). Models used here are public ones in `data/external/models/` (YuNet, FER+, HSEmotion, SFace), **never a model trained in this project** (user's instruction, 2026-11-07).

#### 3.7.1 Status (2026-11-08)

- `data/vtv/review/` holds ~32.7k aligned faces from 14 extraction runs (table in 3.7.2): 23,990 in the 7 class folders, 8,061 in `_unsure/`, 620 in `_reject/`.
- Build of 21:10: **16,550 real faces** (train 11,917 / val 2,292 / test 2,341; 352 people; leak check val 1.8% / test 1.3%) + 614 augmented train copies (disgust +296, fear +105, surprise +213: every class >= 1000 over all splits), `target_dataset.zip` 32.0 MB (md5 14aac9d41773e5e877e7626e7d509029), uploaded to `MyDrive/facial_emotion/` as `target_dataset.zip.part00`-`part03` (9 MB parts, browser limit 10 MB).
- **Transfer on the suggested labels** (user's decision, 2026-11-07, 21:00: train now, review later): Colab notebook `ckplus_colab_vtv.ipynb` on Drive (= `colab/ckplus_colab.ipynb`, section 5 with `TAG = 'vtv_suggested'`), runs in `runs/transfer_vtv_suggested/` (on Drive), summary `reports/transfer_vtv_suggested_summary.md`. Its test numbers are **agreement with the suggestion labels**, not accuracy. All 9 runs done 2026-11-07 23:10 (mean ± std over 3 seeds): resnet18 83.64 ± 0.66% / macro-F1 0.7654 ± 0.0034 (best), cnn_scratch 80.29 ± 1.73% / 0.7314 ± 0.0270, mobilenet_v2 80.21 ± 0.88% / 0.7327 ± 0.0130. Weakest classes (resnet18 s2024): disgust 57% (→ angry 23%), fear 66%, surprise 74% (→ fear 16%), angry 74%.
- **2026-11-08, `vtv_claude`** (user: "lâu quá bạn làm đi cho nhanh": Claude applies its own pre-screen instead of the human review): `python -m src.data.vtv_claude_suggest move` moved the 1,844 flagged faces (rejects → `_reject`, emotion disagreements → `_unsure`; log `data/vtv/claude_suggest_moves.csv`; flagged rejects are measurably blurrier: sharpness median 800 vs 1,276). Rebuild 13:34: **15,030 real faces** (train 10,428 / val 2,404 / test 2,198; 328 people; leak val 0.9% / test 0.5%); label_by: human 30, claude 2,007, ensemble 12,993. Rare classes are now small: real train angry 331, disgust 253, fear 507, surprise 111 (+ augmented copies to 1000; surprise ×7), test surprise only 36 faces. Old build kept as `data/vtv/target_dataset_vtv_suggested.zip`, `report_vtv_suggested.md`, `manifest_vtv_suggested.csv`. Uploaded as `MyDrive/facial_emotion/target_dataset_vtv_claude.zip.part00-03` (md5 57115d0ca28c777d87b6d898bf4b28d2); Colab notebook `ckplus_colab_vtv_claude.ipynb` (TAG `vtv_claude`, runs `runs/transfer_vtv_claude/`) started 13:40. Numbers = agreement with Claude-screened suggestion labels, not accuracy. Colab's GPU quota ran out at 15:10 after 5 runs (cnn_scratch ×3, resnet18 s42/s123); the other 4 (resnet18 s2024, mobilenet_v2 ×3) ran on the laptop RTX 3050, 15:15–16:06, in a copy of the same bundle, dataset zip and CK+ checkpoints (`D:\Claude\vtv-transfer-local`, `run_rest.sh`; their best.pt stay there), and their small files went to Drive as `runs_transfer_vtv_claude_laptop.zip` for the summary (CPU runtime). `src/transfer_compare.py` scores both label versions on the test faces their test sets share (the old models cannot be scored on the new test set: 64% of its faces are of people in the old train set).
- **`vtv_claude` result (2026-11-08 16:30, `reports/transfer_vtv_claude_summary.md`, `reports/transfer_compare_vtv_suggested_vs_vtv_claude.md`):** agreement cnn_scratch 77.31 ± 3.22 / macro-F1 0.614, resnet18 83.64 ± 0.66 / 0.701, mobilenet_v2 81.41 ± 0.85 / 0.663. On the 440 test faces both versions share, every model agrees less after the screen (resnet18 79.7 → 75.8%), with the loss in the screened classes (surprise 63–76 → 26–39%, angry 82–89 → 51–73%, disgust 44–63 → 17–46%). So the screen hurt: it removed too many rare-class faces. `vtv_suggested` stays the main result; restoring the 1,844 moves is proposed to the user (not done). Weak classes need more real faces (FER+), not removal.
- **2026-11-08, VTV + course mix** (user tried the movie demo: "mô hình không nhận diện tốt lắm"; `reports/transfer_mix_summary.md`). `src/eval_wild.py` scores runs and the suggestion ensemble against **human labels** on course faces aligned like the VTV crops: on RAF-DB val+test the vtv_claude resnet18 had 52.8% (mF1 0.379; surprise recall 10%) vs the ensemble's 67.0%. `src/data/mix_build.py` aligns the course dataset's real faces (not `augmented`; `dataset/` untouched) and joins them with the `vtv_suggested` build in `data/mix/` (`course/`, `vtv/dataset/`, `splits/`; train 36,327, rare classes ~4,200 each; splits kept, so RAF-DB/local test stay unseen). `configs/transfer/resnet18_mix.yaml` (runs_dir `runs/transfer_mix`), seed 2024, init = laptop CK+ `resnet18_s2024_20261021-0809`: the run `runs/transfer_mix/resnet18_s2024_20261108-1656` was **killed at epoch 21/23** (Claude Code low-memory reap); best.pt = epoch 21, val mF1 0.710; no `test_metrics.json` (`src.evaluate` refuses unfinished runs). Old → new: RAF-DB test 53.4 → 79.1% (mF1 0.391 → 0.707), local test 41.6 → 60.7% (0.387 → 0.602), 440 shared VTV test faces 77.3 → 78.6% agreement (0.701 → 0.739). RAF-DB/local gains are partly in-distribution.
- **2026-11-09, mix × 3 seeds** (user: "train lại E đủ 3 seed"). Seeds 42/123/2024, init = the Colab CK+ resnet18 checkpoint of the same seed (s42/s123 extracted from `D:/Download/drive-download-20261103T085056Z-1-001.zip` to `D:/Claude/vtv-mix-3seeds/ckplus/`; s2024 identical to the laptop copy), laptop, all 23/23 epochs, runner `D:/Claude/vtv-mix-3seeds/run_seeds.py`. Runs `runs/transfer_mix/resnet18_s{42_20261109-0212,123_20261109-0241,2024_20261109-0310}` (val mF1 0.722/0.715/0.711). Mean ± std: RAF-DB test 79.4 ± 0.9% / 0.713, local test 60.8 ± 0.9% / 0.610, VTV test (= `vtv_suggested` test, 2,341) 82.44 ± 1.24% / 0.748 agreement vs `vtv_suggested` resnet18 83.64 ± 0.66% / 0.765 (slightly lower; loses surprise/sad/happy, gains angry/disgust). Chosen as the best VTV training (human labels); app model = s42 (best val). The 11/08 first try is outside the means. Reports `transfer_mix_summary.md`, `transfer_mix_results.md`, `wild_check_{rafdb,local}_test_mix3.md`. To do: human-check a VTV test subset, score `vtv_suggested` on human labels (checkpoints on Drive), mobilenet_v2 on the mix.
- Open: (1) human review of `review/` in the review app (`duyet_nhan_vtv.bat` = `python -m src.data.vtv_review_app`, step 1e; rules in `review/HUONG_DAN.md`), weakest classes first (disgust, fear, angry, surprise, then `_unsure`), (2) close the app, rebuild (`vtv_pipeline --build`), (3) transfer again with `TAG = 'vtv_reviewed'`: only that run gives reportable accuracy.
- **The labels are still model suggestions** (ensemble + Claude's agreements, 3.7.3); `report.md` of each build says how many a human changed. Never evaluate on them as if they were checked labels (Section 12).

#### 3.7.2 Sources (`D:\Download\New folder\`, read only)

| Source | Eye gate | Faces in `review/` | csv |
|---|---|---|---|
| Main run: Sóng Ở Đáy Sông t10 (old film, upscaled to 1080p), Làng Trong Phố (360p), Trò Đời tập cuối (480p) + 20,718 frame images of `VTV_Emotion_Dataset_Final/final_dataset/images/` (640×360 / 640×252 trailer frames: `playlist_<n>_<youtube id>_<k>.jpg` from 78 videos, `frame_<k>.jpg`) | 20 px | 7,258 | `faces.csv` |
| Người Phán Xử t1, Quỳnh Búp Bê t1, Táo Quân 2016 (1080p, added 2026-11-07) | 20 px | 1,002 / 3,134 / 11,922 | `faces_<name>.csv` |
| Eye-16 re-run (3.7.3, step 1d): frame images, Làng Trong Phố, Người Phán Xử, Trò Đời tập cuối | 16 px | +142 / +168 / +5 / +371 | `faces_images_eye16.csv`, `faces_eye16_<video>.csv` |
| Trò Đời ep. 01, 03, 06, 09, 12, 15 (YouTube playlist `PLtC-L_zqTr6tqhkdlZKG78KagOkEivmbH`, 1080p, `tro_doi_playlist/`; user: ~6 episodes, the other 9 deleted) | 16 px | 1,047 / 1,151 / 1,408 / 1,920 / 1,382 / 1,761 | `faces_tro_doi_tap_XX.csv` |

The 1080p videos gain almost nothing at eye 16 px (Người Phán Xử +5 faces in 25 min), so the eye-16 re-run skipped Sóng Ở Đáy Sông, Quỳnh Búp Bê and Táo Quân. Not used:
- `annotations.json` (COCO boxes + emotion, checked 2026-11-06): most boxes are not faces (walls, paper, shoulders), clear faces are missing, `confidence` is a constant per class: machine-made labels.
- `VTV_Dataset_Training_Ready.zip` (whole frames + rotated/flipped copies) and its unpacked `image_setdata/` (checked 2026-11-07): 8,746 originals byte-identical to `final_dataset/images/` + 10,454 `*_aug` copies padding each class to 3,200; folder labels agree with the ensemble on 15% of 678 faces ("joy": almost no smiles).

#### 3.7.3 Pipeline

| Step | Module (config block) | Does | Writes |
|---|---|---|---|
| 1 | `vtv_faces` (`configs/data/vtv_faces.yaml`) | faces, CK+ alignment, gates, repeat removal, FER+ suggestion | `review/<class>/<id>.png` (192 px), `faces*.csv`, `summary*.md`, `rejected_examples/` |
| 1b | `vtv_relabel` (`relabel:`) | re-sort `review/` by the FER+ + HSEmotion ensemble | moves PNGs, `e_*`/`ensemble` columns |
| 1c | `vtv_claude_review` | Claude's grid review of `_unsure/` | `claude_review.csv`, `claude_review/` |
| 1d | `vtv_supplement` (`vtv_faces_eye16.yaml`) | add only the new faces of a re-run with other gates | `faces_<tag>.csv` |
| 2 | `vtv_build` (`build:`) | person clusters, split by person, 48×48 dataset, zip | `dataset/`, `target_dataset.zip` |
| 1e | `vtv_review_app` | human review in the browser (127.0.0.1:8765): keys set the class, Enter saves a page | moves PNGs in `review/`, `human_review.csv` |
| 2b | `vtv_balance` (`build.balance`, called by 2) | augmented train copies up to ≥ 1000 per class | `dataset/train/<class>/<id>_aug<k>.png` |
| — | `vtv_pipeline` | runs 1 → 1b → 2 one process at a time (3.7.4) | `runs/logs/vtv_pipeline.log` |

Order: `vtv_faces` → `vtv_relabel` → (`vtv_claude_review`) → **human review** → `vtv_build`. Every step is safe to re-run.

**Step 1, `vtv_faces`.** 10 frames per second of video plus every frame image → YuNet faces + 5 landmarks → similarity alignment of the 5 landmarks onto the **mean landmark layout of the CK+ 48×48 crops** (YuNet on the 927 CK+ images, std 0.9 px) → grayscale. Gates, mostly from the range measured on CK+: YuNet score ≥ 0.80; eye distance ≥ 20 px (no upscaling to 48×48); |yaw| ≤ 0.20 (~20°; 0.15–0.20 checked by eye: both eyes clear); pitch and landmark fit within the CK+ range; brightness; contrast; ≤ 12% of the crop outside the frame; sharpness (Laplacian variance ≥ 200 at 48×48 and blur effect ≤ 0.60 at 96×96, the second catches the upscaled old film); then YuNet again on the aligned crop (score ≥ 0.85, landmarks on the template: removes hands and objects). Repeats dropped per video: within 60 s NCC ≥ 0.85, or ≥ 0.80 with the same suggested label; NCC ≥ 0.90 with any of the last 200 kept faces. Gates were tuned by looking at kept and rejected faces of a 2,106-frame sample (`rejected_examples/<gate>/`: 60 per gate). `--append --only <name> --tag <name>` adds a later source with its own `faces_<tag>.csv` (step 2 reads all `faces*.csv`); the tool refuses to overwrite a `review/` that already has faces. Deterministic: ids are `<group>_<frame>_<k>`, a re-run gives the same faces.

**Label suggestion.** FER+ (`emotion-ferplus-8.onnx`) scores on CK+ happy 100%, surprise 92%, sad 65%, disgust 55%, angry 13%, fear 8%, so it only pre-sorts (`review.prelabel: none` puts every face in `_unlabeled/`).

**Step 1b, `vtv_relabel`.** Average of FER+ and two HSEmotion EfficientNet-B2 models (AffectNet, `data/external/models/hsemotion/`), measured on labelled faces prepared exactly like the VTV crops (48 → 192 → YuNet → CK+ alignment): CK+ acc 0.659 → 0.887 (mF1 0.584 → 0.866), RAF-DB val+test acc 0.548 → 0.670 (mF1 0.413 → 0.576). Top probability ≥ 0.4 → class folder (CK+ 97% of faces at 90% acc, RAF-DB 84% at 74%, VTV 74% sorted; 0.5 left 55% of VTV in `_unsure/` though most looked fine), else `_unsure/`. Faces the user moved (folder ≠ `review_folder`) or rejected are never touched; `review_folder` is updated so step 2 still sees the human changes. Idempotent (faces with `ensemble` are not recomputed); batch 8 with the ORT arena off keeps it at ~0.2 GB.

**Step 1c, `vtv_claude_review`** (user asked Claude to review `_unsure/`). Numbered 8×6 grids, rare-class guesses first, labelled by Claude by eye against the CK+ AU criteria; records in `data/vtv/claude_review/` (`labels.txt`, `grid_index.csv`, calibration), moves in `data/vtv/claude_review.csv`. Calibration on 48 RAF-DB faces aligned like the VTV crops: Claude alone 58% right, ensemble 71%, both agreeing (28/48) 82%, so `apply` moves a face only where Claude's label equals the ensemble guess; `x` = reject (blur, eyes closed, hand/microphone/logo over the face, not a face) is applied as given. Round 1 (grids 1–43 = all 2,064 faces guessed fear/disgust/angry/surprise + the first sad grid): 520 sorted (angry 249, fear 124, disgust 100, surprise 33, sad 14), 620 rejected, 9 disagreements and 915 unclear left. The sad/happy/neutral guesses (~4.6k) were not reviewed.

**Step 1d, `vtv_supplement`.** The 360p frame images gave only 1,910 faces (of 31,269 YuNet faces 40% failed `small`, 24% `score`, 18% `yaw`). A re-run at eye 12 px / yaw 0.28, checked by eye at 48×48: eye 16–20 px looks like the 20–25 px faces (scaled up at most 1.16×), 14–16 px soft and some with the logo, 12–14 px blurred; yaw 0.20–0.28 = clearly turned (3/4 view), unlike CK+. So re-runs use eye ≥ 16 px, yaw 0.20 (`vtv_faces_eye16.yaml`) into a temporary folder; `vtv_supplement` copies into `review/` only the faces whose id is new and that do not repeat a face already there (never replaces a file). New episodes are extracted at eye 16 directly with `--append`.

**Step 2, `vtv_build`.** Reads the 7 class folders of `review/` and `faces*.csv` and groups faces **by person** (SFace embeddings cached in `embeddings.npz`; average-linkage clustering at cosine ≥ 0.25) because actors appear in many videos. Threshold calibrated on VTV: two faces in one frame (different people) median 0.11, max 0.46; same person, same shot median 0.79, other shots 0.54–0.76; at 0.40 (near SFace's default 0.363) actors were split into clusters that landed in train and test (6% of test faces had ≥ 5 train faces at cosine ≥ 0.50, same people by eye), at 0.25 0%. Clustering runs in two stages (inside each linked component, then across components; one linkage over 16.6k faces needed ~2.2 GB and was killed, now < 0.7 GB, ARI 0.89 vs. one stage). `frame_*.jpg` are frames of the playlist videos: sources sharing a face at SFace ≥ 0.80 are linked (split together) and copies (SFace ≥ 0.90 and pixel NCC ≥ 0.90) dropped. `max_per_person_class: 100` (without it one Táo Quân performer was 14% of all faces and 86% of test). Split: StratifiedGroupKFold over people, 7 folds (test = fold 0, val = fold 1, train = rest). Writes `dataset/<split>/<class>/<id>.png` (48×48), `manifest.csv` (`person`, `suggested`, `changed_in_review`, `label_by` = human / claude / ensemble), `report.md` (counts, leak check, label sources), `leak_check.png`, `person_check.png` and the zip. Táo Quân costumes (same hat, stage light) give SFace ≥ 0.5 between different performers, so the leak check over-counts there (checked by eye).

**Step 1e, `vtv_review_app`** (2026-11-08). A local page (stdlib HTTP server bound to 127.0.0.1) shows one review folder as a grid that fits the screen, the least certain faces first (probability of the folder's class; `_unsure`: rare-class candidates first as an option). The reviewer fixes the wrong faces with keys 1–7 / U / X and saves the page with Enter: changed faces are moved (like a hand move, never overwritten, renamed or deleted) and **every face on the page, kept or moved, is logged** in `data/vtv/human_review.csv` (`time, batch, action keep|move|undo, id, from, to`; Ctrl+Z appends an `undo` row and moves the page back). Why: a face kept in place looked unchecked on disk. `vtv_build` sets `label_by = human` for logged faces and reports the human-checked share per split; `vtv_relabel` never moves a logged face. It refuses to move while the pipeline lock is held, and the pipeline refuses to start while the app runs (FOREIGN pattern). Ensemble hints are off by default (anchoring); Space zooms with the 48×48 view.

**Claude's pre-screen, `vtv_claude_suggest`** (user's choice 2026-11-08: "Claude lọc trước, tôi xác nhận"). Claude looked at all 90 grids of disgust, fear, surprise, angry (4,196 faces, `data/vtv/claude_suggest/grids/`, its marks in `labels.txt`) and flagged 1,844: angry 754 (723 reject), disgust 267 (201 reject, 50 neutral, 13 happy), fear 190 (170 reject), surprise 633 (407 reject, 193 neutral = talking, brows not raised). `data/vtv/claude_suggest.csv` is only shown in the review page as dashed pending changes "Claude → …" (order "Claude đánh dấu trước", key C toggles); nothing moves until the human saves the page, and the log keeps the shown suggestion in `hint`. Caveats: Claude's eye is weaker than the ensemble (RAF-DB 58% vs 71%), and its blur rejects were stricter than the guide (soft but readable faces are kept), so the human overrules freely; accepting every flag would leave surprise ~234 and angry ~806 faces.

**Step 2b, `vtv_balance`** (user: every emotion ≥ 1000 faces). Classes with fewer faces get offline-augmented copies of their **train** faces (`add_external.augment`: flip, rotation, scale, shift, brightness, contrast), each source used at most ⌈need / n⌉ times so every person keeps its share; `source = augmented`, `origin` = the real file, so `data.use_offline_aug: false` drops them; val/test stay 100% real. Tested on the 16:32 build: angry +136, disgust +407, fear +353, surprise +394.

#### 3.7.4 Running long jobs: `src/data/vtv_pipeline.py`

Long VTV jobs (hours) run unattended, one process at a time (RAM 16 GB; one `vtv_faces` ≈ 1 GB, 25–45 min per 1080p episode at 8 threads). On 2026-11-07 shell-script chains in the scratchpad were killed by Claude Code's low-memory reaper or believed killed and restarted while still alive, so two runs worked on the same episode. The runner fixes this:
- `pipeline.lock` in `data/vtv/`: a second runner exits while the first is alive; any other `src.data.vtv_*` process or old `run_*.sh` chain blocks the start (`--wait` waits for it).
- Resumable: an extraction is done when `data/vtv/<tag>.done` exists; a partial one (csv or PNGs without the marker) is removed and redone (refuses if a face of it was moved by hand). Relabel is idempotent; build is re-run.
- `--detach` starts it as a detached process (not a child of the Claude Code task), log `runs/logs/vtv_pipeline.log`, console output in `vtv_pipeline.out`. **Always run `--status` first**, and never start a step by hand while a runner is alive.

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
2. Face detection: YuNet (`cv2.FaceDetectorYN`, `data/external/models/yunet/`). The installed OpenCV 5.0 has no Haar cascade (`cv2.CascadeClassifier` is gone).
3. Crop as described in Section 3.4 (square face crop → grayscale → 48×48), apply the model's val transform, then `resize_batch`. For a model fine-tuned on VTV, use the CK+ landmark alignment of `src/data/vtv_faces.py` instead (`align`), so demo faces look like its training faces.
4. Classify with `mobilenet_v2` (default) or any model selected via `--run`.
5. Smooth class probabilities over time with an EMA (α = 0.6) so the label does not flicker.
6. Draw the bounding box, label, confidence and FPS. Target ≥ 15 FPS on CPU.
7. Press `q` to quit, `s` to save a screenshot to `reports/demo/`.

**Status (2026-11-08):** written. `nhan_dien_phim.bat` = `python -m src.demo.webcam --pick` (file dialog, opens in `D:/Download/New folder`). Default model: `DEFAULT_RUNS` / `default_run()` in `webcam.py`, the finished (curves.png) `runs/transfer_mix/resnet18_s*` run with the best val macro-F1 (2026-11-09: `resnet18_s42_20261109-0212`, VTV + course mix, Section 3.7.1), else the same rule on `runs/transfer_vtv_*/resnet18_s*`. Faces failing the `vtv_faces` gates (turned, small, blurred, dark…) are drawn grey with the gate name, since the model never saw such faces. Extra options: `--start` s, `--out` annotated mp4, `--csv` per-face rows, `--no-window`, `--max-frames`; arrows/a,d seek ±10 s. The CK+ landmark template is cached in `data/vtv/ckplus_template.npy`. RTX 3050: ~31–35 FPS on 1080p TV (processing only).

**Browser app (2026-11-09, user: "thiết kế 1 app có thể thay file hình tùy ý, có thể tua và thay đổi mốc thời gian tùy ý và đẹp nữa"):** `src/demo/app.py` (Flask on 127.0.0.1, port 7860 or the next free one; opens a Chrome `--app` window, stops when the window closes) + `src/demo/web/index.html` (single file, no external assets). Launchers `app_nhan_dien.bat` (project root) and `tong_hop_ket_qua/3_app/mo_app.bat` (`--run` the s42 copy). Same pipeline as `webcam.py` (imports `Emotion`, `Tracks`, `default_run`). The browser plays the file and POSTs full-resolution JPEG frames to `/api/infer` (gates count pixels, so never downscale), two frames in flight: ~13–17 analysed frames/s on 1080p AV1 TV (JPEG encode ~40 ms + server ~45–50 ms per frame). Features: open/drag any video or image, files of `D:/Download/New folder` (+ `tro_doi_playlist`) served by `/api/file/<i>` (range requests), webcam, timeline seek, jump box (`1:23:45`, `50%`, `+30`, `-1:00`), own marks (B; rename/move/delete), emotion strip on the timeline, whole-film scan (hidden 2nd video, `track=0`, ~0.4 s/frame) with top moments per emotion, click a face to follow it, model switch (`/api/model`, finished 7-class runs), screenshot to `reports/demo/`. Marks, scan results and the last position are kept in the browser's localStorage per file (name + size in MB).

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
│   ├── mobilenet_v2.yaml
│   ├── ckplus/           # the 3 models on CK+ (6 classes)
│   └── transfer/         # CK+ -> target dataset (train with --init)
├── dataset/              # original data (train/val/test + manifest.csv), NEVER modify
├── dataset_removed/      # val/test copies moved out by dedup_split.py, removed.csv, manifest backup
├── data/
│   ├── splits/           # train.csv, val.csv, test.csv
│   └── ckplus/           # images/ (gitignored), index.csv, splits/fold{0-9}/, splits/transfer/
├── src/
│   ├── data/
│   │   ├── inspect.py
│   │   ├── dedup_split.py    # moves cross-split copies to dataset_removed/
│   │   ├── add_external.py   # merged CK+ into dataset/ (2026-10-04), split by person
│   │   ├── ckplus_folds.py   # CK+ stage data: data/ckplus/ (images, folds, transfer split)
│   │   ├── vtv_faces.py      # VTV target data, step 1: aligned faces -> data/vtv/review/ (Section 3.7)
│   │   ├── vtv_review_guide.md   # review rules (Vietnamese), copied to data/vtv/review/HUONG_DAN.md
│   │   ├── vtv_relabel.py    # VTV step 1b: re-sort review/ with FER+ + HSEmotion ensemble
│   │   ├── vtv_claude_review.py # VTV step 1c: Claude's grid review of review/_unsure (agreement with the ensemble only)
│   │   ├── vtv_supplement.py # VTV step 1d: add the new faces of a re-run with other gates (360p images, eye >= 16 px)
│   │   ├── vtv_build.py      # VTV step 2: review/ -> data/vtv/dataset/ (48x48, split by person) + zip
│   │   ├── vtv_claude_suggest.py # VTV step 1e: Claude's pre-screen grids -> claude_suggest.csv (shown in the review page, never applied alone)
│   │   ├── vtv_review_app.py # VTV step 1e: human review page (127.0.0.1:8765) + vtv_review_app.html; log human_review.csv
│   │   ├── vtv_balance.py    # VTV step 2b: augmented train copies so every class has >= 1000 faces
│   │   ├── vtv_pipeline.py   # VTV runner: steps one at a time, lock, resumable, --detach (Section 3.7.4)
│   │   ├── mix_build.py      # VTV build + course real faces aligned like VTV -> data/mix/ (splits for configs/transfer/resnet18_mix.yaml)
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
│   ├── train.py              # --init: start from another run (transfer)
│   ├── ckplus_cv.py          # CK+ subject-independent 10-fold CV + summary
│   ├── evaluate.py
│   ├── aggregate.py
│   ├── predict.py        # labeled image grids + per-image predictions CSV
│   ├── eval_wild.py      # accuracy vs human labels on course faces aligned like VTV (+ suggestion ensemble)
│   ├── transfer_compare.py # two VTV label versions on their shared test faces
│   ├── xai/
│   │   └── gradcam.py
│   └── demo/
│       ├── webcam.py     # webcam or video (--pick dialog, nhan_dien_phim.bat)
│       ├── app.py        # browser app server (app_nhan_dien.bat): open/seek/jump/marks/scan
│       └── web/index.html  # its page (single file)
├── colab/                # ckplus_colab.ipynb, make_bundle.py, upload/ (bundle zip, gitignored)
├── tong_hop_ket_qua/     # all results collected for the user (2026-11-08, gitignored copies): README.md, 1_data, 2_training a-e, 3_app (chay_app.bat, demo clips), models/
├── runs/                 # checkpoints and logs: ckplus/ (transfer checkpoints), ckplus_cv/ (90 fold runs),
│                         # ckplus_tune/ (val-only tuning), transfer/ (fine-tuned on the target), logs/
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

# CK+ stage and transfer (Section 3.6)
python -m src.data.ckplus_folds                                       # data/ckplus/ (re-run if dataset/ changes)
python -m src.train --config configs/ckplus/resnet18.yaml --seed 42   # transfer checkpoint -> runs/ckplus/
python -m src.evaluate --run runs/ckplus/resnet18_s42_<time>          # its 45 CK+ test images
python -m src.ckplus_cv                                               # 10-fold CV, 3 models x 3 seeds, resumable
python -m src.ckplus_cv --summary-only                                # reports/ckplus_cv.md
powershell -ExecutionPolicy Bypass -File runs/logs/ckplus_queue.ps1    # all of the above, in order (local GPU)
python colab/make_bundle.py                                           # Colab: upload colab/upload/*.zip, run colab/ckplus_colab.ipynb
python -m src.train --config configs/transfer/resnet18.yaml --seed 42 --init runs/ckplus/resnet18_s42_<time>

# VTV target dataset (Section 3.7)
python -m src.data.vtv_faces --limit 300 --out data/vtv_test         # quick check of the gates
python -m src.data.vtv_faces                                          # data/vtv/review/ for human review (~1 h)
python -m src.data.vtv_faces --append --only tao_quan --tag tao_quan  # add a video put in the source folder later (prefer vtv_pipeline)
python -m src.data.vtv_relabel                                        # step 1b: ensemble re-sort of review/ (keeps user moves)
python -m src.data.vtv_claude_review apply                           # step 1c: apply Claude's grid labels (labels.txt)
python -m src.data.vtv_supplement --run <tmp> --tag images_eye16      # step 1d: after vtv_faces --config configs/data/vtv_faces_eye16.yaml --out <tmp> --only final_dataset
python -m src.data.vtv_build                                          # step 2: 48x48 dataset split by person + zip
python -m src.data.vtv_review_app                                     # step 1e: review page (or duyet_nhan_vtv.bat); close it before vtv_pipeline
python -m src.data.vtv_balance                                        # step 2b alone (vtv_build calls it)
python -m src.data.vtv_pipeline --status --extract tro_doi_tap_15     # state: lock, other VTV processes, tags, review/ counts
python -m src.data.vtv_pipeline --faces-config configs/data/vtv_faces_eye16.yaml --extract tro_doi_tap_15 --relabel --build --detach
python -m src.data.vtv_pipeline --build --detach                      # rebuild after the human review
python -m src.data.mix_build                                          # VTV (vtv_suggested) + aligned course faces -> data/mix/
python D:/Claude/vtv-mix-3seeds/run_seeds.py                          # mix resnet18 seeds 42/123/2024 + evaluate + eval_wild (log: run.log there)
python -m src.train --config configs/transfer/resnet18_mix.yaml --seed 2024 --init <CK+ resnet18 s2024 run>   # one seed by hand
python -m src.eval_wild --splits test --source rafdb --runs runs/transfer_mix/resnet18_s42_* ...   # human-label check (pass the 3 runs, not the 11/08 first try)
python -m src.aggregate --runs runs/transfer_mix --out reports/transfer_mix_results.csv   # NB: also writes reports/results_runs.csv (rename it to transfer_mix_results_runs.csv)

# Explainability and demo
python -m src.xai.gradcam --run runs/resnet18_s42_<time> --split test
python -m src.demo.webcam --run runs/mobilenet_v2_s42_<time>
python -m src.demo.webcam --pick                                      # video file dialog (nhan_dien_phim.bat)
python -m src.demo.app                                                # browser app window (app_nhan_dien.bat)
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

(Midterm done 2026-10-06: 3 models × 3 seeds on `dataset/`; those runs and tables were deleted on 2026-10-21, see Section 1.)

**Final**
- [x] CK+ stage code: `ckplus_folds.py`, `configs/ckplus/`, `ckplus_cv.py`, `train.py --init`, `configs/transfer/`
- [x] Colab notebook `colab/ckplus_colab.ipynb` + bundle script `colab/make_bundle.py`
- [x] CK+ 10-fold CV (3 models × 3 seeds) and transfer checkpoints (3 × 3) on Colab → `reports/ckplus_cv.md` (2026-11-03; results still on Drive)
- [x] Target dataset chosen: VTV dramas (Section 3.7); step 1 `vtv_faces.py` (aligned faces for review)
- [x] VTV steps 1b–2b: ensemble relabel, Claude's `_unsure` review (round 1), eye-16 supplement, Trò Đời episodes, 48×48 dataset split by person, ≥ 1000 per class, zip; runner `vtv_pipeline.py`
- [x] VTV transfer on the suggested labels (`vtv_suggested`, 2026-11-07): resnet18 83.6% agreement, weakest disgust/fear/surprise/angry
- [x] VTV review page `vtv_review_app.py` (step 1e, logs human checks)
- [ ] VTV human review of `data/vtv/review/` in the review page, then rebuild (`vtv_pipeline --build`) and transfer with `TAG = 'vtv_reviewed'`
- [ ] Optional: FER+ angry/fear/surprise train images (needs `fer2013.csv` from Kaggle; train only, val/test stay VTV)
- [ ] Fine-tune from the CK+ checkpoints on VTV (3 models × 3 seeds)
- [ ] Comparison table of the 3 models (accuracy, macro-F1, params, ms/image, 95% CI)
- [ ] Grad-CAM for all 3 models
- [x] Real-time demo `src/demo/webcam.py` (webcam or video, `nhan_dien_phim.bat`) and the browser app `src/demo/app.py` (`app_nhan_dien.bat`)
- [ ] Recorded presentation
- [ ] (Optional) Ablations

**Suggested split** (two parts with minimal dependencies):
- **Member A:** data (`inspect`, `index_splits`, `dataset`, `transforms`), `cnn_scratch`, `evaluate`, `aggregate`.
- **Member B:** `train.py` (stage mechanism), `pretrained.py` (`resnet18`, `mobilenet_v2`), Grad-CAM, webcam demo.
- **Shared interfaces to agree on early:** the `data/splits/*.csv` format and `build_model(name, num_classes, cfg)`.

---

## 12. Rules for Claude in this repository

- Never delete anything under `dataset/` (the raw data) or `dataset.zip`. The only change made to `dataset/` is the de-duplication by `src/data/dedup_split.py` (Section 3.1: files moved to `dataset_removed/`, logged); any further change needs the user's explicit approval.
- Never re-split the data; use the existing 80 / 15 / 5 split via `data/splits/*.csv`.
- VTV (Section 3.7): never overwrite or re-create `data/vtv/review/` (manual sorting lives there) or `data/vtv/human_review.csv` (the human-check log); run long VTV jobs only through `src.data.vtv_pipeline` (check `--status` first, one process at a time); never evaluate on labels that a model produced without human checking as if they were checked: the only exceptions (user, 2026-11-07 and 2026-11-08) are the `vtv_suggested` and `vtv_claude` transfers, whose test numbers are reported as agreement with those labels, never as accuracy.
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
| Train/val/test split | Fixed in `dataset/`: 47,712 / 4,025 / 1,351 after de-duplication and the CK+ merge (was 47,712 / 4,473 / 1,491) | `index_splits.py` |
| Cross-split copies | None after `dedup_split.py` (`dataset_removed/removed.csv` lists every moved file) | leakage |
| CK+ (checked 2026-10-21) | 927 images, 106 people, 6 classes (no neutral); 922 are in `dataset/`, 5 val frames were moved out by de-duplication | Section 3.6 |
| Folder layout | `dataset/<split>/<class>/*.png` + `dataset/manifest.csv` | `index_splits.py` |
| Images per class in train | 6,816 (3,408 originals + 3,408 offline-augmented), exactly balanced | class weights and focal loss stay off |
| Faces already cropped? | Yes | no `crop_faces.py` |
| Image size and color | 48×48 grayscale | `source_size: 48`, `grayscale: true`, `cache_in_memory: true` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM, CUDA available; 16 GB RAM | batch 64 / 32, AMP on, `num_workers: 0` |

Update this table and the related values in `configs/` if the data or the machine changes.
