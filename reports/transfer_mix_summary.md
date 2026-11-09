# VTV + course faces ("mix") vs VTV only: resnet18 transfer

**Dates:** first try 2026-11-08, three seeds 2026-11-09. **Why:** the user tried the movie demo (`src/demo/webcam.py`) and found the recognition weak.

## Diagnosis

The demo model (`runs/transfer_vtv_claude/resnet18_s2024_20261108-1516`) was trained only on the VTV faces. Their labels come from a model ensemble (FER+ and two HSEmotion networks), and the rare classes are small (real train faces: angry 331, disgust 253, fear 507, surprise 111). The VTV test numbers measure only agreement with those labels.

Scored against **human labels** with `src.eval_wild`, the model is weak. The test faces are RAF-DB val + test (1,849 faces), prepared exactly like the VTV crops and the demo: 48 → 192 → YuNet → CK+ alignment → 48×48.

| Model | Accuracy | Macro-F1 | angry | disgust | fear | surprise |
|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 67.0 | 0.576 | 75 | 36 | 48 | 32 |
| resnet18 vtv_claude (demo model) | 52.8 | 0.379 | 29 | 12 | 21 | 10 |

The last four columns are per-class recall (%). The model predicted surprise for 2% of the faces, against 16% in the labels. The full table is in `reports/wild_check_rafdb.md`.

## Fix: add the course dataset's real faces, aligned the same way

`src.data.mix_build` aligns the real faces of the course dataset (local, RAF-DB, AffectNet, CK+; not its augmented copies; `dataset/` is not modified) the same way as the VTV faces. It joins them with the `vtv_suggested` VTV build in `data/mix/`.

- **Splits:** VTV is split by person, and the course dataset keeps its own splits. The RAF-DB and local test faces are never trained on.
- **Size:** train has 36,327 faces, and the rare classes have about 4,200 each, up from 500–800.
- **Missed faces:** YuNet did not find 71 faces (mostly RAF-DB), and they were skipped.
- **VTV test set:** the mix VTV test faces are the `vtv_suggested` test set (2,341 faces), so the mix model compares directly with the VTV-only `vtv_suggested` runs (C).

## Training

- **Config:** `configs/transfer/resnet18_mix.yaml`, seeds 42, 123 and 2024. Each seed starts from the CK+ resnet18 checkpoint of the same seed, the same Colab checkpoints the `vtv_suggested` runs used. Seeds 42 and 123 come from the Colab results zip; seed 2024 is `resnet18_s2024_20261021-0809`, identical to the copy in the zip.
- **Hardware:** laptop RTX 3050, 2026-11-09 02:11–03:38, about 29 minutes per seed. Every run trained all 23 epochs. They were run one after another by `D:/Claude/vtv-mix-3seeds/run_seeds.py`.
- **Best epochs:** 14, 15 and 17. Val macro-F1: 0.722, 0.715 and 0.711.
- **First try:** seed 2024 on 2026-11-08 (`runs/transfer_mix/resnet18_s2024_20261108-1656`) was killed at epoch 21/23 because the PC ran low on memory. It is not in the means below, and its scores lie inside the three-seed range: RAF-DB test 79.1%, VTV test 83.1%.

## Results (mean ± std over the 3 seeds)

| Test faces | Label source | Old: VTV only (`vtv_claude` resnet18 s2024) | C: `vtv_suggested` resnet18, 3 seeds | **Mix resnet18, 3 seeds** |
|---|---|---|---|---|
| RAF-DB test, 464 faces | human | 53.4% / 0.391 | not scored | **79.4 ± 0.9% / 0.713 ± 0.012** |
| local test, 685 faces | human | 41.6% / 0.387 | not scored | **60.8 ± 0.9% / 0.610 ± 0.014** |
| VTV test of `vtv_suggested`, 2,341 faces | ensemble (agreement only) | not comparable | 83.64 ± 0.66% / 0.765 ± 0.003 | 82.44 ± 1.24% / 0.748 ± 0.015 |
| full mix test, 3,688 faces | human + ensemble | – | – | 77.11 ± 0.59% / 0.728 ± 0.004 |

Each cell gives accuracy (or agreement) and macro-F1.

- **Missing comparisons:** the old model's train set holds some of the `vtv_suggested` test people, so it is not comparable on that test set. The C checkpoints are on Google Drive only, so C has not been scored on human labels.
- **Baselines and averaging:** the suggestion ensemble scores 68.8% / 0.590 on the RAF-DB test faces and 55.0% / 0.538 on the local test faces. Averaging the three seeds' probabilities gives 81.0% / 0.725 on RAF-DB and 62.6% / 0.628 on local.
- **Full tables:** `reports/wild_check_rafdb_test_mix3.md`, `reports/wild_check_local_test_mix3.md` and `reports/transfer_mix_results.md`.

Per-class recall (%), mean over the 3 seeds:

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| RAF-DB test, mix | 65 | 49 | 54 | 87 | 69 | 91 | 81 |
| RAF-DB test, old VTV-only model | 30 | 14 | 25 | 83 | 70 | 59 | 11 |
| VTV test agreement, mix | 77 | 63 | 65 | 88 | 88 | 80 | 60 |
| VTV test agreement, C resnet18 s2024 | 74 | 57 | 66 | 93 | 86 | 88 | 74 |

## Reading

- **On human labels, the mix model is far better than VTV-only training.** RAF-DB test accuracy rises from 53% to 79%, and the rare classes gain the most (surprise 11 → 81%, disgust 14 → 49%, angry 30 → 65%). These gains are partly in-distribution: the mix model trained on other RAF-DB and local faces, and the old model never saw those sources.
- **On VTV agreement, the mix model is slightly below C.** It scores 1.2 points lower, macro-F1 0.018 lower, and the seed ranges touch. It agrees more often on angry and disgust and less often on surprise (−14), sad (−8) and happy (−5). This is agreement with the ensemble's labels, and C learned only from those labels.
- **No VTV test face has a human-checked label,** so there is no accuracy on VTV faces yet.

**Choice:** the mix model is the best VTV training for the app. It is far better on human-labelled faces and close to C on VTV agreement. The app model is **seed 42** (`runs/transfer_mix/resnet18_s42_20261109-0212`), the seed with the best val macro-F1 (0.722). It was chosen on val, not on test.

## Changes

- **App default:** `default_run()` in `src/demo/webcam.py` now takes the finished `runs/transfer_mix/resnet18_s*` run with the best val macro-F1, which is seed 42.
- **Next steps:**
  - Human-check a VTV test subset for a real VTV accuracy.
  - Score C on human labels (needs the C checkpoints from Google Drive).
  - Train mobilenet_v2 on the mix.
