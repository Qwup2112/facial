# Tuning log (seed 42, validation set only)

Every decision below was made on the validation split (4,473 images, 639 per class).
No test metric was computed or used for these choices. Training metrics are measured with
augmentation and dropout on, so they understate the true fit of the training set.

All runs in this log were trained on the split before the de-duplication of 2026-10-04 (CLAUDE.md §3.1)
and are kept in `runs/_leaky_split/`. Their val scores are optimistic: the seed-42 `cnn_scratch` checkpoint
scores macro-F1 0.673 on the old val split and 0.663 on the cleaned one (4,039 images). The comparisons between
configs still hold (see the robustness check below).
The chosen configs (`cnn_scratch` v1, `resnet18` v2, `mobilenet_v2` v2) were retrained on the cleaned split.

## Round 1: configs as first specified (v1)

| Model | Best val macro-F1 | Best epoch (last) | Train macro-F1 at best epoch | Run |
|---|---|---|---|---|
| `cnn_scratch` | 0.6741 | 37 (40) | 0.722 | `_leaky_split/cnn_scratch_s42_20261004-1310` |
| `resnet18` | 0.6925 | 12 (18, early stop) | 0.855 | `_leaky_split/resnet18_s42_20261004-1344` |
| `mobilenet_v2` | 0.6651 | 14 (20, early stop) | 0.751 | `_leaky_split/mobilenet_v2_s42_20261004-1409` |

Findings:
- Both pretrained models overfit. Train macro-F1 kept rising (0.95 for `resnet18` and 0.80 for
  `mobilenet_v2` at the last epoch) while val macro-F1 stopped improving and val loss rose.
- AdamW uses decoupled weight decay scaled by the learning rate (θ ← θ − lr·λ·θ). With λ = 1e-4
  and lr ≤ 5e-4, each step shrank the weights by at most 5e-8, which is practically no regularization.
- `mobilenet_v2` was below the scratch baseline, and only its last blocks (`features[14:]`, 7×7
  resolution) were trained.
- `cnn_scratch` did not underfit: val plateaued at about 0.67 with a small train/val gap. It was
  kept as specified, and its seeds 123 and 2024 were already scheduled with this config.

## Round 2: changes (v2)

| Model | Change | Best val macro-F1 | Run |
|---|---|---|---|
| `resnet18` | weight decay 1e-4 → 5e-2, head dropout 0.3 → 0.5 | 0.6932 (+0.0007) | `_leaky_split/resnet18_s42_20261004-1443` |
| `mobilenet_v2` | same, plus fine-tune `features[7:]` instead of `features[14:]` (lr 5e-5 for `features[7:14]`, 1e-4 for `features[14:]`, 5e-4 for the classifier) | 0.6789 (+0.0138) | `_leaky_split/mobilenet_v2_s42_20261004-1505` |

Decisions:
- `resnet18`: v2 is kept. It ties with v1 on val, uses a meaningful AdamW weight decay, and shares
  the recipe with `mobilenet_v2`. It still overfits (train macro-F1 0.90 at the stop).
- `mobilenet_v2`: v2 is kept (+1.4 points; val loss at the best epoch 1.18 instead of 1.21). Val macro-F1 was still
  rising at the last epoch, so a longer schedule might help a little.
- Seeds 123 and 2024 run with these final configs: `cnn_scratch` v1, `resnet18` v2, `mobilenet_v2` v2.

## Robustness check: near-duplicate leakage in val

`src/data/inspect.py` found val and test images with a near copy in train: dHash Hamming distance ≤ 2,
which gives 104 val and 37 test images (see `reports/data_report.md`). There are no exact copies. The models
get these images right far more often (77–84% vs. about 67–69% for the rest), so they inflate val macro-F1
by about 0.2–0.4 points. With these 104 images removed, the decisions above still hold:

| Run (seed 42) | Val macro-F1, all | Val macro-F1 without the 104 near copies |
|---|---|---|
| `cnn_scratch` v1 | 0.6734 | 0.6694 |
| `resnet18` v1 | 0.6926 | 0.6901 |
| `resnet18` v2 | 0.6927 | 0.6904 |
| `mobilenet_v2` v1 | 0.6641 | 0.6618 |
| `mobilenet_v2` v2 | 0.6778 | 0.6742 |

(fp32 recomputation from `predictions_val.csv`.)

## Per-class val F1 (final seed-42 runs)

Computed from `predictions_val.csv` (fp32 inference, so within 0.001 of the logged AMP values).

| Class | `cnn_scratch` | `resnet18` v2 | `mobilenet_v2` v2 |
|---|---|---|---|
| angry | 0.578 | 0.617 | 0.576 |
| disgust | 0.641 | 0.725 | 0.737 |
| fear | 0.507 | 0.529 | 0.506 |
| happy | 0.871 | 0.869 | 0.860 |
| neutral | 0.658 | 0.679 | 0.637 |
| sad | 0.661 | 0.631 | 0.643 |
| surprise | 0.797 | 0.799 | 0.786 |
| **macro-F1** | **0.673** | **0.693** | **0.678** |

Fear and angry are the hardest classes for every model. Many errors go to neutral, and disgust is often
predicted as angry (21% for `cnn_scratch`).

## Val accuracy by data source (%)

The val split mixes the team's local FER2013-style images (2,478), RAF-DB (1,519) and AffectNet (476;
only angry, disgust and fear).

| Model | AffectNet | Local | RAF-DB | All |
|---|---|---|---|---|
| `cnn_scratch` | 50.8 | 60.9 | 82.7 | 67.2 |
| `resnet18` v1 | 66.4 | 62.4 | 80.4 | 69.0 |
| `resnet18` v2 | 63.7 | 62.9 | 81.1 | 69.2 |
| `mobilenet_v2` v1 | 63.2 | 60.0 | 77.7 | 66.4 |
| `mobilenet_v2` v2 | 67.6 | 59.8 | 80.6 | 67.7 |

- The ImageNet-pretrained models gain mainly on AffectNet images (+12 to +17 points over `cnn_scratch`).
- On RAF-DB, the scratch CNN is as good as or better than the pretrained models.
- The local FER2013-style images are the hardest source (about 60–63%), close to what is usually reported
  for FER2013, whose labels are known to be noisy. This, together with the 48×48 resolution, is the
  likely ceiling for all three models. The labeled test grids
  (`reports/figures/predictions_{model}_s42_test.png`, regenerated for the runs on the cleaned split) show several ambiguous or likely mislabeled faces.
