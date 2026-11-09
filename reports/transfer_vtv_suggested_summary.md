# Transfer CK+ → VTV on the suggested labels (`vtv_suggested`)

> **The VTV labels are model suggestions, not human-checked.** They come from the FER+ + HSEmotion ensemble (plus Claude's agreements on part of `_unsure/`), see CLAUDE.md §3.7.3. Every "accuracy" below is therefore the **agreement with the suggestion labels**, not a measured accuracy. Training on them was the user's decision (2026-11-07) so that the pipeline runs end to end before the human review. The reportable result is the rerun with `TAG = 'vtv_reviewed'` after the review.

Setup: CK+ checkpoint of the same model and seed (section 2 of the Colab notebook) → classifier re-initialised for 7 classes (6 CK+ classes + neutral) → fine-tuned on the VTV train split. Dataset of the 2026-11-07 21:10 build: 16,550 real faces from 352 people (train 11,917 / val 2,292 / test 2,341), split by person (leak check val 1.8% / test 1.3%), plus 614 augmented train copies (disgust +296, fear +105, surprise +213) so that every class has ≥ 1000 faces over all splits. Val and test hold real faces only. Colab T4, 2026-11-07 21:20–23:10, notebook `ckplus_colab_vtv.ipynb`; runs in `MyDrive/facial_emotion/runs/transfer_vtv_suggested/`.

## Test results (mean ± std over seeds 42, 123, 2024)

| Model | Params | Agreement (%) | Macro-F1 | Macro-F1 95% CI* | ms/image (CPU) | ms/image (GPU) |
|---|---|---|---|---|---|---|
| cnn_scratch | 4.69 M | 80.29 ± 1.73 | 0.7314 ± 0.0270 | [0.7060, 0.7549] | 46.29 | 1.87 |
| **resnet18** | 11.18 M | **83.64 ± 0.66** | **0.7654 ± 0.0034** | [0.7411, 0.7894] | 60.70 | 2.82 |
| mobilenet_v2 | 2.23 M | 80.21 ± 0.88 | 0.7327 ± 0.0130 | [0.7076, 0.7559] | 32.15 | 6.00 |

*Bootstrap 95% CI of each seed's test macro-F1 (1,000 resamples), averaged over seeds.

ResNet-18 is the best model and also the most stable across seeds. Its macro-F1 CI does not overlap much with the other two, which are tied. On CK+ (subject-independent 10-fold CV, human labels, 6 posed classes) the same models reached 94.28% / 93.38% / 90.36% (resnet18 / mobilenet_v2 / cnn_scratch). The two numbers measure different things and are not comparable: VTV has spontaneous, often subtle expressions in film lighting, a 7th class (neutral), and noisy reference labels.

## Runs included

| Model | Seed | Run | Agreement | Macro-F1 | Agreement 95% CI | Macro-F1 95% CI | Best epoch | Epochs |
|---|---|---|---|---|---|---|---|---|
| cnn_scratch | 42 | cnn_scratch_s42_20261107-1418 | 0.8125 | 0.7496 | [0.7958, 0.8279] | [0.7238, 0.7726] | 41 | 43/43 |
| cnn_scratch | 123 | cnn_scratch_s123_20261107-1431 | 0.7830 | 0.7003 | [0.7659, 0.7997] | [0.6740, 0.7244] | 32 | 43/43 |
| cnn_scratch | 2024 | cnn_scratch_s2024_20261107-1445 | 0.8133 | 0.7442 | [0.7971, 0.8300] | [0.7202, 0.7676] | 43 | 43/43 |
| mobilenet_v2 | 42 | mobilenet_v2_s42_20261107-1530 | 0.7954 | 0.7329 | [0.7787, 0.8112] | [0.7084, 0.7545] | 18 | 23/23 |
| mobilenet_v2 | 123 | mobilenet_v2_s123_20261107-1543 | 0.7988 | 0.7195 | [0.7830, 0.8150] | [0.6940, 0.7434] | 19 | 23/23 |
| mobilenet_v2 | 2024 | mobilenet_v2_s2024_20261107-1557 | 0.8120 | 0.7455 | [0.7967, 0.8270] | [0.7204, 0.7697] | 21 | 23/23 |
| resnet18 | 42 | resnet18_s42_20261107-1458 | 0.8347 | 0.7642 | [0.8197, 0.8488] | [0.7400, 0.7883] | 19 | 23/23 |
| resnet18 | 123 | resnet18_s123_20261107-1509 | 0.8308 | 0.7627 | [0.8159, 0.8454] | [0.7393, 0.7858] | 17 | 23/23 |
| resnet18 | 2024 | resnet18_s2024_20261107-1519 | 0.8437 | 0.7692 | [0.8278, 0.8599] | [0.7440, 0.7940] | 23 | 23/23 |

## Per-class agreement (resnet18, seed 2024, row-normalised confusion matrix)

| Suggested label | Agreement (%) | Main confusions (%) |
|---|---|---|
| happy | 93 | neutral 5 |
| sad | 88 | neutral 4, angry 3 |
| neutral | 86 | sad 7, angry 3 |
| angry | 74 | sad 11, neutral 6, disgust 5 |
| surprise | 74 | fear 16, neutral 7 |
| fear | 66 | neutral 12, sad 8, surprise 7 |
| disgust | 57 | angry 23, sad 12 |

The weak classes are the ones with the fewest real faces and the ones where the labelling ensemble itself is least reliable (on CK+, FER+ alone recalled angry 13% and fear 8%). disgust↔angry and surprise↔fear are also the classic hard pairs for human annotators.

| Train faces | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| real | 788 | 506 | 672 | 2,816 | 4,629 | 1,920 | 586 |
| augmented copies | 0 | 296 | 105 | 0 | 0 | 0 | 213 |
| test (real) | 156 | 113 | 112 | 563 | 926 | 364 | 107 |

## Next steps

1. Human review of `data/vtv/review/` (rules in `review/HUONG_DAN.md`): first disgust, fear, angry, surprise, and the 8,061 faces in `_unsure/`, which hold many real faces of the rare classes.
2. Rebuild: `python -m src.data.vtv_pipeline --build`.
3. Upload the new zip and rerun section 5 with `TAG = 'vtv_reviewed'`. Only that run gives a reportable accuracy.
