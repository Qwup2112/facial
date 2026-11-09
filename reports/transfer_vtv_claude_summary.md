# CK+ → VTV transfer on Claude-screened suggestion labels (`vtv_claude`)

**Test labels are ensemble suggestions screened by Claude, not human-checked: "accuracy" below is agreement with them, not a measured accuracy.**

## Setup

- Dataset: the `vtv_suggested` faces after Claude's visual pre-screen of the four weakest class folders (2026-11-08, user: "lâu quá bạn làm đi cho nhanh"). 1,844 of 4,196 faces in disgust, fear, surprise and angry were moved out: 1,501 to `_reject` (blurred, dark, occluded, turned) and 343 to `_unsure` (Claude disagreed with the suggested emotion). Rebuilt by `src.data.vtv_build`: 15,030 real faces, 328 people, split by person (train 10,428 / val 2,404 / test 2,198), identity leakage val 0.9% / test 0.5%.
- Real train faces of the rare classes after the screen: angry 331, disgust 253, fear 507, surprise 111 (offline augmentation up to 1,000 each; surprise is mostly augmented copies). Test: angry 127, disgust 99, fear 86, surprise 36.
- Training: same configs, seeds (42, 123, 2024) and CK+ transfer checkpoints as `vtv_suggested`. 5 runs ran on Colab (T4: cnn_scratch ×3, resnet18 s42/s123); Colab's GPU quota then ran out, and the other 4 (resnet18 s2024, mobilenet_v2 ×3) ran on the laptop RTX 3050 with the same code bundle, dataset zip and checkpoints. Summary by `src.aggregate` on Colab (CPU runtime).

## Test results (mean ± std over seeds)

| Model | Agreement (%) | Macro-F1 | Macro-F1 95% CI* | `vtv_suggested` agreement / macro-F1 |
|---|---|---|---|---|
| cnn_scratch | 77.31 ± 3.22 | 0.6135 ± 0.0420 | [0.5769, 0.6464] | 80.29 ± 1.73 / 0.7314 ± 0.0270 |
| resnet18 | 83.64 ± 0.66 | 0.7013 ± 0.0110 | [0.6645, 0.7340] | 83.64 ± 0.66 / 0.7654 ± 0.0034 |
| mobilenet_v2 | 81.41 ± 0.85 | 0.6630 ± 0.0246 | [0.6239, 0.6969] | 80.21 ± 0.88 / 0.7327 ± 0.0130 |

*Bootstrap 95% CI of each seed's test macro-F1 (1,000 resamples), averaged over seeds. The two label versions have different test sets (different faces and people), so this table alone does not compare them; see the head-to-head below.

| Model | Seed | Run | Where | Agreement | Macro-F1 | Best epoch |
|---|---|---|---|---|---|---|
| cnn_scratch | 42 | cnn_scratch_s42_20261108-0647 | Colab | 0.7803 | 0.6157 | 25 |
| cnn_scratch | 123 | cnn_scratch_s123_20261108-0700 | Colab | 0.7379 | 0.5704 | 30 |
| cnn_scratch | 2024 | cnn_scratch_s2024_20261108-0713 | Colab | 0.8012 | 0.6543 | 27 |
| resnet18 | 42 | resnet18_s42_20261108-0727 | Colab | 0.8371 | 0.6997 | 23 |
| resnet18 | 123 | resnet18_s123_20261108-0738 | Colab | 0.8294 | 0.6911 | 21 |
| resnet18 | 2024 | resnet18_s2024_20261108-1516 | laptop | 0.8426 | 0.7130 | 16 |
| mobilenet_v2 | 42 | mobilenet_v2_s42_20261108-1528 | laptop | 0.8230 | 0.6906 | 22 |
| mobilenet_v2 | 123 | mobilenet_v2_s123_20261108-1541 | laptop | 0.8062 | 0.6433 | 21 |
| mobilenet_v2 | 2024 | mobilenet_v2_s2024_20261108-1554 | laptop | 0.8130 | 0.6552 | 15 |

## Head-to-head on shared test faces

`src/transfer_compare.py` (full table: `transfer_compare_vtv_suggested_vs_vtv_claude.md`) scores both model sets on the 440 test faces that are in both test sets (people neither set trained on), against the same labels:

- Agreement is lower after the screen for every model: cnn_scratch 72.7 → 65.6%, resnet18 79.7 → 75.8%, mobilenet_v2 76.9 → 72.7% (macro-F1 0.71/0.76/0.73 → 0.58/0.69/0.64).
- The loss is in the screened classes: surprise 63–76% → 26–39%, angry 82–89% → 51–73%, disgust 44–63% → 17–46% (fear mixed); neutral improves (71–85% → 81–92%), because the smaller rare classes push more faces to neutral.
- On the old test set, the old models agreed with the suggested label about as often on the faces Claude later removed as on the ones it kept (surprise 72% vs 73%, disgust 53% vs 53%; fear 50% vs 73%, angry 73% vs 83%). For surprise and disgust, Claude's rejects were not measurably harder for the models.

The shared set is small for the rare classes (surprise 28, fear 27, angry 34 faces), but the direction is the same for all three models.

## Conclusion

By the only measure available (agreement with the suggestion labels), the screen made the weak classes worse. It removed 45–65% of the real faces of surprise and angry, and the remaining 111–331 real training faces per class are not enough. Two caveats apply. First, both label versions come from the same ensemble, so models trained on more ensemble-labelled faces mimic the ensemble better, whether or not those labels are right. Second, only human-checked test labels can show the true accuracy. Next steps:

1. Keep `vtv_suggested` as the main transfer result. Restore the 1,844 moved faces (logged in `data/vtv/claude_suggest_moves.csv`) unless a human review confirms them.
2. To improve the weak classes, add real labelled faces (FER+: surprise ~4,000 and angry ~2,600 images, from `fer2013.csv` plus the FER+ label file) rather than remove faces.
3. Human-check a test subset of the rare classes, so that accuracy, not agreement, can be reported.
