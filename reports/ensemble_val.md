# Ensembles on val (4,025 images)

Average of the softmax outputs of the final runs (src.aggregate). TTA: each model also sees the horizontally flipped image. Reported separately from the single models.

| Ensemble | Runs | TTA | Accuracy (%) | Macro-F1 | Macro-F1 95% CI |
|---|---|---|---|---|---|
| all final runs | 9 | no | 74.04 | 0.7396 | [0.7268, 0.7528] |
| all final runs | 9 | yes | 74.16 | 0.7410 | [0.7272, 0.7544] |
| cnn_scratch (its seeds) | 3 | no | 71.73 | 0.7173 | [0.7035, 0.7300] |
| cnn_scratch (its seeds) | 3 | yes | 71.55 | 0.7152 | [0.7010, 0.7276] |
| resnet18 (its seeds) | 3 | no | 72.87 | 0.7279 | [0.7146, 0.7420] |
| resnet18 (its seeds) | 3 | yes | 73.61 | 0.7351 | [0.7214, 0.7486] |
| mobilenet_v2 (its seeds) | 3 | no | 70.01 | 0.6999 | [0.6853, 0.7132] |
| mobilenet_v2 (its seeds) | 3 | yes | 70.73 | 0.7071 | [0.6921, 0.7206] |

Runs: cnn_scratch_s42_20261005-1110, cnn_scratch_s123_20261005-2153, cnn_scratch_s2024_20261005-2359, resnet18_s42_20261005-2036, resnet18_s123_20261005-2244, resnet18_s2024_20261006-0051, mobilenet_v2_s42_20261005-2113, mobilenet_v2_s123_20261005-2320, mobilenet_v2_s2024_20261006-0126
