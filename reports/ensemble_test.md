# Ensembles on test (1,351 images)

Average of the softmax outputs of the final runs (src.aggregate). TTA: each model also sees the horizontally flipped image. Reported separately from the single models.

| Ensemble | Runs | TTA | Accuracy (%) | Macro-F1 | Macro-F1 95% CI |
|---|---|---|---|---|---|
| all final runs | 9 | no | 71.80 | 0.7158 | [0.6921, 0.7396] |
| all final runs | 9 | yes | 71.87 | 0.7171 | [0.6941, 0.7410] |
| cnn_scratch (its seeds) | 3 | no | 69.58 | 0.6944 | [0.6694, 0.7182] |
| cnn_scratch (its seeds) | 3 | yes | 69.28 | 0.6918 | [0.6669, 0.7156] |
| resnet18 (its seeds) | 3 | no | 71.43 | 0.7126 | [0.6880, 0.7358] |
| resnet18 (its seeds) | 3 | yes | 72.39 | 0.7228 | [0.6991, 0.7476] |
| mobilenet_v2 (its seeds) | 3 | no | 68.84 | 0.6869 | [0.6634, 0.7106] |
| mobilenet_v2 (its seeds) | 3 | yes | 69.36 | 0.6933 | [0.6679, 0.7192] |

Runs: cnn_scratch_s42_20261005-1110, cnn_scratch_s123_20261005-2153, cnn_scratch_s2024_20261005-2359, resnet18_s42_20261005-2036, resnet18_s123_20261005-2244, resnet18_s2024_20261006-0051, mobilenet_v2_s42_20261005-2113, mobilenet_v2_s123_20261005-2320, mobilenet_v2_s2024_20261006-0126
