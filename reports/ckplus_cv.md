# CK+ subject-independent 10-fold cross-validation

927 CK+ images, 6 classes. Fold k: test = fold k, val = fold k+1 (best.pt), train = the other folds; all frames of one person are in one fold. Pooled = the test folds together, so every image is tested once by a model that never saw its person.

| Model | Params | Pooled accuracy (%) | Pooled macro-F1 | Macro-F1 95% CI* | Fold accuracy (%) | Worst / best fold (%) |
|---|---|---|---|---|---|---|

Mean ± std over seeds (pooled columns) or over all folds of all seeds (fold columns). *Bootstrap 95% CI of each seed's pooled macro-F1 (images resampled), averaged over seeds.

Confusion matrices: `reports/figures/ckplus_cv_confusion_<model>.png` (all seeds pooled). Per-run results: `ckplus_cv_runs.csv`.

## Not included

- cnn_scratch seed 42: 9 of 10 folds not finished, left out
- resnet18 seed 42: 9 of 10 folds not finished, left out
- mobilenet_v2 seed 42: 9 of 10 folds not finished, left out
