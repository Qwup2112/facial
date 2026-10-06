# Facial Emotion Recognition

Seven-class facial emotion recognition (angry, disgust, fear, happy, neutral, sad, surprise) from 48×48 grayscale face images. Midterm project of the Advanced Deep Learning course, Faculty of Electrical and Electronic Engineering, Phenikaa University.

Author: Nguyễn Quang Trung (22011211) · Advisor: Ph.D Minhhuy Le

## Results (test set, 1,351 images, mean ± std over 3 seeds)

| Model | Params | Accuracy (%) | Macro-F1 | CPU ms/image |
|---|---|---|---|---|
| EmotionCNN (from scratch) | 4.69 M | 68.0 ± 1.0 | 0.679 | 13.5 |
| ResNet18 (ImageNet, fine-tuned) | 11.18 M | 68.3 ± 1.0 | 0.683 | 18.2 |
| MobileNetV2 (ImageNet, fine-tuned) | 2.23 M | 67.8 ± 0.9 | 0.678 | 13.0 |
| Ensemble of the 9 runs | | 71.8 | 0.716 | |

Full tables: [`reports/results_summary.md`](reports/results_summary.md) and [`reports/ensemble_test.md`](reports/ensemble_test.md). Midterm report: [`reports/midterm_report_NguyenQuangTrung.pdf`](reports/midterm_report_NguyenQuangTrung.pdf).

## Repository layout

```
configs/          base.yaml + one config per model (all hyperparameters live here)
src/data/         inspect (data report, cross-split copy check), dedup_split, add_external (CK+ merge),
                  index_splits, dataset, transforms
src/models/       EmotionCNN, ResNet18 / MobileNetV2 wrappers
src/              train, evaluate, aggregate, ensemble, predict
data/splits/      train/val/test CSVs (paths, labels, sources)
runs/             the 9 final runs: best.pt, config.yaml, log.csv, curves, test/val metrics
runs/logs/        PowerShell scripts that queue training and evaluation
reports/          reports (Word/PDF), figures, result tables, data report
CLAUDE.md         detailed project specification
```

## Data (not included)

The images come from a FER2013-style local set, RAF-DB, AffectNet and CK+. Their licences do not allow redistribution, so `dataset/` is not in this repository. To reproduce, obtain the datasets from their owners, place them as `dataset/<split>/<class>/*.png` with `dataset/manifest.csv`, then:

```bash
pip install -r requirements.txt
python -m src.data.index_splits     # dataset/ -> data/splits/*.csv
python -m src.data.inspect          # data report and cross-split copy check
```

## Training and evaluation

```bash
python -m src.train --config configs/cnn_scratch.yaml --seed 42 --dry-run
python -m src.train --config configs/cnn_scratch.yaml --seed 42
python -m src.evaluate --run runs/<run_dir> --tta     # test set, once per final model
python -m src.aggregate                              # reports/results_summary.md
python -m src.ensemble --split val                   # then --split test once
```

Tested on Windows with an NVIDIA RTX 3050 Laptop GPU (4 GB); one run takes 37–50 minutes.
