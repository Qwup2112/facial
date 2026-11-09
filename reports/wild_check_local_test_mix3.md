# Accuracy on local faces with human labels (prepared like the VTV crops)

685 faces of the course dataset's test (`source = local`), 48 → 192 → YuNet → CK+ alignment → 48×48 (0 not found by YuNet, skipped). No model here trained on these faces. This is real accuracy (human labels), unlike the VTV test agreement.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| faces | 150 | 23 | 147 | 32 | 146 | 80 | 107 |

| Model | Accuracy (%) | Macro-F1 | angry recall | disgust recall | fear recall | happy recall | neutral recall | sad recall | surprise recall |
|---|---|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 55.0 | 0.538 | 50 | 57 | 31 | 84 | 76 | 34 | 74 |
| resnet18 s42 (mix) | 61.3 | 0.606 | 62 | 57 | 52 | 66 | 71 | 39 | 78 |
| resnet18 s123 (mix) | 59.7 | 0.599 | 53 | 61 | 46 | 66 | 72 | 45 | 81 |
| resnet18 s2024 (mix) | 61.3 | 0.626 | 62 | 57 | 54 | 72 | 63 | 45 | 79 |
| mean of our 3 runs | 62.6 | 0.628 | 60 | 65 | 54 | 66 | 72 | 41 | 80 |

Share of faces predicted as each class (%), true share in the first table:

| Model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 17 | 4 | 11 | 8 | 32 | 10 | 17 |
| resnet18 s42 (mix) | 21 | 3 | 19 | 4 | 24 | 10 | 18 |
| resnet18 s123 (mix) | 18 | 4 | 17 | 4 | 24 | 14 | 20 |
| resnet18 s2024 (mix) | 23 | 2 | 21 | 4 | 20 | 14 | 16 |
| mean of our 3 runs | 21 | 3 | 20 | 4 | 23 | 11 | 18 |
