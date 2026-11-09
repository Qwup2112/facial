# Accuracy on rafdb faces with human labels (prepared like the VTV crops)

464 faces of the course dataset's test (`source = rafdb`), 48 → 192 → YuNet → CK+ alignment → 48×48 (4 not found by YuNet, skipped). No model here trained on these faces. This is real accuracy (human labels), unlike the VTV test agreement.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| faces | 27 | 42 | 16 | 152 | 47 | 106 | 74 |

| Model | Accuracy (%) | Macro-F1 | angry recall | disgust recall | fear recall | happy recall | neutral recall | sad recall | surprise recall |
|---|---|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 68.8 | 0.590 | 67 | 45 | 44 | 88 | 91 | 69 | 34 |
| resnet18 s42 (mix) | 80.4 | 0.727 | 63 | 60 | 50 | 86 | 68 | 92 | 85 |
| resnet18 s123 (mix) | 79.1 | 0.708 | 59 | 48 | 56 | 87 | 72 | 91 | 81 |
| resnet18 s2024 (mix) | 78.7 | 0.705 | 74 | 40 | 56 | 89 | 68 | 89 | 78 |
| mean of our 3 runs | 81.0 | 0.725 | 67 | 50 | 50 | 90 | 70 | 92 | 82 |

Share of faces predicted as each class (%), true share in the first table:

| Model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 8 | 6 | 6 | 32 | 25 | 17 | 6 |
| resnet18 s42 (mix) | 8 | 9 | 2 | 29 | 8 | 28 | 16 |
| resnet18 s123 (mix) | 7 | 8 | 3 | 30 | 8 | 28 | 16 |
| resnet18 s2024 (mix) | 10 | 6 | 3 | 31 | 7 | 28 | 15 |
| mean of our 3 runs | 8 | 7 | 3 | 31 | 8 | 28 | 16 |
