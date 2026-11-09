# Accuracy on rafdb faces with human labels (prepared like the VTV crops)

464 faces of the course dataset's test (`source = rafdb`), 48 → 192 → YuNet → CK+ alignment → 48×48 (4 not found by YuNet, skipped). No model here trained on these faces. This is real accuracy (human labels), unlike the VTV test agreement.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| faces | 27 | 42 | 16 | 152 | 47 | 106 | 74 |

| Model | Accuracy (%) | Macro-F1 | angry recall | disgust recall | fear recall | happy recall | neutral recall | sad recall | surprise recall |
|---|---|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 68.8 | 0.590 | 67 | 45 | 44 | 88 | 91 | 69 | 34 |
| resnet18 s2024 (vtv_claude) | 53.4 | 0.391 | 30 | 14 | 25 | 83 | 70 | 59 | 11 |
| resnet18 s2024 (mix) | 79.1 | 0.707 | 70 | 55 | 50 | 89 | 70 | 84 | 80 |

Share of faces predicted as each class (%), true share in the first table:

| Model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 8 | 6 | 6 | 32 | 25 | 17 | 6 |
| resnet18 s2024 (vtv_claude) | 4 | 4 | 5 | 36 | 28 | 20 | 2 |
| resnet18 s2024 (mix) | 8 | 9 | 3 | 31 | 9 | 25 | 15 |
