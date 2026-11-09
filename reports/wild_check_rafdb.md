# Accuracy on rafdb faces with human labels (prepared like the VTV crops)

1860 faces of the course dataset's val + test (`source = rafdb`), 48 → 192 → YuNet → CK+ alignment → 48×48 (11 not found by YuNet, skipped). No VTV model trained on these faces. This is real accuracy (human labels), unlike the VTV test agreement.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| faces | 112 | 170 | 67 | 593 | 190 | 425 | 303 |

| Model | Accuracy (%) | Macro-F1 | angry recall | disgust recall | fear recall | happy recall | neutral recall | sad recall | surprise recall |
|---|---|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 67.0 | 0.576 | 75 | 36 | 48 | 89 | 91 | 65 | 32 |
| mobilenet_v2 s123 (vtv_claude) | 49.0 | 0.326 | 20 | 8 | 9 | 86 | 68 | 45 | 12 |
| mobilenet_v2 s2024 (vtv_claude) | 50.9 | 0.339 | 15 | 11 | 9 | 87 | 70 | 52 | 12 |
| mobilenet_v2 s42 (vtv_claude) | 50.2 | 0.334 | 17 | 10 | 10 | 88 | 69 | 49 | 10 |
| resnet18 s2024 (vtv_claude) | 52.8 | 0.379 | 29 | 12 | 21 | 85 | 77 | 56 | 10 |
| mean of our 4 runs | 51.8 | 0.352 | 23 | 11 | 9 | 87 | 74 | 53 | 10 |

Share of faces predicted as each class (%), true share in the first table:

| Model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 10 | 5 | 7 | 32 | 26 | 16 | 6 |
| mobilenet_v2 s123 (vtv_claude) | 5 | 3 | 4 | 41 | 30 | 15 | 2 |
| mobilenet_v2 s2024 (vtv_claude) | 4 | 3 | 3 | 39 | 31 | 19 | 2 |
| mobilenet_v2 s42 (vtv_claude) | 3 | 3 | 4 | 40 | 31 | 17 | 2 |
| resnet18 s2024 (vtv_claude) | 5 | 3 | 5 | 36 | 31 | 19 | 2 |
| mean of our 4 runs | 4 | 3 | 3 | 39 | 32 | 17 | 2 |
