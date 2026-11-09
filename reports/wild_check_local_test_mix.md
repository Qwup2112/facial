# Accuracy on local faces with human labels (prepared like the VTV crops)

685 faces of the course dataset's test (`source = local`), 48 → 192 → YuNet → CK+ alignment → 48×48 (0 not found by YuNet, skipped). No model here trained on these faces. This is real accuracy (human labels), unlike the VTV test agreement.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| faces | 150 | 23 | 147 | 32 | 146 | 80 | 107 |

| Model | Accuracy (%) | Macro-F1 | angry recall | disgust recall | fear recall | happy recall | neutral recall | sad recall | surprise recall |
|---|---|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 55.0 | 0.538 | 50 | 57 | 31 | 84 | 76 | 34 | 74 |
| resnet18 s2024 (vtv_claude) | 41.6 | 0.387 | 23 | 22 | 25 | 78 | 73 | 44 | 39 |
| resnet18 s2024 (mix) | 60.7 | 0.602 | 60 | 61 | 50 | 66 | 73 | 31 | 80 |

Share of faces predicted as each class (%), true share in the first table:

| Model | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| suggestion ensemble (FER+ + 2 HSEmotion) | 17 | 4 | 11 | 8 | 32 | 10 | 17 |
| resnet18 s2024 (vtv_claude) | 9 | 3 | 13 | 12 | 38 | 16 | 8 |
| resnet18 s2024 (mix) | 22 | 3 | 18 | 4 | 24 | 10 | 18 |
