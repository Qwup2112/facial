# Transfer: `vtv_suggested` vs `vtv_claude` on the test faces both test sets share

Labels are model suggestions (not human-checked): every number is **agreement** with the `vtv_claude` labels, not accuracy. Each label version was split by person separately, so the test sets differ; a face in both test sets belongs to people neither model set trained on.

Runs: `vtv_suggested` 9, `vtv_claude` 9. Shared test faces: **440** (`vtv_suggested` test 2341, `vtv_claude` test 2198); label differs between the versions on 0 of them.

| Class | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| shared faces | 34 | 81 | 27 | 72 | 152 | 46 | 28 |

## Agreement on the shared faces (mean ± std over seeds, %)

| Model | `vtv_suggested` agreement | `vtv_claude` agreement | `vtv_suggested` macro-F1 | `vtv_claude` macro-F1 |
|---|---|---|---|---|
| cnn_scratch | 72.65 ± 4.23 | 65.61 ± 3.92 | 0.71 ± 0.04 | 0.58 ± 0.03 |
| resnet18 | 79.70 ± 1.07 | 75.76 ± 1.33 | 0.76 ± 0.01 | 0.69 ± 0.01 |
| mobilenet_v2 | 76.89 ± 2.78 | 72.73 ± 3.74 | 0.73 ± 0.02 | 0.64 ± 0.04 |

## Per-class agreement (recall) on the shared faces (mean over seeds, %)

| Class | n | cnn_scratch `vtv_suggested` | cnn_scratch `vtv_claude` | resnet18 `vtv_suggested` | resnet18 `vtv_claude` | mobilenet_v2 `vtv_suggested` | mobilenet_v2 `vtv_claude` |
|---|---|---|---|---|---|---|---|
| angry | 34 | 82 | 51 | 85 | 73 | 89 | 69 |
| disgust | 81 | 44 | 17 | 56 | 42 | 63 | 46 |
| fear | 27 | 79 | 63 | 62 | 64 | 54 | 43 |
| happy | 72 | 99 | 97 | 96 | 97 | 98 | 97 |
| neutral | 152 | 71 | 81 | 85 | 92 | 78 | 88 |
| sad | 46 | 82 | 84 | 88 | 80 | 72 | 82 |
| surprise | 28 | 63 | 33 | 76 | 39 | 73 | 26 |

## `vtv_suggested` test set: faces Claude's pre-screen later moved out

Own test set of `vtv_suggested`, all its runs pooled. "Moved" = moved to `_reject` (blurred, dark, occluded, turned) or `_unsure` (Claude disagreed with the suggested emotion) on 2026-11-08.

| Suggested label | test faces | moved out | agreement on kept (%) | agreement on moved (%) |
|---|---|---|---|---|
| angry | 156 | 77 | 83 | 73 |
| disgust | 113 | 22 | 53 | 53 |
| fear | 112 | 26 | 73 | 50 |
| happy | 563 | 0 | 92 | – |
| neutral | 926 | 0 | 82 | – |
| sad | 364 | 0 | 82 | – |
| surprise | 107 | 70 | 73 | 72 |
