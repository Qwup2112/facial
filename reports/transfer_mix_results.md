# Test results (mean ± std over seeds)

| Model | Seeds | Params | Accuracy (%) | Macro-F1 | Macro-F1 95% CI* | ms/image (CPU) | ms/image (GPU) |
|---|---|---|---|---|---|---|---|
| resnet18 | 42 123 2024 | 11.18 M | 77.11 ± 0.59 | 0.7279 ± 0.0036 | [0.7105, 0.7432] | 16.99 | 2.62 |

*Bootstrap 95% CI of each seed's test macro-F1 (1,000 resamples), averaged over seeds.

## Runs included

| Model | Seed | Run | Accuracy | Macro-F1 | Acc. 95% CI | Macro-F1 95% CI | Best epoch | Epochs |
|---|---|---|---|---|---|---|---|---|
| resnet18 | 42 | resnet18_s42_20261109-0212 | 0.7671 | 0.7261 | [0.7527, 0.7806] | [0.7086, 0.7420] | 14 | 23/23 |
| resnet18 | 123 | resnet18_s123_20261109-0241 | 0.7779 | 0.7320 | [0.7633, 0.7907] | [0.7141, 0.7471] | 15 | 23/23 |
| resnet18 | 2024 | resnet18_s2024_20261109-0310 | 0.7684 | 0.7255 | [0.7541, 0.7812] | [0.7086, 0.7404] | 17 | 23/23 |
