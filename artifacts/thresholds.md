# Threshold trade-offs (out-of-fold, train split, n=833)

Chosen: t_reject=0.8 (wrongly reject <= 0% of real complaints), t_review=0.7 (auto-handled accuracy >= 95%).

## Gate

| t_reject | real complaints wrongly rejected | not_civic correctly rejected |
|---|---|---|
| 0.5 | 1.2% (8/689) | 91.7% (132/144) |
| 0.6 | 0.7% (5/689) | 90.3% (130/144) |
| 0.7 | 0.6% (4/689) | 89.6% (129/144) |
| 0.8 | 0.4% (3/689) | 86.8% (125/144) |
| 0.85 | 0.4% (3/689) | 83.3% (120/144) |
| 0.9 | 0.4% (3/689) | 79.9% (115/144) |
| 0.95 | 0.1% (1/689) | 72.2% (104/144) |
| 0.99 | 0.0% (0/689) | 56.2% (81/144) |

## Review

| t_review | auto-handled share | accuracy of auto-handled | sent to review |
|---|---|---|---|
| 0.3 | 100.0% | 90.5% | 0 |
| 0.4 | 99.3% | 90.8% | 6 |
| 0.5 | 97.2% | 91.9% | 23 |
| 0.6 | 92.1% | 93.7% | 66 |
| 0.7 | 87.0% | 95.0% | 108 |
| 0.8 | 81.9% | 96.2% | 151 |
| 0.9 | 75.0% | 97.9% | 208 |
