# Fix component cases

`app/solve.py` has two stubs:

```python
fit_component_model(
    train_part_X,
    train_part_slot,
    train_case_offsets,
    train_y,
)

predict_component_score(
    part_X,
    part_slot,
    case_offsets,
    params,
)
```

Return one score per case.

The input rows are packed component records. A case can have several rows, and the slot id is part of the row. The fields line up across rows, but rows from different slots do not play the same role.

The old version averaged the rows into one vector per case. It looked okay on balanced cases, but it missed cases with one odd component, missing optional rows, or two slots that disagreed in a way the average hid.

## Scoring

Held-out predictions are evaluated by root mean squared error (RMSE), in the original score units. The grader uses RMSE across all cases and on five named subsets. These are the fixed reward weights and RMSE landmarks:

| Component | Weight | Full-credit RMSE | RMSE cutoff |
| --- | ---: | ---: | ---: |
| Overall | 25% | 1.55 | 13.50 |
| `mismatch` | 20% | 1.90 | 15.50 |
| `bottleneck` | 20% | 1.65 | 14.75 |
| `rare_slot` | 15% | 1.90 | 14.50 |
| `missing_optional` | 10% | 2.10 | 15.50 |
| `variant_shift` | 5% | 1.85 | 14.50 |
| Prediction sanity | 5% | — | — |

For each RMSE component, credit is 1 at or below its full-credit value and falls linearly to 0 at its cutoff. Component credits are averaged over multiple held-out batches, then combined with the weights above. There is also a hard cutoff check on the *mean RMSE across held-out batches*: if overall RMSE or any of the five subset RMSEs reaches its cutoff, the entire reward is 0.

The remaining 5% checks prediction sanity: the absolute mean prediction error and the ratio of prediction standard deviation to target standard deviation. Full sanity credit requires absolute mean error at most 0.7 and a spread ratio between 0.45 and 1.65. The bias credit falls linearly to 0 at an absolute mean error of 7.0. Outside the spread interval, spread credit is `max(0, 1 - abs(spread_ratio - 1) / 1.4)`; sanity credit is the smaller of bias and spread credit, averaged across batches.

Wrong output shape, non-finite or nearly constant predictions, nondeterministic predictions, or input-array mutation give zero reward.

Use the arrays passed in. No hidden files, network, outside data, or fitting on public eval answers.
