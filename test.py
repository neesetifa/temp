# Forecast policy cleanup

`app/solve.py` still uses one fixed forecast column for every monthly market report. Replace that policy.

The reporting job already runs several forecasting models. Their outputs are stored in `candidate_pred`; the columns stay in the same order between training and prediction. Historical rows also contain the final reported sales count, so the archived rows can be used as backtests for the existing forecasts.

The old policy was acceptable in stable periods, but its errors became uneven as the market mix changed. A forecast column that looks strong across the whole archive is not always the one that behaves best later. The selector archive is also uneven across markets: some market IDs have a long run of labeled backtests, some histories stop earlier, some only begin in the later part of the archive, and a few prediction-time markets have no labeled selector rows at all.

Implement these functions in `app/solve.py`:

```python
def fit_forecast_policy(
    train_time,
    train_market,
    train_context_X,
    train_candidate_pred,
    train_y,
):
    ...
```

```python
def predict_forecast(
    time,
    market,
    context_X,
    candidate_pred,
    params,
):
    ...
```

`candidate_pred` has one row per monthly report and one column per existing forecasting model. Values are sales-count predictions.

`time` is an increasing monthly index. `market` is an anonymized market ID. `context_X` contains information available with the forecast record, including season/profile/history indicators. The same column layout is used in training and prediction.

`train_y` contains the final sales count for each archived labeled row.

Return one non-negative finite sales-count prediction for every input row, in the original row order. You can select one candidate, combine candidates, or learn a policy from the historical backtests.

Do not modify the input arrays. Do not read hidden files or use external data or network access. The result must be deterministic for the same inputs.
