# Forecast policy cleanup

`app/solve.py` still uses one fixed forecast column for every monthly market report. Replace that policy.

The reporting job already runs several forecasting models. Their outputs are stored in `candidate_pred`; the columns stay in the same order between training and prediction. Historical rows also contain the final reported sales count, so there is enough backtest history to compare how the existing forecasts behaved.

The old policy was acceptable in stable periods, but its errors became uneven as the market mix changed. A model that looks strong across the whole history is not always the best choice for later months. Some markets also have much more selector history than others, and a few market IDs that appear at prediction time have no labeled selector history in the training rows.

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

`time` is an increasing monthly index. `market` is an anonymized market ID. `context_X` contains only information available with the forecast record, including season/profile/history indicators. The same column layout is used in training and prediction.

`train_y` contains the final sales count for each historical row.

Return one non-negative finite sales-count prediction for every input row, in the original row order. You can select one candidate, combine candidates, or learn a policy from the historical backtests.

Do not modify the input arrays. Do not read hidden files or use external data or network access. The result must be deterministic for the same inputs.

solve
from __future__ import annotations
import numpy as np


def fit_forecast_policy(train_time, train_market, train_context_X, train_candidate_pred, train_y):
    return {}


def predict_forecast(time, market, context_X, candidate_pred, params):
    # Existing fallback policy: use the first production forecast.
    p = np.asarray(candidate_pred, float)
    return np.maximum(p[:, 0], 0.0)

reference
from __future__ import annotations
import numpy as np
import lightgbm as lgb


def _features(time, context_X, candidate_pred):
    p = np.log1p(np.maximum(np.asarray(candidate_pred, float), 0.0))
    c = np.asarray(context_X, float)
    t = np.asarray(time, float).reshape(-1, 1) / 200.0
    parts = [
        p,
        p.mean(axis=1, keepdims=True),
        p.std(axis=1, keepdims=True),
        np.median(p, axis=1, keepdims=True),
        (p.max(axis=1) - p.min(axis=1)).reshape(-1, 1),
    ]
    for i in range(p.shape[1]):
        for j in range(i):
            parts.append((p[:, i] - p[:, j]).reshape(-1, 1))
    parts.extend([c, t])
    return np.concatenate(parts, axis=1)


def fit_forecast_policy(train_time, train_market, train_context_X, train_candidate_pred, train_y):
    X = _features(train_time, train_context_X, train_candidate_pred)
    y = np.log1p(np.maximum(np.asarray(train_y, float), 0.0))
    models = []
    for seed in (3, 5, 7):
        m = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=25,
            reg_lambda=7.0,
            reg_alpha=1.0,
            verbosity=-1,
            random_state=seed,
            n_jobs=1,
        )
        m.fit(X, y)
        models.append(m)
    return {
        "models": models,
        "seen_market": np.unique(np.asarray(train_market, int)),
    }


def predict_forecast(time, market, context_X, candidate_pred, params):
    X = _features(time, context_X, candidate_pred)
    p = np.log1p(np.maximum(np.asarray(candidate_pred, float), 0.0))
    meta = np.mean([m.predict(X) for m in params["models"]], axis=0)
    robust = np.median(p, axis=1)
    z = 0.5 * meta + 0.5 * robust
    unseen = ~np.isin(np.asarray(market, int), np.asarray(params["seen_market"], int))
    z[unseen] = robust[unseen]
    return np.maximum(np.expm1(z), 0.0)

generator
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE / "txhousing.csv"
CANDIDATE_COUNT = 8
SLICE_NAMES = np.array([
    "seen_market",
    "unseen_selector_market",
    "high_candidate_disagreement",
    "low_candidate_disagreement",
    "high_scale_market",
    "low_scale_market",
    "peak_season",
    "late_future",
], dtype=object)


def _market_order(names):
    # Stable anonymization: IDs do not preserve alphabetical city names.
    keyed = sorted((hashlib.sha256(str(x).encode()).hexdigest(), str(x)) for x in names)
    return {name: i for i, (_, name) in enumerate(keyed)}


def _ridge(cols, alpha):
    nums = [c for c in cols if c != "market_id"]
    cats = ["market_id"] if "market_id" in cols else []
    parts = []
    if nums:
        parts.append((
            "num",
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
            ),
            nums,
        ))
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return make_pipeline(ColumnTransformer(parts), Ridge(alpha=float(alpha)))


def _prepare_frame():
    d = pd.read_csv(SEED_PATH)
    order = _market_order(d["city"].unique())
    d["market_id"] = d["city"].map(order).astype(int)
    d["time_index"] = ((d["year"] - int(d["year"].min())) * 12 + (d["month"] - 1)).astype(int)
    d = d.sort_values(["market_id", "time_index"]).reset_index(drop=True)
    d["log_y"] = np.log1p(d["sales"].astype(float))
    for lag in (1, 3, 12):
        d[f"lag{lag}"] = d.groupby("market_id")["log_y"].shift(lag)
    d["roll3"] = d.groupby("market_id")["log_y"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    d["roll12"] = d.groupby("market_id")["log_y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=3).mean()
    )
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12.0)
    d["log_listings"] = np.log1p(d["listings"].astype(float))
    d["log_median_price"] = np.log1p(d["median"].astype(float))

    # A fixed pre-deployment market profile. It uses only 2000-2003 observations.
    pre = d[(d["year"] <= 2003) & d["log_y"].notna()]
    stats = pre.groupby("market_id")["log_y"].agg(["mean", "std", "count"])
    global_mean = float(pre["log_y"].mean())
    global_std = float(pre["log_y"].std())
    d["market_scale"] = d["market_id"].map(stats["mean"]).fillna(global_mean)
    d["market_volatility"] = d["market_id"].map(stats["std"]).fillna(global_std)
    d["history_months"] = d["market_id"].map(stats["count"]).fillna(0).astype(float)
    d["report_missing"] = d[["listings", "inventory", "median"]].isna().sum(axis=1).astype(float)
    return d


@lru_cache(maxsize=1)
def _full_table():
    d = _prepare_frame()
    n = len(d)
    pred_log = np.full((n, CANDIDATE_COUNT), np.nan, dtype=float)

    # Candidate 4: seasonal baseline using the same market one year earlier.
    pred_log[:, 4] = d["lag12"].to_numpy(float)

    full_cols = [
        "market_id", "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price",
        "lag1", "lag3", "lag12", "roll3", "roll12",
    ]
    lag_cols = [
        "market_id", "time_index", "month_sin", "month_cos",
        "lag1", "lag3", "lag12", "roll3", "roll12",
    ]
    pooled_cols = [
        "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price", "lag1", "lag12",
    ]
    cross_cols = [
        "market_id", "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price",
    ]
    trend_cols = ["market_id", "time_index", "month_sin", "month_cos"]
    local_cols = [
        "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price", "lag1", "lag12", "roll3",
    ]

    first_year = int(d["year"].min())
    market_ids = np.sort(d["market_id"].unique())
    for year in range(2004, int(d["year"].max()) + 1):
        test = d["year"].eq(year)
        base = d["year"].lt(year) & d["log_y"].notna() & d["time_index"].ge(12)
        cutoff_t = (year - first_year) * 12

        # 0: expanding full-history pooled model.
        m = _ridge(full_cols, 20.0)
        m.fit(d.loc[base, full_cols], d.loc[base, "log_y"])
        pred_log[test.to_numpy(), 0] = m.predict(d.loc[test, full_cols])

        # 1: recent-window full model.
        tr = base & d["time_index"].ge(cutoff_t - 30)
        m = _ridge(full_cols, 8.0)
        m.fit(d.loc[tr, full_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 1] = m.predict(d.loc[test, full_cols])

        # 2: lag-focused model with a longer recent window.
        tr = base & d["time_index"].ge(cutoff_t - 48)
        m = _ridge(lag_cols, 8.0)
        m.fit(d.loc[tr, lag_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 2] = m.predict(d.loc[test, lag_cols])

        # 3: market-specific recent model. Every prediction is still made from
        # earlier observations only; unseen selector markets can have forecasts.
        for market_id in market_ids:
            te = test & d["market_id"].eq(market_id)
            trm = base & d["market_id"].eq(market_id) & d["time_index"].ge(cutoff_t - 30)
            if int(trm.sum()) < 12:
                continue
            m = _ridge(local_cols, 20.0)
            m.fit(d.loc[trm, local_cols], d.loc[trm, "log_y"])
            pred_log[te.to_numpy(), 3] = m.predict(d.loc[te, local_cols])

        # 5: pooled recent model without market identity.
        tr = base & d["time_index"].ge(cutoff_t - 36)
        m = _ridge(pooled_cols, 8.0)
        m.fit(d.loc[tr, pooled_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 5] = m.predict(d.loc[test, pooled_cols])

        # 6: cross-sectional recent model without target lags.
        tr = base & d["time_index"].ge(cutoff_t - 36)
        m = _ridge(cross_cols, 8.0)
        m.fit(d.loc[tr, cross_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 6] = m.predict(d.loc[test, cross_cols])

        # 7: slower trend/seasonality model with no current report fields.
        tr = base & d["time_index"].ge(cutoff_t - 60)
        m = _ridge(trend_cols, 4.0)
        m.fit(d.loc[tr, trend_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 7] = m.predict(d.loc[test, trend_cols])

    valid = (
        d["year"].ge(2004)
        & d["sales"].notna()
        & np.isfinite(pred_log).all(axis=1)
    )
    d = d.loc[valid].reset_index(drop=True)
    pred_log = pred_log[valid.to_numpy()]
    candidate_pred = np.maximum(np.expm1(pred_log), 0.0)

    context_X = np.column_stack([
        d["month_sin"].to_numpy(float),
        d["month_cos"].to_numpy(float),
        d["market_scale"].to_numpy(float),
        d["market_volatility"].to_numpy(float),
        d["history_months"].to_numpy(float),
        d["report_missing"].to_numpy(float),
        np.std(pred_log, axis=1),
    ]).astype(np.float32)

    y = d["sales"].to_numpy(float)
    out = {
        "time": d["time_index"].to_numpy(np.int32),
        "market": d["market_id"].to_numpy(np.int16),
        "context_X": context_X,
        "candidate_pred": candidate_pred.astype(np.float32),
        "y": y.astype(np.float32),
        "year": d["year"].to_numpy(np.int16),
        "month": d["month"].to_numpy(np.int8),
    }
    return out


def _unseen_markets():
    # 8 of 46 anonymized markets are withheld from selector-label history.
    ids = np.arange(46)
    return set(ids[[2, 7, 13, 18, 24, 31, 37, 43]].tolist())


def _slice_ids(data):
    market = data["market"]
    year = data["year"]
    context = data["context_X"]
    pred_log = np.log1p(data["candidate_pred"].astype(float))
    spread = np.std(pred_log, axis=1)
    scale = context[:, 2]
    missing = context[:, 5]
    unseen = np.isin(market, list(_unseen_markets()))
    hi_spread = spread >= np.quantile(spread, 0.70)
    lo_spread = spread <= np.quantile(spread, 0.35)
    hi_scale = scale >= np.quantile(scale, 0.75)
    lo_scale = scale <= np.quantile(scale, 0.25)
    masks = [
        ~unseen,
        unseen,
        hi_spread,
        lo_spread,
        hi_scale,
        lo_scale,
        np.isin(data["month"], [3,4,5,6,7,8]),
        year >= 2014,
    ]
    return np.vstack(masks).astype(bool)


def generate_train():
    d = _full_table()
    unseen = np.isin(d["market"], list(_unseen_markets()))
    # Q4 2011 is left out for public smoke input.
    train = (d["year"] <= 2011) & ~((d["year"] == 2011) & (d["month"] >= 10)) & ~unseen
    return {k: np.asarray(v[train]).copy() for k, v in d.items() if k in ("time", "market", "context_X", "candidate_pred", "y")}


def generate_public():
    d = _full_table()
    m = (d["year"] == 2011) & (d["month"] >= 10)
    return {k: np.asarray(v[m]).copy() for k, v in d.items() if k in ("time", "market", "context_X", "candidate_pred")}


def generate_hidden():
    d = _full_table()
    m = d["year"] >= 2012
    h = {k: np.asarray(v[m]).copy() for k, v in d.items() if k in ("time", "market", "context_X", "candidate_pred", "y", "year", "month")}
    h["slice_mask"] = _slice_ids(h)
    return h

test main
def test_hidden_module_imports():
    import hidden_eval
    assert callable(hidden_eval.evaluate)

hidden eval
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np
from generator import SLICE_NAMES, generate_hidden

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
OVERALL_GOOD = 0.1395
OVERALL_BAD = 0.1600
SLICE_GOOD = np.array([0.137, 0.145, 0.152, 0.136, 0.098, 0.177, 0.126, 0.130], float)
SLICE_BAD = np.array([0.165, 0.162, 0.183, 0.151, 0.132, 0.202, 0.151, 0.153], float)


def _load_train(app):
    d = np.load(app / "train_data.npz")
    return (
        d["train_time"], d["train_market"], d["train_context_X"],
        d["train_candidate_pred"], d["train_y"],
    )


def _load_module(path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solve module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_process(solve_path, payload):
    mod = _load_module(solve_path)
    params = mod.fit_forecast_policy(*payload["train"])
    return [np.asarray(mod.predict_forecast(*x, params), float) for x in payload["inputs"]]


def _run_candidate(solve_path, payload):
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)
    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get("preds") if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, float) for p in preds]


def _log_rmse(y, p):
    y = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    p = np.log1p(np.maximum(np.asarray(p, float), 0.0))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _score(v, good, bad):
    if not math.isfinite(v):
        return 0.0
    return float(np.clip((bad - v) / (bad - good), 0.0, 1.0))


def _validate(p, n):
    if p.shape != (n,):
        raise ValueError(f"prediction shape must be ({n},), got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    if np.any(p < -1e-8):
        raise ValueError("predictions must be non-negative")


def _metrics(y, p, slice_mask):
    overall = _log_rmse(y, p)
    overall_score = _score(overall, OVERALL_GOOD, OVERALL_BAD)
    per = {}
    scores = []
    for i, name in enumerate(SLICE_NAMES):
        m = np.asarray(slice_mask[i], bool)
        if not np.any(m):
            continue
        r = _log_rmse(y[m], p[m])
        s = _score(r, float(SLICE_GOOD[i]), float(SLICE_BAD[i]))
        per[str(name)] = {"log_rmse": r, "score": s, "n": int(m.sum())}
        scores.append(s)
    slice_mean = float(np.mean(scores)) if scores else 0.0
    reward = float(0.35 * overall_score + 0.65 * slice_mean)
    if overall >= OVERALL_BAD:
        reward = 0.0
    return {
        "reward": reward,
        "overall_log_rmse": overall,
        "overall_score": overall_score,
        "slice_mean": slice_mean,
        "per_slice": per,
    }


def evaluate(solve_path=None, app_dir=None, log_dir=None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"
    train = _load_train(app)
    h = generate_hidden()
    x = (h["time"], h["market"], h["context_X"], h["candidate_pred"])
    payload = {"train": train, "inputs": [x, x]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")
    p0, p1 = preds
    _validate(p0, len(h["y"]))
    _validate(p1, len(h["y"]))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")
    metrics = _metrics(h["y"], p0, h["slice_mask"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    r, m = evaluate()
    print(json.dumps({"reward": r, "overall_log_rmse": m["overall_log_rmse"]}, sort_keys=True))


