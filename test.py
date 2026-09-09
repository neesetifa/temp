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

RECENT_WINDOW = 24
STAT_SHRINK = 40.0
SOFTMAX_TEMP = 0.010
STALE_GAP = 20
LATE_HORIZON = 28
LATE_LOCAL_WEIGHT = 0.40


def _log_pred(candidate_pred):
    return np.log1p(np.maximum(np.asarray(candidate_pred, float), 0.0))


def _softmax_weights(mse, temperature=SOFTMAX_TEMP):
    mse = np.asarray(mse, float)
    z = -(mse - float(np.min(mse))) / max(float(temperature), 1e-8)
    z -= float(np.max(z))
    w = np.exp(z)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return np.full(len(mse), 1.0 / max(len(mse), 1), dtype=float)
    return w / s


def _fit_stats(time, market, candidate_pred, y):
    time = np.asarray(time, int)
    market = np.asarray(market, int)
    p = _log_pred(candidate_pred)
    ylog = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    residual = p - ylog[:, None]
    n_candidates = p.shape[1]

    global_bias = residual.mean(axis=0)
    global_mse = np.mean(residual ** 2, axis=0)
    global_weight = _softmax_weights(global_mse)
    global_tmax = int(np.max(time)) if len(time) else 0

    n_markets = max(46, int(np.max(market)) + 1 if len(market) else 46)
    local_bias = np.tile(global_bias, (n_markets, 1))
    local_mse = np.tile(global_mse, (n_markets, 1))
    local_weight = np.tile(global_weight, (n_markets, 1))
    market_count = np.zeros(n_markets, dtype=int)
    market_last = np.full(n_markets, -10_000, dtype=int)

    for market_id in np.unique(market):
        q = market == market_id
        if not np.any(q):
            continue
        market_count[market_id] = int(q.sum())
        last = int(np.max(time[q]))
        market_last[market_id] = last
        recent = q & (time >= last - (RECENT_WINDOW - 1))
        n = int(recent.sum())
        if n <= 0:
            continue
        rb = residual[recent].mean(axis=0)
        rmse2 = np.mean(residual[recent] ** 2, axis=0)
        local_bias[market_id] = (
            n * rb + STAT_SHRINK * global_bias
        ) / (n + STAT_SHRINK)
        local_mse[market_id] = (
            n * rmse2 + STAT_SHRINK * global_mse
        ) / (n + STAT_SHRINK)
        local_weight[market_id] = _softmax_weights(local_mse[market_id])

    return {
        "global_tmax": global_tmax,
        "global_bias": global_bias,
        "global_weight": global_weight,
        "local_bias": local_bias,
        "local_weight": local_weight,
        "market_count": market_count,
        "market_last": market_last,
    }


def _policy_log(time, market, candidate_pred, stats, use_market=True, use_freshness=True):
    time = np.asarray(time, int)
    market = np.asarray(market, int)
    p = _log_pred(candidate_pred)
    out = np.empty(len(time), dtype=float)
    g_tmax = int(stats["global_tmax"])

    for i, (t, market_id) in enumerate(zip(time, market)):
        global_pred = float(np.dot(stats["global_weight"], p[i] - stats["global_bias"]))
        local_alpha = 0.0
        if (
            use_market
            and 0 <= market_id < len(stats["market_count"])
            and stats["market_count"][market_id] > 0
        ):
            last = int(stats["market_last"][market_id])
            if not use_freshness or last >= g_tmax - STALE_GAP:
                local_alpha = 1.0
                if use_freshness and t >= g_tmax + LATE_HORIZON:
                    local_alpha *= LATE_LOCAL_WEIGHT

        if local_alpha > 0.0:
            local_pred = float(
                np.dot(
                    stats["local_weight"][market_id],
                    p[i] - stats["local_bias"][market_id],
                )
            )
            out[i] = local_alpha * local_pred + (1.0 - local_alpha) * global_pred
        else:
            out[i] = global_pred
    return out


def fit_forecast_policy(
    train_time,
    train_market,
    train_context_X,
    train_candidate_pred,
    train_y,
):
    del train_context_X
    return _fit_stats(
        train_time,
        train_market,
        train_candidate_pred,
        train_y,
    )


def predict_forecast(time, market, context_X, candidate_pred, params):
    del context_X
    z = _policy_log(time, market, candidate_pred, params)
    return np.maximum(np.expm1(z), 0.0)

hidden
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

OVERALL_GOOD = 0.1377
OVERALL_BAD = 0.2050
OVERALL_CATASTROPHIC = 0.30
SCORE_POWER = 1.5
SLICE_GOOD = np.array([
    0.1450,
    0.1261,
    0.1155,
    0.1485,
    0.1330,
    0.1204,
    0.1272,
    0.1278,
], dtype=float)
SLICE_BAD = np.array([
    0.1650,
    0.1400,
    0.1315,
    0.1685,
    0.1490,
    0.1375,
    0.1430,
    0.1420,
], dtype=float)
SLICE_WEIGHTS = np.array([
    0.12,
    0.20,
    0.15,
    0.08,
    0.15,
    0.12,
    0.08,
    0.10,
], dtype=float)


def _load_train(app):
    d = np.load(app / "train_data.npz")
    return (
        d["train_time"],
        d["train_market"],
        d["train_context_X"],
        d["train_candidate_pred"],
        d["train_y"],
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
    return [
        np.asarray(mod.predict_forecast(*x, params), dtype=float)
        for x in payload["inputs"]
    ]


def _run_candidate(solve_path, payload):
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)
    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get("preds") if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, dtype=float) for p in preds]


def _log_rmse(y, p):
    y = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    p = np.log1p(np.maximum(np.asarray(p, float), 0.0))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _score(value, good, bad):
    if not math.isfinite(value) or bad <= good:
        return 0.0
    z = float(np.clip((bad - value) / (bad - good), 0.0, 1.0))
    return float(z ** SCORE_POWER)


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
    weighted = 0.0
    weight_total = 0.0
    for i, name in enumerate(SLICE_NAMES):
        mask = np.asarray(slice_mask[i], bool)
        if not np.any(mask):
            continue
        rmse = _log_rmse(np.asarray(y)[mask], np.asarray(p)[mask])
        score = _score(rmse, float(SLICE_GOOD[i]), float(SLICE_BAD[i]))
        weight = float(SLICE_WEIGHTS[i])
        weighted += weight * score
        weight_total += weight
        per[str(name)] = {
            "log_rmse": rmse,
            "score": score,
            "n": int(mask.sum()),
            "weight": weight,
        }

    slice_score = weighted / max(weight_total, 1e-12)
    reward = float(0.35 * overall_score + 0.65 * slice_score)

    if overall >= OVERALL_CATASTROPHIC:
        reward = 0.0

    return {
        "reward": reward,
        "overall_log_rmse": overall,
        "overall_score": overall_score,
        "slice_score": slice_score,
        "per_slice": per,
    }


def evaluate(solve_path=None, app_dir=None, log_dir=None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"

    train = _load_train(app)
    hidden = generate_hidden()
    x = (
        hidden["time"],
        hidden["market"],
        hidden["context_X"],
        hidden["candidate_pred"],
    )
    payload = {"train": train, "inputs": [x, x]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")

    p0, p1 = preds
    _validate(p0, len(hidden["y"]))
    _validate(p1, len(hidden["y"]))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")

    metrics = _metrics(hidden["y"], p0, hidden["slice_mask"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    reward, metrics = evaluate()
    print(json.dumps({
        "reward": reward,
        "overall_log_rmse": metrics["overall_log_rmse"],
    }, sort_keys=True))

generator
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
SEED_PATH = HERE / "txhousing.csv"
CANDIDATE_COUNT = 8
SLICE_NAMES = np.array([
    "full_selector_history",
    "stale_selector_history",
    "short_selector_history",
    "unseen_selector_market",
    "recent_rank_shift",
    "high_candidate_disagreement",
    "peak_season",
    "late_future",
], dtype=object)


def _market_order(names):
    keyed = sorted((hashlib.sha256(str(x).encode()).hexdigest(), str(x)) for x in names)
    return {name: i for i, (_, name) in enumerate(keyed)}


def _selector_groups():
    # Backtest archives are intentionally uneven: some markets have long selector
    # history, some stop early, some only start recently, and some have none.
    ids = np.arange(46, dtype=int)
    keyed = sorted(
        (hashlib.sha256(f"selector-coverage-{int(i)}".encode()).hexdigest(), int(i))
        for i in ids
    )
    ordered = [i for _, i in keyed]
    return {
        "unseen": set(ordered[:8]),
        "stale": set(ordered[8:16]),
        "short": set(ordered[16:24]),
        "full": set(ordered[24:]),
    }


def _ridge(cols, alpha):
    cats = [c for c in ("market_id", "month") if c in cols]
    nums = [c for c in cols if c not in cats]
    parts = []
    if nums:
        parts.append((
            "num",
            make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
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

    full_cols = [
        "market_id", "month", "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price",
        "lag1", "lag3", "lag12", "roll3", "roll12",
    ]
    local_cols = [
        "month", "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price",
        "lag1", "lag12", "roll3",
    ]
    pooled_full_cols = [
        "month", "time_index", "month_sin", "month_cos",
        "log_listings", "inventory", "log_median_price",
        "lag1", "lag3", "lag12", "roll3", "roll12",
    ]

    first_year = int(d["year"].min())
    market_ids = np.sort(d["market_id"].unique())
    # Persistent market traits from pre-forecast history define actual specialist model pools.
    pre0 = d[d["year"] <= 2003].copy()
    trait_rows = []
    for mid, q in pre0.groupby("market_id"):
        q = q.sort_values("time_index")
        z = q["log_y"].to_numpy(float)
        month_mean = q.groupby("month")["log_y"].mean()
        seas = float(month_mean.std())
        ac = float(np.corrcoef(z[1:], z[:-1])[0, 1]) if len(z) > 3 else 0.0
        x = q["time_index"].to_numpy(float)
        trend = float(np.polyfit(x - x.mean(), z, 1)[0]) if len(z) > 3 else 0.0
        trait_rows.append((int(mid), float(np.mean(z)), float(np.std(z)), seas, ac, trend))
    trait_rows = np.asarray(trait_rows, float)
    T = trait_rows[:, 1:]
    T = np.nan_to_num(T, nan=np.nanmedian(T, axis=0))
    T = (T - T.mean(0)) / np.maximum(T.std(0), 1e-6)
    labels = KMeans(n_clusters=4, random_state=17, n_init=20).fit_predict(T)
    cluster_sets = [set(trait_rows[labels == k, 0].astype(int)) for k in range(4)]

    for year in range(2004, int(d["year"].max()) + 1):
        test = d["year"].eq(year)
        base = d["year"].lt(year) & d["log_y"].notna() & d["time_index"].ge(12)
        cutoff_t = (year - first_year) * 12

        # A broad context-only forecast.
        cols4 = ["month", "time_index", "log_listings", "inventory", "log_median_price"]
        m = _ridge(cols4, 18.0)
        m.fit(d.loc[base, cols4], d.loc[base, "log_y"])
        global_pred = m.predict(d.loc[test, cols4])
        pred_log[test.to_numpy(), 4] = global_pred

        # Four incumbent specialists use market cohorts fixed from pre-2004 behavior.
        for jj in range(4):
            tr = base & d["market_id"].isin(cluster_sets[jj]) & d["time_index"].ge(cutoff_t - 84)
            m = _ridge(full_cols, 8.0)
            m.fit(d.loc[tr, full_cols], d.loc[tr, "log_y"])
            spec = m.predict(d.loc[test, full_cols])
            pred_log[test.to_numpy(), jj] = spec

        # A shorter-window context-only forecast.
        tr = base & d["time_index"].ge(cutoff_t - 30)
        cols5 = ["month", "time_index", "log_listings", "inventory", "log_median_price"]
        m = _ridge(cols5, 8.0)
        m.fit(d.loc[tr, cols5], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 5] = m.predict(d.loc[test, cols5])

        # A local-market forecast with a pooled fallback.
        tr = base & d["time_index"].ge(cutoff_t - 48)
        fallback = _ridge(full_cols, 18.0)
        fallback.fit(d.loc[tr, full_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 6] = fallback.predict(d.loc[test, full_cols])
        for market_id in market_ids:
            te = test & d["market_id"].eq(market_id)
            trm = base & d["market_id"].eq(market_id) & d["time_index"].ge(cutoff_t - 48)
            if int(trm.sum()) < 24:
                continue
            m = _ridge(local_cols, 16.0)
            m.fit(d.loc[trm, local_cols], d.loc[trm, "log_y"])
            pred_log[te.to_numpy(), 6] = m.predict(d.loc[te, local_cols])

        # Same month in the previous year.
        pred_log[test.to_numpy(), 7] = d.loc[test, "lag12"].to_numpy(float)

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

    return {
        "time": d["time_index"].to_numpy(np.int32),
        "market": d["market_id"].to_numpy(np.int16),
        "context_X": context_X,
        "candidate_pred": candidate_pred.astype(np.float32),
        "y": d["sales"].to_numpy(np.float32),
        "year": d["year"].to_numpy(np.int16),
        "month": d["month"].to_numpy(np.int8),
    }


def _recent_rank_shift_markets():
    d = _full_table()
    p = np.log1p(np.maximum(d["candidate_pred"].astype(float), 0.0))
    y = np.log1p(np.maximum(d["y"].astype(float), 0.0))
    out = []
    for market_id in range(46):
        old = (d["market"] == market_id) & (d["year"] >= 2004) & (d["year"] <= 2008)
        recent = (d["market"] == market_id) & (d["year"] >= 2009) & (d["year"] <= 2011)
        if int(old.sum()) < 12 or int(recent.sum()) < 12:
            continue
        old_rmse = np.sqrt(np.mean((p[old] - y[old, None]) ** 2, axis=0))
        recent_rmse = np.sqrt(np.mean((p[recent] - y[recent, None]) ** 2, axis=0))
        old_best = int(np.argmin(old_rmse))
        recent_best = int(np.argmin(recent_rmse))
        gain = float(recent_rmse[old_best] - recent_rmse[recent_best])
        if old_best != recent_best and gain > 0.025:
            out.append(market_id)
    return set(out)


def _slice_ids(data):
    groups = _selector_groups()
    market = np.asarray(data["market"], int)
    pred_log = np.log1p(np.maximum(np.asarray(data["candidate_pred"], float), 0.0))
    spread = np.std(pred_log, axis=1)
    rank_shift = _recent_rank_shift_markets()
    masks = [
        np.isin(market, list(groups["full"])),
        np.isin(market, list(groups["stale"])),
        np.isin(market, list(groups["short"])),
        np.isin(market, list(groups["unseen"])),
        np.isin(market, list(rank_shift)),
        spread >= np.quantile(spread, 0.70),
        np.isin(data["month"], [3, 4, 5, 6, 7, 8]),
        np.asarray(data["year"]) >= 2014,
    ]
    return np.vstack(masks).astype(bool)


def generate_train():
    d = _full_table()
    groups = _selector_groups()
    base = (d["year"] <= 2011) & ~((d["year"] == 2011) & (d["month"] >= 10))
    market = np.asarray(d["market"], int)
    train = base & ~np.isin(market, list(groups["unseen"]))
    train &= ~(np.isin(market, list(groups["stale"])) & (d["year"] > 2008))
    train &= ~(np.isin(market, list(groups["short"])) & (d["year"] < 2009))
    return {
        k: np.asarray(v[train]).copy()
        for k, v in d.items()
        if k in ("time", "market", "context_X", "candidate_pred", "y")
    }


def generate_public():
    d = _full_table()
    m = (d["year"] == 2011) & (d["month"] >= 10)
    return {
        k: np.asarray(v[m]).copy()
        for k, v in d.items()
        if k in ("time", "market", "context_X", "candidate_pred")
    }


def generate_hidden():
    d = _full_table()
    m = d["year"] >= 2012
    h = {
        k: np.asarray(v[m]).copy()
        for k, v in d.items()
        if k in ("time", "market", "context_X", "candidate_pred", "y", "year", "month")
    }
    h["slice_mask"] = _slice_ids(h)
    return h

