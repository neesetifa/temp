reference
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

TREND_SHRINK = 400.0
SEASON_SHRINK = 20.0
RIDGE_ALPHA = 10.0


def _log_pred(candidate_pred):
    return np.log1p(np.maximum(np.asarray(candidate_pred, float), 0.0))


def _fit_temporal_stats(time, market, candidate_pred, y):
    time = np.asarray(time, int)
    market = np.asarray(market, int)
    p = _log_pred(candidate_pred)
    ylog = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    residual = p - ylog[:, None]
    n_candidates = p.shape[1]

    t0 = float(np.mean(time))
    ts = float(np.std(time) + 1e-9)
    x = (time - t0) / ts
    design = np.column_stack([np.ones(len(x)), x])

    global_coef = np.zeros((n_candidates, 2), dtype=float)
    for j in range(n_candidates):
        global_coef[j] = np.linalg.lstsq(design, residual[:, j], rcond=None)[0]

    global_trend = global_coef[:, 0][None, :] + x[:, None] * global_coef[:, 1][None, :]
    detrended = residual - global_trend
    global_season = np.zeros((12, n_candidates), dtype=float)
    for month in range(12):
        q = (time % 12) == month
        if np.any(q):
            global_season[month] = detrended[q].mean(axis=0)

    n_markets = 46
    market_coef = np.tile(global_coef[None, :, :], (n_markets, 1, 1))
    market_season = np.tile(global_season[None, :, :], (n_markets, 1, 1))
    market_count = np.zeros(n_markets, dtype=float)

    for market_id in np.unique(market):
        q = market == market_id
        n = int(q.sum())
        market_count[market_id] = n
        if n < 6:
            continue

        xm = x[q]
        rm = residual[q]
        d = np.column_stack([np.ones(n), xm])
        shrink = n / (n + TREND_SHRINK)
        coef = np.zeros((n_candidates, 2), dtype=float)
        for j in range(n_candidates):
            local = np.linalg.lstsq(d, rm[:, j], rcond=None)[0]
            coef[j] = shrink * local + (1.0 - shrink) * global_coef[j]
        market_coef[market_id] = coef

        remaining = rm - (coef[:, 0][None, :] + xm[:, None] * coef[:, 1][None, :])
        months = time[q] % 12
        for month in range(12):
            z = months == month
            nz = int(z.sum())
            if nz:
                season_shrink = nz / (nz + SEASON_SHRINK)
                market_season[market_id, month] = (
                    season_shrink * remaining[z].mean(axis=0)
                    + (1.0 - season_shrink) * global_season[month]
                )

    return {
        "t0": t0,
        "ts": ts,
        "global_coef": global_coef,
        "global_season": global_season,
        "market_coef": market_coef,
        "market_season": market_season,
        "market_count": market_count,
    }


def _temporal_median(time, market, candidate_pred, stats):
    time = np.asarray(time, int)
    market = np.asarray(market, int)
    p = _log_pred(candidate_pred)
    out = np.empty(len(time), dtype=float)

    for i, (t, market_id) in enumerate(zip(time, market)):
        x = (float(t) - stats["t0"]) / stats["ts"]
        month = int(t) % 12
        if 0 <= market_id < len(stats["market_count"]) and stats["market_count"][market_id] > 0:
            coef = stats["market_coef"][market_id]
            season = stats["market_season"][market_id, month]
        else:
            coef = stats["global_coef"]
            season = stats["global_season"][month]
        expected_residual = coef[:, 0] + x * coef[:, 1] + season
        out[i] = np.median(p[i] - expected_residual)
    return out


def _fit_ridge(candidate_pred, y):
    p = _log_pred(candidate_pred)
    ylog = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    return Ridge(alpha=RIDGE_ALPHA).fit(p, ylog)


def _choose_blend(time, market, context_X, candidate_pred, y):
    del context_X
    time = np.asarray(time, int)
    ylog = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    year = 2000 + time // 12
    folds = []

    for validation_year, weight in ((2009, 0.5), (2010, 2.0), (2011, 5.0)):
        fit = year < validation_year
        val = year == validation_year
        if int(fit.sum()) < 100 or int(val.sum()) < 20:
            continue
        stats = _fit_temporal_stats(time[fit], np.asarray(market)[fit], np.asarray(candidate_pred)[fit], np.asarray(y)[fit])
        temporal = _temporal_median(time[val], np.asarray(market)[val], np.asarray(candidate_pred)[val], stats)
        ridge = _fit_ridge(np.asarray(candidate_pred)[fit], np.asarray(y)[fit])
        ridge_pred = ridge.predict(_log_pred(np.asarray(candidate_pred)[val]))
        folds.append((weight, temporal, ridge_pred, ylog[val]))

    if not folds:
        return 0.75

    best_loss = np.inf
    best_alpha = 0.75
    for alpha in np.linspace(0.0, 1.0, 21):
        total = 0.0
        total_weight = 0.0
        for fold_weight, temporal, ridge_pred, target in folds:
            pred = alpha * temporal + (1.0 - alpha) * ridge_pred
            total += fold_weight * float(np.sum((pred - target) ** 2))
            total_weight += fold_weight * len(target)
        loss = total / max(total_weight, 1.0)
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    return best_alpha


def fit_forecast_policy(
    train_time,
    train_market,
    train_context_X,
    train_candidate_pred,
    train_y,
):
    alpha = _choose_blend(
        train_time,
        train_market,
        train_context_X,
        train_candidate_pred,
        train_y,
    )
    return {
        "temporal": _fit_temporal_stats(
            train_time,
            train_market,
            train_candidate_pred,
            train_y,
        ),
        "ridge": _fit_ridge(train_candidate_pred, train_y),
        "alpha": alpha,
    }


def predict_forecast(time, market, context_X, candidate_pred, params):
    del context_X
    temporal = _temporal_median(time, market, candidate_pred, params["temporal"])
    ridge_pred = params["ridge"].predict(_log_pred(candidate_pred))
    alpha = float(params["alpha"])
    z = alpha * temporal + (1.0 - alpha) * ridge_pred
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
OVERALL_GOOD = 0.1365
OVERALL_BAD = 0.1515
SLICE_GOOD = np.array([0.1445, 0.1275, 0.1155, 0.1365, 0.1295, 0.1505, 0.1265, 0.1215], float)
SLICE_BAD = np.array([0.1600, 0.1450, 0.1350, 0.1530, 0.1480, 0.1680, 0.1430, 0.1430], float)


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

        m = _ridge(full_cols, 20.0)
        m.fit(d.loc[base, full_cols], d.loc[base, "log_y"])
        pred_log[test.to_numpy(), 0] = m.predict(d.loc[test, full_cols])

        tr = base & d["time_index"].ge(cutoff_t - 30)
        m = _ridge(full_cols, 8.0)
        m.fit(d.loc[tr, full_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 1] = m.predict(d.loc[test, full_cols])

        tr = base & d["time_index"].ge(cutoff_t - 48)
        m = _ridge(lag_cols, 8.0)
        m.fit(d.loc[tr, lag_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 2] = m.predict(d.loc[test, lag_cols])

        for market_id in market_ids:
            te = test & d["market_id"].eq(market_id)
            trm = base & d["market_id"].eq(market_id) & d["time_index"].ge(cutoff_t - 30)
            if int(trm.sum()) < 12:
                continue
            m = _ridge(local_cols, 20.0)
            m.fit(d.loc[trm, local_cols], d.loc[trm, "log_y"])
            pred_log[te.to_numpy(), 3] = m.predict(d.loc[te, local_cols])

        tr = base & d["time_index"].ge(cutoff_t - 36)
        m = _ridge(pooled_cols, 8.0)
        m.fit(d.loc[tr, pooled_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 5] = m.predict(d.loc[test, pooled_cols])

        tr = base & d["time_index"].ge(cutoff_t - 36)
        m = _ridge(cross_cols, 8.0)
        m.fit(d.loc[tr, cross_cols], d.loc[tr, "log_y"])
        pred_log[test.to_numpy(), 6] = m.predict(d.loc[test, cross_cols])

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


