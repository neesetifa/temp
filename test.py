test main
import numpy as np

from generator import (
    HIDDEN_VINTAGE, HORIZON_MAX, HORIZON_MIN, SELECTOR_LABEL_END,
    TRAIN_VINTAGES, generate_hidden, generate_train,
)


def test_hidden_module_imports():
    import hidden_eval
    assert callable(hidden_eval.evaluate)


def test_longitudinal_split_preflight():
    train = generate_train()
    hidden = generate_hidden()

    tt = np.asarray(train["time"], int)
    tm = np.asarray(train["market"], int)
    tc = np.asarray(train["context_X"], float)
    ht = np.asarray(hidden["time"], int)
    hm = np.asarray(hidden["market"], int)
    hc = np.asarray(hidden["context_X"], float)

    assert tt.size and ht.size
    assert int(tt.max()) == SELECTOR_LABEL_END
    assert int(tt.max()) < int(ht.min())

    train_keys = set(zip(tt.tolist(), tm.tolist()))
    hidden_keys = set(zip(ht.tolist(), hm.tolist()))
    assert train_keys.isdisjoint(hidden_keys)

    train_vintage = tt - np.rint(tc[:, 0]).astype(int)
    hidden_horizon = np.rint(hc[:, 0]).astype(int)
    assert set(np.unique(train_vintage)).issubset(set(np.asarray(TRAIN_VINTAGES, int).tolist()))
    assert int(train_vintage.max()) < HIDDEN_VINTAGE
    assert np.array_equal(hidden_horizon, ht - HIDDEN_VINTAGE)
    assert int(hidden_horizon.min()) == HORIZON_MIN
    assert int(hidden_horizon.max()) == HORIZON_MAX
    assert int(ht.min()) == SELECTOR_LABEL_END + HORIZON_MIN


def test_no_exact_hidden_label_lookup_path():
    train = generate_train()
    hidden = generate_hidden()
    train_map = {
        (int(t), int(m)): float(y)
        for t, m, y in zip(train["time"], train["market"], train["y"])
    }
    hits = [
        i for i, (t, m) in enumerate(zip(hidden["time"], hidden["market"]))
        if (int(t), int(m)) in train_map
    ]
    assert hits == []

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
    "short_horizon",
    "medium_horizon",
    "long_horizon",
    "rank_shift_market",
], dtype=object)

# Historical pseudo-deployments used as selector backtests.
TRAIN_VINTAGES = np.array([48, 60, 72, 84, 96, 108], dtype=int)   # months from 2000-01
PUBLIC_VINTAGE = 112
HIDDEN_VINTAGE = 120
SELECTOR_LABEL_END = 120
HORIZON_MIN = 4
HORIZON_MAX = 46


def _market_order(names):
    keyed = sorted((hashlib.sha256(str(x).encode()).hexdigest(), str(x)) for x in names)
    return {name: i for i, (_, name) in enumerate(keyed)}


def _selector_groups():
    ids = np.arange(46, dtype=int)
    keyed = sorted(
        (hashlib.sha256(f"selector-v04-{int(i)}".encode()).hexdigest(), int(i))
        for i in ids
    )
    ordered = [i for _, i in keyed]
    return {
        "unseen": set(ordered[:7]),
        "stale": set(ordered[7:15]),
        "short": set(ordered[15:23]),
        "full": set(ordered[23:]),
    }


def _ridge(cols, alpha):
    cats = [c for c in ("market_id", "month") if c in cols]
    nums = [c for c in cols if c not in cats]
    parts = []
    if nums:
        parts.append((
            "num",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            nums,
        ))
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return make_pipeline(ColumnTransformer(parts), Ridge(alpha=float(alpha)))


def _prepare_frame():
    d = pd.read_csv(SEED_PATH)
    d = d[d["city"].notna() & d["sales"].notna()].copy()
    order = _market_order(d["city"].unique())
    d["market_id"] = d["city"].map(order).astype(int)
    first_year = int(d["year"].min())
    d["time_index"] = ((d["year"] - first_year) * 12 + (d["month"] - 1)).astype(int)
    d = d.sort_values(["market_id", "time_index"]).reset_index(drop=True)
    d["log_y"] = np.log1p(d["sales"].astype(float))
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12.0)
    return d


def _market_traits(d, cutoff):
    hist = d[d["time_index"] <= cutoff].copy()
    rows = []
    for mid, q in hist.groupby("market_id"):
        q = q.sort_values("time_index")
        z = q["log_y"].to_numpy(float)
        if len(z) < 18:
            continue
        mm = q.groupby("month")["log_y"].mean()
        seasonality = float(mm.std())
        x = q["time_index"].to_numpy(float)
        trend = float(np.polyfit(x - x.mean(), z, 1)[0]) if len(z) >= 6 else 0.0
        vol = float(np.std(np.diff(z))) if len(z) >= 3 else float(np.std(z))
        rows.append((int(mid), float(np.mean(z)), vol, trend, seasonality, float(len(z))))
    arr = np.asarray(rows, float)
    return arr


def _latest_cycle(q, cutoff):
    q = q[q["time_index"] <= cutoff].sort_values("time_index")
    if q.empty:
        return np.zeros(12), 0.0
    recent = q[q["time_index"] >= cutoff - 35]
    month_mean = recent.groupby("month")["log_y"].mean()
    global_mean = float(recent["log_y"].mean())
    cyc = np.array([float(month_mean.get(m, global_mean)) for m in range(1, 13)], float)
    x = recent["time_index"].to_numpy(float)
    z = recent["log_y"].to_numpy(float)
    trend = float(np.polyfit(x - x.mean(), z, 1)[0]) if len(z) >= 12 else 0.0
    return cyc, trend


def _forecast_vintage(d, cutoff):
    """Generate actual model forecasts using only rows available by cutoff."""
    target = d[(d["time_index"] >= cutoff + HORIZON_MIN) & (d["time_index"] <= cutoff + HORIZON_MAX)].copy()
    if target.empty:
        return None
    hist = d[(d["time_index"] <= cutoff) & d["log_y"].notna()].copy()
    mids = np.sort(d["market_id"].unique())
    pred = np.zeros((len(target), CANDIDATE_COUNT), dtype=float)

    # Features available at issuance: identity, calendar, absolute time, forecast horizon.
    target["horizon"] = target["time_index"] - cutoff
    hist["horizon"] = 0.0
    pooled_cols = ["market_id", "month", "time_index", "month_sin", "month_cos"]
    broad_cols = ["month", "time_index", "month_sin", "month_cos"]

    # 0. Long-history pooled market+seasonal trend model.
    m0 = _ridge(pooled_cols, 18.0)
    m0.fit(hist[pooled_cols], hist["log_y"])
    pred[:, 0] = m0.predict(target[pooled_cols])

    # 1. Recent pooled model: adapts faster, less stable at long horizons.
    h1 = hist[hist["time_index"] >= cutoff - 47]
    m1 = _ridge(pooled_cols, 10.0)
    m1.fit(h1[pooled_cols], h1["log_y"])
    pred[:, 1] = m1.predict(target[pooled_cols])

    # 2. Broad no-market model.
    m2 = _ridge(broad_cols, 22.0)
    m2.fit(hist[broad_cols], hist["log_y"])
    pred[:, 2] = m2.predict(target[broad_cols])

    # 3. Cluster specialist based on pre-cutoff real market behavior.
    traits = _market_traits(d, cutoff)
    T = traits[:, 1:5].copy()
    T = np.nan_to_num(T, nan=np.nanmedian(T, axis=0))
    T = (T - T.mean(0)) / np.maximum(T.std(0), 1e-6)
    labels = KMeans(n_clusters=4, random_state=23, n_init=20).fit_predict(T)
    cluster_by_mid = {int(traits[i, 0]): int(labels[i]) for i in range(len(traits))}
    pred[:, 3] = pred[:, 0]
    for k in range(4):
        train_mids = [mid for mid, lab in cluster_by_mid.items() if lab == k]
        tr = hist[hist["market_id"].isin(train_mids) & (hist["time_index"] >= cutoff - 83)]
        if len(tr) < 100:
            continue
        mk = _ridge(pooled_cols, 8.0)
        mk.fit(tr[pooled_cols], tr["log_y"])
        te = target["market_id"].map(cluster_by_mid).fillna(-1).eq(k)
        if te.any():
            pred[te.to_numpy(), 3] = mk.predict(target.loc[te, pooled_cols])

    # 4. Per-market seasonal profile + local trend, with pooled fallback.
    pred[:, 4] = pred[:, 0]
    for mid in mids:
        te = target["market_id"].eq(mid)
        if not te.any():
            continue
        q = hist[hist["market_id"].eq(mid)]
        if len(q) < 24:
            continue
        cyc, trend = _latest_cycle(q, cutoff)
        hz = target.loc[te, "horizon"].to_numpy(float)
        mo = target.loc[te, "month"].to_numpy(int)
        # Damped extrapolation: deliberately strong short horizon, conservative long horizon.
        damp = (1.0 - np.exp(-hz / 18.0)) * 18.0
        pred[te.to_numpy(), 4] = cyc[mo - 1] + trend * damp

    # 5. Market-month historical mean with shrinkage toward global month profile.
    gm = hist.groupby("month")["log_y"].mean()
    pred[:, 5] = np.array([float(gm.get(int(m), hist["log_y"].mean())) for m in target["month"]])
    for mid in mids:
        q = hist[hist["market_id"].eq(mid)]
        te = target["market_id"].eq(mid)
        if len(q) < 12 or not te.any():
            continue
        lm = q.groupby("month")["log_y"].agg(["mean", "count"])
        vals = []
        for m in target.loc[te, "month"].to_numpy(int):
            if m in lm.index:
                n = float(lm.loc[m, "count"])
                vals.append((n * float(lm.loc[m, "mean"]) + 8.0 * float(gm.get(m, hist["log_y"].mean()))) / (n + 8.0))
            else:
                vals.append(float(gm.get(m, hist["log_y"].mean())))
        pred[te.to_numpy(), 5] = np.asarray(vals)

    # 6. Short-window local trend model, stronger when a market has enough recent rows.
    pred[:, 6] = pred[:, 1]
    for mid in mids:
        tr = hist[hist["market_id"].eq(mid) & (hist["time_index"] >= cutoff - 35)]
        te = target["market_id"].eq(mid)
        if len(tr) < 24 or not te.any():
            continue
        ml = _ridge(["month", "time_index", "month_sin", "month_cos"], 14.0)
        ml.fit(tr[["month", "time_index", "month_sin", "month_cos"]], tr["log_y"])
        pred[te.to_numpy(), 6] = ml.predict(target.loc[te, ["month", "time_index", "month_sin", "month_cos"]])

    # 7. Last observed 12-month cycle repeated forward, with no trend extrapolation.
    pred[:, 7] = pred[:, 2]
    for mid in mids:
        q = hist[hist["market_id"].eq(mid)].sort_values("time_index")
        te = target["market_id"].eq(mid)
        if len(q) < 12 or not te.any():
            continue
        last = q.tail(12)
        by_month = {int(m): float(v) for m, v in zip(last["month"], last["log_y"])}
        fallback = float(last["log_y"].mean())
        pred[te.to_numpy(), 7] = np.array([by_month.get(int(m), fallback) for m in target.loc[te, "month"]], float)

    # Market traits visible to the policy. These are all based on pre-cutoff history.
    trmap = {int(r[0]): r[1:] for r in traits}
    ctx = np.zeros((len(target), 8), dtype=float)
    ctx[:, 0] = target["horizon"].to_numpy(float)
    ctx[:, 1] = target["month_sin"].to_numpy(float)
    ctx[:, 2] = target["month_cos"].to_numpy(float)
    for i, mid in enumerate(target["market_id"].to_numpy(int)):
        vals = trmap.get(int(mid), np.array([hist["log_y"].mean(), 0.2, 0.0, 0.2, 0.0]))
        ctx[i, 3] = vals[0]  # scale
        ctx[i, 4] = vals[1]  # volatility
        ctx[i, 5] = vals[2]  # trend
        ctx[i, 6] = vals[3]  # seasonality
        ctx[i, 7] = vals[4]  # history months

    return {
        "time": target["time_index"].to_numpy(np.int32),
        "market": target["market_id"].to_numpy(np.int16),
        "context_X": ctx.astype(np.float32),
        "candidate_pred": np.maximum(np.expm1(pred), 0.0).astype(np.float32),
        "y": target["sales"].to_numpy(np.float32),
        "year": target["year"].to_numpy(np.int16),
        "month": target["month"].to_numpy(np.int8),
        "horizon": target["horizon"].to_numpy(np.int16),
        "vintage": np.full(len(target), cutoff, dtype=np.int16),
    }


@lru_cache(maxsize=1)
def _all_vintages():
    d = _prepare_frame()
    out = {}
    for v in list(TRAIN_VINTAGES) + [PUBLIC_VINTAGE, HIDDEN_VINTAGE]:
        out[int(v)] = _forecast_vintage(d, int(v))
    return out


def _rank_shift_markets(train_full, hidden):
    pt = np.log1p(np.maximum(train_full["candidate_pred"].astype(float), 0.0))
    yt = np.log1p(np.maximum(train_full["y"].astype(float), 0.0))
    ph = np.log1p(np.maximum(hidden["candidate_pred"].astype(float), 0.0))
    yh = np.log1p(np.maximum(hidden["y"].astype(float), 0.0))
    out = []
    for mid in np.unique(hidden["market"]):
        qt = train_full["market"] == mid
        qh = hidden["market"] == mid
        if qt.sum() < 30 or qh.sum() < 20:
            continue
        old = np.mean((pt[qt] - yt[qt, None]) ** 2, axis=0)
        new = np.mean((ph[qh] - yh[qh, None]) ** 2, axis=0)
        if int(np.argmin(old)) != int(np.argmin(new)) and np.sqrt(old.min()) < np.sqrt(old[int(np.argmin(new))]) + 0.25:
            gain = float(np.sqrt(new[int(np.argmin(old))]) - np.sqrt(new.min()))
            if gain > 0.015:
                out.append(int(mid))
    return set(out)


def _concat(items):
    keys = items[0].keys()
    return {k: np.concatenate([np.asarray(x[k]) for x in items], axis=0) for k in keys}


def _base_train_unfiltered():
    vint = _all_vintages()
    return _concat([vint[int(v)] for v in TRAIN_VINTAGES])


def generate_train():
    x = _base_train_unfiltered()
    groups = _selector_groups()
    market = np.asarray(x["market"], int)
    vintage = np.asarray(x["vintage"], int)
    keep = ~np.isin(market, list(groups["unseen"]))
    keep &= np.asarray(x["time"], int) <= SELECTOR_LABEL_END
    # stale markets only have the first vintage; short-history markets only the latest vintage.
    keep &= ~(np.isin(market, list(groups["stale"])) & (vintage > TRAIN_VINTAGES[0]))
    keep &= ~(np.isin(market, list(groups["short"])) & (vintage < TRAIN_VINTAGES[-1]))
    return {k: np.asarray(v[keep]).copy() for k, v in x.items() if k in ("time", "market", "context_X", "candidate_pred", "y")}


def generate_public():
    x = _all_vintages()[PUBLIC_VINTAGE]
    m = np.asarray(x["horizon"]) <= 8
    return {k: np.asarray(v[m]).copy() for k, v in x.items() if k in ("time", "market", "context_X", "candidate_pred")}


def _slice_ids(hidden):
    groups = _selector_groups()
    market = np.asarray(hidden["market"], int)
    h = np.asarray(hidden["horizon"], int)
    train_full = generate_train()
    rank_shift = _rank_shift_markets(train_full, hidden)
    masks = [
        np.isin(market, list(groups["full"])),
        np.isin(market, list(groups["stale"])),
        np.isin(market, list(groups["short"])),
        np.isin(market, list(groups["unseen"])),
        (h >= 4) & (h <= 12),
        (h >= 13) & (h <= 28),
        (h >= 29) & (h <= 46),
        np.isin(market, list(rank_shift)),
    ]
    return np.vstack(masks).astype(bool)


def generate_hidden():
    x = _all_vintages()[HIDDEN_VINTAGE]
    h = {k: np.asarray(v).copy() for k, v in x.items()}
    h["slice_mask"] = _slice_ids(h)
    return h

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

OVERALL_GOOD = 0.2612
OVERALL_BAD = 0.5000
OVERALL_CATASTROPHIC = 0.80
SCORE_POWER = 2.0
SLICE_GOOD = np.array([
    0.2457, 0.2579, 0.3153, 0.2456,
    0.2294, 0.2377, 0.2940, 0.2561,
], dtype=float)
SLICE_BAD = np.array([
    0.3800, 0.5600, 0.8200, 0.5000,
    0.5000, 0.5200, 0.5600, 0.3800,
], dtype=float)
SLICE_WEIGHTS = np.array([
    0.10, 0.13, 0.16, 0.08,
    0.13, 0.14, 0.16, 0.10,
], dtype=float)


def _load_train(app):
    d = np.load(app / "train_data.npz")
    return (d["train_time"], d["train_market"], d["train_context_X"], d["train_candidate_pred"], d["train_y"])


def _load_module(path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solve module at {path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _run_in_process(solve_path, payload):
    mod = _load_module(solve_path)
    params = mod.fit_forecast_policy(*payload["train"])
    return [np.asarray(mod.predict_forecast(*x, params), dtype=float) for x in payload["inputs"]]


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
    if p.shape != (n,): raise ValueError(f"prediction shape must be ({n},), got {p.shape}")
    if not np.isfinite(p).all(): raise ValueError("predictions must be finite")
    if np.any(p < -1e-8): raise ValueError("predictions must be non-negative")


def _metrics(y, p, slice_mask):
    overall = _log_rmse(y, p)
    overall_score = _score(overall, OVERALL_GOOD, OVERALL_BAD)
    per = {}; weighted = 0.0; weight_total = 0.0
    for i, name in enumerate(SLICE_NAMES):
        mask = np.asarray(slice_mask[i], bool)
        if not np.any(mask): continue
        rmse = _log_rmse(np.asarray(y)[mask], np.asarray(p)[mask])
        sc = _score(rmse, float(SLICE_GOOD[i]), float(SLICE_BAD[i]))
        wt = float(SLICE_WEIGHTS[i]); weighted += wt * sc; weight_total += wt
        per[str(name)] = {"log_rmse": rmse, "score": sc, "n": int(mask.sum()), "weight": wt}
    slice_score = weighted / max(weight_total, 1e-12)
    reward = float(0.35 * overall_score + 0.65 * slice_score)
    if overall >= OVERALL_CATASTROPHIC: reward = 0.0
    return {"reward": reward, "overall_log_rmse": overall, "overall_score": overall_score, "slice_score": slice_score, "per_slice": per}


def evaluate(solve_path=None, app_dir=None, log_dir=None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"
    train = _load_train(app); hidden = generate_hidden()
    x = (hidden["time"], hidden["market"], hidden["context_X"], hidden["candidate_pred"])
    payload = {"train": train, "inputs": [x, x]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2: raise ValueError("candidate must return predictions for both verifier inputs")
    p0, p1 = preds; _validate(p0, len(hidden["y"])); _validate(p1, len(hidden["y"]))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8): raise ValueError("predictions must be deterministic for repeated inputs")
    metrics = _metrics(hidden["y"], p0, hidden["slice_mask"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics

if __name__ == "__main__":
    reward, metrics = evaluate(); print(json.dumps({"reward": reward, "overall_log_rmse": metrics["overall_log_rmse"]}, sort_keys=True))


reference
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HBIN_EDGES = np.array([12.0, 24.0, 36.0], dtype=float)
N_MARKETS = 46
LOCAL_SHRINK = 20.0
SIM_K = 8
SIM_SCALE = 0.65
SOFTMAX_TEMP = 0.001


def _lp(x):
    return np.log1p(np.maximum(np.asarray(x, float), 0.0))


def _pack(time, market, context_X, candidate_pred, y=None):
    out = {
        "time": np.asarray(time),
        "market": np.asarray(market),
        "context_X": np.asarray(context_X),
        "candidate_pred": np.asarray(candidate_pred),
    }
    if y is not None:
        out["y"] = np.asarray(y)
    return out


def _hbin(context_X, use_horizon=True):
    n = len(context_X)
    if not use_horizon:
        return np.ones(n, dtype=int)
    h = np.asarray(context_X, float)[:, 0]
    return np.searchsorted(HBIN_EDGES, h, side="right").astype(int)


def _market_traits(market, context_X):
    market = np.asarray(market, int)
    ctx = np.asarray(context_X, float)
    M = max(N_MARKETS, int(market.max()) + 1 if market.size else N_MARKETS)
    traits = np.zeros((M, 4), dtype=float)
    known = np.zeros(M, dtype=bool)
    for m in range(M):
        q = market == m
        if np.any(q):
            traits[m] = np.median(ctx[q, 3:7], axis=0)
            known[m] = True
    if np.any(known):
        med = np.median(traits[known], axis=0)
        sd = np.maximum(np.std(traits[known], axis=0), 1e-6)
    else:
        med = np.zeros(4)
        sd = np.ones(4)
    return traits, known, med, sd


def _fit_history_stats(train, *, use_market=True, use_horizon=True, use_similarity=True):
    market = np.asarray(train["market"], int)
    ctx = np.asarray(train["context_X"], float)
    p = _lp(train["candidate_pred"])
    y = _lp(train["y"])
    r = p - y[:, None]
    hb = _hbin(ctx, use_horizon)
    B = 4
    C = p.shape[1]
    M = max(N_MARKETS, int(market.max()) + 1 if market.size else N_MARKETS)

    gb = np.zeros((B, C), dtype=float)
    gm = np.zeros((B, C), dtype=float)
    lb = np.zeros((M, B, C), dtype=float)
    lm = np.zeros((M, B, C), dtype=float)
    cnt = np.zeros((M, B), dtype=float)

    for b in range(B):
        q = hb == b
        if not np.any(q):
            q = np.ones(len(hb), dtype=bool)
        gb[b] = np.mean(r[q], axis=0)
        gm[b] = np.mean(r[q] ** 2, axis=0)
        for m in range(M):
            z = q & (market == m)
            if use_market and np.any(z):
                lb[m, b] = np.mean(r[z], axis=0)
                lm[m, b] = np.mean(r[z] ** 2, axis=0)
                cnt[m, b] = float(np.sum(z))
            else:
                lb[m, b] = gb[b]
                lm[m, b] = gm[b]

    traits, known, med, sd = _market_traits(market, ctx)
    ztraits = (traits - med) / sd
    sb = np.zeros_like(lb)
    sm = np.zeros_like(lm)
    for m in range(M):
        if not use_similarity:
            sb[m] = gb
            sm[m] = gm
            continue
        ids = np.where(known & (np.arange(M) != m))[0]
        if len(ids) and m < len(known) and known[m]:
            dist = np.sqrt(np.sum((ztraits[ids] - ztraits[m]) ** 2, axis=1))
            order = np.argsort(dist)[:SIM_K]
            ids = ids[order]
            dist = dist[order]
            simw = np.exp(-SIM_SCALE * dist)
        elif len(ids):
            simw = np.ones(len(ids), dtype=float)
        else:
            simw = np.zeros(0, dtype=float)
        for b in range(B):
            if len(ids):
                w = simw * np.maximum(cnt[ids, b], 0.25)
                if float(np.sum(w)) > 1e-9:
                    sb[m, b] = np.average(lb[ids, b], axis=0, weights=w)
                    sm[m, b] = np.average(lm[ids, b], axis=0, weights=w)
                    continue
            sb[m, b] = gb[b]
            sm[m, b] = gm[b]

    return {
        "gb": gb,
        "gm": gm,
        "lb": lb,
        "lm": lm,
        "cnt": cnt,
        "sb": sb,
        "sm": sm,
        "use_market": bool(use_market),
        "use_horizon": bool(use_horizon),
        "use_similarity": bool(use_similarity),
    }


def _history_predict_log(query, stats, *, use_market=True, use_horizon=True, use_similarity=True):
    market = np.asarray(query["market"], int)
    ctx = np.asarray(query["context_X"], float)
    p = _lp(query["candidate_pred"])
    hb = _hbin(ctx, use_horizon)
    out = np.empty(len(p), dtype=float)
    M = stats["lb"].shape[0]

    for i in range(len(p)):
        m = int(market[i])
        b = int(hb[i])
        if not (0 <= m < M):
            bias = stats["gb"][b]
            mse = stats["gm"][b]
        else:
            n = float(stats["cnt"][m, b]) if use_market else 0.0
            a = n / (n + LOCAL_SHRINK)
            if use_similarity:
                pb = stats["sb"][m, b]
                pm = stats["sm"][m, b]
            else:
                pb = stats["gb"][b]
                pm = stats["gm"][b]
            bias = a * stats["lb"][m, b] + (1.0 - a) * pb
            mse = a * stats["lm"][m, b] + (1.0 - a) * pm
        z = -(mse - float(np.min(mse))) / SOFTMAX_TEMP
        z -= float(np.max(z))
        w = np.exp(z)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= 0.0:
            w = np.full(len(mse), 1.0 / len(mse), dtype=float)
        else:
            w /= sw
        out[i] = float(np.dot(w, p[i] - bias))
    return out


def _meta_features(time, market, context_X, candidate_pred, *, use_market=True, use_horizon=True):
    p = _lp(candidate_pred)
    ctx = np.asarray(context_X, float)
    t = np.asarray(time, float)
    m = np.asarray(market, int)
    if not use_horizon:
        ctx = ctx.copy()
        ctx[:, 0] = 24.0
    parts = [
        p,
        np.mean(p, axis=1, keepdims=True),
        np.std(p, axis=1, keepdims=True),
        np.median(p, axis=1)[:, None],
        ctx,
        (t / 170.0)[:, None],
    ]
    if use_market:
        M = max(N_MARKETS, int(m.max()) + 1 if m.size else N_MARKETS)
        oh = np.zeros((len(m), M), dtype=float)
        ok = (m >= 0) & (m < M)
        oh[np.where(ok)[0], m[ok]] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def _market_archive_state(time, market, context_X):
    time = np.asarray(time, int)
    market = np.asarray(market, int)
    horizon = np.rint(np.asarray(context_X, float)[:, 0]).astype(int)
    vintage = time - horizon
    M = max(N_MARKETS, int(market.max()) + 1 if market.size else N_MARKETS)
    count = np.bincount(np.clip(market, 0, M - 1), minlength=M).astype(int)
    last = np.full(M, -10_000, dtype=int)
    for m in range(M):
        q = market == m
        if np.any(q):
            last[m] = int(np.max(vintage[q]))
    return count, last


def _coverage_mix(count, last_vintage):
    count = int(count)
    last_vintage = int(last_vintage)
    if count <= 0:
        return None
    if last_vintage < 84:
        return 0.70
    if count < 20:
        return 0.0
    return 0.80


def fit_forecast_policy(train_time, train_market, train_context_X, train_candidate_pred, train_y):
    train = _pack(train_time, train_market, train_context_X, train_candidate_pred, train_y)
    stats = _fit_history_stats(train)
    X = _meta_features(train_time, train_market, train_context_X, train_candidate_pred)
    y = _lp(train_y)
    meta = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=20.0,
        random_state=23,
    )
    meta.fit(X, y)
    market_count, market_last = _market_archive_state(train_time, train_market, train_context_X)
    return {"train": train, "stats": stats, "meta": meta, "market_count": market_count, "market_last": market_last}


def _predict_log(time, market, context_X, candidate_pred, params, *,
                 use_market=True, use_freshness=True, use_horizon=True,
                 use_similarity=True, use_meta=True):
    del use_freshness  # archive freshness is represented by the amount of market/bin history available
    query = _pack(time, market, context_X, candidate_pred)
    stats = params.get("stats")
    if stats is None:
        stats = _fit_history_stats(params["train"], use_market=use_market,
                                   use_horizon=use_horizon, use_similarity=use_similarity)
    elif not (use_market and use_horizon and use_similarity):
        stats = _fit_history_stats(params["train"], use_market=use_market,
                                   use_horizon=use_horizon, use_similarity=use_similarity)
    zs = _history_predict_log(query, stats, use_market=use_market,
                              use_horizon=use_horizon, use_similarity=use_similarity)
    if not use_meta or params.get("meta") is None:
        return zs
    X = _meta_features(time, market, context_X, candidate_pred,
                       use_market=True, use_horizon=True)
    zm = np.asarray(params["meta"].predict(X), float)
    med = np.median(_lp(candidate_pred), axis=1)
    counts = np.asarray(params.get("market_count", np.zeros(N_MARKETS, dtype=int)), int)
    last = np.asarray(params.get("market_last", np.full(N_MARKETS, -10000, dtype=int)), int)
    qmarket = np.asarray(market, int)
    out = np.empty(len(zs), dtype=float)
    for i, m in enumerate(qmarket):
        count = int(counts[m]) if 0 <= m < len(counts) else 0
        last_v = int(last[m]) if 0 <= m < len(last) else -10000
        a = _coverage_mix(count, last_v)
        if a is None:
            out[i] = med[i]
        else:
            out[i] = a * zs[i] + (1.0 - a) * zm[i]
    return out


def predict_forecast(time, market, context_X, candidate_pred, params):
    z = _predict_log(time, market, context_X, candidate_pred, params)
    return np.maximum(np.expm1(z), 0.0)


