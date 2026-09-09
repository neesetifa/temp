# Forecast policy update

`app/solve.py` still applies one simple rule to all of the forecast columns. Replace that policy.

The reporting system already has several forecasting models. Their sales-count predictions are stored in `candidate_pred`, and the candidate columns keep the same meaning between the labeled archive and deployment rows. Historical archive rows also include the final observed sales count, so they can be used as backtests for the existing forecasts.

The archive is uneven across markets. Some market IDs have several forecast vintages with labels, some histories are older, some only have a recent portion of the archive, and some deployment-time markets have no labeled selector rows. Candidate quality also changes with how far ahead the forecast is being made, so a policy tuned only to short-range backtests is not enough for the deployment period.

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

`candidate_pred` has one row per monthly forecast record and one column per existing forecasting model. Values are non-negative sales-count predictions.

`time` is a monthly index and `market` is an anonymized market ID. `context_X` contains information available with the forecast record. Its first column is the forecast horizon in months; the remaining columns describe calendar and historical market profile information. The column layout is unchanged between fitting and prediction.

The deployment rows cover forecasts roughly 4 to 46 months beyond the corresponding labeled archive cutoff. Treat the problem as out-of-time deployment rather than an IID random holdout.

Forecast quality is measured using
`RMSE(log1p(prediction), log1p(observed_sales))`.
Evaluation considers both overall performance and performance on several held-out cohorts. The cohort definitions and scoring thresholds are not part of the API contract.

Return one non-negative finite sales-count prediction for every input row, in the original row order. You may select candidates, combine them, or learn a policy from the archived backtests.

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
from sklearn.linear_model import Ridge

HEDGES = np.array([4, 13, 29, 47], dtype=int)
KNN = 8
RECENCY_SCALE = 34.0
LOCAL_SHRINK = 18.0


def _lp(x):
    return np.log1p(np.maximum(np.asarray(x, float), 0.0))


def _hb(h, use_horizon=True):
    if not use_horizon:
        return np.ones(len(np.asarray(h)), dtype=int)
    return np.clip(np.searchsorted(HEDGES[1:-1], np.asarray(h, float), side="right"), 0, 2).astype(int)


def _stats_features(base, query, *, use_market=True, use_freshness=True,
                    use_horizon=True, use_similarity=True):
    bt = np.asarray(base["time"], int)
    bm = np.asarray(base["market"], int)
    bc = np.asarray(base["context_X"], float)
    bp = _lp(base["candidate_pred"])
    by = _lp(base["y"])
    br = bp - by[:, None]

    qt = np.asarray(query["time"], int)
    qm = np.asarray(query["market"], int)
    qc = np.asarray(query["context_X"], float)
    qp = _lp(query["candidate_pred"])
    qh = qc[:, 0]
    qhb = _hb(qh, use_horizon)

    bh = bc[:, 0]
    bhb = _hb(bh, use_horizon)
    bv = bt - np.rint(bh).astype(int)
    vmax = int(np.max(bv)) if len(bv) else 0
    C = bp.shape[1]
    M = max(46, int(max(np.max(bm) if len(bm) else 0, np.max(qm) if len(qm) else 0)) + 1)

    gb = np.zeros((3, C)); gm = np.zeros((3, C))
    for b in range(3):
        z = bhb == b
        if not z.any(): z = np.ones(len(bhb), bool)
        w = np.ones(z.sum())
        sw = max(float(w.sum()), 1e-9)
        gb[b] = (w[:, None] * br[z]).sum(0) / sw
        gm[b] = (w[:, None] * br[z] ** 2).sum(0) / sw

    lb = np.zeros((M, 3, C)); lm = np.zeros((M, 3, C)); cnt = np.zeros((M, 3)); last = np.full(M, -9999, int)
    traits = np.zeros((M, 4)); tok = np.zeros(M, bool)
    for m in range(M):
        z = bm == m
        if z.any():
            last[m] = int(np.max(bv[z]))
            latest = np.max(bv[z])
            zz = z & (bv == latest)
            traits[m] = np.median(bc[zz, 3:7], axis=0)
            tok[m] = True
        for b in range(3):
            q = z & (bhb == b)
            if not q.any():
                lb[m, b] = gb[b]; lm[m, b] = gm[b]
                continue
            w = np.ones(q.sum())
            sw = max(float(w.sum()), 1e-9)
            lb[m, b] = (w[:, None] * br[q]).sum(0) / sw
            lm[m, b] = (w[:, None] * br[q] ** 2).sum(0) / sw
            cnt[m, b] = sw

    if tok.any():
        med = np.median(traits[tok], axis=0); sd = np.maximum(np.std(traits[tok], axis=0), 1e-6)
    else:
        med = np.zeros(4); sd = np.ones(4)
    zt = (traits - med) / sd
    sb = np.zeros_like(lb); sm = np.zeros_like(lm)
    for m in range(M):
        if not use_similarity:
            for b in range(3): sb[m,b]=gb[b]; sm[m,b]=gm[b]
            continue
        ids = np.where(tok & (np.arange(M) != m))[0]
        if len(ids) and tok[m]:
            dist = np.sqrt(np.sum((zt[ids] - zt[m]) ** 2, axis=1))
            order = np.argsort(dist)[:KNN]; ids = ids[order]; dist = dist[order]
            simw = np.exp(-0.65 * dist)
        elif len(ids):
            simw = np.ones(len(ids))
        else:
            simw = np.zeros(0)
        for b in range(3):
            if len(ids):
                w = simw * np.maximum(cnt[ids,b], 0.25)
                if w.sum() > 1e-9:
                    sb[m,b] = np.average(lb[ids,b], axis=0, weights=w)
                    sm[m,b] = np.average(lm[ids,b], axis=0, weights=w)
                    continue
            sb[m,b]=gb[b]; sm[m,b]=gm[b]

    rows=[]
    for i in range(len(qt)):
        m=int(qm[i]); b=int(qhb[i]); h=float(qh[i])
        gbi=gb[b]; gmi=gm[b]
        if 0 <= m < M and use_similarity:
            pri_b=sb[m,b]; pri_m=sm[m,b]
        else:
            pri_b=gbi; pri_m=gmi
        if use_market and 0 <= m < M and cnt[m,b] > 0:
            gap=max(vmax-last[m],0)
            fresh = 1.0 if (not use_freshness or gap <= 24) else float(np.exp(-(gap - 24) / 24.0))
            eff=cnt[m,b]*fresh
            shrink=LOCAL_SHRINK*(1.0 + (max(h-12.0,0.0)/30.0 if use_horizon else 0.0))
            a=eff/(eff+shrink)
            bi=a*lb[m,b]+(1-a)*pri_b; mi=a*lm[m,b]+(1-a)*pri_m
            count=cnt[m,b]; gapf=float(gap)
        else:
            bi=pri_b; mi=pri_m; count=0.0; gapf=96.0
        if not use_freshness: gapf=0.0
        hfeat=h if use_horizon else 20.0
        row=np.r_[qp[i], qp[i]-bi, gbi, gmi, pri_b, pri_m, bi, mi,
                  np.log1p(count), gapf/48.0, hfeat/46.0, qc[i,1:]]
        rows.append(row)
    return np.asarray(rows,float)


def _pack(time, market, context_X, candidate_pred, y=None):
    d={"time":np.asarray(time),"market":np.asarray(market),"context_X":np.asarray(context_X),"candidate_pred":np.asarray(candidate_pred)}
    if y is not None: d["y"]=np.asarray(y)
    return d


def fit_forecast_policy(train_time, train_market, train_context_X, train_candidate_pred, train_y):
    train=_pack(train_time,train_market,train_context_X,train_candidate_pred,train_y)
    ctx=np.asarray(train_context_X,float); time=np.asarray(train_time,int)
    vintage=time-np.rint(ctx[:,0]).astype(int)
    uniq=sorted(np.unique(vintage))
    Xs=[]; ys=[]; pg=[]; yg=[]; blocks=[]
    for vv in uniq:
        q=vintage==vv; prior=vintage<vv
        if prior.sum()<200 or q.sum()<50: continue
        base={k:np.asarray(v)[prior] for k,v in train.items()}
        query={k:np.asarray(v)[q] for k,v in train.items()}
        xb=_stats_features(base,query); yb=_lp(query["y"]); Xs.append(xb); ys.append(yb)
        # out-of-vintage global stack for blend calibration
        gr=Ridge(alpha=12.0).fit(_lp(base["candidate_pred"]),_lp(base["y"]))
        pgp=gr.predict(_lp(query["candidate_pred"])); pg.append(pgp); yg.append(yb); blocks.append((xb,yb,pgp))
    if not Xs:
        # defensive fallback
        gr=Ridge(alpha=12.0).fit(_lp(train_candidate_pred),_lp(train_y))
        return {"train":train,"stats_model":None,"global":gr,"blend":0.0}
    X=np.vstack(Xs); y=np.concatenate(ys)
    stats_model=Ridge(alpha=30.0).fit(X,y)
    # choose blend on the latest pseudo-deployment; it best matches final deployment distance.
    xb,yb,pgp=blocks[-1]
    sp=stats_model.predict(xb)
    best=(np.inf,0.7)
    for a in np.linspace(0.45,0.85,9):
        z=a*sp+(1-a)*pgp
        rm=float(np.sqrt(np.mean((z-yb)**2)))
        if rm<best[0]: best=(rm,float(a))
    global_model=Ridge(alpha=12.0).fit(_lp(train_candidate_pred),_lp(train_y))
    return {"train":train,"stats_model":stats_model,"global":global_model,"blend":best[1]}


def _predict_log(time,market,context_X,candidate_pred,params,**feature_kwargs):
    q=_pack(time,market,context_X,candidate_pred)
    zg=params["global"].predict(_lp(candidate_pred))
    if params.get("stats_model") is None: return zg
    X=_stats_features(params["train"],q,**feature_kwargs)
    zs=params["stats_model"].predict(X)
    a=float(params.get("blend",0.7))
    return a*zs+(1-a)*zg


def predict_forecast(time, market, context_X, candidate_pred, params):
    z=_predict_log(time,market,context_X,candidate_pred,params)
    return np.maximum(np.expm1(z),0.0)

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
TRAIN_VINTAGES = np.array([72, 96, 120], dtype=int)   # months from 2000-01
PUBLIC_VINTAGE = 132
HIDDEN_VINTAGE = 140
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
    train_full = _base_train_unfiltered()
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

OVERALL_GOOD = 0.1923
OVERALL_BAD = 0.3800
OVERALL_CATASTROPHIC = 0.55
SCORE_POWER = 2.0
SLICE_GOOD = np.array([
    0.1662, 0.1678, 0.2291, 0.2453,
    0.1942, 0.1905, 0.1929, 0.1885,
], dtype=float)
SLICE_BAD = np.array([
    0.3400, 0.4300, 0.3900, 0.4000,
    0.3400, 0.3600, 0.4300, 0.3900,
], dtype=float)
SLICE_WEIGHTS = np.array([
    0.14, 0.12, 0.12, 0.10,
    0.14, 0.14, 0.12, 0.12,
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
