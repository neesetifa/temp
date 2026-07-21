# Fix source readings

`app/solve.py` has the two stubs to fill in:

```python
fit_reading_model(train_X, train_y, train_source, train_entity)
predict_reading(X, source, entity, params)
```

Return one prediction per row.

Rows come from several collection sources. The field names line up, but the readings are not equally reliable from every source. The same entity can also show up more than once when it was read by more than one source.

The old pooled model was off on a few sources. A simple average of duplicate rows was not reliable either; it hid which source each reading came from.

Use the arrays passed in. No hidden files, network, outside data, or fitting on public eval answers. Aim to beat the plain pooled baseline on RMSE.


from __future__ import annotations

import time
import numpy as np
from pathlib import Path

from generators import make_eval


def _rmse(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _dec_reward(value, good, cutoff):
    value = float(value)
    if not np.isfinite(value) or value >= cutoff:
        return 0.0
    if value <= good:
        return 1.0
    return float((cutoff - value) / (cutoff - good))


def _entity_counts(entity):
    entity = np.asarray(entity)
    counts = np.zeros(len(entity), dtype=int)
    order = np.argsort(entity, kind="mergesort")
    e = entity[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and e[end] == e[start]:
            end += 1
        counts[order[start:end]] = end - start
        start = end
    return counts


def _entity_consistency(pred, entity):
    pred = np.asarray(pred, dtype=float)
    entity = np.asarray(entity)
    vals = []
    order = np.argsort(entity, kind="mergesort")
    e = entity[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and e[end] == e[start]:
            end += 1
        idx = order[start:end]
        if len(idx) > 1:
            vals.append(float(np.std(pred[idx])))
        start = end
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _metrics_for_case(y, pred, source, entity):
    source = np.asarray(source)
    entity = np.asarray(entity)
    counts = _entity_counts(entity)
    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "common": np.isin(source, [0, 1, 2, 3]),
        "rare": np.isin(source, [4, 6]),
        "rare_single": np.isin(source, [4, 6]) & (counts == 1),
        "source4": source == 4,
        "high_noise": source == 5,
        "bridge": source == 7,
        "paired": counts > 1,
        "single": counts == 1,
    }
    out = {}
    for name, mask in masks.items():
        if mask.any():
            out[name + "_rmse"] = _rmse(y[mask], pred[mask])
        else:
            out[name + "_rmse"] = np.nan
    out["entity_consistency"] = _entity_consistency(pred, entity)
    return out


def _score_from_metrics(m):
    # Hard invalid / catastrophic cutoffs.
    if (
        m["overall_rmse"] > 1.25
        or m["rare_rmse"] > 1.70
        or m["rare_single_rmse"] > 1.70
        or m["source4_rmse"] > 1.55
        or m["bridge_rmse"] > 1.90
        or m["high_noise_rmse"] > 1.65
        or m["single_rmse"] > 1.55
    ):
        return 0.0, {
            "overall": 0.0,
            "rare": 0.0,
            "high_noise": 0.0,
            "paired": 0.0,
            "bridge": 0.0,
            "single": 0.0,
            "consistency": 0.0,
        }

    comps = {
        "overall": _dec_reward(m["overall_rmse"], 0.43, 1.25),
        "rare": _dec_reward(m["rare_rmse"], 0.56, 1.70),
        "rare_single": _dec_reward(m["rare_single_rmse"], 0.70, 1.70),
        "source4": _dec_reward(m["source4_rmse"], 0.52, 1.55),
        "high_noise": _dec_reward(m["high_noise_rmse"], 0.70, 1.65),
        "paired": _dec_reward(m["paired_rmse"], 0.36, 1.20),
        "bridge": _dec_reward(m["bridge_rmse"], 0.55, 1.90),
        "single": _dec_reward(m["single_rmse"], 0.62, 1.55),
        "consistency": _dec_reward(m["entity_consistency"], 0.05, 0.55),
    }
    reward = (
        0.22 * comps["overall"]
        + 0.10 * comps["rare"]
        + 0.13 * comps["rare_single"]
        + 0.10 * comps["source4"]
        + 0.08 * comps["high_noise"]
        + 0.13 * comps["paired"]
        + 0.18 * comps["bridge"]
        + 0.03 * comps["single"]
        + 0.03 * comps["consistency"]
    )
    return float(np.clip(reward, 0.0, 1.0)), comps


def evaluate(fit_fn, predict_fn, train_path=None, seeds=(301, 302, 303, 304)):
    t0 = time.perf_counter()
    if train_path is None:
        train_path = Path(__file__).resolve().parents[1] / "app" / "train_data.npz"
    train = np.load(train_path)
    params = fit_fn(train["train_X"], train["train_y"], train["train_source"], train["train_entity"])

    per_seed = {}
    rewards = []
    for seed in seeds:
        data = make_eval(seed, split="hidden")
        X = data["X"].copy()
        source = data["source"].copy()
        entity = data["entity"].copy()
        pred1 = np.asarray(predict_fn(X, source, entity, params), dtype=float)
        # Determinism/permutation/input mutation checks.
        perm = np.random.default_rng(seed + 999).permutation(len(X))
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        pred_perm = np.asarray(predict_fn(X[perm].copy(), source[perm].copy(), entity[perm].copy(), params), dtype=float)[inv]
        valid = (
            pred1.shape == data["y"].shape
            and np.all(np.isfinite(pred1))
            and np.allclose(pred1, pred_perm, atol=1e-9, rtol=1e-9)
            and np.allclose(X, data["X"])
            and np.array_equal(source, data["source"])
            and np.array_equal(entity, data["entity"])
            and np.std(pred1) > 0.05
        )
        if not valid:
            metrics = {
                "overall_rmse": float("inf"),
                "common_rmse": float("inf"),
                "rare_rmse": float("inf"),
                "rare_single_rmse": float("inf"),
                "source4_rmse": float("inf"),
                "high_noise_rmse": float("inf"),
                "bridge_rmse": float("inf"),
                "paired_rmse": float("inf"),
                "single_rmse": float("inf"),
                "entity_consistency": float("inf"),
                "invalid": 1.0,
            }
            reward, comps = 0.0, {
                "overall": 0.0,
                "rare": 0.0,
                "rare_single": 0.0,
                "source4": 0.0,
                "high_noise": 0.0,
                "paired": 0.0,
                "bridge": 0.0,
                "single": 0.0,
                "consistency": 0.0,
            }
        else:
            metrics = _metrics_for_case(data["y"], pred1, source, entity)
            reward, comps = _score_from_metrics(metrics)
        rewards.append(reward)
        per_seed[str(seed)] = {
            "reward": float(reward),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "component_rewards": {k: float(v) for k, v in comps.items()},
        }

    # Aggregate metrics across seed metrics.
    keys = [k for k in per_seed[str(seeds[0])]["metrics"].keys() if k != "invalid"]
    aggregate_metrics = {}
    for k in keys:
        vals = [per_seed[str(s)]["metrics"].get(k, np.nan) for s in seeds]
        aggregate_metrics[k] = float(np.nanmean(vals))
    agg_reward, agg_comps = _score_from_metrics(aggregate_metrics)
    # Blend aggregate score with mean per-seed score to avoid one seed hiding another.
    final_reward = float(0.70 * agg_reward + 0.30 * np.mean(rewards))
    return {
        "reward": final_reward,
        "passed_cutoff": bool(final_reward > 0.0),
        "aggregate_metrics": aggregate_metrics,
        "aggregate_component_rewards": agg_comps,
        "per_seed": per_seed,
        "elapsed_seconds": time.perf_counter() - t0,
    }

from __future__ import annotations

import numpy as np


def _interactions(X):
    X = np.asarray(X, dtype=float)
    pairs = [(0, 1), (2, 3), (3, 4), (5, 6), (0, 6), (2, 5)]
    return np.column_stack([X[:, i] * X[:, j] for i, j in pairs])


def _base_features(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([
        np.ones(X.shape[0]),
        X,
        X ** 2,
        _interactions(X),
        np.sin(np.clip(X[:, [2, 5]], -5, 5)),
    ])


def _design_reference(X, source, known_sources=None):
    X = np.asarray(X, dtype=float)
    source = np.asarray(source, dtype=int)
    if known_sources is None:
        known_sources = np.arange(7, dtype=int)
    known_sources = np.asarray(known_sources, dtype=int)
    base = _base_features(X)
    parts = [base]
    # Source intercepts and source-specific linear corrections for train-known sources.
    for s in known_sources:
        m = (source == int(s)).astype(float)[:, None]
        parts.append(m)
        parts.append(m * X)
        # selected nonlinear corrections help biased sources without making
        # rare-source fits completely separate.
        parts.append(m * X[:, [0, 1, 3, 6]])
    return np.column_stack(parts)


def _design_base(X):
    return _base_features(X)


def _fit_ridge(Phi, y, lam=1.0):
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = Phi.mean(axis=0)
    sd = Phi.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (Phi - mu) / sd
    Z[:, 0] = Phi[:, 0]  # keep intercept unstandardized
    A = Z.T @ Z
    reg = lam * np.eye(A.shape[0])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A + reg, Z.T @ y)
    return {"coef": coef, "mu": mu, "sd": sd}


def _predict_ridge(model, Phi):
    Phi = np.asarray(Phi, dtype=float)
    Z = (Phi - model["mu"]) / model["sd"]
    Z[:, 0] = Phi[:, 0]
    return Z @ model["coef"]


def _entity_weighted_average(values, weights, entity):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    entity = np.asarray(entity)
    out = np.empty_like(values, dtype=float)
    # Stable deterministic grouping without assuming entity ids are dense.
    order = np.argsort(entity, kind="mergesort")
    ent_sorted = entity[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ent_sorted[end] == ent_sorted[start]:
            end += 1
        idx = order[start:end]
        w = weights[idx]
        if not np.all(np.isfinite(w)) or w.sum() <= 1e-12:
            avg = float(values[idx].mean())
        else:
            avg = float(np.sum(values[idx] * w) / np.sum(w))
        out[idx] = avg
        start = end
    return out


def fit_reading_model(train_X, train_y, train_source, train_entity):
    train_X = np.asarray(train_X, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    train_source = np.asarray(train_source, dtype=int)

    known_sources = np.sort(np.unique(train_source))
    Phi = _design_reference(train_X, train_source, known_sources=known_sources)
    model = _fit_ridge(Phi, train_y, lam=8.0)
    row_pred = _predict_ridge(model, Phi)
    resid = train_y - row_pred

    global_var = float(np.var(resid) + 1e-4)
    source_var = {}
    source_bias = {}
    for s in known_sources:
        mask = train_source == int(s)
        n = int(mask.sum())
        if n == 0:
            continue
        r = resid[mask]
        # Shrink rare-source variance and residual bias toward global.
        alpha = n / (n + 80.0)
        source_var[int(s)] = float(alpha * np.var(r) + (1 - alpha) * global_var + 1e-4)
        source_bias[int(s)] = float(alpha * np.mean(r))

    per_source_models = {}
    for s in known_sources:
        mask = train_source == int(s)
        # Separate models are useful for sources with a strong calibration quirk,
        # but rare sources still share the main model through blending at predict time.
        if int(mask.sum()) >= 90:
            per_source_models[int(s)] = _fit_ridge(_design_base(train_X[mask]), train_y[mask], lam=4.0)

    return {
        "model": model,
        "known_sources": known_sources,
        "global_var": global_var,
        "source_var": source_var,
        "source_bias": source_bias,
        "per_source_models": per_source_models,
    }


def predict_reading(X, source, entity, params):
    X = np.asarray(X, dtype=float)
    source = np.asarray(source, dtype=int)
    entity = np.asarray(entity)
    Phi = _design_reference(X, source, known_sources=params["known_sources"])
    row = _predict_ridge(params["model"], Phi)

    bias = np.array([params["source_bias"].get(int(s), 0.0) for s in source], dtype=float)
    row = row + bias

    # For sources with enough labeled support, blend in a separate local model.
    # This improves heavily biased sources without losing the shared fallback.
    for s, local_model in params.get("per_source_models", {}).items():
        mask = source == int(s)
        if mask.any():
            local = _predict_ridge(local_model, _design_base(X[mask]))
            row[mask] = 0.50 * row[mask] + 0.50 * local

    # Unknown sources are allowed but lower-confidence. Known high-noise sources
    # are downweighted by residual variance.
    var = np.array([
        params["source_var"].get(int(s), params["global_var"] * 3.0)
        for s in source
    ], dtype=float)
    weights = 1.0 / np.maximum(var, 1e-4)
    ent = _entity_weighted_average(row, weights, entity)

    # For duplicated entities, entity evidence is very useful. For singletons,
    # this is identical to the row prediction.
    counts = np.zeros(len(row), dtype=int)
    order = np.argsort(entity, kind="mergesort")
    ent_sorted = entity[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ent_sorted[end] == ent_sorted[start]:
            end += 1
        idx = order[start:end]
        counts[idx] = end - start
        start = end
    blend = np.where(counts > 1, 0.03 * row + 0.97 * ent, row)
    return np.asarray(blend, dtype=float)

from __future__ import annotations

import numpy as np


D = 8
N_SOURCES = 8


def _rng(seed):
    return np.random.default_rng(seed)


SCALE = np.array([
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [1.18, 0.83, 1.04, 1.10, 0.78, 1.00, 0.92, 1.12],
    [0.88, 1.20, 0.90, 1.03, 1.15, 0.84, 1.08, 0.95],
    [1.05, 0.94, 1.10, 0.86, 1.05, 1.20, 0.84, 1.02],
    [1.62, 0.55, 1.42, 0.62, 1.52, 0.54, 1.36, 1.24],
    [0.95, 1.04, 0.98, 1.12, 0.94, 1.02, 1.07, 0.91],
    [0.54, 1.70, 0.62, 1.58, 0.66, 1.38, 0.52, 1.55],
    [1.46, 0.58, 1.34, 0.68, 1.42, 0.60, 1.26, 1.20],
], dtype=float)

OFFSET = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.18, -0.12, 0.05, 0.10, -0.18, 0.02, 0.08, -0.04],
    [-0.15, 0.16, -0.08, 0.04, 0.20, -0.10, -0.02, 0.12],
    [0.05, -0.05, 0.16, -0.14, 0.06, 0.10, -0.16, 0.02],
    [0.62, -0.52, 0.36, -0.44, 0.48, -0.30, 0.28, 0.20],
    [0.02, 0.04, -0.03, 0.08, -0.05, 0.02, 0.00, -0.02],
    [-0.66, 0.58, -0.34, 0.46, -0.52, 0.28, -0.40, 0.42],
    [0.52, -0.46, 0.30, -0.38, 0.42, -0.24, 0.22, 0.18],
], dtype=float)

NOISE = np.array([0.08, 0.13, 0.17, 0.14, 0.22, 0.52, 0.35, 0.38], dtype=float)


def latent_target(z):
    z = np.asarray(z)
    return (
        0.85 * z[..., 0]
        - 0.70 * z[..., 1]
        + 0.55 * z[..., 2] ** 2
        - 0.42 * z[..., 3] * z[..., 4]
        + 0.35 * np.sin(z[..., 5])
        + 0.38 * z[..., 6] * z[..., 0]
        - 0.28 * z[..., 7] ** 2
        + 0.20 * np.cos(z[..., 2] - z[..., 5])
    )


def read_source(z, source, rng):
    z = np.asarray(z, dtype=float)
    s = int(source)
    x = z * SCALE[s] + OFFSET[s]
    # Source-specific cross-talk. Keep it mostly linear so it can be learned
    # from overlap data, but source identity still matters.
    if s == 3:
        x = x.copy()
        x[..., 2] += 0.48 * z[..., 0]
        x[..., 4] -= 0.36 * z[..., 5]
    elif s == 4:
        x = x.copy()
        x[..., 0] += 0.68 * z[..., 1]
        x[..., 6] -= 0.52 * z[..., 2]
        x[..., 4] += 0.34 * z[..., 3]
    elif s == 5:
        x = x.copy()
        x[..., 1] += 0.25 * z[..., 3]
        x[..., 7] -= 0.18 * z[..., 4]
    elif s == 6:
        x = x.copy()
        x[..., 0] += 0.92 * z[..., 1]
        x[..., 3] -= 0.62 * z[..., 6]
        x[..., 2] += 0.40 * z[..., 4]
    elif s == 7:
        x = x.copy()
        x[..., 0] += 0.72 * z[..., 1]
        x[..., 6] -= 0.48 * z[..., 2]
        x[..., 3] += 0.34 * z[..., 5]
    return x + rng.normal(0.0, NOISE[s], size=x.shape)


def sample_latents(n_entities, rng, shift=None):
    z = rng.normal(0.0, 1.0, size=(n_entities, D))
    if shift == "rare":
        z[:, 0] += 0.7
        z[:, 2] -= 0.5
        z[:, 6] += 0.4
    elif shift == "bridge":
        z[:, 1] -= 0.6
        z[:, 4] += 0.6
        z[:, 7] += 0.3
    elif shift == "high_noise":
        z[:, 3] += 0.5
        z[:, 5] -= 0.4
    return z


def _choose_sources_train(rng):
    k = rng.choice([1, 2, 3], p=[0.48, 0.38, 0.14])
    probs = np.array([0.27, 0.21, 0.165, 0.145, 0.075, 0.105, 0.030])
    probs = probs / probs.sum()
    return rng.choice(np.arange(7), size=k, replace=False, p=probs)


def _choose_sources_eval(rng, kind):
    if kind == "common":
        k = rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
        return rng.choice([0, 1, 2, 3], size=k, replace=False)
    if kind == "rare":
        k = rng.choice([1, 2, 3], p=[0.40, 0.45, 0.15])
        first = rng.choice([4, 6], p=[0.72, 0.28])
        pool = [0, 1, 2, 3, 5]
        rest = [] if k == 1 else list(rng.choice(pool, size=k-1, replace=False))
        return np.array([first] + rest, dtype=int)
    if kind == "high_noise":
        k = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
        rest = [] if k == 1 else list(rng.choice([0, 1, 2, 3], size=k-1, replace=False))
        return np.array([5] + rest, dtype=int)
    if kind == "bridge":
        # Source 7 has no labeled training support. Most bridge entities carry
        # a known-source reading too; a few do not.
        if rng.random() < 0.78:
            known = int(rng.choice([0, 1, 2, 3, 5], p=[0.26, 0.22, 0.20, 0.18, 0.14]))
            if rng.random() < 0.22:
                known2 = int(rng.choice([0, 1, 2, 3]))
                if known2 == known:
                    return np.array([7, known], dtype=int)
                return np.array([7, known, known2], dtype=int)
            return np.array([7, known], dtype=int)
        return np.array([7], dtype=int)
    raise ValueError(kind)


def make_train(seed=101, n_entities=3300):
    rng = _rng(seed)
    z = sample_latents(n_entities, rng)
    y_entity = latent_target(z) + rng.normal(0, 0.06, size=n_entities)
    rows = []
    ys = []
    sources = []
    entities = []
    for eid in range(n_entities):
        for s in _choose_sources_train(rng):
            rows.append(read_source(z[eid], int(s), rng))
            ys.append(y_entity[eid] + rng.normal(0, 0.015))
            sources.append(int(s))
            entities.append(eid)
    return {
        "X": np.asarray(rows, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "source": np.asarray(sources, dtype=np.int64),
        "entity": np.asarray(entities, dtype=np.int64),
    }


def make_eval(seed=202, split="public"):
    rng = _rng(seed)
    if split == "public":
        plan = [("common", 520), ("rare", 90), ("high_noise", 120)]
    else:
        plan = [("common", 700), ("rare", 220), ("high_noise", 260), ("bridge", 330)]
    rows = []
    ys = []
    sources = []
    entities = []
    entity_kind = []
    eid = 0
    for kind, n in plan:
        shift = None
        if kind == "rare":
            shift = "rare"
        elif kind == "high_noise":
            shift = "high_noise"
        elif kind == "bridge":
            shift = "bridge"
        z = sample_latents(n, rng, shift=shift)
        y_entity = latent_target(z) + rng.normal(0, 0.06, size=n)
        for j in range(n):
            srcs = _choose_sources_eval(rng, kind)
            for s in srcs:
                rows.append(read_source(z[j], int(s), rng))
                ys.append(y_entity[j] + rng.normal(0, 0.015))
                sources.append(int(s))
                entities.append(eid)
                entity_kind.append(kind)
            eid += 1
    return {
        "X": np.asarray(rows, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "source": np.asarray(sources, dtype=np.int64),
        "entity": np.asarray(entities, dtype=np.int64),
        "kind": np.asarray(entity_kind),
    }

