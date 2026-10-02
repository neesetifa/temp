# ===== BEGIN FILE: tmp/environment/app/solve.py =====
"""Starter implementation for local_field_reconstruction.

Replace fit_field_model and predict_field_value with a stronger model.
"""
from __future__ import annotations

import numpy as np


def fit_field_model(
    train_context_X,
    train_query_X,
    train_obs_X,
    train_obs_code,
    train_obs_value,
    train_case_offsets,
    train_y,
):
    y = np.asarray(train_y, dtype=float)
    mean = float(np.mean(y)) if y.size else 0.0
    context_X = np.asarray(train_context_X, dtype=float)
    cat_means = {}
    if context_X.ndim == 2 and context_X.shape[1] >= 6:
        cats = np.argmax(context_X[:, :6], axis=1)
        for c in range(6):
            m = cats == c
            if np.sum(m) >= 10:
                cat_means[int(c)] = float(np.mean(y[m]))
    # Crude sensor-family calibration from nearby observations to target labels.
    obs_code = np.asarray(train_obs_code, dtype=int)
    obs_value = np.asarray(train_obs_value, dtype=float)
    obs_X = np.asarray(train_obs_X, dtype=float)
    offsets = np.asarray(train_case_offsets, dtype=int)
    diffs = [[] for _ in range(8)]
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        d = obs_X[s:e, 4]
        near = d <= np.quantile(d, 0.35)
        for fam, val in zip(obs_code[s:e, 0][near], obs_value[s:e, 0][near]):
            if 0 <= int(fam) < 8:
                diffs[int(fam)].append(float(val - y[i]))
    sensor_bias = np.array([np.median(v) if len(v) >= 20 else 0.0 for v in diffs], dtype=float)
    sensor_bias -= np.median(sensor_bias)
    return {"mean": mean, "cat_means": cat_means, "sensor_bias": sensor_bias}


def _predict_one(context_row, obs_X, obs_code, obs_value, params):
    mean = float(params.get("mean", 0.0)) if isinstance(params, dict) else 0.0
    if len(obs_value) == 0:
        return mean
    sensor_bias = np.asarray(params.get("sensor_bias", np.zeros(8)), dtype=float) if isinstance(params, dict) else np.zeros(8)
    sensor = np.asarray(obs_code[:, 0], dtype=int)
    values = np.asarray(obs_value[:, 0], dtype=float).copy()
    ok = (sensor >= 0) & (sensor < len(sensor_bias))
    values[ok] = values[ok] - sensor_bias[sensor[ok]]
    dist = np.clip(np.asarray(obs_X[:, 4], dtype=float), 1e-4, 10.0)
    noise = np.clip(np.asarray(obs_value[:, 1], dtype=float), 0.03, 5.0) if obs_value.shape[1] > 1 else 0.2
    w = np.exp(-0.5 * (dist / 0.28) ** 2) / (noise + 0.05)
    if float(np.sum(w)) <= 1e-12:
        pred = float(np.mean(values))
    else:
        pred = float(np.sum(w * values) / np.sum(w))
    # Light shrinkage to category prior for sparse/far cases.
    cat_means = params.get("cat_means", {}) if isinstance(params, dict) else {}
    prior = mean
    if context_row.shape[0] >= 6:
        c = int(np.argmax(context_row[:6]))
        prior = float(cat_means.get(c, mean))
    eff = float(np.sum(w) ** 2 / (np.sum(w * w) + 1e-12))
    lam = np.clip(eff / 5.0, 0.0, 1.0)
    return lam * pred + (1.0 - lam) * prior


def predict_field_value(
    context_X,
    query_X,
    obs_X,
    obs_code,
    obs_value,
    case_offsets,
    params,
):
    context_X = np.asarray(context_X, dtype=float)
    obs_X = np.asarray(obs_X, dtype=float)
    obs_code = np.asarray(obs_code, dtype=int)
    obs_value = np.asarray(obs_value, dtype=float)
    offsets = np.asarray(case_offsets, dtype=int)
    n = len(offsets) - 1
    out = np.empty(n, dtype=float)
    for i in range(n):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out[i] = _predict_one(context_X[i], obs_X[s:e], obs_code[s:e], obs_value[s:e], params)
    return out

# ===== END FILE: tmp/environment/app/solve.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Local field reconstruction

The implementation in `app/solve.py` is only a simple baseline. Improve it.

You are given a collection of cases. Each case has a query location and a packed set of nearby observations. The observations are irregularly placed and come from different sensor families and reading modes. Your job is to predict the true continuous field value at the query location.

Implement these functions in `app/solve.py`:

```python
def fit_field_model(
    train_context_X,
    train_query_X,
    train_obs_X,
    train_obs_code,
    train_obs_value,
    train_case_offsets,
    train_y,
):
    ...


def predict_field_value(
    context_X,
    query_X,
    obs_X,
    obs_code,
    obs_value,
    case_offsets,
    params,
):
    ...
```

`case_offsets` packs observations by case. For case `i`, its observations are in `obs_X[case_offsets[i]:case_offsets[i+1]]`, with matching rows in `obs_code` and `obs_value`.

Return one finite floating-point prediction per case. Predictions must be deterministic and must not mutate input arrays.

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "solve.py"
DATA = np.load(ROOT / "app" / "train_data.npz", allow_pickle=False)
PUBLIC = np.load(ROOT / "app" / "public_eval.npz", allow_pickle=False)


def load_mod():
    spec = importlib.util.spec_from_file_location("solve", str(APP))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def fit_args():
    return [
        DATA["train_context_X"], DATA["train_query_X"], DATA["train_obs_X"], DATA["train_obs_code"],
        DATA["train_obs_value"], DATA["train_case_offsets"], DATA["train_y"],
    ]


def pred_args():
    return [
        PUBLIC["context_X"], PUBLIC["query_X"], PUBLIC["obs_X"], PUBLIC["obs_code"],
        PUBLIC["obs_value"], PUBLIC["case_offsets"],
    ]


def test_api_shape_and_finite():
    mod = load_mod()
    params = mod.fit_field_model(*fit_args())
    pred = np.asarray(mod.predict_field_value(*pred_args(), params), dtype=float)
    assert pred.shape == PUBLIC["y"].shape
    assert np.all(np.isfinite(pred))


def test_predict_is_deterministic():
    mod = load_mod()
    params = mod.fit_field_model(*fit_args())
    p1 = np.asarray(mod.predict_field_value(*pred_args(), params), dtype=float)
    p2 = np.asarray(mod.predict_field_value(*pred_args(), params), dtype=float)
    assert np.allclose(p1, p2, rtol=0, atol=1e-12)


def test_does_not_mutate_inputs():
    mod = load_mod()
    fa = [x.copy() for x in fit_args()]
    before = [x.copy() for x in fa]
    params = mod.fit_field_model(*fa)
    for a, b in zip(fa, before):
        assert np.array_equal(a, b)
    pa = [x.copy() for x in pred_args()]
    before = [x.copy() for x in pa]
    _ = mod.predict_field_value(*pa, params)
    for a, b in zip(pa, before):
        assert np.array_equal(a, b)


def test_offsets_are_consistent():
    assert DATA["train_case_offsets"][0] == 0
    assert DATA["train_case_offsets"][-1] == DATA["train_obs_X"].shape[0]
    assert DATA["train_obs_X"].shape[0] == DATA["train_obs_code"].shape[0] == DATA["train_obs_value"].shape[0]
    assert DATA["train_case_offsets"].shape[0] == DATA["train_context_X"].shape[0] + 1
    assert PUBLIC["case_offsets"][0] == 0
    assert PUBLIC["case_offsets"][-1] == PUBLIC["obs_X"].shape[0]
    assert PUBLIC["obs_X"].shape[0] == PUBLIC["obs_code"].shape[0] == PUBLIC["obs_value"].shape[0]
    assert PUBLIC["case_offsets"].shape[0] == PUBLIC["context_X"].shape[0] + 1
    assert np.all(np.diff(DATA["train_case_offsets"]) > 0)
    assert np.all(np.diff(PUBLIC["case_offsets"]) > 0)

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/reference_solution.py =====
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def _safe_weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    sw = float(np.sum(weights))
    if sw <= 1e-12 or len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.sum(weights * values) / sw)
    var = float(np.sum(weights * (values - mu) ** 2) / sw)
    eff = float(sw * sw / (np.sum(weights * weights) + 1e-12))
    return mu, float(np.sqrt(max(var, 0.0))), eff, sw


def _weighted_local_linear(obs_X: np.ndarray, values: np.ndarray, weights: np.ndarray) -> float:
    sw = float(weights.sum())
    if sw <= 1e-9 or len(values) < 3:
        return float(np.average(values, weights=weights + 1e-9)) if len(values) else 0.0
    X = np.column_stack([np.ones(len(values)), obs_X[:, 2], obs_X[:, 3]])
    sqrtw = np.sqrt(weights + 1e-12)
    Xw = X * sqrtw[:, None]
    yw = values * sqrtw
    try:
        beta = np.linalg.solve(Xw.T @ Xw + 1e-4 * np.eye(3), Xw.T @ yw)
        return float(beta[0])
    except np.linalg.LinAlgError:
        return float(np.average(values, weights=weights + 1e-9))


def _estimate_sensor_bias(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, y):
    pairs_by_fam = [[] for _ in range(8)]
    vals_by_mode = [[] for _ in range(4)]
    for i in range(len(case_offsets)-1):
        s, e = int(case_offsets[i]), int(case_offsets[i+1])
        if e <= s:
            continue
        d = obs_X[s:e, 4].astype(float)
        near = d <= np.quantile(d, min(0.50, max(0.22, 7.0 / max(len(d), 1))))
        # Favor closer points as noisy target proxies; many cases make a robust calibration possible.
        for fam, md, val in zip(obs_code[s:e,0].astype(int)[near], obs_code[s:e,1].astype(int)[near], obs_value[s:e,0].astype(float)[near]):
            pairs_by_fam[int(fam)].append((float(y[i]), float(val)))
            vals_by_mode[int(md)].append(float(val - y[i]))
    fam_bias = np.zeros(8, dtype=float)
    fam_scale = np.ones(8, dtype=float)
    for f, pairs in enumerate(pairs_by_fam):
        if len(pairs) < 35:
            continue
        arr = np.asarray(pairs, dtype=float)
        yy = arr[:, 0]
        rr = arr[:, 1]
        # Robust-ish trimmed linear calibration: reading ~= scale * true + bias.
        lo, hi = np.quantile(rr - yy, [0.08, 0.92])
        keep = (rr - yy >= lo) & (rr - yy <= hi)
        yy2, rr2 = yy[keep], rr[keep]
        vy = float(np.var(yy2))
        if vy > 1e-5:
            slope = float(np.cov(yy2, rr2, bias=True)[0, 1] / vy)
            fam_scale[f] = float(np.clip(slope, 0.62, 1.42))
        fam_bias[f] = float(np.median(rr2 - fam_scale[f] * yy2))
    fam_bias = fam_bias - np.median(fam_bias)
    mode_bias = np.array([np.median(v) if len(v) >= 30 else 0.0 for v in vals_by_mode], dtype=float)
    mode_bias = mode_bias - np.median(mode_bias)
    return fam_bias, fam_scale, 0.40 * mode_bias


def _extract_features(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, *, variant="full", sensor_bias=None, mode_bias=None, sensor_scale=None):
    n = len(case_offsets) - 1
    feats = []
    bandwidths = [0.06, 0.11, 0.20, 0.38, 0.72]
    if sensor_bias is None:
        sensor_bias = np.zeros(8, dtype=float)
    if mode_bias is None:
        mode_bias = np.zeros(4, dtype=float)
    if sensor_scale is None:
        sensor_scale = np.ones(8, dtype=float)
    for i in range(n):
        s, e = int(case_offsets[i]), int(case_offsets[i + 1])
        ox = obs_X[s:e]
        oc = obs_code[s:e]
        ov = obs_value[s:e]
        raw_values = ov[:, 0].astype(float)
        noise = np.clip(ov[:, 1].astype(float), 0.03, 2.5)
        quality = np.clip(ov[:, 2].astype(float), 0.01, 60.0)
        dist = np.clip(ox[:, 4].astype(float), 0.0, 5.0)
        sensor = oc[:, 0].astype(int)
        mode = oc[:, 1].astype(int)
        if variant == "no_sensor":
            values = raw_values.copy()
        else:
            values = (raw_values - sensor_bias[sensor] - mode_bias[mode]) / np.clip(sensor_scale[sensor], 0.5, 1.8)
        row = []
        ctx = context_X[i].astype(float).copy()
        if variant == "no_sparse":
            # Remove the context/cohort fallback used when local observations are sparse or far.
            ctx[:6] = 0.0
            ctx[12:] = 0.0
            if len(raw_values) <= 7 or (len(dist) and float(np.min(dist)) > 0.22):
                ctx[:] = 0.0
        if variant == "no_anisotropy":
            # Do not let the ablation recover directionality from the visible orientation fields.
            ctx[6:8] = 0.0
            ctx[10:12] = 0.0
        row.extend(ctx.tolist())
        row.extend(query_X[i].astype(float).tolist())
        row.extend([len(values), np.log1p(len(values)), dist.min() if len(dist) else 3.0, dist.mean() if len(dist) else 3.0, dist.std() if len(dist) else 0.0])
        if len(values) == 0:
            feats.append(row + [0.0] * 340)
            continue
        order = np.argsort(dist)
        for k in [1, 2, 3, 5, 8]:
            kk = min(k, len(order))
            row.extend([float(values[order[:kk]].mean()), float(values[order[:kk]].std() if kk > 1 else 0.0), float(dist[order[kk - 1]])])
            if variant != "no_sensor":
                row.append(float(raw_values[order[:kk]].mean() - values[order[:kk]].mean()))
        row.extend([float(values.mean()), float(values.std()), float(values.min()), float(values.max()), float(np.median(values))])
        # Raw isotropic multi-scale kernels, quality-weighted variants, and local linear estimates.
        for bw in bandwidths:
            w = np.exp(-0.5 * (dist / bw) ** 2)
            if variant != "no_sensor":
                wq = w * np.sqrt(quality) / (noise + 0.04)
            else:
                wq = w
            row.extend(_safe_weighted_stats(values, w))
            row.extend(_safe_weighted_stats(values, wq))
            row.append(_weighted_local_linear(ox, values, wq))
        # Anisotropic kernels based on visible context orientation.
        if variant != "no_anisotropy":
            cos_t = float(context_X[i, 6])
            sin_t = float(context_X[i, 7])
            dx, dy = ox[:, 2].astype(float), ox[:, 3].astype(float)
            along = dx * cos_t + dy * sin_t
            across = -dx * sin_t + dy * cos_t
            for along_bw, across_bw in [(0.12, 0.045), (0.22, 0.055), (0.36, 0.070), (0.80, 0.055), (0.60, 0.085), (0.42, 0.13), (0.95, 0.20), (0.15, 0.55)]:
                w = np.exp(-0.5 * ((along / along_bw) ** 2 + (across / across_bw) ** 2))
                if variant != "no_sensor":
                    w = w * np.sqrt(quality) / (noise + 0.04)
                row.extend(_safe_weighted_stats(values, w))
                row.append(_weighted_local_linear(ox, values, w))
            # Directional contrasts and cross-boundary disagreement.
            masks = [along >= 0, along < 0, across >= 0, across < 0, np.abs(across) < 0.10, np.abs(across) > 0.22]
            side_stats = []
            for signmask in masks:
                if signmask.any():
                    ww = np.exp(-0.5 * (dist[signmask] / 0.32) ** 2)
                    st = _safe_weighted_stats(values[signmask], ww)[:3]
                    row.extend(st)
                    side_stats.append(st)
                else:
                    row.extend([0.0, 0.0, 0.0])
                    side_stats.append((0.0, 0.0, 0.0))
            # Explicit directional contrasts; generic aggregates do not know which side of a ridge/boundary produced the readings.
            row.extend([
                side_stats[0][0] - side_stats[1][0],
                side_stats[2][0] - side_stats[3][0],
                side_stats[4][0] - side_stats[5][0],
                0.5 * (side_stats[2][0] + side_stats[3][0]),
                0.5 * (side_stats[0][0] + side_stats[1][0]),
                abs(side_stats[2][0] - side_stats[3][0]),
                abs(side_stats[0][0] - side_stats[1][0]),
                side_stats[4][2] - side_stats[5][2],
                (side_stats[2][0] - side_stats[3][0]) * float(context_X[i, 11]),
                (side_stats[4][0] - side_stats[5][0]) * float(context_X[i, 10]),
                0.5 * (side_stats[2][0] + side_stats[3][0]) * float(context_X[i, 11]),
            ])
        # Sensor family-specific local estimates.
        if variant != "no_sensor":
            for fam in range(8):
                m = sensor == fam
                row.append(float(m.mean()))
                row.append(float(m.sum()))
                if m.any():
                    d = dist[m]
                    v = values[m]
                    nse = noise[m]
                    for bw in [0.13, 0.30, 0.70]:
                        w = np.exp(-0.5 * (d / bw) ** 2) / (nse + 0.04)
                        row.extend(_safe_weighted_stats(v, w)[:3])
                    row.extend([float(v.mean()), float(v.std() if len(v) > 1 else 0.0), float(d.min())])
                else:
                    row.extend([0.0] * (3 * 3 + 3))
            for md in range(4):
                m = mode == md
                row.append(float(m.mean()))
                if m.any():
                    w = np.exp(-0.5 * (dist[m] / 0.32) ** 2)
                    row.extend(_safe_weighted_stats(values[m], w)[:2])
                else:
                    row.extend([0.0, 0.0])
        # Sparse/context fallback indicators.
        if variant != "no_sparse":
            sparse = float(len(values) <= 7)
            far = float((dist.min() if len(dist) else 99.0) > 0.22)
            eff_short = _safe_weighted_stats(values, np.exp(-0.5 * (dist / 0.16) ** 2))[2]
            row.extend([sparse, far, eff_short, sparse * context_X[i, -1], sparse * values.mean(), far * values.mean()])
            # Explicit sparse/cohort interactions. These help only when examples are poorly observed.
            row.extend((sparse * context_X[i, :6]).astype(float).tolist())
            row.extend((far * context_X[i, :6]).astype(float).tolist())
            row.extend((max(0.0, 2.5 - eff_short) * context_X[i, :6]).astype(float).tolist())
        feats.append(row)
    maxlen = max(len(r) for r in feats)
    out = np.zeros((len(feats), maxlen), dtype=np.float64)
    for i, r in enumerate(feats):
        out[i, :len(r)] = r
    out[~np.isfinite(out)] = 0.0
    return out


class FieldModel:
    def __init__(self, variant="full"):
        self.variant = variant
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.y_mean = 0.0
        self.sensor_bias = np.zeros(8, dtype=float)
        self.sensor_scale = np.ones(8, dtype=float)
        self.mode_bias = np.zeros(4, dtype=float)

    def fit(self, context_X, query_X, obs_X, obs_code, obs_value, case_offsets, y):
        self.y_mean = float(np.mean(y))
        if self.variant != "no_sensor":
            self.sensor_bias, self.sensor_scale, self.mode_bias = _estimate_sensor_bias(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, y)
        X = _extract_features(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, variant=self.variant, sensor_bias=self.sensor_bias, mode_bias=self.mode_bias, sensor_scale=self.sensor_scale)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, context_X, query_X, obs_X, obs_code, obs_value, case_offsets):
        X = _extract_features(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, variant=self.variant, sensor_bias=self.sensor_bias, mode_bias=self.mode_bias, sensor_scale=self.sensor_scale)
        Xs = self.scaler.transform(X)
        return np.asarray(self.model.predict(Xs), dtype=np.float64)


def fit_field_model(train_context_X, train_query_X, train_obs_X, train_obs_code, train_obs_value, train_case_offsets, train_y):
    return FieldModel("full").fit(train_context_X, train_query_X, train_obs_X, train_obs_code, train_obs_value, train_case_offsets, train_y)


def predict_field_value(context_X, query_X, obs_X, obs_code, obs_value, case_offsets, params):
    return params.predict(context_X, query_X, obs_X, obs_code, obs_value, case_offsets)

# ===== END FILE: tmp/solution/reference_solution.py =====

# ===== BEGIN FILE: tmp/tests/agent_call.py =====
#!/usr/bin/env python3
"""Sandbox-side agent runner for local_field_reconstruction."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REQUIRED = ("fit_field_model", "predict_field_value")


def _load(path: str | os.PathLike):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_solution", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(path: str | os.PathLike) -> bool:
    try:
        mod = _load(path)
        return all(callable(getattr(mod, name, None)) for name in REQUIRED)
    except Exception:
        return False


def _arrays_equal(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _save_error(out_path: str | os.PathLike, code: str, detail: str = "") -> None:
    np.savez(out_path, n=np.array(0, dtype=np.int64), error=np.array(str(code)), detail=np.array(str(detail)[:1200]))


def run_eval(solve_path: str | os.PathLike, in_path: str | os.PathLike, out_path: str | os.PathLike) -> int:
    try:
        with np.load(in_path, allow_pickle=False) as z:
            train_context_X = np.asarray(z["train_context_X"], dtype=np.float64)
            train_query_X = np.asarray(z["train_query_X"], dtype=np.float64)
            train_obs_X = np.asarray(z["train_obs_X"], dtype=np.float64)
            train_obs_code = np.asarray(z["train_obs_code"], dtype=np.int64)
            train_obs_value = np.asarray(z["train_obs_value"], dtype=np.float64)
            train_case_offsets = np.asarray(z["train_case_offsets"], dtype=np.int64)
            train_y = np.asarray(z["train_y"], dtype=np.float64)
            context_X = np.asarray(z["context_X"], dtype=np.float64)
            query_X = np.asarray(z["query_X"], dtype=np.float64)
            obs_X = np.asarray(z["obs_X"], dtype=np.float64)
            obs_code = np.asarray(z["obs_code"], dtype=np.int64)
            obs_value = np.asarray(z["obs_value"], dtype=np.float64)
            case_offsets = np.asarray(z["case_offsets"], dtype=np.int64)
        mod = _load(solve_path)
        if not all(callable(getattr(mod, name, None)) for name in REQUIRED):
            _save_error(out_path, "api_missing", "required fit_field_model/predict_field_value not found")
            return 0

        fit_args = [
            train_context_X.copy(), train_query_X.copy(), train_obs_X.copy(), train_obs_code.copy(),
            train_obs_value.copy(), train_case_offsets.copy(), train_y.copy(),
        ]
        fit_before = [x.copy() for x in fit_args]
        params = mod.fit_field_model(*fit_args)
        if not all(_arrays_equal(a, b) for a, b in zip(fit_args, fit_before)):
            _save_error(out_path, "input_mutation_in_fit", "fit_field_model modified one or more input arrays")
            return 0

        pred_args = [context_X.copy(), query_X.copy(), obs_X.copy(), obs_code.copy(), obs_value.copy(), case_offsets.copy()]
        pred_before = [x.copy() for x in pred_args]
        pred1 = np.asarray(mod.predict_field_value(*pred_args, params), dtype=np.float64)
        if not all(_arrays_equal(a, b) for a, b in zip(pred_args, pred_before)):
            _save_error(out_path, "input_mutation_in_predict", "predict_field_value modified one or more input arrays")
            return 0
        pred2_args = [context_X.copy(), query_X.copy(), obs_X.copy(), obs_code.copy(), obs_value.copy(), case_offsets.copy()]
        pred2 = np.asarray(mod.predict_field_value(*pred2_args, params), dtype=np.float64)
        if pred1.shape != pred2.shape or not np.allclose(pred1, pred2, rtol=0, atol=1e-10, equal_nan=True):
            _save_error(out_path, "nondeterministic_prediction", "two predict calls with the same params differed")
            return 0
        np.savez(out_path, n=np.array(1, dtype=np.int64), p0=pred1.astype(np.float64))
        return 0
    except Exception:
        _save_error(out_path, "runtime_error", traceback.format_exc())
        return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: agent_call.py SOLVE_PATH {check|eval} [IN_NPZ OUT_NPZ]", file=sys.stderr)
        return 2
    solve_path = argv[1]
    cmd = argv[2]
    if cmd == "check":
        ok = check(solve_path)
        if ok:
            print(json.dumps({"ok": True}))
            return 0
        print(json.dumps({"ok": False}), file=sys.stderr)
        return 1
    if cmd == "eval":
        if len(argv) != 5:
            print("eval requires IN_NPZ and OUT_NPZ", file=sys.stderr)
            return 2
        return run_eval(solve_path, argv[3], argv[4])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# ===== END FILE: tmp/tests/agent_call.py =====

# ===== BEGIN FILE: tmp/tests/generator.py =====
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SLICE_NAMES = np.array([
    "stable_dense_local",
    "sparse_observations",
    "nearest_sensor_noisy",
    "biased_sensor_cluster",
    "anisotropic_ridge",
    "wrong_side_boundary",
    "local_bump_near_query",
    "trend_dominated",
    "rare_sensor_code",
    "mixed_scale_field",
    "short_range_vs_long_range_conflict",
], dtype=object)

CATEGORY_EFFECT = np.array([-0.85, -0.42, 0.00, 0.42, 0.78, 1.18], dtype=float)
SENSOR_BIAS = np.array([-0.26, 0.22, 0.02, 0.42, -0.22, 0.78, -0.78, 0.12], dtype=float)
SENSOR_NOISE = np.array([0.07, 0.10, 0.17, 0.28, 0.11, 0.40, 0.26, 0.06], dtype=float)
SENSOR_SCALE = np.array([1.00, 0.92, 1.08, 1.15, 0.86, 1.25, 0.78, 1.02], dtype=float)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _field(params: Dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = params["theta"]
    ct, st = np.cos(theta), np.sin(theta)
    u = x * ct + y * st
    v = -x * st + y * ct
    base = params["base"] + params["trend_x"] * x + params["trend_y"] * y
    ridge = params["ridge_amp"] * np.sin(params["ridge_freq"] * u + params["phase"]) * np.exp(-0.5 * ((v - params["ridge_center"]) / params["ridge_width"]) ** 2)
    bump = params["bump_amp"] * np.exp(-0.5 * (((x - params["bump_x"]) ** 2 + (y - params["bump_y"]) ** 2) / (params["bump_sigma"] ** 2)))
    boundary = params["boundary_amp"] * np.tanh((v - params["boundary_center"]) / params["boundary_width"])
    weak = 0.08 * np.sin(5.0 * x - 2.2 * y + 0.6 * params["phase"])
    return base + ridge + bump + boundary + weak


def _case_params(rng: np.random.Generator, slice_id: int) -> Dict[str, float]:
    cat = int(rng.integers(0, 6))
    theta = rng.uniform(-np.pi, np.pi)
    trend_scale = 0.28
    ridge_amp = rng.normal(0.0, 0.35)
    ridge_width = rng.uniform(0.25, 0.55)
    bump_amp = rng.normal(0.0, 0.30)
    bump_sigma = rng.uniform(0.15, 0.35)
    boundary_amp = rng.normal(0.0, 0.16)
    boundary_width = rng.uniform(0.20, 0.45)

    if slice_id == 1:  # sparse observations: context/cohort fallback should matter
        trend_scale = 0.62
        ridge_amp *= 0.20
        bump_amp *= 0.20
        boundary_amp *= 0.15
    elif slice_id == 4:  # anisotropic_ridge
        ridge_amp = rng.choice([-1.0, 1.0]) * rng.uniform(1.35, 2.05)
        ridge_width = rng.uniform(0.025, 0.070)
    elif slice_id == 5:  # boundary
        boundary_amp = rng.choice([-1.0, 1.0]) * rng.uniform(1.15, 1.85)
        boundary_width = rng.uniform(0.020, 0.055)
        ridge_amp *= 0.25
    elif slice_id == 6:  # local bump
        bump_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.85, 1.35)
        bump_sigma = rng.uniform(0.045, 0.105)
    elif slice_id == 7:  # trend dominated
        trend_scale = 0.70
        ridge_amp *= 0.20
        bump_amp *= 0.20
        boundary_amp *= 0.10
    elif slice_id == 9:  # mixed scale
        ridge_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.75, 1.15)
        bump_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.65, 1.10)
        boundary_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.50)
    elif slice_id == 10:  # conflict
        trend_scale = 0.65
        bump_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.55, 1.05)
        ridge_amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 0.90)

    trend_x = rng.normal(0.0, trend_scale)
    trend_y = rng.normal(0.0, trend_scale)
    base = CATEGORY_EFFECT[cat] + rng.normal(0.0, 0.25)
    return {
        "category": cat,
        "theta": theta,
        "base": base,
        "trend_x": trend_x,
        "trend_y": trend_y,
        "ridge_amp": ridge_amp,
        "ridge_width": ridge_width,
        "ridge_center": rng.normal(0.0, 0.18),
        "ridge_freq": rng.uniform(2.8, 5.8) if slice_id == 4 else rng.uniform(2.2, 4.5),
        "phase": rng.uniform(-np.pi, np.pi),
        "bump_amp": bump_amp,
        "bump_sigma": bump_sigma,
        "bump_x": rng.uniform(-0.8, 0.8),
        "bump_y": rng.uniform(-0.8, 0.8),
        "boundary_amp": boundary_amp,
        "boundary_width": boundary_width,
        "boundary_center": rng.normal(0.0, 0.15),
    }


def _sample_observations(rng: np.random.Generator, params: Dict[str, float], qx: float, qy: float, slice_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = params["theta"]
    ct, st = np.cos(theta), np.sin(theta)

    if slice_id == 1:
        n = int(rng.integers(2, 6))
    elif slice_id == 0:
        n = int(rng.integers(22, 34))
    elif slice_id in (3, 4, 5, 9, 10):
        n = int(rng.integers(18, 32))
    else:
        n = int(rng.integers(10, 24))

    dx = rng.normal(0.0, 0.48, size=n)
    dy = rng.normal(0.0, 0.48, size=n)

    if slice_id == 0:  # dense and local
        dx = rng.normal(0.0, 0.20, size=n)
        dy = rng.normal(0.0, 0.20, size=n)
    elif slice_id == 1:  # sparse, farther, often one-sided
        dx = rng.normal(0.35, 0.68, size=n)
        dy = rng.normal(-0.28, 0.68, size=n)
    elif slice_id == 2:  # nearest is noisy; others ring around query
        dx = rng.normal(0.0, 0.42, size=n)
        dy = rng.normal(0.0, 0.42, size=n)
        dx[0] = rng.normal(0.0, 0.025)
        dy[0] = rng.normal(0.0, 0.025)
    elif slice_id == 3:  # biased near cluster plus farther useful sensors
        m = max(6, n // 2)
        dx[:m] = rng.normal(0.05, 0.08, size=m)
        dy[:m] = rng.normal(-0.02, 0.08, size=m)
        dx[m:] = rng.normal(-0.45, 0.28, size=n - m)
        dy[m:] = rng.normal(0.30, 0.28, size=n - m)
    elif slice_id == 4:  # anisotropic samples; broad along-ridge, narrow across-ridge
        along = rng.normal(0.0, 0.82, size=n)
        across = rng.normal(0.0, 0.075, size=n)
        # a minority of off-ridge points would mislead isotropic smoothers
        off = rng.random(n) < 0.45
        across[off] += rng.choice([-1.0, 1.0], size=off.sum()) * rng.uniform(0.22, 0.42, size=off.sum())
        dx = along * ct - across * st
        dy = along * st + across * ct
    elif slice_id == 5:  # many points on wrong side of boundary
        along = rng.normal(0.0, 0.52, size=n)
        across = rng.normal(-0.36, 0.10, size=n)
        few = max(1, n // 7)
        across[:few] = rng.normal(0.12, 0.055, size=few)
        dx = along * ct - across * st
        dy = along * st + across * ct
    elif slice_id == 6:  # local bump near query with uneven coverage
        params["bump_x"] = qx + rng.normal(0.0, 0.08)
        params["bump_y"] = qy + rng.normal(0.0, 0.08)
        dx = rng.normal(0.0, 0.32, size=n)
        dy = rng.normal(0.0, 0.32, size=n)
    elif slice_id == 7:  # broad spread helps trend
        dx = rng.uniform(-0.95, 0.95, size=n)
        dy = rng.uniform(-0.95, 0.95, size=n)
    elif slice_id == 10:
        m = max(5, n // 3)
        dx[:m] = rng.normal(0.0, 0.07, size=m)
        dy[:m] = rng.normal(0.0, 0.07, size=m)
        dx[m:] = rng.normal(0.0, 0.65, size=n - m)
        dy[m:] = rng.normal(0.0, 0.65, size=n - m)

    ox = np.clip(qx + dx, -1.15, 1.15)
    oy = np.clip(qy + dy, -1.15, 1.15)
    dx = ox - qx
    dy = oy - qy
    dist = np.sqrt(dx * dx + dy * dy)
    obs_X = np.column_stack([ox, oy, dx, dy, dist]).astype(np.float32)

    # Sensor mix. Codes 0/1/4/7 are relatively reliable; 3/5/6 can be biased/noisy.
    if slice_id == 2:
        sensor = rng.choice(np.arange(8), size=n, p=[0.16, 0.16, 0.16, 0.10, 0.16, 0.14, 0.06, 0.06])
        sensor[0] = 5
    elif slice_id == 3:
        sensor = rng.choice(np.arange(8), size=n, p=[0.12, 0.10, 0.12, 0.06, 0.14, 0.08, 0.32, 0.06])
        sensor[: max(6, n // 2)] = 6
    elif slice_id == 1:
        sensor = rng.choice(np.arange(8), size=n, p=[0.06, 0.08, 0.08, 0.18, 0.08, 0.30, 0.16, 0.06])
    elif slice_id == 8:
        sensor = rng.choice(np.arange(8), size=n, p=[0.09, 0.09, 0.09, 0.10, 0.09, 0.08, 0.06, 0.40])
    elif slice_id == 10:
        sensor = rng.choice(np.arange(8), size=n, p=[0.12, 0.12, 0.14, 0.10, 0.12, 0.20, 0.14, 0.06])
        sensor[: max(5, n // 3)] = rng.choice([3, 5, 6], size=max(5, n // 3), p=[0.25, 0.45, 0.30])
    else:
        sensor = rng.choice(np.arange(8), size=n, p=[0.18, 0.16, 0.15, 0.11, 0.15, 0.10, 0.09, 0.06])

    mode = rng.choice(np.arange(4), size=n, p=[0.55, 0.22, 0.16, 0.07])
    obs_code = np.column_stack([sensor, mode]).astype(np.int64)

    true_val = _field(params, ox, oy)
    cat = int(params["category"])
    rare_cat_bias = np.where(sensor == 7, 0.18 * (cat - 2.5), 0.0)
    mode_shift = np.array([0.0, 0.07, -0.06, 0.14])[mode]
    noise_sd = SENSOR_NOISE[sensor] * np.array([1.0, 1.25, 0.9, 1.55])[mode]
    # Sparse cases deliberately have less trustworthy observations; context/cohort fallback should help.
    if slice_id == 1:
        noise_sd = noise_sd * 1.85 + 0.10
    reading = SENSOR_SCALE[sensor] * true_val + SENSOR_BIAS[sensor] + rare_cat_bias + mode_shift + rng.normal(0.0, noise_sd)
    # An exposed reliability/quality auxiliary: helpful but not sufficient.
    quality = 1.0 / (0.08 + noise_sd) + rng.normal(0.0, 0.15, size=n)
    aux = np.column_stack([reading, noise_sd, quality]).astype(np.float32)
    return obs_X, obs_code, aux


def _make_context(params: Dict[str, float], slice_id: int, qx: float, qy: float, n_obs: int) -> np.ndarray:
    cat = int(params["category"])
    one = np.zeros(6, dtype=float)
    one[cat] = 1.0
    theta = params["theta"]
    # Context fields are visible but noisy enough that observations matter.
    ctx = np.concatenate([
        one,
        np.array([
            np.cos(theta), np.sin(theta),
            params["trend_x"] + 0.05 * np.sin(3 * qx),
            params["trend_y"] + 0.05 * np.cos(2 * qy),
            abs(params["ridge_amp"]),
            abs(params["boundary_amp"]),
            np.log1p(n_obs),
            float(slice_id in (1, 8)),
        ], dtype=float),
    ])
    return ctx.astype(np.float32)


def generate_split(n_cases: int, seed: int, split: str = "train") -> Dict[str, np.ndarray]:
    rng = _rng(seed)
    slice_probs = np.array([0.09, 0.10, 0.08, 0.10, 0.13, 0.12, 0.10, 0.08, 0.07, 0.07, 0.06], dtype=float)
    slice_probs = slice_probs / slice_probs.sum()
    slice_ids = rng.choice(np.arange(len(SLICE_NAMES)), size=n_cases, p=slice_probs)

    contexts: List[np.ndarray] = []
    queries: List[np.ndarray] = []
    obs_Xs: List[np.ndarray] = []
    obs_codes: List[np.ndarray] = []
    obs_values: List[np.ndarray] = []
    offsets = [0]
    y = np.zeros(n_cases, dtype=np.float32)

    for i, sid in enumerate(slice_ids):
        qx, qy = rng.uniform(-0.75, 0.75, size=2)
        params = _case_params(rng, int(sid))
        if int(sid) == 5:
            # Put boundary near the query in the visible orientation frame.
            theta = params["theta"]
            vq = -qx * np.sin(theta) + qy * np.cos(theta)
            # Query lies just on the sparsely sampled side of the transition.
            params["boundary_center"] = vq - 0.055 + rng.normal(0.0, 0.018)
        if int(sid) == 4:
            theta = params["theta"]
            vq = -qx * np.sin(theta) + qy * np.cos(theta)
            params["ridge_center"] = vq + rng.normal(0.0, 0.025)
        obs_X, obs_code, obs_value = _sample_observations(rng, params, qx, qy, int(sid))
        true_y = _field(params, np.array([qx]), np.array([qy]))[0]
        target_noise = rng.normal(0.0, 0.022 + 0.012 * (int(sid) in (1, 8)))
        y[i] = np.float32(true_y + target_noise)
        contexts.append(_make_context(params, int(sid), qx, qy, obs_X.shape[0]))
        queries.append(np.array([qx, qy], dtype=np.float32))
        obs_Xs.append(obs_X)
        obs_codes.append(obs_code)
        obs_values.append(obs_value)
        offsets.append(offsets[-1] + obs_X.shape[0])

    return {
        "context_X": np.vstack(contexts).astype(np.float32),
        "query_X": np.vstack(queries).astype(np.float32),
        "obs_X": np.vstack(obs_Xs).astype(np.float32),
        "obs_code": np.vstack(obs_codes).astype(np.int64),
        "obs_value": np.vstack(obs_values).astype(np.float32),
        "case_offsets": np.asarray(offsets, dtype=np.int64),
        "y": y.astype(np.float32),
        "slice_id": slice_ids.astype(np.int64),
        "slice_name": SLICE_NAMES[slice_ids],
    }


@dataclass(frozen=True)
class GeneratedSplit:
    context_X: np.ndarray
    query_X: np.ndarray
    obs_X: np.ndarray
    obs_code: np.ndarray
    obs_value: np.ndarray
    case_offsets: np.ndarray
    y: np.ndarray
    slice_id: np.ndarray
    slice_masks: Dict[str, np.ndarray]


def _to_generated(d: Dict[str, np.ndarray]) -> GeneratedSplit:
    sid = np.asarray(d["slice_id"], dtype=np.int64)
    masks = {str(name): sid == i for i, name in enumerate(SLICE_NAMES.tolist())}
    return GeneratedSplit(
        context_X=np.asarray(d["context_X"], dtype=np.float32),
        query_X=np.asarray(d["query_X"], dtype=np.float32),
        obs_X=np.asarray(d["obs_X"], dtype=np.float32),
        obs_code=np.asarray(d["obs_code"], dtype=np.int64),
        obs_value=np.asarray(d["obs_value"], dtype=np.float32),
        case_offsets=np.asarray(d["case_offsets"], dtype=np.int64),
        y=np.asarray(d["y"], dtype=np.float32),
        slice_id=sid,
        slice_masks=masks,
    )


def make_train_public() -> Tuple[GeneratedSplit, GeneratedSplit]:
    return _to_generated(generate_split(3200, 1123, "train")), _to_generated(generate_split(700, 2123, "public"))


def make_hidden() -> GeneratedSplit:
    return _to_generated(generate_split(1300, 3123, "hidden"))


def save_trial_npz(out_dir: str | Path, train_n: int = 3200, public_n: int = 700, hidden_n: int = 1300) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = generate_split(train_n, 1123, "train")
    public = generate_split(public_n, 2123, "public")
    hidden = generate_split(hidden_n, 3123, "hidden")

    np.savez_compressed(
        out / "train_data.npz",
        train_context_X=train["context_X"],
        train_query_X=train["query_X"],
        train_obs_X=train["obs_X"],
        train_obs_code=train["obs_code"],
        train_obs_value=train["obs_value"],
        train_case_offsets=train["case_offsets"],
        train_y=train["y"],
    )
    np.savez_compressed(
        out / "public_eval.npz",
        context_X=public["context_X"],
        query_X=public["query_X"],
        obs_X=public["obs_X"],
        obs_code=public["obs_code"],
        obs_value=public["obs_value"],
        case_offsets=public["case_offsets"],
        y=public["y"],
    )
    np.savez_compressed(
        out / "hidden_eval_payload.npz",
        context_X=hidden["context_X"],
        query_X=hidden["query_X"],
        obs_X=hidden["obs_X"],
        obs_code=hidden["obs_code"],
        obs_value=hidden["obs_value"],
        case_offsets=hidden["case_offsets"],
        y=hidden["y"],
        slice_id=hidden["slice_id"],
        slice_name=hidden["slice_name"],
    )
    meta = {
        "slice_names": SLICE_NAMES.tolist(),
        "train_n": train_n,
        "public_n": public_n,
        "hidden_n": hidden_n,
        "seeds": {"train": 1123, "public": 2123, "hidden": 3123},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    save_trial_npz(Path(__file__).resolve().parent / "trial_npz")

# ===== END FILE: tmp/tests/generator.py =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
"""Verifier scoring for local_field_reconstruction v0.2."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

try:
    from generator import make_hidden, make_train_public, SLICE_NAMES
    from sandbox_utils import run_eval, check_solution
except Exception:  # pragma: no cover
    from .generator import make_hidden, make_train_public, SLICE_NAMES
    from .sandbox_utils import run_eval, check_solution

# v0.2 anchors, calibrated after anisotropy/boundary/sparse hardening.
# GOOD is slightly stricter than the measured reference so the reference does not clip to 1.0.
OVERALL_GOOD = 0.358395506426
OVERALL_BAD = 1.109707550683
SLICE_GOOD = {
    "stable_dense_local": 0.0883134318370451,
    "sparse_observations": 0.41563690222506244,
    "nearest_sensor_noisy": 0.17403014500950145,
    "biased_sensor_cluster": 0.15742030001023877,
    "anisotropic_ridge": 0.6163137375527532,
    "wrong_side_boundary": 0.45285538419646343,
    "local_bump_near_query": 0.5092121321030808,
    "trend_dominated": 0.18963539640337215,
    "rare_sensor_code": 0.1884437022775738,
    "mixed_scale_field": 0.21037732492489422,
    "short_range_vs_long_range_conflict": 0.14305961047567559
}
SLICE_BAD = {
    "stable_dense_local": 0.7511000343885513,
    "sparse_observations": 1.0415679944098497,
    "nearest_sensor_noisy": 0.9605297547366998,
    "biased_sensor_cluster": 0.8886263312115448,
    "anisotropic_ridge": 1.392500429200827,
    "wrong_side_boundary": 2.09172121230338,
    "local_bump_near_query": 1.0644522054685837,
    "trend_dominated": 0.8019946233882154,
    "rare_sensor_code": 0.7444135770412587,
    "mixed_scale_field": 1.0570130333199435,
    "short_range_vs_long_range_conflict": 0.8938658839283856
}
SLICE_WEIGHTS = {
    "stable_dense_local": 0.55,
    "sparse_observations": 1.75,
    "nearest_sensor_noisy": 0.8,
    "biased_sensor_cluster": 1.05,
    "anisotropic_ridge": 1.45,
    "wrong_side_boundary": 1.25,
    "local_bump_near_query": 1.0,
    "trend_dominated": 0.7,
    "rare_sensor_code": 0.95,
    "mixed_scale_field": 0.9,
    "short_range_vs_long_range_conflict": 1.0
}
SCORE_POWER = 2.3



def _rmse(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _score_rmse(value, good, bad):
    value = float(value)
    good = max(float(good), 1e-9)
    bad = max(float(bad), good * 1.0001)
    if not np.isfinite(value):
        return 0.0
    raw = (np.log(bad) - np.log(max(value, 1e-9))) / (np.log(bad) - np.log(good))
    return float(np.clip(raw, 0.0, 1.0) ** SCORE_POWER)


def _payload(train, hidden):
    return {
        "train_context_X": train.context_X,
        "train_query_X": train.query_X,
        "train_obs_X": train.obs_X,
        "train_obs_code": train.obs_code,
        "train_obs_value": train.obs_value,
        "train_case_offsets": train.case_offsets,
        "train_y": train.y,
        "context_X": hidden.context_X,
        "query_X": hidden.query_X,
        "obs_X": hidden.obs_X,
        "obs_code": hidden.obs_code,
        "obs_value": hidden.obs_value,
        "case_offsets": hidden.case_offsets,
    }


def evaluate(solve_path: str | Path, write_dir: str | Path | None = None):
    solve_path = Path(solve_path)
    if not check_solution(solve_path):
        return _finish(0.0, {"error": "api_missing"}, write_dir)
    train, _ = make_train_public()
    hidden = make_hidden()
    try:
        out = run_eval(solve_path, _payload(train, hidden))
    except Exception as exc:
        return _finish(0.0, {"error": "runtime_error", "detail": str(exc)[:1200]}, write_dir)
    if "error" in out:
        return _finish(0.0, {"error": out.get("error"), "detail": out.get("detail", "")}, write_dir)
    pred = np.asarray(out["preds"][0], dtype=float)
    if pred.shape != hidden.y.shape:
        return _finish(0.0, {"error": "wrong_shape", "shape": list(pred.shape), "expected": list(hidden.y.shape)}, write_dir)
    if not np.all(np.isfinite(pred)):
        return _finish(0.0, {"error": "nonfinite_prediction"}, write_dir)
    if np.nanmax(np.abs(pred)) > 1e9:
        return _finish(0.0, {"error": "catastrophic_scale"}, write_dir)

    overall_rmse = _rmse(hidden.y, pred)
    overall_score = _score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    slice_metrics = {}
    weighted = []
    for sid, name in enumerate(SLICE_NAMES.tolist()):
        mask = hidden.slice_masks.get(str(name))
        if mask is None or not np.any(mask):
            continue
        r = _rmse(hidden.y, pred, mask)
        s = _score_rmse(r, SLICE_GOOD[str(name)], SLICE_BAD[str(name)])
        w = min(1.0, max(0.25, float(np.sum(mask)) / 120.0)) * SLICE_WEIGHTS.get(str(name), 1.0)
        slice_metrics[str(name)] = {"n": int(np.sum(mask)), "rmse": r, "score": s, "weight": w}
        weighted.append((w, s))
    if weighted:
        ws = np.array([x[0] for x in weighted], dtype=float)
        ss = np.array([x[1] for x in weighted], dtype=float)
        slice_score = float(np.sum(ws * ss) / np.sum(ws))
    else:
        slice_score = overall_score
    reward = 0.55 * overall_score + 0.45 * slice_score
    metrics = {
        "reward": float(np.clip(reward, 0.0, 1.0)),
        "overall_rmse": overall_rmse,
        "overall_score": overall_score,
        "slice_score": slice_score,
        "slice_metrics": slice_metrics,
    }
    return _finish(metrics["reward"], metrics, write_dir)


def _finish(reward, metrics, write_dir=None):
    if write_dir is not None:
        p = Path(write_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "reward.txt").write_text(f"{float(reward):.12f}\n")
        (p / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return float(reward), metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("solve_path")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    reward, metrics = evaluate(ns.solve_path, ns.out)
    print(json.dumps(metrics, indent=2, sort_keys=True))

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/sandbox_utils.py =====
"""Root-side sandbox helper for local_field_reconstruction."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

RUNNER = "/sandbox/agent_call.py"
IO_DIR = "/sandbox/io"
_counter = [0]
REQUIRED = ("fit_field_model", "predict_field_value")


def _sandbox_ready() -> bool:
    return os.path.exists(RUNNER) and getattr(os, "geteuid", lambda: 1000)() == 0


def _local_runner() -> str:
    return str(Path(__file__).resolve().parent / "agent_call.py")


def _demote():  # pragma: no cover
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(38, 1, 0, 0, 0)
    except Exception:
        pass
    try:
        import pwd
        p = pwd.getpwnam("nobody")
        try:
            os.setgroups([])
        except Exception:
            pass
        os.setgid(p.pw_gid)
        os.setuid(p.pw_uid)
    except Exception:
        pass


def _nobody_command(args: list[str]) -> tuple[list[str], object | None]:
    if shutil.which("runuser"):
        return ["runuser", "-u", "nobody", "--", *args], None
    if shutil.which("su"):
        quoted = " ".join(subprocess.list2cmdline([a]) for a in args)
        return ["su", "-s", "/bin/sh", "nobody", "-c", quoted], None
    return args, _demote


def _payload_to_npz(path: str | os.PathLike, payload: dict) -> None:
    np.savez(
        path,
        train_context_X=np.asarray(payload["train_context_X"], dtype=np.float64),
        train_query_X=np.asarray(payload["train_query_X"], dtype=np.float64),
        train_obs_X=np.asarray(payload["train_obs_X"], dtype=np.float64),
        train_obs_code=np.asarray(payload["train_obs_code"], dtype=np.int64),
        train_obs_value=np.asarray(payload["train_obs_value"], dtype=np.float64),
        train_case_offsets=np.asarray(payload["train_case_offsets"], dtype=np.int64),
        train_y=np.asarray(payload["train_y"], dtype=np.float64),
        context_X=np.asarray(payload["context_X"], dtype=np.float64),
        query_X=np.asarray(payload["query_X"], dtype=np.float64),
        obs_X=np.asarray(payload["obs_X"], dtype=np.float64),
        obs_code=np.asarray(payload["obs_code"], dtype=np.int64),
        obs_value=np.asarray(payload["obs_value"], dtype=np.float64),
        case_offsets=np.asarray(payload["case_offsets"], dtype=np.int64),
    )


def _read_output(out_path: str | os.PathLike) -> dict:
    with np.load(out_path, allow_pickle=False) as z:
        if "error" in z.files:
            return {"error": str(z["error"].item()), "detail": str(z.get("detail", ""))}
        n = int(z["n"])
        return {"preds": [np.asarray(z[f"p{i}"], dtype=np.float64) for i in range(n)]}


def run_eval(solve_path: str | os.PathLike, payload: dict, timeout: int = 180) -> dict:
    if not _sandbox_ready():
        import tempfile
        runner = _local_runner()
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in_eval.npz")
            out_path = os.path.join(td, "out_eval.npz")
            _payload_to_npz(in_path, payload)
            r = subprocess.run([sys.executable, runner, str(solve_path), "eval", in_path, out_path], capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0 or not os.path.exists(out_path):
                raise RuntimeError(f"local eval failed (rc={r.returncode}): {r.stderr[-800:]}")
            return _read_output(out_path)

    Path(IO_DIR).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(IO_DIR, 0o777)
    except Exception:
        pass
    _counter[0] += 1
    tag = f"eval_{os.getpid()}_{_counter[0]}"
    in_path = os.path.join(IO_DIR, f"{tag}_in.npz")
    out_path = os.path.join(IO_DIR, f"{tag}_out.npz")
    _payload_to_npz(in_path, payload)
    try:
        os.chmod(in_path, 0o644)
    except Exception:
        pass
    args = [sys.executable, RUNNER, str(solve_path), "eval", in_path, out_path]
    cmd, preexec = _nobody_command(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=preexec)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"sandbox eval failed (rc={r.returncode}): {r.stderr[-800:]}")
    return _read_output(out_path)


def check_solution(solve_path: str | os.PathLike, timeout: int = 60) -> bool:
    runner = RUNNER if _sandbox_ready() else _local_runner()
    try:
        r = subprocess.run([sys.executable, runner, str(solve_path), "check"], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False

# ===== END FILE: tmp/tests/sandbox_utils.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
set -uo pipefail

TEST_DIR=/tests
if [ ! -d "$TEST_DIR" ]; then
  TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

mkdir -p /logs/verifier 2>/dev/null || true
mkdir -p /sandbox/io 2>/dev/null || true

if [ -f "$TEST_DIR/agent_call.py" ]; then
  cp "$TEST_DIR/agent_call.py" /sandbox/agent_call.py 2>/dev/null || true
fi
chmod 755 /sandbox/agent_call.py 2>/dev/null || true
chmod 777 /sandbox/io 2>/dev/null || true

START_SEC=$(date +%s)

set +e
pytest "$TEST_DIR/test_smoke.py" -v --tb=short 2> /logs/verifier/smoke_stderr.txt
SMOKE_EXIT=$?
set -e
if [ $SMOKE_EXIT -ne 0 ]; then
  echo pipeline_broken > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

set +e
pytest "$TEST_DIR/test_schema.py" -v --tb=short 2>> /logs/verifier/smoke_stderr.txt
SCHEMA_EXIT=$?
set -e
if [ $SCHEMA_EXIT -ne 0 ]; then
  echo schema_violation > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

chmod 700 /tests 2>/dev/null || true

set +e
pytest "$TEST_DIR/test_main.py" \
  --json-report --json-report-file=/logs/verifier/ctrf.json \
  -v --tb=short \
  2> /logs/verifier/stderr.txt
EXIT=$?
set -e

if [ $EXIT -ne 0 ]; then
  echo runtime_error > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
elif [ ! -f /logs/verifier/reward.txt ]; then
  echo missing_reward > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
fi

END_SEC=$(date +%s)
echo $(( END_SEC - START_SEC )) > /logs/verifier/wall_clock_sec.txt
if [ -f /proc/self/status ]; then
  awk '/VmHWM/ {printf "%.0f\n", $2/1024}' /proc/self/status \
    > /logs/verifier/peak_memory_mb.txt 2>/dev/null || true
fi

exit 0

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_main.py =====
from __future__ import annotations

from pathlib import Path

from hidden_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]


def app_solve_path() -> Path:
    p = Path("/app/solve.py")
    return p if p.exists() else ROOT / "environment" / "app" / "solve.py"


def verifier_log_dir() -> Path:
    p = Path("/logs/verifier")
    if p.exists() or Path("/logs").exists():
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = ROOT / "logs" / "verifier"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_main_scoring_writes_reward_and_metrics():
    out_dir = verifier_log_dir()
    reward, metrics = evaluate(app_solve_path(), out_dir)
    assert (out_dir / "reward.txt").exists(), "reward.txt was not written"
    assert (out_dir / "metrics.json").exists(), "metrics.json was not written"
    assert 0.0 <= reward <= 1.0
    assert "reward" in metrics

# ===== END FILE: tmp/tests/test_main.py =====

# ===== BEGIN FILE: tmp/tests/test_schema.py =====
from __future__ import annotations

from pathlib import Path
import numpy as np

from generator import make_hidden, make_train_public

ROOT = Path(__file__).resolve().parents[1]
N_CONTEXT = 14
N_QUERY = 2
N_OBS_X = 5
N_OBS_CODE = 2
N_OBS_VALUE = 3


def app_data_path(name: str) -> Path:
    p = Path("/app") / name
    return p if p.exists() else ROOT / "environment" / "app" / name


def _check_dataset(d, *, train: bool, has_y: bool, has_slices: bool = False):
    ckey = "train_context_X" if train else "context_X"
    qkey = "train_query_X" if train else "query_X"
    oxkey = "train_obs_X" if train else "obs_X"
    ockey = "train_obs_code" if train else "obs_code"
    ovkey = "train_obs_value" if train else "obs_value"
    offkey = "train_case_offsets" if train else "case_offsets"
    ykey = "train_y" if train else "y"
    assert d[ckey].ndim == 2 and d[ckey].shape[1] == N_CONTEXT
    assert d[qkey].ndim == 2 and d[qkey].shape[1] == N_QUERY
    assert d[oxkey].ndim == 2 and d[oxkey].shape[1] == N_OBS_X
    assert d[ockey].ndim == 2 and d[ockey].shape[1] == N_OBS_CODE
    assert d[ovkey].ndim == 2 and d[ovkey].shape[1] == N_OBS_VALUE
    assert d[offkey].ndim == 1
    assert d[offkey][0] == 0
    assert d[offkey][-1] == d[oxkey].shape[0]
    assert d[oxkey].shape[0] == d[ockey].shape[0] == d[ovkey].shape[0]
    assert d[offkey].shape[0] == d[ckey].shape[0] + 1
    assert d[qkey].shape[0] == d[ckey].shape[0]
    assert np.all(np.diff(d[offkey]) > 0)
    assert np.all(np.isfinite(d[ckey]))
    assert np.all(np.isfinite(d[qkey]))
    assert np.all(np.isfinite(d[oxkey]))
    assert np.all(np.isfinite(d[ovkey]))
    if has_y:
        assert ykey in d
        assert d[ykey].shape == (d[ckey].shape[0],)
        assert np.all(np.isfinite(d[ykey]))
    if has_slices:
        slice_keys = [k for k in d.keys() if str(k).startswith("slice__")]
        assert len(slice_keys) >= 6
        for k in slice_keys:
            assert d[k].shape == (d[ckey].shape[0],)


def test_agent_visible_schema():
    with np.load(app_data_path("train_data.npz"), allow_pickle=False) as z:
        _check_dataset(z, train=True, has_y=True)
    with np.load(app_data_path("public_eval.npz"), allow_pickle=False) as z:
        _check_dataset(z, train=False, has_y=True)


def test_verifier_generated_schema():
    train, public = make_train_public()
    hidden = make_hidden()
    td = {
        "train_context_X": train.context_X,
        "train_query_X": train.query_X,
        "train_obs_X": train.obs_X,
        "train_obs_code": train.obs_code,
        "train_obs_value": train.obs_value,
        "train_case_offsets": train.case_offsets,
        "train_y": train.y,
    }
    _check_dataset(td, train=True, has_y=True)
    for split in (public, hidden):
        d = {
            "context_X": split.context_X,
            "query_X": split.query_X,
            "obs_X": split.obs_X,
            "obs_code": split.obs_code,
            "obs_value": split.obs_value,
            "case_offsets": split.case_offsets,
            "y": split.y,
        }
        for k, m in split.slice_masks.items():
            d[f"slice__{k}"] = m
        _check_dataset(d, train=False, has_y=True, has_slices=True)


def test_train_public_hidden_are_distinct_sizes():
    train, public = make_train_public()
    hidden = make_hidden()
    assert train.context_X.shape[0] > public.context_X.shape[0] > 0
    assert hidden.context_X.shape[0] > public.context_X.shape[0] > 0

# ===== END FILE: tmp/tests/test_schema.py =====

# ===== BEGIN FILE: tmp/tests/test_smoke.py =====
from __future__ import annotations

from pathlib import Path
import numpy as np

from sandbox_utils import check_solution

ROOT = Path(__file__).resolve().parents[1]


def app_solve_path() -> Path:
    p = Path("/app/solve.py")
    return p if p.exists() else ROOT / "environment" / "app" / "solve.py"


def app_data_path(name: str) -> Path:
    p = Path("/app") / name
    return p if p.exists() else ROOT / "environment" / "app" / name


def test_candidate_file_exists():
    assert app_solve_path().exists(), "solve.py not found"


def test_required_functions_present():
    assert check_solution(app_solve_path()), "required fit_field_model / predict_field_value functions not found"


def test_visible_data_loadable():
    for name in ("train_data.npz", "public_eval.npz"):
        path = app_data_path(name)
        assert path.exists(), f"{name} not found"
        with np.load(path, allow_pickle=False) as z:
            assert (("obs_X" in z.files) or ("train_obs_X" in z.files)) and (("train_case_offsets" in z.files) or ("case_offsets" in z.files)), f"{name} missing core arrays"

# ===== END FILE: tmp/tests/test_smoke.py =====

