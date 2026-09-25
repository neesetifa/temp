Inspection event risk

The current model in app/solve.py is a weak fallback. Replace it with a stronger deterministic model.

You are given packed histories of prior inspection-like events for a set of entities. Each entity has one future continuous risk target. Higher predictions should correspond to more severe future outcomes.

Implement these two functions in app/solve.py:

def fit_inspection_model(
    train_entity_X,
    train_entity_code,
    train_event_time,
    train_event_code,
    train_event_value,
    train_entity_offsets,
    train_y,
):
    ...

def predict_inspection_risk(
    entity_X,
    entity_code,
    event_time,
    event_code,
    event_value,
    entity_offsets,
    params,
):
    ...

Data

entity_X is a numeric matrix with one row per entity. entity_code is an integer-coded matrix with one row per entity. The event arrays are packed across all entities, and entity_offsets[i]:entity_offsets[i+1] gives the history rows for entity i.

event_time gives the event time relative to the prediction point. The histories are ordered within each entity. event_code gives the event type. event_value contains numeric measurements attached to each event.

train_y is the future risk value for each training entity. The public evaluation file includes labels only for local sanity checks; hidden scoring uses separate held-out data.

Notes

Simple averages, last-event rules, or category-only fallbacks are intentionally weak. Useful signal can come from event order, event type, numeric event values, recency, repeated patterns, gaps between events, and short-history fallbacks.

Your predictions must be a one-dimensional finite NumPy-compatible array of shape (n_entities,). The functions must be deterministic and must not mutate any input arrays. Do not read hidden files, use the network, or depend on external data.


  solve
  """Starter implementation for inspection_event_risk.

Replace fit_inspection_model and predict_inspection_risk with a stronger model.
"""
from __future__ import annotations
import numpy as np


def fit_inspection_model(
    train_entity_X,
    train_entity_code,
    train_event_time,
    train_event_code,
    train_event_value,
    train_entity_offsets,
    train_y,
):
    y = np.asarray(train_y, dtype=float)
    codes = np.asarray(train_entity_code, dtype=int)
    mean = float(np.mean(y)) if y.size else 0.0
    cat_means = {}
    pair_means = {}
    for cat in np.unique(codes[:, 0]) if codes.size else []:
        m = codes[:, 0] == cat
        if np.sum(m) >= 3:
            cat_means[int(cat)] = float(np.mean(y[m]))
    if codes.ndim == 2 and codes.shape[1] >= 2:
        for cat in np.unique(codes[:, 0]) if codes.size else []:
            for reg in np.unique(codes[:, 1]):
                m = (codes[:, 0] == cat) & (codes[:, 1] == reg)
                if np.sum(m) >= 6:
                    pair_means[(int(cat), int(reg))] = float(np.mean(y[m]))
    return {"mean": mean, "cat_means": cat_means, "pair_means": pair_means}


def predict_inspection_risk(
    entity_X,
    entity_code,
    event_time,
    event_code,
    event_value,
    entity_offsets,
    params,
):
    codes = np.asarray(entity_code, dtype=int)
    n = codes.shape[0]
    mean = float(params.get("mean", 0.0)) if isinstance(params, dict) else 0.0
    cat_means = params.get("cat_means", {}) if isinstance(params, dict) else {}
    pair_means = params.get("pair_means", {}) if isinstance(params, dict) else {}
    pred = np.full(n, mean, dtype=float)
    for i in range(n):
        cat = int(codes[i, 0]) if codes.ndim == 2 and codes.shape[1] >= 1 else -1
        reg = int(codes[i, 1]) if codes.ndim == 2 and codes.shape[1] >= 2 else -1
        if (cat, reg) in pair_means:
            pred[i] = pair_means[(cat, reg)]
        elif cat in cat_means:
            pred[i] = cat_means[cat]
    return np.maximum(pred, 0.0)

reference

"""Reference solution for inspection_event_risk v0.1 core trial.

This file is intentionally self-contained enough to be copied into app/solve.py.
It avoids using HGB as the core mechanism; the main signal is a structured
sequence/history featurizer plus shrinkage-aware ridge models.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

try:
    from sklearn.linear_model import RidgeCV, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception:  # pragma: no cover - platform sanity fallback
    RidgeCV = None
    Ridge = None
    StandardScaler = None
    make_pipeline = None


_EPS = 1e-9


def _one_hot(values: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(values), n), dtype=np.float64)
    mask = (values >= 0) & (values < n)
    out[np.arange(len(values))[mask], values[mask].astype(int)] = 1.0
    return out


def _safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))


def _build_features(entity_X, entity_code, event_time, event_code, event_value, entity_offsets, level: str = "full"):
    n = entity_X.shape[0]
    rows = []
    cats = entity_code[:, 0].astype(int)
    regs = entity_code[:, 1].astype(int)
    modes = entity_code[:, 2].astype(int)
    histb = entity_code[:, 3].astype(int) if entity_code.shape[1] > 3 else np.zeros(n, dtype=int)

    cat_oh = _one_hot(cats, 10)
    reg_oh = _one_hot(regs, 6)
    mode_oh = _one_hot(modes, 4)
    hist_oh = _one_hot(histb, 5)

    half_lives = [45.0, 120.0, 300.0, 750.0]
    windows = [75.0, 180.0, 365.0]

    for i in range(n):
        a, b = int(entity_offsets[i]), int(entity_offsets[i + 1])
        t = event_time[a:b]
        c = event_code[a:b]
        val = event_value[a:b]
        fam = (c // 5).astype(int) if c.size else np.array([], dtype=int)
        days = -t.astype(float)
        v0 = val[:, 0] if val.size else np.array([], dtype=float)
        v1 = val[:, 1] if val.size else np.array([], dtype=float)
        v2 = val[:, 2] if val.size else np.array([], dtype=float)
        v3 = val[:, 3] if val.size else np.array([], dtype=float)

        h = len(c)
        last_days = float(days[-1]) if h else 999.0
        first_days = float(days[0]) if h else 999.0
        gaps = np.diff(days[::-1]) if h >= 2 else np.array([], dtype=float)
        mean_gap, sd_gap, min_gap, max_gap = _safe_stats(gaps)
        mean_v, sd_v, min_v, max_v = _safe_stats(v0)
        last_v = float(v0[-1]) if h else 0.0
        prev_v = float(v0[-2]) if h >= 2 else last_v
        trend_last = last_v - prev_v

        feats = []
        feats.extend(entity_X[i].tolist())
        feats.extend([h, math.log1p(h), last_days / 365.0, first_days / 365.0, math.log1p(last_days), math.log1p(first_days)])
        feats.extend([mean_gap / 180.0, sd_gap / 180.0, min_gap / 180.0, max_gap / 180.0])
        feats.extend([mean_v, sd_v, min_v, max_v, last_v, trend_last])
        feats.extend([float(np.mean(v1)) if h else 0.0, float(np.max(v1)) if h else 0.0, float(np.mean(v3)) if h else 0.0])

        # Ordered recency kernels by family. This is the load-bearing part.
        family_kernel_values = []
        for f in range(6):
            m = fam == f
            feats.append(float(np.sum(m)))
            feats.append(float(np.mean(v0[m])) if np.any(m) else 0.0)
            feats.append(float(v0[np.where(m)[0][-1]]) if np.any(m) else 0.0)
            feats.append(float(days[np.where(m)[0][-1]]) / 365.0 if np.any(m) else 3.0)
            for w in windows:
                mw = m & (days <= w)
                feats.append(float(np.sum(mw)))
                feats.append(float(np.sum(v0[mw])))
                feats.append(float(np.sum(v1[mw])))
            for hl in half_lives:
                k = np.exp(-days / hl) if h else np.array([], dtype=float)
                kv = float(np.sum(v0[m] * k[m])) if np.any(m) else 0.0
                kc = float(np.sum(k[m])) if np.any(m) else 0.0
                feats.extend([kv, kc, kv / (kc + _EPS)])
                family_kernel_values.append(kv)

        # Recurrence / mitigation / follow-up hand features.
        if h:
            symptom_recent = float(np.sum((fam == 1) & (days < 260)))
            symptom_pressure = float(np.sum((fam == 1) * v0 * np.exp(-days / 150.0)))
            repeat_feature = math.log1p(max(0.0, symptom_recent - 1.5))
            high_recent = float(np.sum((fam == 2) * np.minimum(v0, 2.5) * np.exp(-days / 60.0)))
            mitigation = float(np.sum((fam == 3) * (0.7 + v0) * np.exp(-days / 150.0)))
            follow = float(np.sum((fam == 4) * (0.35 + v2) * np.exp(-days / 150.0)))
            rare = float(np.sum((fam == 5) * v0 * np.exp(-days / 365.0)))
            long_memory = float(np.sum(v0 * np.exp(-days / 900.0)) / (0.75 + np.sum(np.exp(-days / 900.0))))
            routine_mit_share = float(np.mean(np.isin(fam, [0, 3])))

            mitigation_after_symptom = 0.0
            followup_context = 0.0
            relapse_after_mitigation = 0.0
            clean_after_mitigation = 0.0
            for idx in range(h):
                prior_recent = (event_time[a:a+idx] > event_time[a+idx] - 240) if idx > 0 else np.array([], dtype=bool)
                if fam[idx] == 3 and idx > 0 and np.any(((fam[:idx] == 1) | (fam[:idx] == 2))):
                    mitigation_after_symptom += float(np.exp(-days[idx] / 220.0))
                if fam[idx] == 4:
                    if idx > 0 and np.any(prior_recent & ((fam[:idx] == 1) | (fam[:idx] == 2))):
                        followup_context += float(np.exp(-days[idx] / 210.0))
                    else:
                        followup_context -= 0.25 * float(np.exp(-days[idx] / 210.0))
                prior_mit = idx > 0 and np.any((event_time[a:a+idx] > event_time[a+idx] - 360) & (fam[:idx] == 3))
                if prior_mit and fam[idx] in (1, 2):
                    relapse_after_mitigation += float((0.65 + 0.20 * v0[idx]) * np.exp(-days[idx] / 260.0))
                elif prior_mit and fam[idx] == 0:
                    clean_after_mitigation += float(np.exp(-days[idx] / 260.0))
        else:
            symptom_recent = symptom_pressure = repeat_feature = high_recent = mitigation = follow = rare = long_memory = routine_mit_share = 0.0
            mitigation_after_symptom = followup_context = relapse_after_mitigation = clean_after_mitigation = 0.0

        if level == "no_recurrence":
            repeat_feature = 0.0
            relapse_after_mitigation = 0.0
        if level == "no_mitigation":
            mitigation = 0.0
            mitigation_after_symptom = 0.0
            relapse_after_mitigation = 0.0
            clean_after_mitigation = 0.0

        feats.extend([
            symptom_recent, symptom_pressure, repeat_feature, high_recent, mitigation, follow, rare, long_memory,
            mitigation_after_symptom, followup_context, relapse_after_mitigation, clean_after_mitigation,
            routine_mit_share, float(h >= 14) * routine_mit_share, float(h <= 4) * (symptom_pressure + high_recent + abs(rare)),
            math.log1p(max(0.0, last_days - 280.0)) / 3.0,
        ])

        if level == "full":
            # Visible categorical interactions; these are structured, not tree-specific.
            cat = cats[i]
            mode = modes[i]
            region = regs[i]
            feats.extend([
                symptom_pressure * (mode == 3),
                repeat_feature * (mode >= 2),
                mitigation * (cat in (3, 5, 7)),
                relapse_after_mitigation * (mode >= 2),
                clean_after_mitigation * (cat in (0, 2, 6)),
            ])
            feats.extend([rare * (cat == k) for k in range(10)])
            feats.extend([symptom_pressure * (region == k) for k in range(6)])
            feats.extend([follow * (mode == k) for k in range(4)])
            # Code-level counts break pure family-level simplifications.
            for code in range(30):
                cm = c == code
                if np.any(cm):
                    feats.append(float(np.sum(cm)))
                    feats.append(float(np.sum(v0[cm] * np.exp(-days[cm] / 180.0))))
                else:
                    feats.extend([0.0, 0.0])
        elif level == "agent_recovered":
            # A strong but imperfect hand-recovered feature family: it knows about
            # ordered recovery/relapse and broad interactions, but not exact rare-code
            # category signs or all code-level details.
            cat = cats[i]
            mode = modes[i]
            feats.extend([
                symptom_pressure * (mode >= 2),
                repeat_feature * (mode >= 2),
                mitigation * (cat in (3, 5, 7)),
                relapse_after_mitigation * (mode >= 2),
                followup_context * (mode == 3),
            ])
        elif level in ("no_recurrence", "no_mitigation", "no_cohort"):
            pass

        rows.append(feats)

    X = np.asarray(rows, dtype=np.float64)
    # Most models see coarse categorical one-hots.  The no_cohort ablation removes
    # both explicit one-hots and the later cohort fallback.
    if level == "no_cohort":
        return X
    return np.hstack([X, cat_oh, reg_oh, mode_oh, hist_oh]).astype(np.float64)


class _MeanFallback:
    def __init__(self, pred: float):
        self.pred = float(pred)
    def predict(self, X):
        return np.full(X.shape[0], self.pred, dtype=float)


def _fit_ridge(X, y):
    y = np.asarray(y, dtype=float)
    if RidgeCV is None or StandardScaler is None or make_pipeline is None:
        # Minimal numpy ridge fallback.
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-6
        Xs = (X - mu) / sd
        lam = 3.0
        coef = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
        intercept = float(y.mean() - (mu / sd) @ coef)
        return {"kind": "np_ridge", "mu": mu, "sd": sd, "coef": coef, "intercept": intercept}
    # Fixed-alpha ridge keeps the reference deterministic and fast for calibration.
    model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    model.fit(X, y)
    return model


def _predict_model(model, X):
    if isinstance(model, dict) and model.get("kind") == "np_ridge":
        return model["intercept"] + ((X - model["mu"]) / model["sd"]) @ model["coef"]
    return model.predict(X)


def _group_means(entity_code, y):
    # Smoothed category, region, category-region, and hist-bucket means.
    global_mean = float(np.mean(y))
    groups = {}
    specs = {
        "cat": entity_code[:, [0]],
        "region": entity_code[:, [1]],
        "mode": entity_code[:, [2]],
        "hist": entity_code[:, [3]] if entity_code.shape[1] > 3 else np.zeros((len(y), 1), dtype=int),
        "cat_region": entity_code[:, [0, 1]],
    }
    for name, arr in specs.items():
        d = {}
        for key, yi in zip(map(tuple, arr.tolist()), y):
            if key not in d:
                d[key] = [0, 0.0]
            d[key][0] += 1
            d[key][1] += float(yi)
        groups[name] = {k: (cnt, s / cnt) for k, (cnt, s) in d.items()}
    return global_mean, groups


def _cohort_predict(entity_code, global_mean, groups):
    out = np.full(entity_code.shape[0], global_mean, dtype=float)
    for i, row in enumerate(entity_code):
        pieces = [(12.0, global_mean)]
        for name, key in [
            ("cat", tuple(row[[0]].tolist())),
            ("region", tuple(row[[1]].tolist())),
            ("mode", tuple(row[[2]].tolist())),
            ("hist", tuple(row[[3]].tolist() if len(row) > 3 else [0])),
            ("cat_region", tuple(row[[0, 1]].tolist())),
        ]:
            item = groups.get(name, {}).get(key)
            if item is not None:
                cnt, mean = item
                pieces.append((min(float(cnt), 80.0), mean))
        w = np.array([p[0] for p in pieces])
        v = np.array([p[1] for p in pieces])
        out[i] = float(np.sum(w * v) / np.sum(w))
    return out


def fit_inspection_model(
    train_entity_X,
    train_entity_code,
    train_event_time,
    train_event_code,
    train_event_value,
    train_entity_offsets,
    train_y,
):
    X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="full")
    model = _fit_ridge(X, train_y)
    global_mean, groups = _group_means(train_entity_code, np.asarray(train_y, dtype=float))
    # A no-shrinkage model is kept to construct adaptive blending weight at prediction time.
    return {"model": model, "global_mean": global_mean, "groups": groups}


def predict_inspection_risk(
    entity_X,
    entity_code,
    event_time,
    event_code,
    event_value,
    entity_offsets,
    params,
):
    X = _build_features(entity_X, entity_code, event_time, event_code, event_value, entity_offsets, level="full")
    pred_model = _predict_model(params["model"], X)
    pred_cohort = _cohort_predict(entity_code, params["global_mean"], params["groups"])
    hist_len = np.diff(entity_offsets).astype(float)
    last_days = np.zeros(len(hist_len), dtype=float)
    for i in range(len(hist_len)):
        a, b = int(entity_offsets[i]), int(entity_offsets[i + 1])
        last_days[i] = -float(event_time[b - 1]) if b > a else 999.0
    # Mostly trust the structured sequence model.  Apply only mild shrinkage for
    # extremely short or stale histories; heavy shrinkage was empirically worse.
    short = np.clip((3.0 - hist_len) / 3.0, 0.0, 1.0)
    stale = np.clip((last_days - 520.0) / 900.0, 0.0, 1.0)
    shrink = np.clip(0.10 * short + 0.08 * stale, 0.0, 0.16)
    pred = (1.0 - shrink) * pred_model + shrink * pred_cohort
    return np.maximum(pred, 0.0).astype(float)


# Ablation helpers used by calibration, not part of the required public API.
def fit_ablation(level, train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, train_y):
    if level == "no_recurrence":
        # Reuse feature builder but zero recurrence-like columns by using a thinner feature family.
        X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="no_recurrence")
    elif level == "no_mitigation":
        X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="no_mitigation")
    elif level == "no_cohort":
        X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="no_cohort")
    elif level == "agent_recovered":
        X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="agent_recovered")
    else:
        X = _build_features(train_entity_X, train_entity_code, train_event_time, train_event_code, train_event_value, train_entity_offsets, level="full")
    return {"level": level, "model": _fit_ridge(X, train_y)}


def predict_ablation(params, entity_X, entity_code, event_time, event_code, event_value, entity_offsets):
    X = _build_features(entity_X, entity_code, event_time, event_code, event_value, entity_offsets, level=params.get("level", "full"))
    return np.maximum(_predict_model(params["model"], X), 0.0).astype(float)

test schema
from __future__ import annotations

from pathlib import Path
import numpy as np

from generator import make_hidden, make_train_public

ROOT = Path(__file__).resolve().parents[1]
N_ENTITY_X = 4
N_ENTITY_CODE = 4
N_EVENT_VALUE = 4


def app_data_path(name: str) -> Path:
    p = Path("/app") / name
    return p if p.exists() else ROOT / "environment" / "app" / name


def _check_dataset(d, *, has_y: bool, has_slices: bool = False):
    assert d["entity_X"].ndim == 2 and d["entity_X"].shape[1] == N_ENTITY_X
    assert d["entity_code"].ndim == 2 and d["entity_code"].shape[1] == N_ENTITY_CODE
    assert d["event_time"].ndim == 1
    assert d["event_code"].ndim == 1
    assert d["event_value"].ndim == 2 and d["event_value"].shape[1] == N_EVENT_VALUE
    assert d["entity_offsets"].ndim == 1
    assert d["entity_offsets"][0] == 0
    assert d["entity_offsets"][-1] == d["event_code"].shape[0]
    assert d["event_time"].shape[0] == d["event_code"].shape[0] == d["event_value"].shape[0]
    assert d["entity_offsets"].shape[0] == d["entity_X"].shape[0] + 1
    assert d["entity_code"].shape[0] == d["entity_X"].shape[0]
    assert np.all(np.diff(d["entity_offsets"]) > 0)
    assert np.all(np.isfinite(d["entity_X"]))
    assert np.all(np.isfinite(d["event_time"]))
    assert np.all(np.isfinite(d["event_value"]))
    if has_y:
        assert "y" in d
        assert d["y"].shape == (d["entity_X"].shape[0],)
        assert np.all(np.isfinite(d["y"]))
    if has_slices:
        slice_keys = [k for k in d.keys() if str(k).startswith("slice__")]
        assert len(slice_keys) >= 6
        for k in slice_keys:
            assert d[k].shape == (d["entity_X"].shape[0],)


def test_agent_visible_schema():
    for name in ("train_data.npz", "public_eval.npz"):
        with np.load(app_data_path(name), allow_pickle=False) as z:
            _check_dataset(z, has_y=True)


def test_verifier_generated_schema():
    train, public = make_train_public()
    hidden = make_hidden()
    for split in (train, public):
        d = {
            "entity_X": split.entity_X,
            "entity_code": split.entity_code,
            "event_time": split.event_time,
            "event_code": split.event_code,
            "event_value": split.event_value,
            "entity_offsets": split.entity_offsets,
            "y": split.y,
        }
        _check_dataset(d, has_y=True)
    hd = {
        "entity_X": hidden.entity_X,
        "entity_code": hidden.entity_code,
        "event_time": hidden.event_time,
        "event_code": hidden.event_code,
        "event_value": hidden.event_value,
        "entity_offsets": hidden.entity_offsets,
        "y": hidden.y,
    }
    for k, m in hidden.slice_masks.items():
        hd[f"slice__{k}"] = m
    _check_dataset(hd, has_y=True, has_slices=True)


def test_train_public_hidden_are_distinct_sizes():
    train, public = make_train_public()
    hidden = make_hidden()
    assert train.entity_X.shape[0] > public.entity_X.shape[0] > 0
    assert hidden.entity_X.shape[0] > public.entity_X.shape[0] > 0

sandbox
"""Root-side sandbox helper for inspection_event_risk."""
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
REQUIRED = ("fit_inspection_model", "predict_inspection_risk")


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
        train_entity_X=np.asarray(payload["train_entity_X"], dtype=np.float64),
        train_entity_code=np.asarray(payload["train_entity_code"], dtype=np.int64),
        train_event_time=np.asarray(payload["train_event_time"], dtype=np.float64),
        train_event_code=np.asarray(payload["train_event_code"], dtype=np.int64),
        train_event_value=np.asarray(payload["train_event_value"], dtype=np.float64),
        train_entity_offsets=np.asarray(payload["train_entity_offsets"], dtype=np.int64),
        train_y=np.asarray(payload["train_y"], dtype=np.float64),
        entity_X=np.asarray(payload["entity_X"], dtype=np.float64),
        entity_code=np.asarray(payload["entity_code"], dtype=np.int64),
        event_time=np.asarray(payload["event_time"], dtype=np.float64),
        event_code=np.asarray(payload["event_code"], dtype=np.int64),
        event_value=np.asarray(payload["event_value"], dtype=np.float64),
        entity_offsets=np.asarray(payload["entity_offsets"], dtype=np.int64),
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

test.sh
#!/bin/bash
set -uo pipefail

TEST_DIR=/tests
if [ ! -d "$TEST_DIR" ]; then
  TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

mkdir -p /logs/verifier 2>/dev/null || true
mkdir -p /sandbox/io 2>/dev/null || true

# Make the sandbox runner available outside /tests before /tests is locked down.
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
pytest "$TEST_DIR/test_main.py"   --json-report --json-report-file=/logs/verifier/ctrf.json   -v --tb=short   2> /logs/verifier/stderr.txt
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
  awk '/VmHWM/ {printf "%.0f
", $2/1024}' /proc/self/status     > /logs/verifier/peak_memory_mb.txt 2>/dev/null || true
fi

exit 0

test_main
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


generator
"""Synthetic generator for inspection_event_risk v0.1.

Core-only trial package.  No README/instruction artifacts are included.
The public agent-visible interface is the set of arrays produced by make_dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GeneratedSplit:
    entity_X: np.ndarray
    entity_code: np.ndarray
    event_time: np.ndarray
    event_code: np.ndarray
    event_value: np.ndarray
    entity_offsets: np.ndarray
    y: np.ndarray
    slice_masks: Dict[str, np.ndarray]
    component_table: Dict[str, np.ndarray]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _make_global_params(seed: int = 1729) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Fixed effects visible only through repeated data patterns.
    category_effect = np.array([-0.50, -0.25, 0.00, 0.20, 0.45, 0.70, -0.10, 0.35, -0.35, 0.15], dtype=float)
    region_effect = np.array([-0.25, 0.10, 0.25, -0.05, 0.35, -0.15], dtype=float)
    mode_effect = np.array([-0.20, 0.05, 0.28, 0.55], dtype=float)
    # Rare-code family sign changes with category. This is the main non-global effect.
    rare_category_sign = np.array([0.80, -0.55, 0.30, 0.95, -0.80, 0.45, -0.30, 0.70, -0.65, 0.15], dtype=float)
    # Code-level effects are visible only statistically.  They make pure family
    # recovery incomplete without code-level modeling.
    code_effect = rng.normal(0.0, 0.32, size=30)
    return {
        "category_effect": category_effect,
        "region_effect": region_effect,
        "mode_effect": mode_effect,
        "rare_category_sign": rare_category_sign,
        "code_effect": code_effect,
    }


def _sample_event_sequence(
    rng: np.random.Generator,
    n_events: int,
    base_risk: float,
    volatility: float,
    category: int,
    mode: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event_time, event_code, event_value for one entity.

    event_time is negative days before the target event. Later history is closer to 0.
    event_value has 4 columns:
      0: observed severity-like signal
      1: secondary count/intensity signal
      2: follow-up/selection flag intensity
      3: noisy operation-state proxy
    """
    # Irregular backward gaps: many small gaps and a few long absences.
    gap_scale = np.clip(70.0 - 10.0 * base_risk + 20.0 * (mode == 0), 22.0, 160.0)
    gaps = rng.gamma(shape=1.45, scale=gap_scale, size=n_events)
    # Make very short histories and long gaps naturally possible.
    gaps *= rng.lognormal(mean=0.0, sigma=0.35, size=n_events)
    t_back = np.cumsum(gaps[::-1])[::-1]
    event_time = -np.maximum(1.0, t_back).astype(np.float64)

    # Family probabilities depend on latent risk but not deterministically.
    # Families: 0 routine, 1 recurring symptom, 2 high-severity noisy,
    # 3 mitigation/recovery, 4 follow-up/selection, 5 rare category-specific.
    logits = np.array([
        1.30 - 0.20 * base_risk,
        -0.10 + 0.55 * base_risk + 0.15 * volatility,
        -0.50 + 0.40 * base_risk + 0.25 * (mode >= 2),
        -0.55 + 0.20 * base_risk + 0.20 * (category in (3, 5, 7)),
        -0.85 + 0.50 * base_risk + 0.35 * (mode == 3),
        -1.25 + 0.25 * volatility + 0.22 * (category in (1, 3, 4, 7, 8)),
    ], dtype=float)
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    families = rng.choice(6, size=n_events, p=probs)

    # Selection structure: follow-up family more likely soon after severe/symptom events.
    for j in range(1, n_events):
        prev_fam = families[j - 1]
        recent_gap = event_time[j] - event_time[j - 1]
        if prev_fam in (1, 2) and recent_gap < 95 and rng.random() < 0.32:
            families[j] = 4
        if prev_fam == 3 and recent_gap < 160 and rng.random() < 0.25:
            families[j] = 0
        # Some entities relapse after a mitigation event; the order matters and
        # is deliberately not reducible to bagged code counts.
        if prev_fam == 3 and recent_gap < 320 and rng.random() < (0.10 + 0.08 * (volatility > 1.2) + 0.04 * (mode >= 2)):
            families[j] = 1

    # Five codes per family, with category/mode-biased sub-code use.
    sub = rng.integers(0, 5, size=n_events)
    sub = (sub + (category % 3) + (mode == 3)) % 5
    event_code = families * 5 + sub

    values = np.zeros((n_events, 4), dtype=np.float64)
    # A slowly varying observed state creates useful but imperfect continuity.
    state = base_risk + rng.normal(0.0, 0.35)
    last_family = -1
    repeat_run = 0
    for j, fam in enumerate(families):
        if fam == last_family:
            repeat_run += 1
        else:
            repeat_run = 0
        last_family = int(fam)
        # State dynamics are not exactly the target formula.
        drift = 0.035 * base_risk + rng.normal(0.0, 0.12 + 0.05 * volatility)
        if fam == 1:
            drift += 0.15 + 0.08 * repeat_run
        elif fam == 2:
            drift += 0.10 + rng.normal(0.0, 0.18)
        elif fam == 3:
            drift -= 0.22 + 0.04 * min(repeat_run, 2)
        elif fam == 4:
            drift += 0.08
        elif fam == 5:
            drift += rng.normal(0.0, 0.12)
        state = 0.86 * state + drift

        severity_loc = {
            0: -0.35,
            1: 0.45,
            2: 0.75,
            3: -0.20,
            4: 0.18,
            5: 0.35,
        }[int(fam)] + 0.42 * state
        obs = _softplus(np.array([severity_loc + rng.normal(0.0, 0.35 + 0.08 * volatility)]))[0]
        count_like = rng.poisson(np.clip(np.exp(-0.2 + obs * 0.35 + 0.18 * (fam in (1, 2, 4))), 0.2, 8.0))
        follow_intensity = (fam == 4) * (0.8 + 0.25 * obs + rng.normal(0.0, 0.1))
        noisy_state_proxy = state + rng.normal(0.0, 0.55)
        values[j] = [obs, count_like, follow_intensity, noisy_state_proxy]
    return event_time, event_code.astype(np.int64), values


def _target_from_history(
    event_time: np.ndarray,
    event_code: np.ndarray,
    event_value: np.ndarray,
    entity_x: np.ndarray,
    entity_code_row: np.ndarray,
    gp: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[float, Dict[str, float]]:
    cat, region, mode = map(int, entity_code_row[:3])
    fam = event_code // 5
    days = -event_time
    value = event_value[:, 0]
    countv = event_value[:, 1]

    base = (
        1.15
        + gp["category_effect"][cat]
        + gp["region_effect"][region]
        + gp["mode_effect"][mode]
        + 0.22 * entity_x[0]
        - 0.18 * entity_x[1]
        + 0.13 * entity_x[2]
    )

    k60 = np.exp(-days / 60.0)
    k150 = np.exp(-days / 150.0)
    k365 = np.exp(-days / 365.0)
    k900 = np.exp(-days / 900.0)

    symptom = np.sum((fam == 1) * value * k150)
    symptom_count_recent = np.sum((fam == 1) * (days < 260))
    repeat_escalation = np.log1p(max(0.0, symptom_count_recent - 1.5)) * (0.65 + 0.10 * mode)

    high_noisy = np.sum((fam == 2) * np.minimum(value, 2.5) * k60)
    long_memory = np.sum(value * k900) / (0.75 + np.sum(k900))

    mitigation = np.sum((fam == 3) * (0.7 + value) * k150)
    recent_mitigation_after_symptom = 0.0
    if len(fam):
        mit_idx = np.where((fam == 3) & (days < 300))[0]
        for idx in mit_idx:
            prior = np.any((fam[:idx] == 1) | (fam[:idx] == 2))
            if prior:
                recent_mitigation_after_symptom += float(np.exp(-days[idx] / 220.0))

    followup = np.sum((fam == 4) * (0.35 + event_value[:, 2]) * k150)
    followup_context = 0.0
    relapse_after_mitigation = 0.0
    clean_after_mitigation = 0.0
    mitigation_after_symptom = recent_mitigation_after_symptom
    for idx in np.where(fam == 4)[0]:
        # Follow-up means much more if it follows severe/symptom history.
        prior_recent = (event_time[:idx] > event_time[idx] - 240) if idx > 0 else np.array([], dtype=bool)
        if idx > 0 and np.any(prior_recent & ((fam[:idx] == 1) | (fam[:idx] == 2))):
            followup_context += float(np.exp(-days[idx] / 210.0))
        else:
            followup_context -= 0.25 * float(np.exp(-days[idx] / 210.0))

    # Ordered transition effects: symptoms after a recent mitigation are a relapse
    # signal; routine observations after mitigation are a durable recovery signal.
    for idx in range(len(fam)):
        prior_mit = (idx > 0) and np.any((event_time[:idx] > event_time[idx] - 360) & (fam[:idx] == 3))
        if prior_mit and fam[idx] in (1, 2):
            relapse_after_mitigation += float((0.65 + 0.20 * value[idx]) * np.exp(-days[idx] / 260.0))
        elif prior_mit and fam[idx] == 0:
            clean_after_mitigation += float(np.exp(-days[idx] / 260.0))

    rare_raw = np.sum((fam == 5) * value * k365)
    rare = gp["rare_category_sign"][cat] * rare_raw
    code_signal = float(np.sum(gp["code_effect"][event_code] * value * np.exp(-days / 210.0)))

    last_days = float(days[-1]) if len(days) else 999.0
    last_value = float(value[-1]) if len(value) else 0.0
    long_gap = np.log1p(max(0.0, last_days - 280.0)) / 3.0
    hist_len = len(event_code)
    short_hist_penalty = 0.25 if hist_len <= 2 else 0.0

    total_count = len(event_code)
    count_proxy_breaker = 0.0
    # Anti-proxy: many routine/mitigation rows can imply lower future risk, while a few rare high-signal rows can imply high risk.
    routine_mitigation_share = np.mean(np.isin(fam, [0, 3])) if total_count else 0.0
    if total_count >= 14 and routine_mitigation_share > 0.72:
        count_proxy_breaker -= 0.55
    if total_count <= 4 and (symptom + high_noisy + abs(rare)) > 1.25:
        count_proxy_breaker += 0.45

    raw = (
        base
        + 0.70 * symptom
        + 0.76 * repeat_escalation
        + 0.30 * high_noisy
        + 0.45 * followup
        + 0.70 * followup_context
        + 0.72 * relapse_after_mitigation
        - 0.35 * clean_after_mitigation
        + 0.48 * rare
        + 1.05 * code_signal
        + 0.38 * long_memory
        + 0.10 * last_value * np.exp(-last_days / 90.0)
        - 0.82 * mitigation
        - 0.55 * recent_mitigation_after_symptom
        - 0.22 * long_gap * (0.5 + max(0.0, base))
        - short_hist_penalty
        + count_proxy_breaker
    )
    # Heteroscedastic but bounded-ish noise.
    noise_scale = 0.18 + 0.04 * entity_x[2] + 0.035 * (hist_len <= 2) + 0.035 * (cat in (1, 4, 8))
    y = _softplus(np.array([raw + rng.normal(0.0, noise_scale)]))[0]
    components = {
        "base": float(base),
        "symptom": float(symptom),
        "repeat": float(repeat_escalation),
        "high_noisy": float(high_noisy),
        "mitigation": float(mitigation + mitigation_after_symptom),
        "followup": float(followup + followup_context),
        "relapse_after_mitigation": float(relapse_after_mitigation),
        "clean_after_mitigation": float(clean_after_mitigation),
        "rare": float(rare),
        "code_signal": float(code_signal),
        "long_gap": float(long_gap),
        "count_proxy_breaker": float(count_proxy_breaker),
        "last_days": float(last_days),
        "last_value": float(last_value),
        "hist_len": float(hist_len),
        "routine_mitigation_share": float(routine_mitigation_share),
    }
    return float(y), components


def make_split(n_examples: int, seed: int, global_seed: int = 1729) -> GeneratedSplit:
    rng = np.random.default_rng(seed)
    gp = _make_global_params(global_seed)

    entity_X = np.zeros((n_examples, 4), dtype=np.float64)
    entity_code = np.zeros((n_examples, 4), dtype=np.int64)
    all_times = []
    all_codes = []
    all_values = []
    offsets = [0]
    y = np.zeros(n_examples, dtype=np.float64)
    comp_lists: Dict[str, list] = {}

    for i in range(n_examples):
        category = int(rng.choice(10, p=np.array([.13,.09,.12,.11,.08,.07,.13,.10,.07,.10])))
        region = int(rng.choice(6, p=np.array([.20,.17,.15,.18,.13,.17])))
        mode = int(rng.choice(4, p=np.array([.30,.36,.22,.12])))
        # numeric profile features: size/exposure, compliance proxy, volatility, service complexity
        size = rng.normal(0.0, 1.0) + 0.25 * (category in (4, 5, 7)) + 0.15 * region
        compliance = rng.normal(0.0, 1.0) - 0.35 * mode + 0.15 * (category in (0, 8))
        volatility = np.clip(rng.gamma(1.8, 0.55) + 0.12 * mode + 0.07 * abs(size), 0.15, 3.5)
        complexity = rng.normal(0.0, 0.8) + 0.35 * (category in (3, 5, 7)) + 0.20 * (mode == 3)
        entity_X[i] = [size, compliance, volatility, complexity]
        hist_bucket = 0  # Filled after sampling length.
        entity_code[i, :3] = [category, region, mode]

        base_for_sampling = (
            0.75 + gp["category_effect"][category] + gp["region_effect"][region]
            + gp["mode_effect"][mode] + 0.20 * size - 0.14 * compliance + 0.08 * complexity
        )
        # Long-tailed history lengths, with nontrivial short-history slice.
        mean_len = np.clip(5.0 + 2.2 * _softplus(np.array([base_for_sampling]))[0] + 1.1 * volatility, 1.5, 18.0)
        n_events = int(np.clip(rng.negative_binomial(n=3, p=3 / (3 + mean_len)) + 1, 1, 32))
        if rng.random() < 0.13:
            n_events = int(rng.integers(1, 3))
        elif rng.random() < 0.10:
            n_events = int(rng.integers(18, 34))
        hist_bucket = int(np.digitize(n_events, [2, 5, 10, 18]))
        entity_code[i, 3] = hist_bucket

        t, c, v = _sample_event_sequence(rng, n_events, base_for_sampling, volatility, category, mode)
        yi, comps = _target_from_history(t, c, v, entity_X[i], entity_code[i], gp, rng)
        y[i] = yi
        all_times.append(t)
        all_codes.append(c)
        all_values.append(v)
        offsets.append(offsets[-1] + n_events)
        for k, val in comps.items():
            comp_lists.setdefault(k, []).append(val)

    event_time = np.concatenate(all_times).astype(np.float64)
    event_code = np.concatenate(all_codes).astype(np.int64)
    event_value = np.vstack(all_values).astype(np.float64)
    entity_offsets = np.array(offsets, dtype=np.int64)
    comp_table = {k: np.array(v, dtype=np.float64) for k, v in comp_lists.items()}
    slices = make_slice_masks(entity_code, comp_table)
    return GeneratedSplit(entity_X, entity_code, event_time, event_code, event_value, entity_offsets, y, slices, comp_table)


def make_slice_masks(entity_code: np.ndarray, comp: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    h = comp["hist_len"]
    symptom = comp["symptom"]
    repeat = comp["repeat"]
    mitigation = comp["mitigation"]
    followup = comp["followup"]
    rare = np.abs(comp["rare"])
    last_days = comp["last_days"]
    last_value = comp["last_value"]
    high_noisy = comp["high_noisy"]
    routine_share = comp["routine_mitigation_share"]
    count_breaker = comp["count_proxy_breaker"]
    cat = entity_code[:, 0]

    masks = {
        "stable_history": (h >= 6) & (last_days < 180) & (np.abs(count_breaker) < 1e-6) & (mitigation < 0.5),
        "recent_noisy_outlier": (last_days < 90) & (last_value > np.quantile(last_value, 0.72)) & (high_noisy > np.quantile(high_noisy, 0.70)) & (repeat < 0.4),
        "long_gap_decay": (last_days > 420) & (h >= 3),
        "repeat_symptom_escalation": (repeat > np.quantile(repeat, 0.72)) & (symptom > np.quantile(symptom, 0.65)),
        "mitigation_recovery": (mitigation > np.quantile(mitigation, 0.72)),
        "followup_selection_bias": (np.abs(followup) > np.quantile(np.abs(followup), 0.72)),
        "short_history_fallback": (h <= 2),
        "rare_code_category_interaction": (rare > np.quantile(rare, 0.70)) & np.isin(cat, [1, 3, 4, 7, 8]),
        "high_count_low_risk": (h >= 14) & (routine_share > 0.72),
        "low_count_high_risk": (h <= 4) & ((symptom + high_noisy + rare) > np.quantile(symptom + high_noisy + rare, 0.70)),
        "mixed_signal_history": (symptom > np.quantile(symptom, 0.55)) & (mitigation > np.quantile(mitigation, 0.55)),
    }
    # Make sure no slice is empty. If very small under a particular seed, broaden slightly.
    n = len(entity_code)
    for name, m in list(masks.items()):
        if int(m.sum()) < max(25, int(0.015 * n)):
            if name == "recent_noisy_outlier":
                masks[name] = (last_days < 140) & (high_noisy > np.quantile(high_noisy, 0.55))
            elif name == "rare_code_category_interaction":
                masks[name] = rare > np.quantile(rare, 0.60)
            elif name == "mixed_signal_history":
                masks[name] = (symptom > np.quantile(symptom, 0.45)) & (mitigation > np.quantile(mitigation, 0.45))
            else:
                # Top relevant component fallback; still natural, not duplicated.
                score = {
                    "long_gap_decay": last_days,
                    "repeat_symptom_escalation": repeat + symptom,
                    "mitigation_recovery": mitigation,
                    "followup_selection_bias": np.abs(followup),
                    "short_history_fallback": -h,
                    "high_count_low_risk": h + 5 * routine_share,
                    "low_count_high_risk": symptom + high_noisy + rare - 0.05 * h,
                    "stable_history": h - 0.01 * last_days,
                }.get(name, h)
                thresh = np.quantile(score, 0.94)
                masks[name] = score >= thresh
    return {k: v.astype(bool) for k, v in masks.items()}


def make_dataset(
    n_train: int = 5000,
    n_public: int = 1200,
    n_hidden: int = 1800,
    seed: int = 20260924,
) -> Dict[str, GeneratedSplit]:
    return {
        "train": make_split(n_train, seed + 11),
        "public": make_split(n_public, seed + 29),
        "hidden": make_split(n_hidden, seed + 47),
    }


def split_to_npz_payload(split: GeneratedSplit) -> Dict[str, np.ndarray]:
    payload = {
        "entity_X": split.entity_X,
        "entity_code": split.entity_code,
        "event_time": split.event_time,
        "event_code": split.event_code,
        "event_value": split.event_value,
        "entity_offsets": split.entity_offsets,
        "y": split.y,
    }
    for k, m in split.slice_masks.items():
        payload[f"slice__{k}"] = m.astype(np.int8)
    for k, v in split.component_table.items():
        payload[f"component__{k}"] = v
    return payload


if __name__ == "__main__":
    ds = make_dataset(n_train=2000, n_public=500, n_hidden=700)
    for name, split in ds.items():
        print(name, split.entity_X.shape, split.event_value.shape, split.y.mean(), split.y.std())
        print({k: int(v.sum()) for k, v in split.slice_masks.items()})


# Verifier convenience wrappers.  These sizes match the v0.1b calibration pass.
_CACHED_DATASET = None


def _cached_dataset():
    global _CACHED_DATASET
    if _CACHED_DATASET is None:
        _CACHED_DATASET = make_dataset(n_train=1800, n_public=450, n_hidden=650, seed=20260924)
    return _CACHED_DATASET


def make_train_public():
    ds = _cached_dataset()
    return ds["train"], ds["public"]


def make_hidden():
    ds = _cached_dataset()
    return ds["hidden"]


hidden
"""Verifier scoring for inspection_event_risk v0.1b."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

try:
    from generator import make_hidden, make_train_public
    from sandbox_utils import run_eval, check_solution
except Exception:  # pragma: no cover
    from .generator import make_hidden, make_train_public
    from .sandbox_utils import run_eval, check_solution

ANCHORS = {'overall': {'bad': 1.09216481385893, 'good': 0.253048124169406},
 'slices': {'followup_selection_bias': {'bad': 1.4498235336880947, 'good': 0.2657088347128341},
            'high_count_low_risk': {'bad': 1.4014502890480054, 'good': 0.30070426810293477},
            'long_gap_decay': {'bad': 1.04815570452228, 'good': 0.3046664064140489},
            'low_count_high_risk': {'bad': 1.311878921178568, 'good': 0.2986916308587213},
            'mitigation_recovery': {'bad': 1.1755334979530483, 'good': 0.24298918384362977},
            'mixed_signal_history': {'bad': 1.4282982089260159, 'good': 0.27163294852920283},
            'rare_code_category_interaction': {'bad': 1.1291763666122163,
                                               'good': 0.28148549736915063},
            'recent_noisy_outlier': {'bad': 1.2555231832837817, 'good': 0.25342895821876327},
            'repeat_symptom_escalation': {'bad': 1.9121114825796348, 'good': 0.29079409464126316},
            'short_history_fallback': {'bad': 0.9867298567492181, 'good': 0.2489592702624272},
            'stable_history': {'bad': 1.1084190288893694, 'good': 0.2572722710364862}}}
SLICE_WEIGHT = 0.45
SCORE_POWER = 1.20


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
        "train_entity_X": train.entity_X,
        "train_entity_code": train.entity_code,
        "train_event_time": train.event_time,
        "train_event_code": train.event_code,
        "train_event_value": train.event_value,
        "train_entity_offsets": train.entity_offsets,
        "train_y": train.y,
        "entity_X": hidden.entity_X,
        "entity_code": hidden.entity_code,
        "event_time": hidden.event_time,
        "event_code": hidden.event_code,
        "event_value": hidden.event_value,
        "entity_offsets": hidden.entity_offsets,
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
    overall_score = _score_rmse(overall_rmse, ANCHORS["overall"]["good"], ANCHORS["overall"]["bad"])
    slice_metrics = {}
    weighted = []
    for name, a in ANCHORS["slices"].items():
        mask = hidden.slice_masks.get(name)
        if mask is None or not np.any(mask):
            continue
        r = _rmse(hidden.y, pred, mask)
        s = _score_rmse(r, a["good"], a["bad"])
        w = min(1.0, max(0.25, float(np.sum(mask)) / 120.0))
        slice_metrics[name] = {"n": int(np.sum(mask)), "rmse": r, "score": s, "weight": w}
        weighted.append((w, s))
    if weighted:
        ws = np.array([x[0] for x in weighted], dtype=float)
        ss = np.array([x[1] for x in weighted], dtype=float)
        slice_score = float(np.sum(ws * ss) / np.sum(ws))
    else:
        slice_score = overall_score
    reward = (1.0 - SLICE_WEIGHT) * overall_score + SLICE_WEIGHT * slice_score
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

agent call
#!/usr/bin/env python3
"""Sandbox-side agent runner for inspection_event_risk."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REQUIRED = ("fit_inspection_model", "predict_inspection_risk")


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
            train_entity_X = np.asarray(z["train_entity_X"], dtype=np.float64)
            train_entity_code = np.asarray(z["train_entity_code"], dtype=np.int64)
            train_event_time = np.asarray(z["train_event_time"], dtype=np.float64)
            train_event_code = np.asarray(z["train_event_code"], dtype=np.int64)
            train_event_value = np.asarray(z["train_event_value"], dtype=np.float64)
            train_entity_offsets = np.asarray(z["train_entity_offsets"], dtype=np.int64)
            train_y = np.asarray(z["train_y"], dtype=np.float64)
            entity_X = np.asarray(z["entity_X"], dtype=np.float64)
            entity_code = np.asarray(z["entity_code"], dtype=np.int64)
            event_time = np.asarray(z["event_time"], dtype=np.float64)
            event_code = np.asarray(z["event_code"], dtype=np.int64)
            event_value = np.asarray(z["event_value"], dtype=np.float64)
            entity_offsets = np.asarray(z["entity_offsets"], dtype=np.int64)
        mod = _load(solve_path)
        if not all(callable(getattr(mod, name, None)) for name in REQUIRED):
            _save_error(out_path, "api_missing", "required fit_inspection_model/predict_inspection_risk not found")
            return 0

        fit_args = [
            train_entity_X.copy(), train_entity_code.copy(), train_event_time.copy(), train_event_code.copy(),
            train_event_value.copy(), train_entity_offsets.copy(), train_y.copy(),
        ]
        fit_before = [x.copy() for x in fit_args]
        params = mod.fit_inspection_model(*fit_args)
        if not all(_arrays_equal(a, b) for a, b in zip(fit_args, fit_before)):
            _save_error(out_path, "input_mutation_in_fit", "fit_inspection_model modified one or more input arrays")
            return 0

        pred_args = [entity_X.copy(), entity_code.copy(), event_time.copy(), event_code.copy(), event_value.copy(), entity_offsets.copy()]
        pred_before = [x.copy() for x in pred_args]
        pred1 = np.asarray(mod.predict_inspection_risk(*pred_args, params), dtype=np.float64)
        if not all(_arrays_equal(a, b) for a, b in zip(pred_args, pred_before)):
            _save_error(out_path, "input_mutation_in_predict", "predict_inspection_risk modified one or more input arrays")
            return 0
        pred2_args = [entity_X.copy(), entity_code.copy(), event_time.copy(), event_code.copy(), event_value.copy(), entity_offsets.copy()]
        pred2 = np.asarray(mod.predict_inspection_risk(*pred2_args, params), dtype=np.float64)
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

test smoke
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
    assert check_solution(app_solve_path()), "required fit_inspection_model / predict_inspection_risk functions not found"


def test_visible_data_loadable():
    for name in ("train_data.npz", "public_eval.npz"):
        path = app_data_path(name)
        assert path.exists(), f"{name} not found"
        with np.load(path, allow_pickle=False) as z:
            assert "entity_X" in z.files and "entity_offsets" in z.files, f"{name} missing core arrays"
