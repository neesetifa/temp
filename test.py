# Batch wait model

There are two functions in `app/solve.py` to fill in:

```python
fit_wait_model(
    train_job_X,
    train_job_code,
    train_batch_X,
    train_batch_offsets,
    train_y,
)

predict_wait_time(
    job_X,
    job_code,
    batch_X,
    batch_offsets,
    params,
)
```

The job rows are packed by batch. For batch `i`, the rows are

```python
batch_offsets[i] : batch_offsets[i + 1]
```

Return one predicted wait value for each job row.

The old model scored each job row by itself. It was fine on quiet batches, but
it missed cases where the same-looking job landed very differently depending on
what arrived with it. Some batches have rows that mostly get in each other's way.
Some have a lot of busy-looking rows that do not matter much for a given job.
Some have a few urgent rows mixed into a slower group.

Use the training batches and their wait values to build the model. At prediction
time, use the rows and batch offsets passed into `predict_wait_time`.

Keep the input arrays unchanged. Do not read hidden files or use outside data.


solve
  """Starter implementation for shared_capacity_waits.

This deliberately uses only the training mean. It is here to make the public
checks import and run; it is not intended to score well on hidden data.
"""

from __future__ import annotations

import numpy as np


def fit_wait_model(train_job_X, train_job_code, train_batch_X, train_batch_offsets, train_y):
    y = np.asarray(train_y, dtype=float)
    return {"mean": float(np.mean(y)) if y.size else 0.0}


def predict_wait_time(job_X, job_code, batch_X, batch_offsets, params):
    n = int(np.asarray(job_X).shape[0])
    return np.full(n, float(params.get("mean", 0.0)), dtype=float)

reference
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

N_FAMILY = 16
N_STATION = 10
N_HANDLING = 7
N_PRODUCT = 80
N_RES = 5


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -30.0, 30.0)))


def onehot(x, n):
    x = np.asarray(x, dtype=int)
    out = np.zeros((len(x), n), dtype=float)
    good = (x >= 0) & (x < n)
    out[np.arange(len(x))[good], x[good]] = 1.0
    return out


def _static():
    rng = np.random.default_rng(92717)
    base_groups = np.array([
        [1.25, 0.15, -0.15, 0.30, -0.25],
        [-0.10, 1.30, 0.25, -0.30, 0.10],
        [0.15, -0.25, 1.25, 0.10, 0.30],
        [0.45, 0.15, -0.35, 1.20, 0.20],
        [-0.20, 0.35, 0.20, 0.10, 1.25],
    ])
    family = rng.normal(0, 0.20, (N_FAMILY, N_RES))
    for f in range(N_FAMILY):
        family[f] += base_groups[f % 5] + 0.10 * base_groups[(f // 5 + f) % 5]
    station = rng.normal(0, 0.22, (N_STATION, N_RES))
    for s in range(N_STATION):
        station[s] += 0.78 * base_groups[(2 * s + 1) % 5]
        station[s, (s + 2) % 5] += 0.30
    handling = rng.normal(0, 0.18, (N_HANDLING, N_RES))
    handling[0] += np.array([0.20, 0.05, 0.05, 0.00, 0.00])
    handling[1] += np.array([0.05, 0.50, 0.10, -0.10, 0.10])
    handling[2] += np.array([-0.05, 0.15, 0.55, 0.05, 0.00])
    handling[3] += np.array([0.10, -0.05, 0.05, 0.55, 0.10])
    handling[4] += np.array([0.10, 0.00, 0.05, 0.10, 0.55])
    handling[5] += np.array([0.30, 0.15, 0.00, 0.25, 0.15])
    handling[6] += np.array([-0.80, -0.60, 1.20, -0.50, -0.35])
    product = rng.normal(0, 0.15, (N_PRODUCT, N_RES))
    for p in range(N_PRODUCT):
        product[p] += 0.40 * base_groups[(p // N_FAMILY + 2 * p) % 5]
        product[p, p % 5] += 0.28
    lane_pair = 0.65 + 0.18 * rng.normal(size=(5, 5))
    for i in range(5):
        lane_pair[i, i] += 0.45
        lane_pair[i, (i + 1) % 5] += 0.25
        lane_pair[i, (i + 3) % 5] -= 0.18
    lane_pair = np.clip(lane_pair, 0.25, 1.55)
    fam_pair = 0.70 + 0.14 * rng.normal(size=(N_FAMILY, N_FAMILY))
    for i in range(N_FAMILY):
        for j in range(N_FAMILY):
            if i == j:
                fam_pair[i, j] += 0.35
            if i % 5 == j % 5:
                fam_pair[i, j] += 0.30
            if {i % 5, j % 5} in ({0, 3}, {1, 4}):
                fam_pair[i, j] += 0.18
    fam_pair = np.clip(fam_pair, 0.20, 1.60)
    return family, station, handling, product, lane_pair, fam_pair


def _demand(X, C):
    family, station, handling, product, _, _ = _static()
    size, comp, priority, slack, frag, rush, decoy = X.T
    fam, st, hand, prod = C.T.astype(int)
    fam = np.clip(fam, 0, N_FAMILY - 1)
    st = np.clip(st, 0, N_STATION - 1)
    hand = np.clip(hand, 0, N_HANDLING - 1)
    prod = np.clip(prod, 0, N_PRODUCT - 1)
    g = fam % 5
    z = family[fam] + station[st] + handling[hand] + product[prod]
    z += np.stack([
        0.55 * np.log1p(size) + 0.34 * comp - 0.10 * slack + 0.22 * (g == 0),
        0.44 * size + 0.28 * frag + 0.18 * (hand == 1) + 0.20 * (g == 1),
        0.30 * comp + 0.40 * (hand == 6) + 0.25 * (st % 3 == 0) + 0.16 * (g == 2),
        0.42 * frag + 0.24 * priority + 0.18 * (hand == 3) + 0.20 * (g == 3),
        0.25 * rush + 0.28 * (1.0 - slack) + 0.22 * (hand == 4) + 0.20 * (g == 4),
    ], axis=1)
    dem = 0.25 + softplus(z)
    dem *= (0.60 + 0.50 * np.log1p(size))[:, None]
    dec = hand == 6
    dem[dec] *= np.array([0.22, 0.28, 1.35, 0.30, 0.33])
    return dem


def _capacity(bx):
    staff, ea, eb, ec, shift, maint, mode, day = bx
    cap = np.array([18.5, 17.0, 16.0, 16.8, 15.8]) * (0.62 + 0.82 * staff)
    cap += np.array([5.2 * ea, 2.0 * eb, 4.5 * eb + 1.6 * ec, 2.4 * ea + 2.8 * ec, 4.0 * ec])
    cap *= (1.0 - 0.20 * maint * np.array([0.35, 0.85, 0.50, 0.55, 0.70]))
    if mode > 0.67:
        cap *= np.array([1.18 - 0.22 * eb, 0.78 + 0.22 * ea, 1.10, 0.90 + 0.18 * ec, 0.82 + 0.25 * eb])
    elif mode < 0.30:
        cap *= np.array([0.92 + 0.25 * ea, 1.08, 0.86 + 0.22 * eb, 1.12, 0.94 + 0.20 * ec])
    return np.maximum(cap, 4.0)


def reference_features(job_X, job_code, batch_X, offsets):
    X = np.asarray(job_X, dtype=float)
    C = np.asarray(job_code, dtype=int)
    B = np.asarray(batch_X, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    fam, st, hand, prod = C.T
    fam = np.clip(fam, 0, N_FAMILY - 1)
    st = np.clip(st, 0, N_STATION - 1)
    hand = np.clip(hand, 0, N_HANDLING - 1)
    prod = np.clip(prod, 0, N_PRODUCT - 1)
    D = _demand(X, np.stack([fam, st, hand, prod], axis=1))
    dnorm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-6)
    _, _, _, _, lane_pair, fam_pair = _static()
    lane = prod % 5
    n = len(X)
    rel = np.zeros((n, N_RES))
    stress = np.zeros((n, N_RES))
    total = np.zeros((n, N_RES))
    same = np.zeros((n, 10))
    peak = np.zeros((n, 4))
    decstress = np.zeros((n, 3))
    batch_reps = []
    for bi in range(len(offsets) - 1):
        a, b = offsets[bi], offsets[bi + 1]
        if b <= a:
            continue
        idx = slice(a, b)
        m = b - a
        Xi = X[idx]
        Di = D[idx]
        dni = dnorm[idx]
        fami, sti, handi, prodi = fam[idx], st[idx], hand[idx], prod[idx]
        lanei = lane[idx]
        pri = Xi[:, 2]
        cap = _capacity(B[bi])
        tot = Di.sum(axis=0)
        dec_load = Di[handi == 6].sum(axis=0) if np.any(handi == 6) else np.zeros(N_RES)
        fg = np.bincount(fami % 5, minlength=5) / m
        fc = np.bincount(fami, minlength=N_FAMILY) / m
        sc = np.bincount(sti, minlength=N_STATION) / m
        hc = np.bincount(handi, minlength=N_HANDLING) / m
        rep = np.tile(np.r_[B[bi], m, tot, tot / (cap + 1e-6), fg, sc, hc], (m, 1))
        batch_reps.append(rep)
        for jj in range(m):
            global_i = a + jj
            dot = dni @ dni[jj]
            compat = 0.16 + 0.46 * dot
            compat *= fam_pair[fami[jj], fami]
            compat *= lane_pair[lanei[jj], lanei]
            compat *= (1.0 + 0.30 * (sti == sti[jj]) + 0.16 * (handi == handi[jj]))
            compat *= (1.0 + 0.18 * (((prodi // N_FAMILY) == (prodi[jj] // N_FAMILY)) & (fami != fami[jj])))
            pf = 0.45 + 1.05 * np.maximum(pri - pri[jj], 0.0)
            pf += 0.30 * (pri > 0.82) + 0.22 * (pri[jj] < 0.28)
            pf *= (1.0 - 0.26 * np.maximum(pri[jj] - pri, 0.0))
            weights = compat * pf
            weights[jj] = 0.30
            rel[global_i] = (Di * weights[:, None]).sum(axis=0)
            stress[global_i] = (rel[global_i] - cap * (0.42 + 0.012 * m)) / (cap * (0.58 + 0.008 * m))
            total[global_i] = tot
            hi = Di[:, lanei[jj]] * compat * (pri > pri[jj] + 0.18)
            top = np.sort(hi)[-4:]
            peak[global_i, :len(top)] = top
            same[global_i] = [
                m,
                (fami == fami[jj]).sum() - 1,
                ((fami % 5) == (fami[jj] % 5)).sum() - 1,
                (sti == sti[jj]).sum() - 1,
                (handi == handi[jj]).sum() - 1,
                (prodi == prodi[jj]).sum() - 1,
                (pri > pri[jj] + 0.25).sum(),
                ((sti == sti[jj]) & (pri > pri[jj] + 0.25)).sum(),
                (handi == 6).sum(),
                ((compat > 0.85) & (pri > pri[jj] + 0.12)).sum(),
            ]
            decstress[global_i] = [dec_load[2] / (cap[2] + 1e-6), dec_load[lanei[jj]] / (cap[lanei[jj]] + 1e-6), lane_pair[lanei[jj], (lanei[jj] + 1) % 5]]
    batch_feat = np.vstack(batch_reps) if batch_reps else np.empty((n, 0))
    feats = [
        X,
        np.stack([fam / 15.0, st / 9.0, hand / 6.0, prod / 79.0, lane / 4.0, (prod // N_FAMILY) / 4.0], axis=1),
        onehot(fam % 5, 5), onehot(st, N_STATION), onehot(hand, N_HANDLING), onehot(lane, 5),
        D, dnorm, total, rel, softplus(stress), D * softplus(stress),
        dnorm * rel / (np.maximum(total, 1e-6)), same, peak, decstress, batch_feat,
    ]
    return np.hstack(feats).astype(float)


def fit_wait_model(train_job_X, train_job_code, train_batch_X, train_batch_offsets, train_y):
    X = reference_features(train_job_X, train_job_code, train_batch_X, train_batch_offsets)
    y = np.asarray(train_y, dtype=float)
    model = HistGradientBoostingRegressor(
        max_iter=320,
        learning_rate=0.043,
        max_leaf_nodes=47,
        l2_regularization=0.035,
        random_state=19,
    )
    model.fit(X, y)
    return {"model": model}


def predict_wait_time(job_X, job_code, batch_X, batch_offsets, params):
    X = reference_features(job_X, job_code, batch_X, batch_offsets)
    pred = params["model"].predict(X)
    return np.asarray(pred, dtype=float)

hidden
"""Hidden verifier for shared_capacity_waits."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from generator import SLICE_NAMES, generate_dataset

HIDDEN_SEED = 8041
HIDDEN_N_BATCHES = 220
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))

# Slice RMSE anchors. These are intentionally verifier-side: public tests only
# check the interface, while the hidden run uses per-slice continuous scoring.
SLICE_GOOD = np.array([2.266, 3.084, 2.815, 3.020, 2.854, 2.797, 2.391, 3.360, 3.866, 3.220], dtype=float)
SLICE_BAD = np.array([3.899, 13.340, 6.282, 7.941, 6.242, 9.702, 5.289, 18.360, 16.877, 11.175], dtype=float)
OVERALL_GOOD = 3.018
OVERALL_BAD = 10.492


def _load_train(app_dir: Path) -> tuple[Any, ...]:
    data = np.load(app_dir / "train_data.npz")
    return (
        data["train_job_X"],
        data["train_job_code"],
        data["train_batch_X"],
        data["train_batch_offsets"],
        data["train_y"],
    )


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solve module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_in_process(solve_path: Path, payload: dict[str, Any]) -> list[np.ndarray]:
    module = _load_module(solve_path)
    params = module.fit_wait_model(*payload["train"])
    preds = []
    for args in payload["inputs"]:
        preds.append(np.asarray(module.predict_wait_time(*args, params), dtype=float))
    return preds


def _run_candidate(solve_path: Path, payload: dict[str, Any]) -> list[np.ndarray]:
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)

    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get("preds") if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, dtype=float) for p in preds]


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2)))


def _score_rmse(rmse: float, good: float, bad: float) -> float:
    if not math.isfinite(rmse):
        return 0.0
    return float(np.clip((bad - rmse) / (bad - good), 0.0, 1.0))


def _validate_predictions(pred: np.ndarray, expected_n: int) -> None:
    if pred.shape != (expected_n,):
        raise ValueError(f"prediction shape must be ({expected_n},), got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions must be finite")


def _compute_metrics(y: np.ndarray, pred: np.ndarray, slice_id: np.ndarray) -> dict[str, Any]:
    overall_rmse = _rmse(y, pred)
    overall_score = _score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    per_slice: dict[str, Any] = {}
    slice_scores = []
    for sid, name in enumerate(SLICE_NAMES):
        mask = slice_id == sid
        if not bool(mask.any()):
            continue
        rmse = _rmse(y[mask], pred[mask])
        score = _score_rmse(rmse, float(SLICE_GOOD[sid]), float(SLICE_BAD[sid]))
        per_slice[str(name)] = {"rmse": rmse, "score": score, "n": int(mask.sum())}
        slice_scores.append(score)
    slice_mean = float(np.mean(slice_scores)) if slice_scores else 0.0
    reward = float(0.35 * overall_score + 0.65 * slice_mean)
    return {
        "reward": reward,
        "overall_rmse": overall_rmse,
        "overall_score": overall_score,
        "slice_mean": slice_mean,
        "per_slice": per_slice,
    }


def evaluate(solve_path: str | Path | None = None, app_dir: str | Path | None = None, log_dir: str | Path | None = None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"

    train = _load_train(app)
    hidden = generate_dataset(HIDDEN_N_BATCHES, HIDDEN_SEED, "hidden")
    inputs = (
        hidden["job_X"],
        hidden["job_code"],
        hidden["batch_X"],
        hidden["batch_offsets"],
    )
    payload = {"train": train, "inputs": [inputs, inputs]}

    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")
    pred0, pred1 = preds
    _validate_predictions(pred0, len(hidden["y"]))
    _validate_predictions(pred1, len(hidden["y"]))
    if not np.allclose(pred0, pred1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")

    metrics = _compute_metrics(hidden["y"], pred0, hidden["slice_id"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    reward, metrics = evaluate()
    print(json.dumps({"reward": reward, "overall_rmse": metrics["overall_rmse"]}, sort_keys=True))

generator
import numpy as np

N_FAMILY = 16
N_STATION = 10
N_HANDLING = 7
N_PRODUCT = 80
N_RES = 5

SLICE_NAMES = np.array([
    "sparse_clean",
    "crowded_same_family",
    "crowded_mixed_family",
    "same_count_different_mix",
    "rare_family",
    "priority_conflict",
    "capacity_shift",
    "decoy_load",
    "long_tail_batch_size",
    "matched_proxy_trap",
])


def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30.0, 30.0)))


def _static():
    rng = np.random.default_rng(92717)
    base_groups = np.array([
        [1.25, 0.15, -0.15, 0.30, -0.25],
        [-0.10, 1.30, 0.25, -0.30, 0.10],
        [0.15, -0.25, 1.25, 0.10, 0.30],
        [0.45, 0.15, -0.35, 1.20, 0.20],
        [-0.20, 0.35, 0.20, 0.10, 1.25],
    ])
    family = rng.normal(0, 0.20, (N_FAMILY, N_RES))
    for f in range(N_FAMILY):
        family[f] += base_groups[f % 5] + 0.10 * base_groups[(f // 5 + f) % 5]

    station = rng.normal(0, 0.22, (N_STATION, N_RES))
    for s in range(N_STATION):
        station[s] += 0.78 * base_groups[(2 * s + 1) % 5]
        station[s, (s + 2) % 5] += 0.30

    handling = rng.normal(0, 0.18, (N_HANDLING, N_RES))
    handling[0] += np.array([0.20, 0.05, 0.05, 0.00, 0.00])
    handling[1] += np.array([0.05, 0.50, 0.10, -0.10, 0.10])
    handling[2] += np.array([-0.05, 0.15, 0.55, 0.05, 0.00])
    handling[3] += np.array([0.10, -0.05, 0.05, 0.55, 0.10])
    handling[4] += np.array([0.10, 0.00, 0.05, 0.10, 0.55])
    handling[5] += np.array([0.30, 0.15, 0.00, 0.25, 0.15])
    # decoy handling: visible size can be high, but it mostly occupies a different lane.
    handling[6] += np.array([-0.80, -0.60, 1.20, -0.50, -0.35])

    product = rng.normal(0, 0.15, (N_PRODUCT, N_RES))
    for p in range(N_PRODUCT):
        product[p] += 0.40 * base_groups[(p // N_FAMILY + 2 * p) % 5]
        product[p, p % 5] += 0.28

    # Directional compatibility. Same visible aggregates can have different effects when
    # product lanes disagree.
    lane_pair = 0.65 + 0.18 * rng.normal(size=(5, 5))
    for i in range(5):
        lane_pair[i, i] += 0.45
        lane_pair[i, (i + 1) % 5] += 0.25
        lane_pair[i, (i + 3) % 5] -= 0.18
    lane_pair = np.clip(lane_pair, 0.25, 1.55)

    fam_pair = 0.70 + 0.14 * rng.normal(size=(N_FAMILY, N_FAMILY))
    for i in range(N_FAMILY):
        for j in range(N_FAMILY):
            if i == j:
                fam_pair[i, j] += 0.35
            if i % 5 == j % 5:
                fam_pair[i, j] += 0.30
            if {i % 5, j % 5} in ({0, 3}, {1, 4}):
                fam_pair[i, j] += 0.18
    fam_pair = np.clip(fam_pair, 0.20, 1.60)
    return family, station, handling, product, lane_pair, fam_pair


def _sample_slice(rng, split):
    if split == "hidden":
        p = np.array([0.08, 0.10, 0.12, 0.13, 0.12, 0.12, 0.12, 0.11, 0.09, 0.11])
    else:
        p = np.array([0.19, 0.10, 0.10, 0.11, 0.08, 0.10, 0.10, 0.09, 0.07, 0.06])
    p = p / p.sum()
    return int(rng.choice(len(SLICE_NAMES), p=p))


def _choose_family(rng, slice_id, n):
    base = np.array([0.095, 0.090, 0.087, 0.083, 0.080, 0.075, 0.070, 0.065,
                     0.060, 0.055, 0.050, 0.045, 0.040, 0.035, 0.030, 0.030])
    base = base / base.sum()
    if slice_id == 1:
        f0 = int(rng.choice(np.arange(12), p=(base[:12] / base[:12].sum())))
        out = np.where(rng.random(n) < 0.74, f0, rng.choice(N_FAMILY, size=n, p=base))
    elif slice_id == 4:
        p = base.copy()
        p[13:] += 0.09
        p[:10] *= 0.72
        p = p / p.sum()
        out = rng.choice(N_FAMILY, size=n, p=p)
    elif slice_id in (3, 9):
        g0 = int(rng.integers(0, 5))
        g1 = int((g0 + rng.choice([1, 2, 3])) % 5)
        fams0 = np.array([f for f in range(N_FAMILY) if f % 5 == g0])
        fams1 = np.array([f for f in range(N_FAMILY) if f % 5 == g1])
        out = np.empty(n, dtype=int)
        pick = rng.random(n) < 0.52
        out[pick] = rng.choice(fams0, size=pick.sum())
        out[~pick] = rng.choice(fams1, size=(~pick).sum())
        mask = rng.random(n) < 0.18
        out[mask] = rng.choice(N_FAMILY, size=mask.sum(), p=base)
    elif slice_id == 2:
        p = np.ones(N_FAMILY) / N_FAMILY
        out = rng.choice(N_FAMILY, size=n, p=p)
    else:
        out = rng.choice(N_FAMILY, size=n, p=base)
    return out.astype(np.int64)


def _demand_from_visible(X, C, family, station, handling, product):
    size, comp, priority, slack, frag, rush, decoy = X.T
    fam, st, hand, prod = C.T.astype(int)
    g = fam % 5
    z = family[fam] + station[st] + handling[hand] + product[prod]
    z += np.stack([
        0.55 * np.log1p(size) + 0.34 * comp - 0.10 * slack + 0.22 * (g == 0),
        0.44 * size + 0.28 * frag + 0.18 * (hand == 1) + 0.20 * (g == 1),
        0.30 * comp + 0.40 * (hand == 6) + 0.25 * (st % 3 == 0) + 0.16 * (g == 2),
        0.42 * frag + 0.24 * priority + 0.18 * (hand == 3) + 0.20 * (g == 3),
        0.25 * rush + 0.28 * (1.0 - slack) + 0.22 * (hand == 4) + 0.20 * (g == 4),
    ], axis=1)
    dem = 0.25 + _softplus(z)
    dem *= (0.60 + 0.50 * np.log1p(size))[:, None]
    # Decoy rows are visibly large but mostly only hit resource/lane 2.
    dec = hand == 6
    dem[dec] *= np.array([0.22, 0.28, 1.35, 0.30, 0.33])
    return dem


def _batch_capacity(bx, slice_id):
    staff, ea, eb, ec, shift, maint, mode, day = bx
    cap = np.array([18.5, 17.0, 16.0, 16.8, 15.8]) * (0.62 + 0.82 * staff)
    cap += np.array([5.2 * ea, 2.0 * eb, 4.5 * eb + 1.6 * ec, 2.4 * ea + 2.8 * ec, 4.0 * ec])
    cap *= (1.0 - 0.20 * maint * np.array([0.35, 0.85, 0.50, 0.55, 0.70]))
    # Capacity shift is visible but not a simple global scalar.
    if slice_id == 6 or mode > 0.67:
        cap *= np.array([1.18 - 0.22 * eb, 0.78 + 0.22 * ea, 1.10, 0.90 + 0.18 * ec, 0.82 + 0.25 * eb])
    elif mode < 0.30:
        cap *= np.array([0.92 + 0.25 * ea, 1.08, 0.86 + 0.22 * eb, 1.12, 0.94 + 0.20 * ec])
    return np.maximum(cap, 4.0)


def generate_dataset(n_batches, seed, split="train"):
    rng = np.random.default_rng(seed)
    family, station, handling, product, lane_pair, fam_pair = _static()
    job_Xs = []
    job_codes = []
    batch_X = []
    ys = []
    offsets = [0]
    slices = []

    for b in range(n_batches):
        sid = _sample_slice(rng, split)
        if sid == 0:
            n = int(rng.integers(4, 9))
        elif sid in (1, 2, 3, 5, 7, 9):
            n = int(rng.integers(16, 32))
        elif sid == 8:
            n = int(rng.choice([rng.integers(2, 5), rng.integers(32, 55)]))
        else:
            n = int(rng.integers(10, 25))

        staff = rng.uniform(0.35, 1.0)
        ea = rng.integers(0, 3) / 2.0
        eb = rng.integers(0, 3) / 2.0
        ec = rng.integers(0, 3) / 2.0
        shift = rng.uniform(0, 1)
        maint = float(rng.random() < (0.27 if sid == 6 else 0.09))
        mode = rng.uniform(0, 1)
        day = rng.integers(0, 7) / 6.0
        bx = np.array([staff, ea, eb, ec, shift, maint, mode, day], dtype=float)
        batch_X.append(bx)

        fam = _choose_family(rng, sid, n)
        st = (2 * fam + rng.integers(0, 5, size=n) + (sid == 6) * rng.integers(0, 3, size=n)) % N_STATION
        if sid in (2, 9):
            st = rng.choice(N_STATION, size=n)
        hand = rng.choice(N_HANDLING, size=n, p=np.array([.19, .17, .16, .16, .15, .10, .07]))
        if sid == 7:
            hand = np.where(rng.random(n) < 0.48, 6, hand)
        prod_group = rng.integers(0, 5, size=n)
        prod = (fam + N_FAMILY * prod_group) % N_PRODUCT
        # In matched proxy slices, keep marginals similar while changing lane composition.
        if sid == 9:
            half = n // 2
            prod[:half] = (fam[:half] + N_FAMILY * ((fam[:half] + 1) % 5)) % N_PRODUCT
            prod[half:] = (fam[half:] + N_FAMILY * ((fam[half:] + 3) % 5)) % N_PRODUCT

        size = rng.lognormal(mean=0.0, sigma=0.43, size=n)
        if sid == 7:
            size = np.where(hand == 6, size * rng.uniform(1.8, 3.0, size=n), size)
        comp = np.clip(rng.beta(2.1, 2.25, size=n) + 0.10 * (fam % 5 == 2), 0, 1.35)
        if sid == 5:
            priority = np.where(rng.random(n) < 0.34, rng.uniform(0.80, 1.0, size=n), rng.uniform(0.02, 0.38, size=n))
        else:
            priority = rng.beta(1.45, 2.55, size=n)
        slack = rng.beta(2.1, 2.0, size=n)
        frag = np.clip(rng.beta(1.7, 3.0, size=n) + 0.20 * (hand == 4), 0, 1.25)
        rush = np.clip(priority + rng.normal(0, 0.18, size=n) - 0.15 * slack, 0, 1.2)
        decoy_visible = (hand == 6).astype(float) + rng.normal(0, 0.04, size=n)
        X = np.stack([size, comp, priority, slack, frag, rush, decoy_visible], axis=1).astype(np.float64)
        C = np.stack([fam, st, hand, prod], axis=1).astype(np.int64)
        D = _demand_from_visible(X, C, family, station, handling, product)
        cap = _batch_capacity(bx, sid)
        dnorm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-6)
        lane = prod % 5

        related = np.zeros((n, N_RES), dtype=float)
        peak_pressure = np.zeros(n, dtype=float)
        merge_credit = np.zeros(n, dtype=float)
        rival_count = np.zeros(n, dtype=float)
        dec_load = D[hand == 6].sum(axis=0) if np.any(hand == 6) else np.zeros(N_RES)

        for i in range(n):
            dot = dnorm @ dnorm[i]
            compat = 0.16 + 0.46 * dot
            compat *= fam_pair[fam[i], fam]
            compat *= lane_pair[lane[i], lane]
            compat *= (1.0 + 0.30 * (st == st[i]) + 0.16 * (hand == hand[i]))
            # Jobs close in requested lane but not exact same code are the hard cases.
            compat *= (1.0 + 0.18 * (((prod // N_FAMILY) == (prod[i] // N_FAMILY)) & (fam != fam[i])))
            # Directional priority. Higher priority neighbors hurt more; lower priority neighbors can be mostly harmless.
            pf = 0.45 + 1.05 * np.maximum(priority - priority[i], 0.0)
            pf += 0.30 * (priority > 0.82) + 0.22 * (priority[i] < 0.28)
            pf *= (1.0 - 0.26 * np.maximum(priority[i] - priority, 0.0))
            # Some similar low-rush jobs combine efficiently rather than hurt each other.
            merge = ((fam % 5) == (fam[i] % 5)) & (hand == hand[i]) & (priority < 0.42) & (priority[i] < 0.42)
            weights = compat * pf
            weights[i] = 0.30
            related[i] = (D * weights[:, None]).sum(axis=0)
            merge_credit[i] = 0.55 * np.sqrt(np.maximum(0.0, (D[merge, lane[i]].sum() if np.any(merge) else 0.0)))
            top = np.sort((D[:, lane[i]] * compat * (priority > priority[i] + 0.18)))[-4:]
            peak_pressure[i] = top.sum() if len(top) else 0.0
            rival_count[i] = np.sum((compat > 0.85) & (priority > priority[i] + 0.12))

        # Job-specific stress, not a batch-wide addend.
        nscale = 0.42 + 0.012 * n
        dscale = 0.58 + 0.008 * n
        stress = (related - cap[None, :] * nscale) / (cap[None, :] * dscale)
        sens = 0.24 + dnorm + 0.18 * frag[:, None] * np.array([0.25, 0.30, 0.20, 0.55, 0.45])
        cong = (sens * _softplus(1.18 * stress)).sum(axis=1)

        same_family_count = np.array([(fam == fam[i]).sum() - 1 for i in range(n)])
        high_prio_same_station = np.array([((st == st[i]) & (priority > priority[i] + 0.25)).sum() for i in range(n)])
        base = 17.0 + 9.0 * np.log1p(size) + 6.5 * comp + 4.8 * frag - 4.6 * priority + 3.2 * (1.0 - slack) + 1.6 * rush
        y = base + 12.8 * cong
        y += 1.05 * np.sqrt(np.maximum(same_family_count, 0)) * ((fam % 5 == 1) + 0.35)
        y += 1.75 * high_prio_same_station
        y += 1.90 * _softplus((peak_pressure - cap[lane] * 0.22) / (cap[lane] * 0.26))
        y -= 1.30 * merge_credit
        y += 0.65 * rival_count * (priority < 0.35)

        if sid == 4:
            # Rare families require borrowing structure from family/station/product lanes.
            y += 4.2 * (fam >= 13) * (0.45 + _softplus(stress[np.arange(n), fam % 5]))
        if sid == 2:
            y += 1.8 * np.maximum(0, D[:, 1] - D[:, 0]) * _softplus(stress[:, 1])
        if sid == 3:
            y += 3.1 * ((lane == lane[0]) | (fam % 5 == fam[0] % 5)) * (cong > 1.25)
        if sid == 5:
            y += 2.8 * (priority < 0.30) * _softplus((peak_pressure - cap[lane] * 0.18) / (cap[lane] * 0.20))
        if sid == 6:
            y += 2.2 * ((mode > 0.67) & (dnorm[:, 1] > 0.45)) * _softplus(stress[:, 1])
            y += 1.8 * ((mode < 0.30) & (dnorm[:, 2] > 0.45)) * _softplus(stress[:, 2])
        if sid == 7:
            y += 2.4 * (dnorm[:, 2] > 0.55) * _softplus((dec_load[2] - cap[2] * 0.25) / (cap[2] * 0.38))
        if sid == 8:
            y += 2.3 * (n > 30) * _softplus(stress[:, 4])
        if sid == 9:
            # Same aggregate traps: lane interactions dominate even when family counts and total size match.
            y += 3.4 * lane_pair[lane, (lane + 1) % 5] * _softplus(stress[np.arange(n), lane])

        noise = rng.normal(0, 1.45 + 0.22 * np.sqrt(n), size=n)
        y = y + noise
        job_Xs.append(X)
        job_codes.append(C)
        ys.append(y.astype(np.float64))
        slices.extend([sid] * n)
        offsets.append(offsets[-1] + n)

    return {
        "job_X": np.vstack(job_Xs).astype(np.float64),
        "job_code": np.vstack(job_codes).astype(np.int64),
        "batch_X": np.vstack(batch_X).astype(np.float64),
        "batch_offsets": np.array(offsets, dtype=np.int64),
        "y": np.concatenate(ys).astype(np.float64),
        "slice_id": np.array(slices, dtype=np.int64),
        "slice_names": SLICE_NAMES,
    }

test_main
from hidden_eval import evaluate


def test_hidden_eval_runs():
    reward, metrics = evaluate()
    assert 0.0 <= reward <= 1.0
    assert metrics["overall_rmse"] >= 0.0

