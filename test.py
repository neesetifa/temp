reference
"""Canonical self-contained reference solution for component_fit_regression v0.2.

This file is intentionally self-contained. During oracle evaluation the verifier may
copy only this file to app/solve.py, so it must not import sibling/private files.
It exposes exactly the same public functions/signatures as app/solve.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SLOTS = 8
D = 8


def slot_tensor(part_X, part_slot, case_offsets, n_slots=N_SLOTS):
    part_X = np.asarray(part_X, dtype=float)
    part_slot = np.asarray(part_slot, dtype=int)
    case_offsets = np.asarray(case_offsets, dtype=int)
    n = len(case_offsets) - 1
    d = part_X.shape[1]
    Xs = np.zeros((n, n_slots, d), dtype=float)
    present = np.zeros((n, n_slots), dtype=float)
    counts = np.zeros((n, n_slots), dtype=float)
    # Sum-and-normalize protects against repeated component records.
    for i in range(n):
        lo, hi = int(case_offsets[i]), int(case_offsets[i + 1])
        for j in range(lo, hi):
            s = int(part_slot[j])
            if 0 <= s < n_slots:
                Xs[i, s] += part_X[j]
                counts[i, s] += 1.0
    nz = counts > 0
    Xs[nz] /= counts[nz][..., None]
    present[nz] = 1.0
    return Xs, present, counts


def mean_pooled_features(part_X, part_slot, case_offsets):
    n = len(case_offsets) - 1
    feats = []
    for i in range(n):
        lo, hi = int(case_offsets[i]), int(case_offsets[i + 1])
        X = np.asarray(part_X[lo:hi], dtype=float)
        if len(X) == 0:
            row = np.zeros(4 * D + 1)
        else:
            row = np.concatenate([
                X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0), [len(X)]
            ])
        feats.append(row)
    return np.vstack(feats)


def rich_aggregate_features(part_X, part_slot, case_offsets):
    base = mean_pooled_features(part_X, part_slot, case_offsets)
    _, present, counts = slot_tensor(part_X, part_slot, case_offsets)
    n_rows = counts.sum(axis=1, keepdims=True)
    return np.hstack([base, present, counts, n_rows, present.sum(axis=1, keepdims=True)])


def slot_additive_features(part_X, part_slot, case_offsets, include_squares=True):
    Xs, present, counts = slot_tensor(part_X, part_slot, case_offsets)
    parts = [Xs.reshape(len(Xs), -1), present, counts]
    if include_squares:
        parts.append((Xs ** 2).reshape(len(Xs), -1))
    return np.hstack(parts)


def all_pair_features(part_X, part_slot, case_offsets):
    Xs, present, counts = slot_tensor(part_X, part_slot, case_offsets)
    n = len(Xs)
    parts = [slot_additive_features(part_X, part_slot, case_offsets, include_squares=True)]
    pair_feats = []
    for a in range(N_SLOTS):
        for b in range(a + 1, N_SLOTS):
            pa = present[:, a:a+1]
            pb = present[:, b:b+1]
            pp = pa * pb
            xa = Xs[:, a, :]
            xb = Xs[:, b, :]
            pair_feats.extend([
                pp,
                pp * np.abs(xa - xb),
                pp * (xa - xb),
                pp * (xa * xb),
            ])
    parts.append(np.hstack(pair_feats))
    return np.hstack(parts)


def reference_features(part_X, part_slot, case_offsets):
    Xs, present, counts = slot_tensor(part_X, part_slot, case_offsets)
    n = len(Xs)
    feats = [slot_additive_features(part_X, part_slot, case_offsets, include_squares=True)]

    def x(s, k):
        return Xs[:, s, k]

    def p(s):
        return present[:, s]

    # Target-aligned quality proxies. These are still inferred from visible columns; no hidden labels/slices.
    q0 = 1.05*x(0,0) - 0.30*x(0,1) + 0.18*x(0,4)
    q1 = 1.10*x(1,0) - 0.25*np.abs(x(1,2)-x(0,1)) + 0.12*x(1,5)
    q2 = 0.95*x(2,1) - 0.45*x(2,3)
    q3 = 0.95*x(3,1) - 0.45*x(3,3)
    q4 = p(4) * (0.90*x(4,0) + 0.20*x(4,1))
    q5 = p(5) * (1.10*x(5,4) + 0.15*x(5,0))
    q = np.vstack([q0, q1, q2, q3, q4, q5]).T
    min_crit = np.min(q[:, :4], axis=1)
    bad_count = np.sum(q[:, :4] < -1.1, axis=1)

    align2 = x(2,0) + 0.35*x(2,2)
    align3 = x(3,0) - 0.30*x(3,2)
    mismatch = np.abs(align2 - align3)
    load_gap = np.abs(x(0,1) - x(1,2))
    variant_marker = 0.55*x(0,2) + 0.45*x(1,3)
    extended_like = (variant_marker > 0.35).astype(float)
    compact_like = (variant_marker < -0.55).astype(float)

    selected = [
        q,
        q ** 2,
        np.tanh(q),
        min_crit[:, None],
        np.maximum(0.0, -0.75 - min_crit)[:, None],
        (np.maximum(0.0, -0.75 - min_crit) ** 1.5)[:, None],
        bad_count[:, None],
        mismatch[:, None],
        mismatch[:, None] ** 2,
        np.maximum(0.0, mismatch - 0.8)[:, None],
        np.maximum(0.0, mismatch - 1.2)[:, None],
        load_gap[:, None],
        np.maximum(0.0, load_gap - 0.9)[:, None],
        variant_marker[:, None],
        extended_like[:, None],
        compact_like[:, None],
        (extended_like * (1.0 - p(4)))[:, None],
        (p(5) * q5)[:, None],
        (p(5) * q5 * (q1 < -0.25))[:, None],
        (p(4) * q4 * mismatch)[:, None],
        (p(7) * np.abs(x(7,1) - 0.4*x(3,1)))[:, None],
        present.sum(axis=1, keepdims=True),
    ]
    # A few selected role pairs, not all pairs.
    for a, b in [(2,3), (0,1), (1,5), (0,4), (3,7)]:
        pp = (p(a) * p(b))[:, None]
        diff = Xs[:, a, :] - Xs[:, b, :]
        selected.extend([pp, pp * np.abs(diff), pp * diff[:, [0,1,2,3]], pp * (Xs[:, a, :] * Xs[:, b, :])[:, [0,1,2,3]]])
    feats.append(np.hstack(selected))
    return np.hstack([*feats, operating_features(part_X,part_slot,case_offsets)])


def operating_features(part_X, part_slot, case_offsets):
    """Physical-summary proxies built from visible role readings.

    These are not labels or hidden-slice flags. The model estimates coefficients
    using the supplied training cases.
    """
    X, P, _ = slot_tensor(part_X, part_slot, case_offsets)
    n = len(X)
    def x(s, k): return X[:, s, k]
    def p(s): return P[:, s]
    def plus(z): return np.logaddexp(0., 2.*z) / 2.
    def sig(z): return 1./(1.+np.exp(-np.clip(z,-30,30)))
    q0 = 1.00*x(0,0) - 0.30*x(0,1) + 0.18*x(0,4)
    q1 = 1.05*x(1,0) - 0.22*np.abs(x(1,2)-x(0,1)) + 0.12*x(1,5)
    q2 = 1.0*x(2,1) - 0.42*x(2,3) + 0.12*x(2,5)
    q3 = 1.0*x(3,1) - 0.42*x(3,3) + 0.12*x(3,5)
    q4 = p(4)*(0.9*x(4,0)+0.2*x(4,1))
    q5 = p(5)*(1.1*x(5,4)+0.15*x(5,0))
    qs = np.stack([q0,q1,q2,q3],axis=1)
    duty = x(0,5)
    demand = 0.8*x(0,1) + 0.4*duty + 0.15*x(1,2)
    stress = plus(demand)
    marker = .5*x(0,2)+.5*x(1,3)
    ext = sig(2.2*(marker-.25))
    compact = sig(-2.7*(marker+.4))
    stiffness_gap=x(2,4)-x(3,4)
    routing_mode=np.tanh(1.4*marker+.4*p(7)*x(7,6))
    left_share=sig((1.2+.7*routing_mode)*stiffness_gap
                   +(.27+.4*routing_mode)*demand+.2*marker
                   +.4*p(4)*np.tanh(q4)*stress)
    right_share=1-left_share
    eff = np.stack([
        q0-(.25+.45*sig(x(0,6)))*stress,
        q1-(.35+.48*sig(x(1,7)))*stress+.20*np.tanh(q4),
        q2-(.23+1.1*left_share*sig(x(2,6)))*stress+.13*np.tanh(q1),
        q3-(.28+1.1*right_share*sig(x(3,6)))*stress+.12*np.tanh(q0),
    ],axis=1)
    soft_mins=[]
    for temp in (.40,.65,1.0):
        arg = -eff / temp
        z = -temp*(np.logaddexp.reduce(arg,axis=1)-np.log(4.0))
        soft_mins.append(z)
    softmin=soft_mins[1]
    align2=x(2,0)+0.35*x(2,2)
    align3=x(3,0)-0.30*x(3,2)
    signed_gap=align2-align3
    fit_gap=np.abs(signed_gap-(.25+.25*routing_mode)*demand
                  -.55*stress*(left_share-.5)-.30*routing_mode*np.tanh(stiffness_gap))
    fit_tolerance=1.0+.25*np.tanh(q4)-.14*sig(demand)-.17*sig(stiffness_gap)*(1-p(4))
    excess=plus(fit_gap-fit_tolerance)
    load_gap=np.abs(x(0,1)-x(1,2))
    deflection=np.abs(x(0,4)-.50*x(1,5))
    # Candidate terms encode general shape, not the target's exact coefficients.
    scalar = [q0,q1,q2,q3,q4,q5, demand,stress,duty,marker,ext,compact,
        softmin, signed_gap,fit_gap,fit_tolerance,excess,load_gap,deflection,
        sig(demand),sig(-q1),np.mean(eff,axis=1),
        np.min(eff,axis=1),np.max(eff,axis=1),
        p(4)*q4,p(5)*q5, (1-p(4))*ext, p(7)*x(7,1),
        stiffness_gap,left_share,right_share,stress*left_share,
        stress*right_share,stress*left_share*sig(x(2,6)),
        stress*right_share*sig(x(3,6)),
        sig(demand)*plus(.3-eff[:,2])*plus(.3-eff[:,3]),
        routing_mode, routing_mode*demand,routing_mode*stiffness_gap,
        routing_mode*np.tanh(stiffness_gap),routing_mode*stress,
        (sig(marker)*sig(demand)*plus(.3-eff[:,2])*plus(.3-eff[:,3]))]
    scalar.extend(soft_mins)
    scalar.extend([plus(-.5-softmin), plus(-1.-softmin),
                   plus(fit_gap-.75),plus(fit_gap-1.2),
                   plus(load_gap-.9),plus(deflection-.9)])
    for j,q in enumerate([q0,q1,q2,q3]):
        sensitivity=x(j,6) if j!=1 else x(1,7)
        scalar.extend([q*stress, plus(-.9-q), plus(-.9-q)**1.5,
                       stress*sig(sensitivity), stress*sig(sensitivity)**2,
                       eff[:,j],plus(-1.15-eff[:,j])**1.25])
    scalar.extend([stress*np.tanh(q4),
                   p(5)*np.tanh(q5)*sig(stress+plus(-q1)-.9),
                   excess*sig(demand),excess**1.5,excess**2,
                   softmin*sig(demand),plus(-.55-softmin)**1.5,
                   plus(-.55-softmin)**2,
                   compact*plus(load_gap-.9)**1.3,
                   sig(demand)*plus(deflection-.9)**1.3,
                   (1-p(4))*ext*sig(demand),
                   p(7)*sig(demand)*plus(np.abs(x(7,1)-x(3,1))-.9)])
    return np.stack(scalar,axis=1)


ALPHAS = np.logspace(-2, 4, 16)


class _ReferenceModel:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

    def fit(self, train_part_X, train_part_slot, train_case_offsets, train_y):
        F = reference_features(train_part_X, train_part_slot, train_case_offsets)
        self.model.fit(F, np.asarray(train_y, dtype=float))
        return self

    def predict(self, part_X, part_slot, case_offsets):
        F = reference_features(part_X, part_slot, case_offsets)
        return np.asarray(self.model.predict(F), dtype=float)


def fit_component_model(train_part_X, train_part_slot, train_case_offsets, train_y):
    """Fit the reference model.

    Signature matches app/solve.py:
        fit_component_model(train_part_X, train_part_slot, train_case_offsets, train_y)
    """
    model = _ReferenceModel()
    model.fit(train_part_X, train_part_slot, train_case_offsets, train_y)
    return {"model": model}


def predict_component_score(part_X, part_slot, case_offsets, params):
    """Return one prediction per case.

    Signature matches app/solve.py:
        predict_component_score(part_X, part_slot, case_offsets, params)
    """
    return params["model"].predict(part_X, part_slot, case_offsets)


hidden
"""Hidden evaluator for component_fit_regression v0.2.

This file is under top-level tests/ and is not visible to agents. It is self-contained
with tests/generator.py and does not import dev/ or solution/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Callable, Dict, Any

import numpy as np

try:
    from .generator import generate_split
except ImportError:  # allow running this file directly from tests/
    from generator import generate_split

# Reward anchors rechecked against v0.2 reference and non-oracle baselines.
RMSE_GOOD = {
    "overall": 1.70,
    "mismatch": 2.52,
    "bottleneck": 1.88,
    "rare_slot": 2.05,
    "missing_optional": 2.22,
    "variant_shift": 2.03,
}
RMSE_CUTOFF = {
    "overall": 13.50,
    "mismatch": 18.25,
    "bottleneck": 14.75,
    "rare_slot": 14.50,
    "missing_optional": 15.50,
    "variant_shift": 14.50,
}
WEIGHTS = {
    "overall": 0.25,
    "mismatch": 0.20,
    "bottleneck": 0.20,
    "rare_slot": 0.15,
    "missing_optional": 0.10,
    "variant_shift": 0.05,
    "sanity": 0.05,
}


def decreasing_reward(value: float, good: float, cutoff: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value >= cutoff:
        return 0.0
    if value <= good:
        return 1.0
    return float((cutoff - value) / (cutoff - good))


def _rmse(y, pred, mask=None) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if mask is None:
        mask = np.ones_like(y, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))


def _check_prediction(pred, n_cases):
    pred = np.asarray(pred)
    if pred.shape != (n_cases,):
        return False, f"wrong shape: expected {(n_cases,)}, got {pred.shape}"
    if not np.all(np.isfinite(pred)):
        return False, "NaN or Inf prediction"
    if float(np.std(pred)) < 1e-8:
        return False, "prediction std almost zero"
    return True, "ok"


def _load_train(app_dir: str):
    data = np.load(os.path.join(app_dir, "train_data.npz"))
    return data["train_part_X"], data["train_part_slot"], data["train_case_offsets"], data["train_y"]


def evaluate(fit_fn: Callable, predict_fn: Callable, app_dir: str = "app", seeds=(901, 902, 903, 904, 905)) -> Dict[str, Any]:
    train_part_X, train_part_slot, train_case_offsets, train_y = _load_train(app_dir)

    # Check fit input mutation.
    originals = [arr.copy() for arr in (train_part_X, train_part_slot, train_case_offsets, train_y)]
    params = fit_fn(train_part_X, train_part_slot, train_case_offsets, train_y)
    for arr, orig, name in zip((train_part_X, train_part_slot, train_case_offsets, train_y), originals, ["train_part_X", "train_part_slot", "train_case_offsets", "train_y"]):
        if not np.array_equal(arr, orig):
            return {"reward": 0.0, "passed_cutoff": False, "error": f"input mutation during fit: {name}"}

    per_seed = []
    component_values = {k: [] for k in ["overall", "mismatch", "bottleneck", "rare_slot", "missing_optional", "variant_shift"]}
    component_rewards = {k: [] for k in WEIGHTS}

    for seed in seeds:
        split = generate_split(620, int(seed), "hidden")
        X = split.part_X.copy()
        slot = split.part_slot.copy()
        offsets = split.case_offsets.copy()
        X0, slot0, offsets0 = X.copy(), slot.copy(), offsets.copy()
        pred1 = predict_fn(X, slot, offsets, params)
        ok, msg = _check_prediction(pred1, len(offsets) - 1)
        if not ok:
            return {"reward": 0.0, "passed_cutoff": False, "error": msg, "seed": int(seed)}
        if not (np.array_equal(X, X0) and np.array_equal(slot, slot0) and np.array_equal(offsets, offsets0)):
            return {"reward": 0.0, "passed_cutoff": False, "error": "input mutation during predict", "seed": int(seed)}
        pred2 = predict_fn(X.copy(), slot.copy(), offsets.copy(), params)
        if not np.allclose(pred1, pred2, atol=1e-10, rtol=1e-10):
            return {"reward": 0.0, "passed_cutoff": False, "error": "nondeterministic prediction", "seed": int(seed)}

        metrics = {
            "overall_rmse": _rmse(split.y, pred1),
            "mismatch_rmse": _rmse(split.y, pred1, split.slices["mismatch"]),
            "bottleneck_rmse": _rmse(split.y, pred1, split.slices["bottleneck"]),
            "rare_slot_rmse": _rmse(split.y, pred1, split.slices["rare_slot"]),
            "missing_optional_rmse": _rmse(split.y, pred1, split.slices["missing_optional"]),
            "variant_shift_rmse": _rmse(split.y, pred1, split.slices["variant_shift"]),
            "short_case_rmse": _rmse(split.y, pred1, split.slices["short_case"]),
            "long_case_rmse": _rmse(split.y, pred1, split.slices["long_case"]),
        }
        seed_rewards = {}
        for k in ["overall", "mismatch", "bottleneck", "rare_slot", "missing_optional", "variant_shift"]:
            r = decreasing_reward(metrics[f"{k}_rmse"], RMSE_GOOD[k], RMSE_CUTOFF[k])
            seed_rewards[k] = r
            component_values[k].append(metrics[f"{k}_rmse"])
            component_rewards[k].append(r)
        # Sanity reward: no huge bias and variance in plausible range.
        bias = abs(float(np.mean(pred1 - split.y)))
        spread_ratio = float(np.std(pred1) / (np.std(split.y) + 1e-12))
        sanity = min(
            decreasing_reward(bias, 0.7, 7.0),
            1.0 if 0.45 <= spread_ratio <= 1.65 else max(0.0, 1.0 - abs(spread_ratio - 1.0) / 1.4),
        )
        seed_rewards["sanity"] = float(sanity)
        component_rewards["sanity"].append(float(sanity))
        per_seed.append({"seed": int(seed), "metrics": metrics, "component_rewards": seed_rewards})

    aggregate_metrics = {f"{k}_rmse": float(np.mean(v)) for k, v in component_values.items()}
    aggregate_component_rewards = {k: float(np.mean(v)) for k, v in component_rewards.items()}
    reward = float(sum(WEIGHTS[k] * aggregate_component_rewards[k] for k in WEIGHTS))

    # Catastrophic guards.
    passed_cutoff = True
    for k in ["overall", "mismatch", "bottleneck", "rare_slot", "missing_optional", "variant_shift"]:
        if aggregate_metrics[f"{k}_rmse"] >= RMSE_CUTOFF[k]:
            passed_cutoff = False
    if not passed_cutoff:
        reward = 0.0

    return {
        "reward": reward,
        "passed_cutoff": bool(passed_cutoff),
        "aggregate_metrics": aggregate_metrics,
        "aggregate_component_rewards": aggregate_component_rewards,
        "per_seed": per_seed,
    }


def evaluate_solution_file(solution_path: str, app_dir: str = "app") -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("candidate_solve", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {solution_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solve"] = mod
    spec.loader.exec_module(mod)
    return evaluate(mod.fit_component_model, mod.predict_component_score, app_dir=app_dir)


if __name__ == "__main__":
    # Usage: python -m private.hidden_eval app/solve.py
    solution = sys.argv[1] if len(sys.argv) > 1 else os.path.join("app", "solve.py")
    print(json.dumps(evaluate_solution_file(solution), indent=2, sort_keys=True))

generator

"""Deterministic synthetic component-case generator for component_fit v0.2.

Private file. The visible task only exposes app/train_data.npz and app/public_eval.npz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

N_SLOTS = 8
D = 8


@dataclass(frozen=True)
class GeneratedSplit:
    part_X: np.ndarray
    part_slot: np.ndarray
    case_offsets: np.ndarray
    y: np.ndarray
    slices: Dict[str, np.ndarray]


def _soft_hinge(x: float) -> float:
    return float(np.logaddexp(0.0, 2.0 * x) / 2.0)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0))))


def _row_noise(rng: np.random.Generator, scale: float = 0.20) -> np.ndarray:
    return rng.normal(0.0, scale, size=D)


def generate_split(n_cases: int, seed: int, split: str = "train") -> GeneratedSplit:
    """Generate one deterministic split.

    split controls mixture weights only; the underlying semantics are stable across train/hidden.
    """
    rng = np.random.default_rng(seed)
    rows = []
    slots = []
    offsets = [0]
    y = np.zeros(n_cases, dtype=np.float64)

    slice_flags = {
        "balanced": np.zeros(n_cases, dtype=bool),
        "mismatch": np.zeros(n_cases, dtype=bool),
        "bottleneck": np.zeros(n_cases, dtype=bool),
        "rare_slot": np.zeros(n_cases, dtype=bool),
        "missing_optional": np.zeros(n_cases, dtype=bool),
        "variant_shift": np.zeros(n_cases, dtype=bool),
        "short_case": np.zeros(n_cases, dtype=bool),
        "long_case": np.zeros(n_cases, dtype=bool),
    }

    # Train has all regimes. Hidden is a little heavier on the slices that diagnose failures.
    if split == "train":
        variant_p = np.array([0.49, 0.23, 0.20, 0.08])
        mismatch_base = 0.16
        bottleneck_base = 0.17
        rare_mult = 1.0
    elif split == "public":
        variant_p = np.array([0.53, 0.23, 0.18, 0.06])
        mismatch_base = 0.12
        bottleneck_base = 0.14
        rare_mult = 0.75
    else:
        variant_p = np.array([0.41, 0.21, 0.26, 0.12])
        mismatch_base = 0.21
        bottleneck_base = 0.21
        rare_mult = 1.25

    for i in range(n_cases):
        variant = int(rng.choice(4, p=variant_p))
        # Variant marker is visible but noisy; presence pattern also carries subtype evidence.
        variant_signal = [-0.9, -0.35, 0.85, 0.35][variant] + rng.normal(0, 0.25)

        present = {0, 1, 2, 3}
        if variant == 0:  # standard
            if rng.random() < 0.36:
                present.add(4)
            if rng.random() < 0.10 * rare_mult:
                present.add(5)
            if rng.random() < 0.12:
                present.add(6)
        elif variant == 1:  # compact
            if rng.random() < 0.12:
                present.add(4)
            if rng.random() < 0.06 * rare_mult:
                present.add(5)
            if rng.random() < 0.28:
                present.add(6)
        elif variant == 2:  # extended: slot 4 is expected, but sometimes missing
            if rng.random() < 0.82:
                present.add(4)
            if rng.random() < 0.24 * rare_mult:
                present.add(5)
            if rng.random() < 0.30:
                present.add(6)
            if rng.random() < 0.18:
                present.add(7)
        else:  # asymmetric
            if rng.random() < 0.55:
                present.add(4)
            if rng.random() < 0.17 * rare_mult:
                present.add(5)
            if rng.random() < 0.20:
                present.add(6)
            if rng.random() < 0.36:
                present.add(7)

        mismatch = rng.random() < (mismatch_base + (0.08 if variant in (2, 3) else 0.0))
        bottleneck = rng.random() < (bottleneck_base + (0.05 if variant == 1 else 0.0))
        bottleneck_slot = int(rng.choice([0, 1, 2, 3])) if bottleneck else -1

        frame_q = rng.normal(0.0, 1.0)
        support_q = 0.35 * frame_q + rng.normal(0.0, 0.95)
        left_strength = rng.normal(0.0, 1.0)
        right_strength = 0.25 * left_strength + rng.normal(0.0, 0.95)
        # Mildly skewed operating demand and correlated operating duty.
        load = 0.68 * rng.normal() + 0.58 * (rng.lognormal(0.0, 0.65) - 1.235)
        duty = 0.62 * load + 0.72 * rng.normal()
        align = rng.normal(0.0, 0.85)

        # Inject visible regimes.
        if bottleneck_slot == 0:
            frame_q -= rng.uniform(1.8, 3.0)
        elif bottleneck_slot == 1:
            support_q -= rng.uniform(1.8, 3.0)
        elif bottleneck_slot == 2:
            left_strength -= rng.uniform(1.8, 3.1)
        elif bottleneck_slot == 3:
            right_strength -= rng.uniform(1.8, 3.1)

        if mismatch:
            delta = rng.choice([-1.0, 1.0]) * rng.uniform(1.3, 2.8)
        else:
            delta = rng.normal(0.0, 0.35)
        left_align = align + 0.5 * delta + rng.normal(0, 0.10)
        right_align = align - 0.5 * delta + rng.normal(0, 0.10)

        stabilizer_q = rng.normal(0.1 + 0.35 * (variant == 2), 0.9)
        rare_q = rng.normal(0.0, 1.0)
        aux_q = rng.normal(0.0, 1.0)

        case_rows = {}

        x0 = _row_noise(rng)
        x0[0] += frame_q
        x0[1] += load
        x0[2] += variant_signal
        x0[4] += 0.4 * support_q
        x0[5] += duty
        x0[6] += 0.30 * frame_q + rng.normal(0, 0.55)
        case_rows[0] = x0

        x1 = _row_noise(rng)
        x1[0] += support_q
        x1[2] += load + rng.normal(0, 0.15)
        x1[3] += variant_signal + rng.normal(0, 0.20)
        x1[5] += 0.25 * frame_q
        x1[6] += 0.60 * duty + rng.normal(0, 0.12)
        x1[7] += 0.30 * support_q + rng.normal(0, 0.55)
        case_rows[1] = x1

        x2 = _row_noise(rng)
        x2[0] += left_align
        x2[1] += left_strength
        x2[2] += 0.45 * variant_signal
        x2[3] += -0.35 * left_strength + rng.normal(0, 0.15)
        x2[5] += 0.28 * duty + rng.normal(0, 0.15)
        x2[6] += -0.20 * left_strength + rng.normal(0, 0.55)
        x2[4] += 0.70 * left_strength + rng.normal(0, 0.30)
        case_rows[2] = x2

        x3 = _row_noise(rng)
        x3[0] += right_align
        x3[1] += right_strength
        x3[2] += -0.35 * variant_signal
        x3[3] += -0.35 * right_strength + rng.normal(0, 0.15)
        x3[5] += 0.24 * duty + rng.normal(0, 0.15)
        x3[6] += -0.20 * right_strength + rng.normal(0, 0.55)
        x3[4] += 0.70 * right_strength + rng.normal(0, 0.30)
        case_rows[3] = x3

        if 4 in present:
            x4 = _row_noise(rng)
            x4[0] += stabilizer_q
            x4[1] += 0.4 * frame_q + 0.2 * support_q
            x4[2] += variant_signal
            x4[6] += 0.5 * load
            case_rows[4] = x4
        if 5 in present:
            x5 = _row_noise(rng, 0.24)
            x5[0] += rare_q
            x5[4] += rare_q + 0.35 * (support_q < -0.4)
            x5[6] += -0.4 * load
            case_rows[5] = x5
        if 6 in present:
            x6 = _row_noise(rng, 0.28)
            x6[0] += aux_q
            x6[2] += 0.4 * variant_signal
            x6[7] += rng.normal(0, 1.0)
            case_rows[6] = x6
        if 7 in present:
            x7 = _row_noise(rng, 0.26)
            x7[1] += 0.45 * right_strength - 0.25 * left_strength
            x7[5] += rng.normal(0, 1.0)
            x7[6] += variant_signal
            case_rows[7] = x7

        # Only observed part fields and slot presence affect expected score.
        # Operating demand interacts with capacity, fit, and optional equipment.
        def get(slot: int, dim: int, default: float = 0.0) -> float:
            return float(case_rows[slot][dim]) if slot in case_rows else default

        p4, p5, p6, p7 = (float(j in case_rows) for j in (4, 5, 6, 7))
        q0 = 1.05 * get(0, 0) - 0.30 * get(0, 1) + 0.18 * get(0, 4)
        q1 = 1.10 * get(1, 0) - 0.25 * abs(get(1, 2) - get(0, 1)) + 0.12 * get(1, 5)
        q2 = 0.95 * get(2, 1) - 0.45 * get(2, 3) + 0.15 * get(2, 5)
        q3 = 0.95 * get(3, 1) - 0.45 * get(3, 3) + 0.15 * get(3, 5)
        q4 = p4 * (0.90 * get(4, 0) + 0.20 * get(4, 1))
        q5 = p5 * (1.10 * get(5, 4) + 0.15 * get(5, 0))

        demand = 0.78 * get(0, 1) + 0.40 * get(0, 5) + 0.14 * get(1, 2)
        stress = _soft_hinge(demand + 0.12)
        variant_marker = 0.52 * get(0, 2) + 0.48 * get(1, 3)
        extended = _sigmoid(2.4 * (variant_marker - 0.22))
        compact = _sigmoid(-2.8 * (variant_marker + 0.40))

        # Role capacities are condition-dependent: the weakest part changes
        # with demand rather than being a fixed minimum across four slots.
        eq0 = q0 - (0.24 + 0.48 * _sigmoid(get(0, 6))) * stress
        eq1 = q1 - (0.35 + 0.48 * _sigmoid(get(1, 7))) * stress + 0.22 * np.tanh(q4)
        # The two parallel arms take different shares of demand depending on
        # measured stiffness and geometry. Equal total capacity is not enough.
        stiffness_gap = get(2, 4) - get(3, 4)
        # Configuration changes the load path, not just an additive score.
        routing_mode = np.tanh(1.45 * variant_marker + 0.42 * p7 * get(7, 6))
        left_share = _sigmoid((1.35 + 0.78 * routing_mode) * stiffness_gap
                              + (0.30 + 0.45 * routing_mode) * demand
                              + 0.20 * variant_marker
                              + 0.45 * p4 * np.tanh(q4) * stress)
        right_share = 1.0 - left_share
        eq2 = q2 - (0.22 + 1.18 * left_share * _sigmoid(get(2, 6) + 0.2 * get(2, 2))) * stress + 0.15 * np.tanh(q1)
        eq3 = q3 - (0.27 + 1.18 * right_share * _sigmoid(get(3, 6) - 0.2 * get(3, 2))) * stress + 0.12 * np.tanh(q0)
        effective = np.array([eq0, eq1, eq2, eq3], dtype=float)
        temp = 0.55 + 0.20 * _sigmoid(demand)
        soft_min = float(-temp * np.log(np.mean(np.exp(-effective / temp))))

        score = 53.0 + 1.8 * (q0 + q1 + q2 + q3)
        score += 1.5 * np.mean(effective) + 2.9 * soft_min
        score -= 7.0 * _soft_hinge(-0.55 - soft_min) ** 1.50
        score -= 1.65 * np.sum([_soft_hinge(-1.15 - v) ** 1.25 for v in effective])
        score -= (2.3 + 1.3 * _sigmoid(variant_marker)) * _sigmoid(demand) * _soft_hinge(0.3 - eq2) * _soft_hinge(0.3 - eq3)

        # Directional fit under demand: high load can require a different
        # alignment offset, and a fitted stabilizer may widen/narrow tolerance.
        align2 = get(2, 0) + 0.35 * get(2, 2)
        align3 = get(3, 0) - 0.30 * get(3, 2)
        observed_gap = align2 - align3
        expected_gap = ((0.24 + 0.29 * routing_mode) * demand
                        + 0.16 * np.tanh(q2 - q3)
                        + 0.58 * stress * (left_share - 0.5)
                        + 0.35 * routing_mode * np.tanh(stiffness_gap))
        gap = abs(observed_gap - expected_gap)
        tolerance = (0.98 + 0.26 * np.tanh(q4) - 0.15 * _sigmoid(demand)
                     - 0.20 * _sigmoid(stiffness_gap) * (1.0 - p4))
        fit_excess = _soft_hinge(gap - tolerance)
        score -= (6.6 + 2.8 * _sigmoid(demand)) * fit_excess ** 1.40

        # Agreement of load measurements and compatibility of support/frame
        # become more important under sustained demand.
        load_gap = abs(get(0, 1) - get(1, 2))
        score -= (1.3 + 1.35 * compact + 0.40 * _sigmoid(demand)) * _soft_hinge(load_gap - 0.95) ** 1.3
        deflection_gap = abs(get(0, 4) - 0.50 * get(1, 5))
        score -= (1.0 + 2.3 * _sigmoid(demand)) * _soft_hinge(deflection_gap - 0.95) ** 1.35

        # Optional components interact with the stress and support state.
        score += p4 * (1.25 * q4 + 0.70 * np.tanh(q4) * stress)
        score += p5 * (1.50 * q5 + 3.0 * np.tanh(q5) * _sigmoid(1.3 * (stress + _soft_hinge(-q1) - 0.95)))
        score -= (1.0 - p4) * 4.7 * extended * (0.7 + 0.55 * _sigmoid(demand))
        score += p6 * (0.45 * np.tanh(get(6, 0)) - 0.35 * compact * abs(get(6, 2) - get(0, 2)))
        score += p7 * (0.85 * np.tanh(get(7, 1) - 0.4 * get(3, 1))
                       - 0.65 * _sigmoid(demand) * _soft_hinge(abs(get(7, 1) - get(3, 1)) - 0.9))

        # Heteroscedastic observation noise: severe, high-stress cases are
        # naturally somewhat less repeatable, but train/hidden share this law.
        noise_sd = 0.95 + 0.15 * _sigmoid(demand) + 0.12 * fit_excess + 0.10 * p5
        score += rng.normal(0.0, noise_sd)

        # Sort rows by slot in generated data. Agents must still use case_offsets.
        for s in sorted(case_rows):
            rows.append(case_rows[s])
            slots.append(s)
        offsets.append(len(rows))
        y[i] = score

        slice_flags["mismatch"][i] = bool(gap > 1.30)
        slice_flags["bottleneck"][i] = bool(soft_min < -1.25)
        slice_flags["rare_slot"][i] = 5 in case_rows
        slice_flags["missing_optional"][i] = bool(extended > 0.67 and 4 not in case_rows)
        slice_flags["variant_shift"][i] = bool(variant_marker > 0.33)
        n_rows = len(case_rows)
        slice_flags["short_case"][i] = n_rows <= 4
        slice_flags["long_case"][i] = n_rows >= 7
        slice_flags["balanced"][i] = not (slice_flags["mismatch"][i] or slice_flags["bottleneck"][i] or slice_flags["rare_slot"][i])

    return GeneratedSplit(
        part_X=np.asarray(rows, dtype=np.float64),
        part_slot=np.asarray(slots, dtype=np.int64),
        case_offsets=np.asarray(offsets, dtype=np.int64),
        y=y.astype(np.float64),
        slices=slice_flags,
    )


def write_visible_data(app_dir: str, train_seed: int = 101, public_seed: int = 202) -> None:
    import os

    os.makedirs(app_dir, exist_ok=True)
    train = generate_split(2200, train_seed, "train")
    public = generate_split(260, public_seed, "public")
    np.savez(
        os.path.join(app_dir, "train_data.npz"),
        train_part_X=train.part_X,
        train_part_slot=train.part_slot,
        train_case_offsets=train.case_offsets,
        train_y=train.y,
    )
    # Do not include public_y in the visible app. Public tests are API/sanity only.
    np.savez(
        os.path.join(app_dir, "public_eval.npz"),
        public_part_X=public.part_X,
        public_part_slot=public.part_slot,
        public_case_offsets=public.case_offsets,
    )


