# Fix component cases

`app/solve.py` has two stubs:

```python
fit_component_model(
    train_part_X,
    train_part_slot,
    train_case_offsets,
    train_y,
)

predict_component_score(
    part_X,
    part_slot,
    case_offsets,
    params,
)
```

Return one score per case.

The input rows are packed component records. A case can have several rows, and the slot id is part of the row. The fields line up across rows, but rows from different slots do not play the same role.

The old version averaged the rows into one vector per case. It looked okay on balanced cases, but it missed cases with one odd component, missing optional rows, or two slots that disagreed in a way the average hid.

Use the arrays passed in. No hidden files, network, outside data, or fitting on public eval answers.


hidden
"""Hidden evaluator for component_fit_regression v0.1f.

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

# Reward anchors from v0.1 calibration. They are intentionally coarse and slice-specific.
RMSE_GOOD = {
    "overall": 1.55,
    "mismatch": 1.90,
    "bottleneck": 1.65,
    "rare_slot": 1.90,
    "missing_optional": 2.10,
    "variant_shift": 1.85,
}
RMSE_CUTOFF = {
    "overall": 13.50,
    "mismatch": 15.50,
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
"""Synthetic data generator for component_fit_regression v0.1.

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
    # Smooth enough that learned approximations are possible, but still bottleneck-like.
    return float(np.log1p(np.exp(2.3 * x)) / 2.3)


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
        variant_p = np.array([0.45, 0.24, 0.21, 0.10])
        mismatch_base = 0.22
        bottleneck_base = 0.20
        rare_mult = 1.0
    elif split == "public":
        variant_p = np.array([0.50, 0.24, 0.18, 0.08])
        mismatch_base = 0.16
        bottleneck_base = 0.14
        rare_mult = 0.75
    else:
        variant_p = np.array([0.34, 0.20, 0.31, 0.15])
        mismatch_base = 0.30
        bottleneck_base = 0.26
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
        load = rng.normal(0.0, 1.0)
        align = rng.normal(0.0, 1.0)

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
        case_rows[0] = x0

        x1 = _row_noise(rng)
        x1[0] += support_q
        x1[2] += load + rng.normal(0, 0.15)
        x1[3] += variant_signal + rng.normal(0, 0.20)
        x1[5] += 0.25 * frame_q
        case_rows[1] = x1

        x2 = _row_noise(rng)
        x2[0] += left_align
        x2[1] += left_strength
        x2[2] += 0.45 * variant_signal
        x2[3] += -0.35 * left_strength + rng.normal(0, 0.15)
        case_rows[2] = x2

        x3 = _row_noise(rng)
        x3[0] += right_align
        x3[1] += right_strength
        x3[2] += -0.35 * variant_signal
        x3[3] += -0.35 * right_strength + rng.normal(0, 0.15)
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

        # Build target using the observed row features plus stable semantics.
        def get(slot: int, dim: int, default: float = 0.0) -> float:
            return float(case_rows[slot][dim]) if slot in case_rows else default

        q0 = 1.05 * get(0, 0) - 0.30 * get(0, 1) + 0.18 * get(0, 4)
        q1 = 1.10 * get(1, 0) - 0.25 * abs(get(1, 2) - get(0, 1)) + 0.12 * get(1, 5)
        q2 = 0.95 * get(2, 1) - 0.45 * get(2, 3)
        q3 = 0.95 * get(3, 1) - 0.45 * get(3, 3)
        q4 = 0.90 * get(4, 0) + 0.20 * get(4, 1) if 4 in case_rows else 0.0
        q5 = 1.10 * get(5, 4) + 0.15 * get(5, 0) if 5 in case_rows else 0.0

        score = 52.0
        score += 3.2 * q0 + 2.7 * q1 + 2.1 * q2 + 2.1 * q3
        if 4 in case_rows:
            score += (1.6 + 0.8 * (variant == 2)) * q4
        if 5 in case_rows:
            # Rare slot is high impact mainly when support is weak; otherwise it is modest.
            score += 2.0 * q5 + 3.6 * np.tanh(q5) * (1.0 if q1 < -0.25 else 0.35)

        align2 = get(2, 0) + 0.35 * get(2, 2)
        align3 = get(3, 0) - 0.30 * get(3, 2)
        raw_mismatch = abs(align2 - align3)
        stabilizer_relief = 0.28 * np.tanh(q4) if 4 in case_rows else 0.0
        variant_tolerance = [1.05, 0.90, 1.20, 0.78][variant]
        mismatch_penalty = 7.8 * _soft_hinge(raw_mismatch - variant_tolerance - stabilizer_relief) ** 1.25
        score -= mismatch_penalty

        load_gap = abs(get(0, 1) - get(1, 2))
        gate01 = 1.35 if variant in (1, 3) else 0.75
        score -= gate01 * 2.3 * _soft_hinge(load_gap - 0.95) ** 1.35

        # Weakest-link effect. One bad critical slot drags the case down.
        crit = np.array([q0, q1, q2, q3], dtype=float)
        min_crit = float(np.min(crit))
        score -= 8.5 * _soft_hinge(-0.75 - min_crit) ** 1.45
        score -= 1.1 * np.sum(np.maximum(0.0, -1.2 - crit))

        # Missing optional rows are only bad when the case looks extended-like.
        if variant == 2 and 4 not in case_rows:
            score -= 5.2 + 0.9 * max(0.0, variant_signal)
        if variant == 3 and 7 not in case_rows and raw_mismatch > 1.0:
            score -= 2.2

        # Mild auxiliary effect, deliberately weaker than main structure.
        if 6 in case_rows:
            score += 0.6 * np.tanh(get(6, 0)) - 0.5 * (variant == 1) * abs(get(6, 2) - get(0, 2))
        if 7 in case_rows:
            score += 0.9 * np.tanh(get(7, 1) - 0.4 * get(3, 1))

        score += rng.normal(0.0, 1.0)

        # Sort rows by slot in generated data. Agents must still use case_offsets.
        for s in sorted(case_rows):
            rows.append(case_rows[s])
            slots.append(s)
        offsets.append(len(rows))
        y[i] = score

        slice_flags["mismatch"][i] = bool(mismatch or raw_mismatch > 1.35)
        slice_flags["bottleneck"][i] = bool(min_crit < -1.35 or bottleneck)
        slice_flags["rare_slot"][i] = 5 in case_rows
        slice_flags["missing_optional"][i] = bool(variant == 2 and 4 not in case_rows)
        slice_flags["variant_shift"][i] = variant in (2, 3)
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

reference
"""Canonical self-contained reference solution for component_fit_regression v0.1e.

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
    # v0.1 generated data has one row per slot, but this is robust to duplicates.
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
    return np.hstack(feats)


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

solve
"""Starter solution for component_fit_regression.

Agents should replace the simple baseline below.
"""
from __future__ import annotations

import numpy as np


def fit_component_model(train_part_X, train_part_slot, train_case_offsets, train_y):
    """Fit a model from packed component rows to one score per case."""
    return {"mean": float(np.mean(train_y))}


def predict_component_score(part_X, part_slot, case_offsets, params):
    """Return one prediction per case."""
    n_cases = len(case_offsets) - 1
    return np.full(n_cases, float(params.get("mean", 0.0)), dtype=float)



