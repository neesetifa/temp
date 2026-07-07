from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

from simulator import make_dataset, make_rollout, inventory, expected_inventory_change

THRESHOLDS = {
    "mix_one_step_mse": 3.0e-5,
    "rare_one_step_mse": 3.0e-5,
    "boundary_one_step_mse": 1.0e-5,
    "low_mass_one_step_mse": 2.0e-5,
    "shift_one_step_mse": 2.0e-5,
    "mix_rollout_mse": 2.5e-4,
    "low_mass_rollout_mse": 1.5e-4,
    "max_inventory_error": 1e-8,
    "min_prediction": -1e-10,
    "predict_repeat_absdiff": 1e-12,
    "rollout_repeat_absdiff": 1e-12,
    "batch_permutation_absdiff": 1e-12,
}

ONE_STEP_CASES = [
    ("mix", 101, 1200, "mix"),
    ("rare", 102, 800, "rare"),
    ("boundary", 103, 600, "boundary"),
    ("low_mass", 104, 800, "low"),
    ("shift", 105, 800, "shift"),
]
ROLL_CASES = [
    ("mix", 201, "mix"),
    ("low_mass", 202, "low"),
]


def load_solution(path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solve"] = mod
    spec.loader.exec_module(mod)
    return mod


def inventory_error(before, after, controls):
    return np.max(np.abs(inventory(after) - inventory(before) - expected_inventory_change(controls)))


def _max_absdiff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def evaluate_solution(solve_module):
    train_path = pathlib.Path(__file__).parents[1] / "app" / "train_data.npz"
    train = np.load(train_path)
    params = solve_module.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])

    metrics = {}
    min_pred = 1e9
    max_inv = 0.0

    # One-step hidden slices.  These are generated from private seeds and include
    # common, rare-control, gate-boundary, low-mass, and shifted-state regimes.
    saved_mix_inputs = None
    for label, seed, n, kind in ONE_STEP_CASES:
        states, controls, truth = make_dataset(seed, n, kind)
        states_before = states.copy()
        controls_before = controls.copy()
        pred = np.asarray(solve_module.predict_next(states, controls, params), dtype=float)
        if pred.shape != truth.shape:
            raise AssertionError(f"predict_next returned shape {pred.shape}, expected {truth.shape}")
        if not np.allclose(states, states_before) or not np.allclose(controls, controls_before):
            raise AssertionError("predict_next must not mutate its input arrays")
        if not np.all(np.isfinite(pred)):
            raise AssertionError("predict_next returned non-finite values")
        metrics[f"{label}_one_step_mse"] = float(np.mean((pred - truth) ** 2))
        min_pred = min(min_pred, float(pred.min()))
        max_inv = max(max_inv, float(inventory_error(states, pred, controls)))
        if label == "mix":
            saved_mix_inputs = (states[:64].copy(), controls[:64].copy(), pred[:64].copy())

    # Determinism and row-permutation invariance for prediction.  These catch
    # solutions that cache row order, consume random numbers at inference time,
    # or make prediction depend on batch composition.
    if saved_mix_inputs is not None:
        states, controls, first_pred = saved_mix_inputs
        second_pred = np.asarray(solve_module.predict_next(states.copy(), controls.copy(), params), dtype=float)
        metrics["predict_repeat_absdiff"] = _max_absdiff(first_pred, second_pred)
        perm = np.array([7, 1, 15, 0, 23, 3, 31, 9, 2, 17, 5, 29, 11, 13, 19, 21])
        pred_perm = np.asarray(solve_module.predict_next(states[perm].copy(), controls[perm].copy(), params), dtype=float)
        metrics["batch_permutation_absdiff"] = _max_absdiff(first_pred[perm], pred_perm)

    # Rollout slices.  These evaluate accumulated drift, not just isolated one-step fit.
    saved_roll = None
    for label, seed, kind in ROLL_CASES:
        init, controls, truth = make_rollout(seed, 128, 40, kind)
        init_before = init.copy()
        controls_before = controls.copy()
        pred = np.asarray(solve_module.rollout(init, controls, params), dtype=float)
        if pred.shape != truth.shape:
            raise AssertionError(f"rollout returned shape {pred.shape}, expected {truth.shape}")
        if not np.allclose(pred[:, 0], init_before):
            raise AssertionError("rollout must include the initial state at index 0")
        if not np.allclose(init, init_before) or not np.allclose(controls, controls_before):
            raise AssertionError("rollout must not mutate its input arrays")
        if not np.all(np.isfinite(pred)):
            raise AssertionError("rollout returned non-finite values")
        metrics[f"{label}_rollout_mse"] = float(np.mean((pred - truth) ** 2))
        min_pred = min(min_pred, float(pred.min()))
        for t in range(controls.shape[1]):
            max_inv = max(max_inv, float(inventory_error(pred[:, t], pred[:, t + 1], controls[:, t])))
        if label == "mix":
            saved_roll = (init[:16].copy(), controls[:16, :12].copy(), pred[:16, :13].copy())

    if saved_roll is not None:
        init, controls, first_roll = saved_roll
        second_roll = np.asarray(solve_module.rollout(init.copy(), controls.copy(), params), dtype=float)
        metrics["rollout_repeat_absdiff"] = _max_absdiff(first_roll, second_roll)

    metrics["max_inventory_error"] = max_inv
    metrics["min_prediction"] = min_pred

    passed = all(
        (metrics[name] < limit if name != "min_prediction" else metrics[name] >= limit)
        for name, limit in THRESHOLDS.items()
    )
    return {"passed": passed, "metrics": metrics, "thresholds": THRESHOLDS}


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("reference_solution.py")
    print(json.dumps(evaluate_solution(load_solution(path)), indent=2, sort_keys=True))

