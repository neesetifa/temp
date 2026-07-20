
`app/solve.py` has the three stubs to fill in:

```python
fit_model(train_states, train_controls, train_next_states)
predict_next(states, controls, params)
rollout(init_states, controls, params)
```

The data comes from a six-compartment simulator with three controls. The six state columns are not on the same accounting scale. `forward_process.py` has the inventory helper and the known control-driven inventory change.

Fit from the transition rows in `train_data.npz`. The rows now include normal operation plus the less common low-mass, shifted-control, and partly-on transfer cases, so the behavior should be learnable from the data rather than guessed from a formula. Public eval rows are only for the public tests.

A good step prediction needs to keep inventory accounting tight and stay nonnegative. Rollout matters too: a one-step fit that only patches the inventory at the end can still drift after many steps.

Use the passed arrays only. No hidden files, network, outside data, or fitting on public eval answers.

"""Visible helpers for the compartment-transfer surrogate task.

The simulator moves a calibrated inventory among six compartments while a
small control-dependent amount may enter or leave the system.  This file gives
only the public accounting rules and array conventions; it does not contain the
simulator dynamics.
"""
from __future__ import annotations

import numpy as np

NUM_COMPARTMENTS = 6
NUM_CONTROLS = 3

# Public calibration factors for the conserved inventory.  A unit in one
# compartment does not necessarily correspond to the same amount of inventory
# as a unit in another compartment.
INVENTORY_WEIGHTS = np.array([1.00, 1.23, 0.82, 1.41, 0.73, 1.56], dtype=float)


def as_state_array(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != NUM_COMPARTMENTS:
        raise ValueError(f"states must have shape (n, {NUM_COMPARTMENTS})")
    return arr


def as_control_array(controls: np.ndarray) -> np.ndarray:
    arr = np.asarray(controls, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != NUM_CONTROLS:
        raise ValueError(f"controls must have shape (n, {NUM_CONTROLS})")
    return arr


def inventory(states: np.ndarray) -> np.ndarray:
    """Return calibrated inventory for each state row."""
    states = as_state_array(states)
    return states @ INVENTORY_WEIGHTS


def expected_inventory_change(controls: np.ndarray) -> np.ndarray:
    """Known inventory entering/leaving during one step.

    The controls include a small external loading/unloading term.  Internal
    transfers should account for the rest of the state change.
    """
    controls = as_control_array(controls)
    c0, c1, c2 = controls[:, 0], controls[:, 1], controls[:, 2]
    return 0.018 * np.tanh(0.9 * c0 - 0.55 * c1 + 0.65 * c2) + 0.006 * c0 * c2 - 0.003 * c1**2


def inventory_residual(before: np.ndarray, after: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Signed inventory accounting residual for predicted one-step transitions."""
    before = as_state_array(before)
    after = as_state_array(after)
    controls = as_control_array(controls)
    if len(before) != len(after) or len(before) != len(controls):
        raise ValueError("before, after, and controls must have the same row count")
    return inventory(after) - inventory(before) - expected_inventory_change(controls)


def max_inventory_error(before: np.ndarray, after: np.ndarray, controls: np.ndarray) -> float:
    return float(np.max(np.abs(inventory_residual(before, after, controls))))



def stack_state_control(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return raw state and control columns side by side.

    This is only a shape convenience for small experiments; it is not a
    simulator model.
    """
    states = as_state_array(states)
    controls = as_control_array(controls)
    return np.concatenate([states, controls], axis=1)


def rollout_with_predictor(predict_next, init_states: np.ndarray, controls: np.ndarray, params):
    """Apply a predict_next function over batched control sequences.

    init_states has shape (batch, 6), controls has shape (batch, horizon, 3),
    and the return value has shape (batch, horizon + 1, 6).
    """
    init_states = as_state_array(init_states)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 3 or controls.shape[0] != init_states.shape[0] or controls.shape[2] != NUM_CONTROLS:
        raise ValueError("controls must have shape (batch, horizon, 3)")
    out = np.empty((init_states.shape[0], controls.shape[1] + 1, NUM_COMPARTMENTS), dtype=float)
    out[:, 0, :] = init_states
    current = init_states.copy()
    for t in range(controls.shape[1]):
        current = np.asarray(predict_next(current, controls[:, t, :], params), dtype=float)
        current = as_state_array(current)
        out[:, t + 1, :] = current
    return out


from __future__ import annotations

import numpy as np

from forward_process import max_inventory_error
import solve


def test_public_sanity():
    train = np.load("train_data.npz")
    public = np.load("public_eval.npz")
    params = solve.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])
    pred = solve.predict_next(public["states"], public["controls"], params)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == public["next_states"].shape
    assert np.all(np.isfinite(pred))
    assert pred.min() >= -1e-8
    assert max_inventory_error(public["states"], pred, public["controls"]) < 1e-6
    mse = np.mean((pred - public["next_states"]) ** 2)
    assert mse < 5e-4


def test_public_rollout_shape_and_accounting():
    train = np.load("train_data.npz")
    public = np.load("public_eval.npz")
    params = solve.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])
    init = public["states"][:8]
    controls = np.stack([public["controls"][:8], public["controls"][8:16], public["controls"][16:24]], axis=1)
    out = solve.rollout(init, controls, params)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 4, 6)
    assert np.allclose(out[:, 0], init)
    for t in range(controls.shape[1]):
        assert max_inventory_error(out[:, t], out[:, t + 1], controls[:, t]) < 1e-6


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
    "mix_rollout_mse": 3.2e-4,
    "low_mass_rollout_mse": 2.0e-4,
    "max_inventory_error": 1e-8,
    "min_prediction": -1e-10,
}


def load_solution(path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solve"] = mod
    spec.loader.exec_module(mod)
    return mod


def inventory_error(before, after, controls):
    return np.max(np.abs(inventory(after) - inventory(before) - expected_inventory_change(controls)))


def evaluate_solution(solve_module):
    train = np.load(pathlib.Path(__file__).parents[1] / "app" / "train_data.npz")
    params = solve_module.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])
    metrics = {}
    min_pred = 1e9
    max_inv = 0.0
    for key, seed, n in [
        ("mix", 101, 1200),
        ("rare", 102, 800),
        ("boundary", 103, 600),
        ("low_mass", 104, 800),
        ("shift", 105, 800),
    ]:
        kind = "low" if key == "low_mass" else key
        states, controls, truth = make_dataset(seed, n, kind)
        pred = np.asarray(solve_module.predict_next(states, controls, params), dtype=float)
        if pred.shape != truth.shape:
            raise AssertionError(f"predict_next returned shape {pred.shape}, expected {truth.shape}")
        metrics[f"{key}_one_step_mse"] = float(np.mean((pred - truth) ** 2))
        min_pred = min(min_pred, float(pred.min()))
        max_inv = max(max_inv, float(inventory_error(states, pred, controls)))
    for key, seed in [("mix", 201), ("low_mass", 202)]:
        kind = "low" if key == "low_mass" else key
        init, controls, truth = make_rollout(seed, 128, 40, kind)
        pred = np.asarray(solve_module.rollout(init, controls, params), dtype=float)
        if pred.shape != truth.shape:
            raise AssertionError(f"rollout returned shape {pred.shape}, expected {truth.shape}")
        metrics[f"{key}_rollout_mse"] = float(np.mean((pred - truth) ** 2))
        min_pred = min(min_pred, float(pred.min()))
        for t in range(controls.shape[1]):
            max_inv = max(max_inv, float(inventory_error(pred[:, t], pred[:, t + 1], controls[:, t])))
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

from __future__ import annotations

import numpy as np

DIM = 6
WEIGHTS = np.array([1.00, 1.23, 0.82, 1.41, 0.73, 1.56], dtype=float)
EDGES = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 0), (0, 3), (3, 1), (2, 5), (4, 0)]
BASE = np.array([0.030, -0.020, 0.025, -0.015, 0.018, 0.022, -0.026, 0.017, -0.021])
C0 = np.array([0.018, -0.012, 0.011, 0.014, -0.010, 0.016, -0.013, 0.009, -0.011])
C1 = np.array([-0.011, 0.017, -0.014, 0.010, 0.013, -0.009, 0.012, -0.016, 0.015])
C2 = np.array([0.012, 0.010, -0.009, 0.015, -0.012, 0.011, 0.013, -0.010, 0.014])
BETA = np.array([1.1, -0.8, 0.9, -1.0, 0.7, -0.6, 0.75, -0.9, 0.65])
GB = np.array([-0.15, 0.05, 0.2, -0.25, 0.0, 0.15, -0.1, 0.25, -0.2])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def inventory(x):
    return np.asarray(x) @ WEIGHTS


def expected_inventory_change(u):
    u = np.asarray(u)
    return 0.018 * np.tanh(0.9 * u[..., 0] - 0.55 * u[..., 1] + 0.65 * u[..., 2]) + 0.006 * u[..., 0] * u[..., 2] - 0.003 * u[..., 1] ** 2


def external_distribution(x, u):
    u = np.asarray(u)
    x = np.asarray(x)
    logits = np.stack(
        [
            0.4 * u[..., 0] - 0.2 * u[..., 1] + 0.1 * (x[..., 0] - x[..., 3]),
            -0.3 * u[..., 0] + 0.5 * u[..., 2] + 0.1 * (x[..., 1] - x[..., 4]),
            0.2 * u[..., 1] - 0.4 * u[..., 2] + 0.08 * (x[..., 2] - x[..., 5]),
            -0.1 * u[..., 0] + 0.3 * u[..., 1] - 0.2 * u[..., 2],
            0.3 * u[..., 0] + 0.1 * u[..., 1] + 0.2 * u[..., 2],
            -0.2 * u[..., 0] - 0.1 * u[..., 1] + 0.4 * u[..., 2],
        ],
        axis=-1,
    )
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = exp / exp.sum(axis=-1, keepdims=True)
    return expected_inventory_change(u)[..., None] * p / WEIGHTS


def true_step(x, u):
    x = np.asarray(x, dtype=float).copy()
    u = np.asarray(u, dtype=float)
    scalar = False
    if x.ndim == 1:
        x = x[None, :]
        u = u[None, :]
        scalar = True
    inv_amt = x * WEIGHTS
    delta = np.zeros_like(x)
    for k, (i, j) in enumerate(EDGES):
        local = inv_amt[:, i] - inv_amt[:, j]
        gate = sigmoid(5.0 * (0.75 * local + 0.55 * u[:, 0] * BETA[k] - 0.35 * u[:, 1] + 0.25 * u[:, 2] + GB[k]))
        drive = np.tanh(1.4 * local + 24.0 * (0.8 * u[:, 0] * C0[k] + 0.6 * u[:, 1] * C1[k] + 0.4 * u[:, 2] * C2[k]))
        q = BASE[k] * local + (C0[k] * u[:, 0] + C1[k] * u[:, 1] + C2[k] * u[:, 2]) * gate + 0.010 * drive * gate
        q = np.clip(q, -0.06, 0.06)
        delta[:, i] -= q / WEIGHTS[i]
        delta[:, j] += q / WEIGHTS[j]
    delta += external_distribution(x, u)
    y = x + delta
    target = inventory(x) + expected_inventory_change(u)
    for n in range(y.shape[0]):
        if y[n].min() < 1e-8:
            y[n] = np.maximum(y[n], 1e-8)
        diff = target[n] - inventory(y[n])
        y[n] += diff / WEIGHTS.sum()
        if y[n].min() < 1e-8:
            y[n] = np.maximum(y[n], 1e-8)
            y[n] *= target[n] / inventory(y[n])
    return y[0] if scalar else y


def sample_states(rng, n, kind="mix"):
    if kind == "low":
        alpha = np.array([0.25, 0.35, 0.3, 0.5, 0.25, 0.45])
        target = rng.uniform(0.9, 1.4, size=n)
    elif kind == "shift":
        alpha = np.array([2.0, 0.5, 1.8, 0.4, 1.6, 0.5])
        target = rng.uniform(1.1, 1.8, size=n)
    else:
        alpha = np.array([1.4, 1.0, 1.2, 0.9, 1.1, 0.8])
        target = rng.uniform(0.8, 1.6, size=n)
    z = rng.gamma(alpha, 1.0, size=(n, DIM))
    z = z / z.sum(axis=1, keepdims=True)
    return z * (target[:, None] / inventory(z)[:, None])


def sample_controls(rng, n, kind="mix"):
    if kind == "rare":
        return np.column_stack([rng.uniform(0.55, 1.0, n), rng.uniform(-1.0, -0.35, n), rng.uniform(0.3, 1.0, n)])
    if kind == "boundary":
        return rng.uniform(-0.8, 0.8, size=(n, 3))
    if kind == "low":
        return np.column_stack([rng.uniform(-1.0, 0.2, n), rng.uniform(0.2, 1.0, n), rng.uniform(-0.5, 0.8, n)])
    if kind == "shift":
        return np.column_stack([rng.uniform(0.2, 1.0, n), rng.uniform(-0.8, 0.4, n), rng.uniform(0.6, 1.0, n)])
    return rng.uniform(-1.0, 1.0, size=(n, 3))


def gate_activity(x, u):
    inv_amt = x * WEIGHTS
    vals = []
    for k, (i, j) in enumerate(EDGES):
        local = inv_amt[:, i] - inv_amt[:, j]
        vals.append(0.75 * local + 0.55 * u[:, 0] * BETA[k] - 0.35 * u[:, 1] + 0.25 * u[:, 2] + GB[k])
    return np.stack(vals, axis=1)


def make_dataset(seed, n, kind="mix"):
    rng = np.random.default_rng(seed)
    if kind == "boundary":
        xs, us = [], []
        while len(xs) < n:
            m = max(1000, n * 5)
            x = sample_states(rng, m, "mix")
            u = sample_controls(rng, m, "boundary")
            g = gate_activity(x, u)
            mask = np.abs(g).min(axis=1) < 0.035
            for xi, ui in zip(x[mask], u[mask]):
                xs.append(xi)
                us.append(ui)
                if len(xs) >= n:
                    break
        x = np.array(xs)
        u = np.array(us)
    elif kind == "rare":
        x = sample_states(rng, n, "shift")
        u = sample_controls(rng, n, "rare")
    elif kind == "low":
        x = sample_states(rng, n, "low")
        u = sample_controls(rng, n, "low")
    elif kind == "shift":
        x = sample_states(rng, n, "shift")
        u = sample_controls(rng, n, "shift")
    else:
        x = sample_states(rng, n, "mix")
        u = sample_controls(rng, n, "mix")
    y = true_step(x, u)
    return x, u, y


def make_rollout(seed, batch, horizon, kind="mix"):
    rng = np.random.default_rng(seed)
    x0 = sample_states(rng, batch, "low" if kind == "low" else ("shift" if kind in ("rare", "shift") else "mix"))
    controls = []
    for _ in range(horizon):
        controls.append(sample_controls(rng, batch, kind if kind in ("low", "rare", "shift") else "mix"))
    U = np.stack(controls, axis=1)
    X = np.empty((batch, horizon + 1, DIM), dtype=float)
    X[:, 0] = x0
    for t in range(horizon):
        X[:, t + 1] = true_step(X[:, t], U[:, t])
    return x0, U, X


from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

# This reference intentionally does not use the hidden edge list.  It fits a
# structured transfer basis over all candidate compartment pairs, then enforces
# the public inventory accounting rule.

DIM = 6
WEIGHTS = np.array([1.00, 1.23, 0.82, 1.41, 0.73, 1.56], dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def inventory(x):
    return np.asarray(x) @ WEIGHTS


def expected_inventory_change(u):
    u = np.asarray(u)
    return 0.018 * np.tanh(0.9 * u[..., 0] - 0.55 * u[..., 1] + 0.65 * u[..., 2]) + 0.006 * u[..., 0] * u[..., 2] - 0.003 * u[..., 1] ** 2


def adjust_inventory_nonnegative(y, x, u):
    y = np.maximum(np.asarray(y, dtype=float), 1e-10).copy()
    target = inventory(x) + expected_inventory_change(u)
    diff = target - inventory(y)
    y += diff[:, None] / WEIGHTS.sum()
    bad = y.min(axis=1) < 0
    if np.any(bad):
        y[bad] = np.maximum(y[bad], 1e-10)
        y[bad] *= (target[bad] / inventory(y[bad]))[:, None]
    return y


def pair_phi(x, u, i, j):
    inv = x * WEIGHTS
    local = inv[:, i] - inv[:, j]
    pieces = [local, u[:, 0], u[:, 1], u[:, 2], local * u[:, 0], local * u[:, 1], local * u[:, 2]]
    for offset in (-0.3, 0.0, 0.3):
        gate = sigmoid(5.0 * (0.7 * local + 0.35 * u[:, 0] - 0.25 * u[:, 1] + 0.2 * u[:, 2] + offset))
        pieces.extend([gate, gate * local, gate * u[:, 0], gate * u[:, 1], gate * u[:, 2]])
    return np.stack(pieces, axis=1)


def design_matrix(x, u):
    columns = []
    for i in range(DIM):
        for j in range(i + 1, DIM):
            phi = pair_phi(x, u, i, j)
            basis = np.zeros(DIM)
            basis[i] -= 1.0 / WEIGHTS[i]
            basis[j] += 1.0 / WEIGHTS[j]
            for k in range(phi.shape[1]):
                columns.append((phi[:, k, None] * basis[None, :]).reshape(-1))
    dI = expected_inventory_change(u)
    ext = np.stack([dI, dI * u[:, 0], dI * u[:, 1], dI * u[:, 2]], axis=1)
    for m in range(DIM):
        basis = np.zeros(DIM)
        basis[m] = 1.0 / WEIGHTS[m]
        for k in range(ext.shape[1]):
            columns.append((ext[:, k, None] * basis[None, :]).reshape(-1))
    return np.stack(columns, axis=1)


def fit_model(train_states, train_controls, train_next_states):
    A = design_matrix(train_states, train_controls)
    target = (train_next_states - train_states).reshape(-1)
    model = Ridge(alpha=1e-5, fit_intercept=False)
    model.fit(A, target)
    return {"coef": model.coef_}


def predict_next(states, controls, params):
    A = design_matrix(np.asarray(states, dtype=float), np.asarray(controls, dtype=float))
    delta = A @ params["coef"]
    pred = np.asarray(states, dtype=float) + delta.reshape(-1, DIM)
    return adjust_inventory_nonnegative(pred, np.asarray(states, dtype=float), np.asarray(controls, dtype=float))


def rollout(init_states, controls, params):
    init_states = np.asarray(init_states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    out = np.empty((init_states.shape[0], controls.shape[1] + 1, DIM), dtype=float)
    out[:, 0] = init_states
    current = init_states.copy()
    for t in range(controls.shape[1]):
        current = predict_next(current, controls[:, t], params)
        out[:, t + 1] = current
    return out

