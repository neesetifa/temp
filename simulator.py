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
