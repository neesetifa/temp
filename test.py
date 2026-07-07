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
