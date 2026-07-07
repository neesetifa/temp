from __future__ import annotations

import numpy as np

from forward_process import max_inventory_error
import solve


def _load_and_fit():
    train = np.load("train_data.npz")
    params = solve.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])
    return params


def test_public_training_file_and_one_step_sanity():
    train = np.load("train_data.npz")
    assert {"train_states", "train_controls", "train_next_states"}.issubset(set(train.files))
    assert train["train_states"].shape[1] == 6
    assert train["train_controls"].shape[1] == 3
    assert train["train_next_states"].shape == train["train_states"].shape

    public = np.load("public_eval.npz")
    params = solve.fit_model(train["train_states"], train["train_controls"], train["train_next_states"])
    states = public["states"][:120].copy()
    controls = public["controls"][:120].copy()
    expected_states = states.copy()
    expected_controls = controls.copy()
    pred = solve.predict_next(states, controls, params)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == public["next_states"][:120].shape
    assert np.all(np.isfinite(pred))
    assert np.allclose(states, expected_states)
    assert np.allclose(controls, expected_controls)
    assert pred.min() >= -1e-8
    assert max_inventory_error(states, pred, controls) < 1e-6
    assert np.mean((pred - public["next_states"][:120]) ** 2) < 5e-4


def test_public_rollout_contract_and_repeatability():
    public = np.load("public_eval.npz")
    params = _load_and_fit()

    init = public["states"][:8].copy()
    controls = np.stack(
        [public["controls"][:8], public["controls"][8:16], public["controls"][16:24], public["controls"][24:32]],
        axis=1,
    ).copy()
    init_before = init.copy()
    controls_before = controls.copy()

    out1 = solve.rollout(init, controls, params)
    out2 = solve.rollout(init, controls, params)
    assert isinstance(out1, np.ndarray)
    assert out1.shape == (8, 5, 6)
    assert np.allclose(out1, out2)
    assert np.allclose(out1[:, 0], init_before)
    assert np.allclose(init, init_before)
    assert np.allclose(controls, controls_before)

    for t in range(controls.shape[1]):
        assert max_inventory_error(out1[:, t], out1[:, t + 1], controls[:, t]) < 1e-6
