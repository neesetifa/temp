"""Canonical reference solution for component_fit_regression v0.1d.

This top-level file is intentionally named reference_solution.py and exposes
exactly the same public functions/signatures as app/solve.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from private.features import reference_features

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
