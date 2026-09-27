"""Verifier scoring for inspection_event_risk v0.1c."""
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

ANCHORS = {'overall': {'bad': 1.2036102030282088, 'good': 0.23684307194673465},
 'slices': {'followup_selection_bias': {'bad': 1.523794122141569, 'good': 0.24626672492896812},
            'high_count_low_risk': {'bad': 1.4729528548157609, 'good': 0.2787015168515005},
            'long_gap_decay': {'bad': 1.1016330363856617, 'good': 0.28237374260326487},
            'low_count_high_risk': {'bad': 1.3788115191978827, 'good': 0.2768361457471076},
            'mitigation_recovery': {'bad': 1.235509696828204, 'good': 0.22520948753799833},
            'mixed_signal_history': {'bad': 1.501170566524282, 'good': 0.2517573670026758},
            'rare_code_category_interaction': {'bad': 1.1867874057250845,
                                               'good': 0.26088899763482254},
            'recent_noisy_outlier': {'bad': 1.3195804885533624, 'good': 0.23488537598324397},
            'repeat_symptom_escalation': {'bad': 2.009668190874514, 'good': 0.26951647803336587},
            'short_history_fallback': {'bad': 1.0370732167874437, 'good': 0.23074273836517645},
            'stable_history': {'bad': 1.1649710201592351, 'good': 0.23844747078991402}}}
SLICE_WEIGHT = 0.45
SCORE_POWER = 1.20
SLICE_WEIGHT_OVERRIDES = {"long_gap_decay": 0.35}


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
        w *= SLICE_WEIGHT_OVERRIDES.get(name, 1.0)
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
