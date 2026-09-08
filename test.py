from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np
from generator import SLICE_NAMES, generate_hidden

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))

# Reward anchors intentionally leave room for valid-but-weak modelling attempts.
# BAD is the lower end of the graded band, not a hard invalidity threshold.
OVERALL_GOOD = 0.1365
OVERALL_BAD = 0.1650
OVERALL_CATASTROPHIC = 0.25
SCORE_POWER = 1.5
SLICE_GOOD = np.array([0.1445, 0.1275, 0.1155, 0.1365, 0.1295, 0.1505, 0.1265, 0.1215], float)
SLICE_BAD = np.array([0.1680, 0.1540, 0.1450, 0.1615, 0.1575, 0.1770, 0.1515, 0.1540], float)


def _load_train(app):
    d = np.load(app / "train_data.npz")
    return (
        d["train_time"], d["train_market"], d["train_context_X"],
        d["train_candidate_pred"], d["train_y"],
    )


def _load_module(path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solve module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_process(solve_path, payload):
    mod = _load_module(solve_path)
    params = mod.fit_forecast_policy(*payload["train"])
    return [np.asarray(mod.predict_forecast(*x, params), float) for x in payload["inputs"]]


def _run_candidate(solve_path, payload):
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)
    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get("preds") if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, float) for p in preds]


def _log_rmse(y, p):
    y = np.log1p(np.maximum(np.asarray(y, float), 0.0))
    p = np.log1p(np.maximum(np.asarray(p, float), 0.0))
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _score(v, good, bad):
    if not math.isfinite(v) or bad <= good:
        return 0.0
    z = float(np.clip((bad - v) / (bad - good), 0.0, 1.0))
    return float(z ** SCORE_POWER)


def _validate(p, n):
    if p.shape != (n,):
        raise ValueError(f"prediction shape must be ({n},), got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    if np.any(p < -1e-8):
        raise ValueError("predictions must be non-negative")


def _metrics(y, p, slice_mask):
    overall = _log_rmse(y, p)
    overall_score = _score(overall, OVERALL_GOOD, OVERALL_BAD)
    per = {}
    scores = []
    for i, name in enumerate(SLICE_NAMES):
        m = np.asarray(slice_mask[i], bool)
        if not np.any(m):
            continue
        r = _log_rmse(y[m], p[m])
        s = _score(r, float(SLICE_GOOD[i]), float(SLICE_BAD[i]))
        per[str(name)] = {"log_rmse": r, "score": s, "n": int(m.sum())}
        scores.append(s)
    slice_mean = float(np.mean(scores)) if scores else 0.0
    reward = float(0.35 * overall_score + 0.65 * slice_mean)

    # Reserve hard zero for grossly broken predictions. Competent but weak
    # policies remain inside the continuous reward band.
    if overall >= OVERALL_CATASTROPHIC:
        reward = 0.0
    return {
        "reward": reward,
        "overall_log_rmse": overall,
        "overall_score": overall_score,
        "slice_mean": slice_mean,
        "per_slice": per,
    }


def evaluate(solve_path=None, app_dir=None, log_dir=None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"
    train = _load_train(app)
    h = generate_hidden()
    x = (h["time"], h["market"], h["context_X"], h["candidate_pred"])
    payload = {"train": train, "inputs": [x, x]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")
    p0, p1 = preds
    _validate(p0, len(h["y"]))
    _validate(p1, len(h["y"]))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")
    metrics = _metrics(h["y"], p0, h["slice_mask"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    r, m = evaluate()
    print(json.dumps({"reward": r, "overall_log_rmse": m["overall_log_rmse"]}, sort_keys=True))
