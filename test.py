from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from generator import build

OUTCOME_GOOD = 0.105
OUTCOME_BAD = 0.280
WORST_GOOD = 0.290
WORST_BAD = 0.550
OUTCOME_POWER = 1.5
WORST_POWER = 1.2
ESS_GOOD = 0.85
ESS_BAD = 0.45
TOP1_GOOD = 0.026
TOP1_BAD = 0.060


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    c = np.cumsum(w)
    return float(np.interp(q * c[-1], c, x))


def _estimands(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    y = df["mdvis"].to_numpy(float)
    w = np.asarray(w, float)
    w = w / w.sum()
    out = [
        np.sum(w * y),
        np.sum(w * (y == 0)),
        np.sum(w * (y >= 5)),
        np.sum(w * (y >= 10)),
        _weighted_quantile(y, w, 0.75),
        _weighted_quantile(y, w, 0.90),
    ]
    groups = [
        df["physlm"].to_numpy(float) > 0.5,
        (df["hlthf"].to_numpy(float) > 0.5) | (df["hlthp"].to_numpy(float) > 0.5),
        df["idp"].to_numpy(float) > 0.5,
        df["disea"].to_numpy(float) >= 14,
        df["lncoins"].to_numpy(float) >= 3.5,
    ]
    for mask in groups:
        ww = w[mask]
        yy = y[mask]
        out.append(np.sum(ww * yy) / ww.sum())
        out.append(np.sum(ww * (yy >= 5)) / ww.sum())
    return np.asarray(out, float)


SCALES = np.asarray([
    1.0, 0.08, 0.08, 0.04, 1.0, 1.5,
    1.5, 0.10, 1.5, 0.10, 1.5, 0.10, 1.5, 0.10, 1.5, 0.10,
])


def _invalid(reason: str) -> dict:
    return {"reward": 0.0, "invalid": reason}


def score_weights(weights_path: str | Path) -> dict:
    here = Path(__file__).resolve().parent
    analysis, _, hidden_target = build(here / "source_randhie.csv")

    p = Path(weights_path)
    if not p.is_file():
        return _invalid("missing /output/weights.npy")
    try:
        w = np.load(p, allow_pickle=False)
    except Exception as exc:
        return _invalid(f"unable to load weights: {type(exc).__name__}")

    if w.ndim != 1 or len(w) != len(analysis):
        return _invalid("wrong shape")
    if not np.isfinite(w).all() or np.any(w < 0) or float(w.sum()) <= 0:
        return _invalid("weights must be finite, nonnegative, and nonzero")

    w = w.astype(float)
    w /= w.mean()
    ess = float(w.sum() ** 2 / np.dot(w, w) / len(w))
    top_n = max(1, int(0.01 * len(w)))
    top1_mass = float(np.sort(w)[-top_n:].sum() / w.sum())
    if ess < 0.15 or top1_mass > 0.20 or float(w.max()) > 100.0:
        return _invalid("catastrophically concentrated weights")

    truth = _estimands(hidden_target, np.ones(len(hidden_target)))
    pred = _estimands(analysis, w)
    z = np.abs((pred - truth) / SCALES)
    rmse = float(np.sqrt(np.mean(z ** 2)))
    worst = float(np.max(z))

    outcome_score = np.clip((OUTCOME_BAD - rmse) / (OUTCOME_BAD - OUTCOME_GOOD), 0.0, 1.0) ** OUTCOME_POWER
    worst_score = np.clip((WORST_BAD - worst) / (WORST_BAD - WORST_GOOD), 0.0, 1.0) ** WORST_POWER
    ess_score = float(np.clip((ess - ESS_BAD) / (ESS_GOOD - ESS_BAD), 0.0, 1.0))
    tail_score = float(np.clip((TOP1_BAD - top1_mass) / (TOP1_BAD - TOP1_GOOD), 0.0, 1.0))
    stability = 0.60 * ess_score + 0.40 * tail_score
    reward = float(0.65 * outcome_score + 0.25 * worst_score + 0.10 * stability)

    return {
        "reward": reward,
        "normalized_estimand_rmse": rmse,
        "worst_normalized_error": worst,
        "ess_fraction": ess,
        "top1_weight_mass": top1_mass,
    }


def write_result(result: dict, logs_dir: str | Path = "/logs/verifier") -> None:
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{float(result.get('reward', 0.0)):.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", type=Path, default=Path("/output/weights.npy"))
    ap.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = ap.parse_args()
    result = score_weights(args.weights)
    write_result(result, args.logs_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
