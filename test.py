reference
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def make_weights(app_dir: Path) -> np.ndarray:
    data = pd.read_csv(app_dir / "analysis_sample.csv")
    targets = pd.read_csv(app_dir / "frame_targets.csv")
    cols = targets["column"].tolist()

    X = data[cols].to_numpy(float)
    target = targets["target_mean"].to_numpy(float)
    target_se = targets["std_error"].to_numpy(float)

    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    tz = (target - mu) / sd
    sez = target_se / sd

    y = data["mdvis"].to_numpy(float)
    Ys = np.column_stack([
        y,
        (y == 0).astype(float),
        (y >= 5).astype(float),
        (y >= 10).astype(float),
    ])
    relevance = np.zeros(Z.shape[1], dtype=float)
    for j in range(Ys.shape[1]):
        yy = Ys[:, j]
        yy = (yy - yy.mean()) / (yy.std() + 1e-8)
        model = Ridge(alpha=20.0).fit(Z, yy)
        relevance += model.coef_ ** 2
    relevance = np.sqrt(relevance)

    penalty = 3.0 * (sez + 0.01) / (relevance + 0.05)
    gram = (Z.T @ Z) / len(Z)
    beta = np.linalg.solve(gram + np.diag(penalty), tz)
    w = 1.0 + Z @ beta
    w = np.clip(w, 0.05, 10.0)
    return w.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", type=Path, default=Path("/app"))
    ap.add_argument("--output", type=Path, default=Path("/output/weights.npy"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, make_weights(args.app_dir))


if __name__ == "__main__":
    main()

generator
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260914
TARGET_N = 6000
FRAME_N = 1200

RAW_COVARIATES = [
    "lncoins", "idp", "lpi", "fmde", "physlm", "disea",
    "hlthg", "hlthf", "hlthp",
]

TARGET_COLUMNS = [
    "lncoins", "lpi", "fmde", "disea", "idp", "physlim",
    "hlthg", "hlthf", "hlthp", "disease_ge_8", "disease_ge_14",
    "disease_ge_22", "coins_zero", "coins_high", "incentive_high",
    "fmde_high", "idp_x_high_disease", "phys_x_high_disease",
    "idp_x_fairpoor", "good_x_zero_coins", "high_coins_x_high_disease",
]


def add_calibration_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["physlim"] = (out["physlm"] > 0.5).astype(float)
    out["disease_ge_8"] = (out["disea"] >= 8).astype(float)
    out["disease_ge_14"] = (out["disea"] >= 14).astype(float)
    out["disease_ge_22"] = (out["disea"] >= 22).astype(float)
    out["coins_zero"] = (out["lncoins"] < 0.05).astype(float)
    out["coins_high"] = (out["lncoins"] >= 3.5).astype(float)
    out["incentive_high"] = (out["lpi"] >= 6.3).astype(float)
    out["fmde_high"] = (out["fmde"] >= 6.5).astype(float)
    out["idp_x_high_disease"] = out["idp"] * out["disease_ge_14"]
    out["phys_x_high_disease"] = out["physlim"] * out["disease_ge_14"]
    fairpoor = ((out["hlthf"] > 0.5) | (out["hlthp"] > 0.5)).astype(float)
    out["idp_x_fairpoor"] = out["idp"] * fairpoor
    out["good_x_zero_coins"] = out["hlthg"] * out["coins_zero"]
    out["high_coins_x_high_disease"] = out["coins_high"] * out["disease_ge_14"]
    return out


def _biased_keep(pool: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    def z(c: str) -> np.ndarray:
        x = pool[c].to_numpy(float)
        return (x - x.mean()) / (x.std() + 1e-9)

    dz, cz, lz, fz = [z(c) for c in ["disea", "lncoins", "lpi", "fmde"]]
    pl = pool["physlm"].to_numpy(float) > 0.5
    idp = pool["idp"].to_numpy(float) > 0.5
    hg = pool["hlthg"].to_numpy(float) > 0.5
    hf = pool["hlthf"].to_numpy(float) > 0.5
    hp = pool["hlthp"].to_numpy(float) > 0.5

    # Deterministic coverage/nonresponse mechanism. It uses only observed
    # covariates; outcomes never enter sample selection.
    logit = (
        -0.15 + 0.35 * idp - 0.12 * dz + 0.10 * cz + 0.10 * hg
        - 0.25 * hf - 0.80 * hp
        - 1.40 * (pl & (dz > 0.35))
        - 0.85 * (hf & idp)
        + 0.95 * (idp & (cz > 0.25))
        + 0.75 * ((fz > 0.50) & (~idp))
        + 0.65 * (hg & (lz > 0.30) & (dz < 0.15))
        - 0.75 * ((cz < -0.45) & (dz > 0.45))
        + 0.55 * ((dz < -0.50) & (~idp) & (cz > 0.10))
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(len(pool)) < (0.90 * p)


def build(source_csv: str | Path):
    source = pd.read_csv(source_csv)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(source))

    hidden_target = source.iloc[order[:TARGET_N]].reset_index(drop=True)
    frame = source.iloc[order[TARGET_N:TARGET_N + FRAME_N]].reset_index(drop=True)
    pool = source.iloc[order[TARGET_N + FRAME_N:]].reset_index(drop=True)
    keep = _biased_keep(pool, rng)
    analysis = pool.loc[keep].reset_index(drop=True)

    analysis = add_calibration_columns(analysis)
    frame = add_calibration_columns(frame)
    hidden_target = add_calibration_columns(hidden_target)

    frame_targets = []
    for col in TARGET_COLUMNS:
        values = frame[col].to_numpy(float)
        frame_targets.append({
            "column": col,
            "target_mean": float(values.mean()),
            "std_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        })
    frame_targets = pd.DataFrame(frame_targets)

    visible_cols = ["mdvis"] + RAW_COVARIATES + [c for c in TARGET_COLUMNS if c not in RAW_COVARIATES]
    analysis_visible = analysis[visible_cols].copy()
    return analysis_visible, frame_targets, hidden_target


def write_environment(source_csv: str | Path, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis, frame_targets, _ = build(source_csv)
    analysis.to_csv(data_dir / "analysis_sample.csv", index=False)
    frame_targets.to_csv(data_dir / "frame_targets.csv", index=False)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parent
    write_environment(here / "source_randhie.csv", root / "environment" / "app")

hidden
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
    ess_score = float(np.clip((ess - 0.45) / (0.85 - 0.45), 0.0, 1.0))
    tail_score = float(np.clip((0.060 - top1_mass) / (0.060 - 0.025), 0.0, 1.0))
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

test main
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd

from generator import build, TARGET_COLUMNS
from hidden_eval import score_weights, write_result

APP_DIR = Path(os.environ.get("TASK_APP_DIR", "/app"))
OUTPUT_DIR = Path(os.environ.get("TASK_OUTPUT_DIR", "/output"))
LOGS_DIR = Path(os.environ.get("TASK_LOGS_DIR", "/logs/verifier"))
HERE = Path(__file__).resolve().parent


def test_00_agent_output_is_scored():
    result = score_weights(OUTPUT_DIR / "weights.npy")
    write_result(result, LOGS_DIR)
    assert 0.0 <= float(result["reward"]) <= 1.0


def test_shipped_visible_data_match_generator():
    analysis, targets, _ = build(HERE / "source_randhie.csv")
    shipped_analysis = pd.read_csv(APP_DIR / "analysis_sample.csv")
    shipped_targets = pd.read_csv(APP_DIR / "frame_targets.csv")
    pd.testing.assert_frame_equal(analysis, shipped_analysis, check_dtype=False)
    pd.testing.assert_frame_equal(targets, shipped_targets, check_dtype=False)


def test_target_columns_exist():
    analysis = pd.read_csv(APP_DIR / "analysis_sample.csv")
    assert set(TARGET_COLUMNS).issubset(analysis.columns)

