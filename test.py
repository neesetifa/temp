# Build transport weights for the analysis sample

`/app/analysis_sample.csv` contains one row per observed record. `mdvis` is the observed number of outpatient physician visits. The remaining columns describe insurance terms, health status, and calibration indicators. The observed sample is not representative of the target analysis population.

`/app/frame_targets.csv` contains external calibration information. Each row gives a column name, an estimated target-population mean for that column, and the standard error of that estimate. The frame estimates have different sampling precision, so treat them as estimates rather than exact census totals.

Produce one nonnegative analysis weight for every row of `analysis_sample.csv` and save the vector to:

`/output/weights.npy`

The array must be one-dimensional, finite, and have the same length and row order as `analysis_sample.csv`. Weight scale is arbitrary; the evaluator normalizes the weights before use.

The submitted weights are used to estimate held-out physician-utilization summaries in a separate representative target panel. Evaluation considers overall utilization, zero/high-use rates, several health/plan subgroups, and weight stability. Extremely concentrated weights are not useful even when they match the supplied frame targets closely.

Use any analysis or calibration method available in the environment. Do not use external data or network access.


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

    # Estimate which margins are useful for the downstream physician-use
    # summaries from the analysis sample itself.  No hidden target outcomes are
    # used.  Multiple views of utilization reduce dependence on one scale.
    y = data["mdvis"].to_numpy(float)
    views = np.column_stack([
        y,
        (y == 0).astype(float),
        (y >= 5).astype(float),
        (y >= 10).astype(float),
    ])
    relevance = np.zeros(Z.shape[1], dtype=float)
    for j in range(views.shape[1]):
        yy = views[:, j]
        yy = (yy - yy.mean()) / (yy.std() + 1e-8)
        model = Ridge(alpha=20.0).fit(Z, yy)
        relevance += model.coef_ ** 2
    relevance = np.sqrt(relevance)

    # Trust differs by margin: noisy external estimates and weakly relevant
    # margins receive much stronger shrinkage than precise, useful margins.
    penalty = 32.0 * (sez + 0.01) ** 1.5 / (relevance + 0.03) ** 2
    gram = (Z.T @ Z) / len(Z)
    beta = np.linalg.solve(gram + np.diag(penalty), tz)
    w = 1.0 + Z @ beta
    return np.clip(w, 0.05, 10.0).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", type=Path, default=Path("/app"))
    ap.add_argument("--output", type=Path, default=Path("/output/weights.npy"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, make_weights(args.app_dir))


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

generator
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260914
TARGET_N = 5000
FRAME_RESERVOIR_N = 5000

RAW_COVARIATES = [
    "lncoins", "idp", "lpi", "fmde", "physlm", "disea",
    "hlthg", "hlthf", "hlthp",
]

BASE_TARGETS = [
    "lncoins", "lpi", "fmde", "disea", "idp", "physlim",
    "hlthg", "hlthf", "hlthp",
]
THRESHOLD_TARGETS = [
    "disease_ge_6", "disease_ge_10", "disease_ge_14", "disease_ge_18",
    "disease_ge_22", "disease_ge_28",
    "coins_zero", "coins_low", "coins_mid", "coins_high", "coins_vhigh",
    "incentive_low", "incentive_mid", "incentive_high",
    "fmde_low", "fmde_mid", "fmde_high",
]
INTERACTION_TARGETS = [
    "idp_x_high_disease", "idp_x_vhigh_disease", "phys_x_high_disease",
    "phys_x_fairpoor", "idp_x_fairpoor", "good_x_zero_coins",
    "high_coins_x_high_disease", "zero_coins_x_high_disease",
    "idp_x_zero_coins", "idp_x_high_coins", "phys_x_idp",
    "fairpoor_x_high_disease", "good_x_low_disease",
    "high_incentive_x_idp", "high_fmde_x_phys",
    "disea_x_idp", "disea_x_phys", "coins_x_idp",
]
TARGET_COLUMNS = BASE_TARGETS + THRESHOLD_TARGETS + INTERACTION_TARGETS

# The external frame statistics do not all have the same precision.  These
# sizes represent separate representative frame extracts used for different
# published margins.  They are fixed construction metadata; only the resulting
# target mean and standard error are shipped to the agent.
FRAME_SAMPLE_SIZE = {c: 450 for c in TARGET_COLUMNS}
for c in [
    "disea", "lpi", "fmde_mid", "physlim", "hlthg", "idp", "coins_mid",
    "coins_high", "incentive_low", "fmde_high", "good_x_zero_coins",
    "idp_x_high_coins",
]:
    FRAME_SAMPLE_SIZE[c] = 3500
for c in [
    "disease_ge_14", "disease_ge_18", "disease_ge_22",
    "phys_x_high_disease", "idp_x_high_disease", "coins_zero",
    "incentive_high", "fairpoor_x_high_disease",
    "high_coins_x_high_disease",
]:
    FRAME_SAMPLE_SIZE[c] = 1200
for c in [
    "fmde", "disea_x_phys", "disease_ge_10", "disease_ge_6",
    "incentive_mid", "high_incentive_x_idp", "disea_x_idp",
]:
    FRAME_SAMPLE_SIZE[c] = 120
for c in [
    "hlthf", "hlthp", "coins_low", "coins_vhigh", "fmde_low",
    "phys_x_fairpoor", "idp_x_fairpoor", "phys_x_idp",
]:
    FRAME_SAMPLE_SIZE[c] = 180


def add_calibration_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["physlim"] = (out["physlm"] > 0.5).astype(float)
    out["fairpoor"] = ((out["hlthf"] > 0.5) | (out["hlthp"] > 0.5)).astype(float)

    for threshold in [6, 10, 14, 18, 22, 28]:
        out[f"disease_ge_{threshold}"] = (out["disea"] >= threshold).astype(float)

    out["coins_zero"] = (out["lncoins"] < 0.05).astype(float)
    for threshold, name in [(1.2, "low"), (2.5, "mid"), (3.5, "high"), (4.2, "vhigh")]:
        out[f"coins_{name}"] = (out["lncoins"] >= threshold).astype(float)
    for threshold, name in [(2.5, "low"), (4.8, "mid"), (6.3, "high")]:
        out[f"incentive_{name}"] = (out["lpi"] >= threshold).astype(float)
    for threshold, name in [(2.0, "low"), (4.5, "mid"), (6.5, "high")]:
        out[f"fmde_{name}"] = (out["fmde"] >= threshold).astype(float)

    out["idp_x_high_disease"] = out["idp"] * out["disease_ge_14"]
    out["idp_x_vhigh_disease"] = out["idp"] * out["disease_ge_22"]
    out["phys_x_high_disease"] = out["physlim"] * out["disease_ge_14"]
    out["phys_x_fairpoor"] = out["physlim"] * out["fairpoor"]
    out["idp_x_fairpoor"] = out["idp"] * out["fairpoor"]
    out["good_x_zero_coins"] = out["hlthg"] * out["coins_zero"]
    out["high_coins_x_high_disease"] = out["coins_high"] * out["disease_ge_14"]
    out["zero_coins_x_high_disease"] = out["coins_zero"] * out["disease_ge_14"]
    out["idp_x_zero_coins"] = out["idp"] * out["coins_zero"]
    out["idp_x_high_coins"] = out["idp"] * out["coins_high"]
    out["phys_x_idp"] = out["physlim"] * out["idp"]
    out["fairpoor_x_high_disease"] = out["fairpoor"] * out["disease_ge_14"]
    out["good_x_low_disease"] = out["hlthg"] * (1.0 - out["disease_ge_10"])
    out["high_incentive_x_idp"] = out["incentive_high"] * out["idp"]
    out["high_fmde_x_phys"] = out["fmde_high"] * out["physlim"]
    out["disea_x_idp"] = out["disea"] * out["idp"]
    out["disea_x_phys"] = out["disea"] * out["physlim"]
    out["coins_x_idp"] = out["lncoins"] * out["idp"]
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

    # Coverage/nonresponse depends only on observed covariates, but in several
    # different directions.  No outcome enters sample selection.
    logit = (
        -0.05 + 0.55 * idp - 0.20 * dz + 0.15 * cz + 0.15 * hg
        - 0.40 * hf - 0.95 * hp
        - 1.70 * (pl & (dz > 0.20))
        - 1.15 * (hf & idp)
        + 1.10 * (idp & (cz > 0.20))
        + 0.80 * ((fz > 0.35) & (~idp))
        + 0.70 * (hg & (lz > 0.15) & (dz < 0.10))
        - 1.00 * ((cz < -0.35) & (dz > 0.30))
        + 0.75 * ((dz < -0.35) & (~idp) & (cz > 0.05))
        - 0.80 * (pl & (~idp) & (cz < 0.0))
        + 0.65 * ((hp | hf) & (cz > 0.20))
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(len(pool)) < (0.88 * p)


def _frame_targets(frame_reservoir: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(TARGET_COLUMNS):
        n = min(int(FRAME_SAMPLE_SIZE[col]), len(frame_reservoir))
        rng = np.random.default_rng(1701 + 53 * j)
        idx = rng.choice(len(frame_reservoir), size=n, replace=False)
        values = frame_reservoir[col].to_numpy(float)[idx]
        rows.append({
            "column": col,
            "target_mean": float(values.mean()),
            "std_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        })
    return pd.DataFrame(rows)


def build(source_csv: str | Path):
    source = pd.read_csv(source_csv)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(source))

    hidden_target = source.iloc[order[:TARGET_N]].reset_index(drop=True)
    frame_reservoir = source.iloc[order[TARGET_N:TARGET_N + FRAME_RESERVOIR_N]].reset_index(drop=True)
    pool = source.iloc[order[TARGET_N + FRAME_RESERVOIR_N:]].reset_index(drop=True)

    hidden_target = add_calibration_columns(hidden_target)
    frame_reservoir = add_calibration_columns(frame_reservoir)
    pool = add_calibration_columns(pool)
    analysis = pool.loc[_biased_keep(pool, rng)].reset_index(drop=True)

    frame_targets = _frame_targets(frame_reservoir)
    visible_cols = ["mdvis"] + RAW_COVARIATES + [c for c in TARGET_COLUMNS if c not in RAW_COVARIATES]
    return analysis[visible_cols].copy(), frame_targets, hidden_target


def write_environment(source_csv: str | Path, app_dir: str | Path) -> None:
    app_dir = Path(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    analysis, frame_targets, _ = build(source_csv)
    analysis.to_csv(app_dir / "analysis_sample.csv", index=False)
    frame_targets.to_csv(app_dir / "frame_targets.csv", index=False)


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

# Calibrated so the reference has modest headroom while useful but imperfect
# weighting strategies keep a continuous gradient instead of falling off a
# worst-slice cliff.
OUTCOME_GOOD = 0.390
OUTCOME_BAD = 0.520
OUTCOME_POWER = 1.4
WORST_GOOD = 1.15
WORST_BAD = 1.80
WORST_POWER = 1.1
ESS_GOOD = 0.90
ESS_BAD = 0.45
TOP1_GOOD = 0.025
TOP1_BAD = 0.080


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
        df["disea"].to_numpy(float) >= 22,
        df["lncoins"].to_numpy(float) >= 3.5,
        df["lncoins"].to_numpy(float) < 0.05,
        (df["idp"].to_numpy(float) > 0.5) & (df["disea"].to_numpy(float) >= 14),
    ]
    for mask in groups:
        ww = w[mask]
        yy = y[mask]
        out.append(np.sum(ww * yy) / ww.sum())
        out.append(np.sum(ww * (yy >= 5)) / ww.sum())
    return np.asarray(out, float)


SCALES = np.asarray([
    1.0, 0.08, 0.08, 0.04, 1.0, 1.5,
    1.5, 0.10, 1.5, 0.10, 1.5, 0.10, 1.5, 0.10,
    1.5, 0.10, 1.5, 0.10, 1.5, 0.10, 1.5, 0.10,
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

    outcome_score = np.clip(
        (OUTCOME_BAD - rmse) / (OUTCOME_BAD - OUTCOME_GOOD), 0.0, 1.0
    ) ** OUTCOME_POWER
    worst_score = np.clip(
        (WORST_BAD - worst) / (WORST_BAD - WORST_GOOD), 0.0, 1.0
    ) ** WORST_POWER
    ess_score = float(np.clip((ess - ESS_BAD) / (ESS_GOOD - ESS_BAD), 0.0, 1.0))
    tail_score = float(np.clip((TOP1_BAD - top1_mass) / (TOP1_BAD - TOP1_GOOD), 0.0, 1.0))
    stability = 0.60 * ess_score + 0.40 * tail_score
    reward = float(0.90 * outcome_score + 0.06 * worst_score + 0.04 * stability)

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


