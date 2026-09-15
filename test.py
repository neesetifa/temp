# Build transport weights for the analysis sample

`/app/analysis_sample.csv` contains an observed analysis sample. `mdvis` is the observed number of outpatient physician visits. The remaining columns describe insurance terms, health status, and derived balancing features. This sample is not representative of the target population.

`/app/frame_sample.csv` is an independent representative frame sample from the target population. It contains the same covariates and balancing features, but no physician-utilization outcome.

Produce one nonnegative analysis weight for every row of `analysis_sample.csv` and save the vector to:

`/output/weights.npy`

The array must be one-dimensional, finite, and have the same length and row order as `analysis_sample.csv`. Weight scale is arbitrary; the evaluator normalizes the weights before use.

The submitted weights are used to estimate physician-utilization summaries in a separate held-out representative target panel. Evaluation covers overall utilization, zero/high-use rates, several health and plan subgroups, and weight stability. Extremely concentrated weights are not useful even if they reproduce the visible frame sample closely.

Use any analysis, transport, calibration, density-ratio, or outcome-assisted weighting method available in the environment. Do not use external data or network access.

reference
  from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


def make_weights(app_dir: Path) -> np.ndarray:
    analysis = pd.read_csv(app_dir / "analysis_sample.csv")
    frame = pd.read_csv(app_dir / "frame_sample.csv")

    cols = frame.columns.tolist()
    Xa = analysis[cols].to_numpy(float)
    Xf = frame[cols].to_numpy(float)
    y = analysis["mdvis"].to_numpy(float)

    # Outcome-assisted balancing.  The frame has covariates but no utilization
    # outcome, so learn several smooth utilization views in the analysis sample
    # and balance their frame predictions together with the observed covariates.
    views = [
        y,
        (y == 0).astype(float),
        (y >= 5).astype(float),
        (y >= 10).astype(float),
    ]
    pred_a = []
    pred_f = []
    for j, target in enumerate(views):
        model = ExtraTreesRegressor(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.70,
            random_state=110 + j,
            n_jobs=1,
        )
        model.fit(Xa, target)
        pred_a.append(model.predict(Xa))
        pred_f.append(model.predict(Xf))

    A = np.column_stack([Xa, np.column_stack(pred_a)])
    F = np.column_stack([Xf, np.column_stack(pred_f)])
    mu = A.mean(axis=0)
    sd = A.std(axis=0) + 1e-8
    Z = (A - mu) / sd
    target = (F.mean(axis=0) - mu) / sd
    gram = (Z.T @ Z) / len(Z)

    reg = 2.0
    beta = np.linalg.solve(gram + reg * np.eye(Z.shape[1]), target)
    w = 1.0 + Z @ beta

    # Preserve overlap and keep the estimator stable in sparse health/plan cells.
    return np.clip(w, 2.0 / 3.0, 1.5).astype(np.float64)


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
TARGET_N = 5000
FRAME_N = 5000

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
BALANCE_COLUMNS = BASE_TARGETS + THRESHOLD_TARGETS + INTERACTION_TARGETS


def add_balance_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    # Coverage/nonresponse depends only on observed covariates.  The structure
    # is deliberately nonlinear, so matching a few first moments is useful but
    # not sufficient for the downstream target population.
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


def build(source_csv: str | Path):
    source = pd.read_csv(source_csv)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(source))

    hidden_target = add_balance_columns(source.iloc[order[:TARGET_N]].reset_index(drop=True))
    frame = add_balance_columns(source.iloc[order[TARGET_N:TARGET_N + FRAME_N]].reset_index(drop=True))
    pool = add_balance_columns(source.iloc[order[TARGET_N + FRAME_N:]].reset_index(drop=True))
    analysis = pool.loc[_biased_keep(pool, rng)].reset_index(drop=True)

    visible_analysis = ["mdvis"] + RAW_COVARIATES + [c for c in BALANCE_COLUMNS if c not in RAW_COVARIATES]
    visible_frame = RAW_COVARIATES + [c for c in BALANCE_COLUMNS if c not in RAW_COVARIATES]
    return analysis[visible_analysis].copy(), frame[visible_frame].copy(), hidden_target


def write_environment(source_csv: str | Path, app_dir: str | Path) -> None:
    app_dir = Path(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)
    analysis, frame, _ = build(source_csv)
    analysis.to_csv(app_dir / "analysis_sample.csv", index=False)
    frame.to_csv(app_dir / "frame_sample.csv", index=False)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parent
    write_environment(here / "source_randhie.csv", root / "environment" / "app")


test main
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd

from generator import build, BALANCE_COLUMNS
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
    analysis, frame, _ = build(HERE / "source_randhie.csv")
    shipped_analysis = pd.read_csv(APP_DIR / "analysis_sample.csv")
    shipped_frame = pd.read_csv(APP_DIR / "frame_sample.csv")
    pd.testing.assert_frame_equal(analysis, shipped_analysis, check_dtype=False)
    pd.testing.assert_frame_equal(frame, shipped_frame, check_dtype=False)


def test_balance_columns_exist():
    analysis = pd.read_csv(APP_DIR / "analysis_sample.csv")
    frame = pd.read_csv(APP_DIR / "frame_sample.csv")
    assert set(BALANCE_COLUMNS).issubset(analysis.columns)
    assert set(BALANCE_COLUMNS).issubset(frame.columns)
    assert "mdvis" in analysis.columns
    assert "mdvis" not in frame.columns

hiddenfrom __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from generator import build

OUTCOME_GOOD = 0.358
OUTCOME_BAD = 0.520
OUTCOME_POWER = 2.0
WORST_GOOD = 0.90
WORST_BAD = 1.60
WORST_POWER = 1.1
ESS_GOOD = 0.92
ESS_BAD = 0.45
TOP1_GOOD = 0.016
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
    reward = float(0.90 * outcome_score + 0.05 * worst_score + 0.05 * stability)

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

