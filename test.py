# Build general-purpose transport weights for the analysis sample

`/app/analysis_sample.csv` contains an observed analysis sample described by seven insurance and baseline-health covariates. The sample is not representative of the target population.

`/app/frame_sample.csv` is an independent representative frame sample from the target population. It contains the same seven covariates. Exact covariate profiles in the two visible files need not overlap.

Produce one nonnegative transport weight for every row of `analysis_sample.csv` and save the vector to:

`/output/weights.npy`

The array must be one-dimensional, finite, and have the same length and row order as `analysis_sample.csv`. Weight scale is arbitrary; the evaluator normalizes the weights before use.

The submitted weights will be used as general-purpose population weights for downstream summaries in a separate held-out representative target panel. Evaluation covers utilization and withheld health measurements overall and within natural insurance/health subgroups. The downstream variables themselves are not available when constructing the weights.

Good weights should therefore transport the joint covariate distribution, not merely match a few marginal means. At the same time, extremely concentrated weights are not useful: overlap and effective sample size matter.

Use any transport, calibration, entropy-balancing, density-ratio, or related weighting method available in the environment. Do not use external data or network access.


reference
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

REG_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CAP_GRID = (1.5, 2.0, 3.0, 5.0, 8.0)
STABILITY_PENALTY = 0.055


def _candidate(X: np.ndarray, F: np.ndarray, reg: float, cap: float):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    A = poly.fit_transform(X)
    R = poly.transform(F)
    mu = A.mean(axis=0)
    sd = A.std(axis=0) + 1e-8
    Z = (A - mu) / sd
    target = (R.mean(axis=0) - mu) / sd
    gram = (Z.T @ Z) / len(Z)
    beta = np.linalg.solve(gram + reg * np.eye(Z.shape[1]), target)
    w = np.clip(1.0 + Z @ beta, 0.2, cap)
    wn = w / w.mean()
    residual = (wn[:, None] * Z).mean(axis=0) - target
    balance_rmse = float(np.sqrt(np.mean(residual * residual)))
    ess = float(wn.sum() ** 2 / np.dot(wn, wn) / len(wn))
    criterion = balance_rmse + STABILITY_PENALTY * (1.0 / ess - 1.0)
    return w, criterion


def make_weights(app_dir: Path) -> np.ndarray:
    analysis = pd.read_csv(app_dir / "analysis_sample.csv")
    frame = pd.read_csv(app_dir / "frame_sample.csv")
    cols = frame.columns.tolist()
    X = analysis[cols].to_numpy(float)
    F = frame[cols].to_numpy(float)

    best = None
    for reg in REG_GRID:
        for cap in CAP_GRID:
            w, criterion = _candidate(X, F, reg, cap)
            if best is None or criterion < best[0]:
                best = (criterion, reg, cap, w)
    return np.asarray(best[3], dtype=np.float64)


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
from generator import build, VISIBLE
from hidden_eval import score_weights, write_result

APP_DIR=Path(os.environ.get('TASK_APP_DIR','/app'))
OUTPUT_DIR=Path(os.environ.get('TASK_OUTPUT_DIR','/output'))
LOGS_DIR=Path(os.environ.get('TASK_LOGS_DIR','/logs/verifier'))
HERE=Path(__file__).resolve().parent


def test_00_agent_output_is_scored():
    result=score_weights(OUTPUT_DIR/'weights.npy'); write_result(result,LOGS_DIR)
    assert 0.0 <= float(result['reward']) <= 1.0


def test_shipped_visible_data_match_generator():
    analysis,frame,_=build(HERE/'source_randhie.csv')
    pd.testing.assert_frame_equal(analysis,pd.read_csv(APP_DIR/'analysis_sample.csv'),check_dtype=False)
    pd.testing.assert_frame_equal(frame,pd.read_csv(APP_DIR/'frame_sample.csv'),check_dtype=False)


def test_visible_schema_is_transport_only():
    a=pd.read_csv(APP_DIR/'analysis_sample.csv'); f=pd.read_csv(APP_DIR/'frame_sample.csv')
    assert a.columns.tolist()==VISIBLE
    assert f.columns.tolist()==VISIBLE
    assert 'mdvis' not in a and 'hlthf' not in a and 'hlthp' not in a


def test_visible_profiles_are_disjoint():
    a=pd.read_csv(APP_DIR/'analysis_sample.csv'); f=pd.read_csv(APP_DIR/'frame_sample.csv')
    ka=set(map(tuple,a[VISIBLE].to_numpy())); kf=set(map(tuple,f[VISIBLE].to_numpy()))
    assert ka.isdisjoint(kf)

hidden
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from generator import build

# Reward anchors are calibrated after the visible-only reference is frozen.
OUTCOME_GOOD = 0.265
OUTCOME_BAD = 0.440
WORST_GOOD = 0.950
WORST_BAD = 1.250
ESS_GOOD = 0.650
ESS_BAD = 0.300
TOP1_GOOD = 0.060
TOP1_BAD = 0.120


def _weighted_quantile(x, w, q):
    order=np.argsort(x); x=x[order]; w=w[order]; c=np.cumsum(w)
    return float(np.interp(q*c[-1], c, x))


def _wm(x,w):
    w=np.asarray(w,float); return float(np.dot(w,x)/w.sum())


def _estimands(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    w=np.asarray(w,float); y=df['mdvis'].to_numpy(float)
    fair=df['hlthf'].to_numpy(float); poor=df['hlthp'].to_numpy(float)
    fp=np.clip(fair+poor,0,1)
    out=[
        _wm(y,w), _wm((y==0).astype(float),w), _wm((y>=5).astype(float),w),
        _wm((y>=10).astype(float),w), _weighted_quantile(y,w,0.75), _weighted_quantile(y,w,0.90),
        _wm(fair,w), _wm(poor,w), _wm(fp,w),
    ]
    groups=[
        df['physlm'].to_numpy(float)>0.5,
        df['idp'].to_numpy(float)>0.5,
        df['disea'].to_numpy(float)>=14,
        df['lncoins'].to_numpy(float)>=3.5,
        df['lncoins'].to_numpy(float)<0.05,
        df['hlthg'].to_numpy(float)>0.5,
    ]
    for m in groups:
        ww=w[m]; yy=y[m]
        out.extend([
            _wm(yy,ww),
            _wm((yy>=5).astype(float),ww),
            _wm(fp[m],ww),
        ])
    return np.asarray(out,float)

SCALES=np.asarray([
    1.0, .08, .08, .04, 1.0, 1.5, .08, .05, .08,
    1.5,.10,.10, 1.5,.10,.10, 1.5,.10,.10,
    1.5,.10,.10, 1.5,.10,.10, 1.5,.10,.10,
],float)


def _invalid(reason): return {'reward':0.0,'invalid':reason}


def score_array(w: np.ndarray) -> dict:
    here=Path(__file__).resolve().parent
    analysis,_,hidden=build(here/'source_randhie.csv')
    w=np.asarray(w,float)
    if w.ndim!=1 or len(w)!=len(analysis) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        return _invalid('invalid weights')
    w=w/w.mean(); ess=float(w.sum()**2/np.dot(w,w)/len(w))
    top_n=max(1,int(.01*len(w))); top1=float(np.sort(w)[-top_n:].sum()/w.sum())
    if ess<.12 or top1>.22 or w.max()>100: return _invalid('catastrophically concentrated weights')
    truth=_estimands(hidden,np.ones(len(hidden))); pred=_estimands(hidden.iloc[:0] if False else _analysis_full(here),w)
    z=np.abs((pred-truth)/SCALES); rmse=float(np.sqrt(np.mean(z*z))); worst=float(z.max())
    os=float(np.clip((OUTCOME_BAD-rmse)/(OUTCOME_BAD-OUTCOME_GOOD),0,1))**2
    ws=float(np.clip((WORST_BAD-worst)/(WORST_BAD-WORST_GOOD),0,1))**1.1
    es=float(np.clip((ess-ESS_BAD)/(ESS_GOOD-ESS_BAD),0,1)); ts=float(np.clip((TOP1_BAD-top1)/(TOP1_BAD-TOP1_GOOD),0,1))
    reward=float(.90*os+.05*ws+.05*(.6*es+.4*ts))
    return {'reward':reward,'normalized_estimand_rmse':rmse,'worst_normalized_error':worst,'ess_fraction':ess,'top1_weight_mass':top1}


def _analysis_full(here: Path) -> pd.DataFrame:
    # Reconstruct the selected source rows, retaining withheld columns only in verifier memory.
    import generator
    source=pd.read_csv(here/'source_randhie.csv').reset_index(drop=True)
    ih, iff, ip=generator._profile_split(source)
    pool=source.iloc[ip].reset_index(drop=True)
    rng=np.random.default_rng(generator.SEED)
    return pool.loc[generator._biased_keep(pool,rng)].reset_index(drop=True)


def score_weights(weights_path: str|Path) -> dict:
    p=Path(weights_path)
    if not p.is_file(): return _invalid('missing /output/weights.npy')
    try: w=np.load(p,allow_pickle=False)
    except Exception as exc: return _invalid(f'unable to load weights: {type(exc).__name__}')
    return score_array(w)


def write_result(result, logs_dir='/logs/verifier'):
    logs=Path(logs_dir); logs.mkdir(parents=True,exist_ok=True)
    (logs/'reward.txt').write_text(f"{float(result.get('reward',0)):.12f}\n")
    (logs/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('weights',nargs='?',type=Path,default=Path('/output/weights.npy')); ap.add_argument('--logs-dir',type=Path,default=Path('/logs/verifier'))
    a=ap.parse_args(); r=score_weights(a.weights); write_result(r,a.logs_dir); print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__': main()

generator

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260915
SPLIT_SEED = 385
TARGET_N = 5000
FRAME_N = 5000
VISIBLE = ["lncoins", "idp", "lpi", "fmde", "physlm", "disea", "hlthg"]
WITHHELD = ["mdvis", "hlthf", "hlthp"]


def _profile_split(source: pd.DataFrame):
    # Whole visible-covariate profiles are assigned to exactly one partition.
    groups = source.groupby(VISIBLE, sort=False, dropna=False).indices
    keys = list(groups)
    labels_path = Path(__file__).resolve().parent / "profile_partition.npy"
    labels = np.load(labels_path, allow_pickle=False)
    if len(labels) != len(keys):
        raise RuntimeError("profile partition does not match source profiles")
    def rows(label: int):
        js = np.flatnonzero(labels == label)
        return np.concatenate([np.asarray(groups[keys[j]], dtype=int) for j in js])
    h, f, p = rows(0), rows(1), rows(2)
    if len(h) != TARGET_N or len(f) != FRAME_N:
        raise RuntimeError(f"profile split has wrong sizes: hidden={len(h)}, frame={len(f)}")
    return h, f, p


def _biased_keep(pool: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    # Coverage depends only on agent-visible variables, but includes nonlinear
    # interactions that a purely linear propensity model cannot capture well.
    def z(c: str) -> np.ndarray:
        x = pool[c].to_numpy(float)
        return (x - x.mean()) / (x.std() + 1e-9)
    dz, cz, lz, fz = [z(c) for c in ["disea", "lncoins", "lpi", "fmde"]]
    pl = pool["physlm"].to_numpy(float) > 0.5
    idp = pool["idp"].to_numpy(float) > 0.5
    hg = pool["hlthg"].to_numpy(float) > 0.5
    logit = (
        -0.10 + 0.48 * idp - 0.22 * dz + 0.10 * cz + 0.12 * hg
        - 1.35 * (pl & (dz > 0.15))
        + 1.00 * (idp & (cz > 0.20))
        + 0.82 * ((fz > 0.30) & (~idp))
        + 0.68 * (hg & (lz > 0.15) & (dz < 0.10))
        - 0.90 * ((cz < -0.30) & (dz > 0.25))
        + 0.72 * ((dz < -0.35) & (~idp) & (cz > 0.05))
        - 0.70 * (pl & (~idp) & (cz < 0.0))
        - 0.60 * ((lz < -0.45) & (fz > 0.20))
        + 0.65 * ((dz > 0.45) & idp & (fz < 0.15))
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(len(pool)) < (0.92 * p)


def build(source_csv: str | Path):
    source = pd.read_csv(source_csv).reset_index(drop=True)
    ih, iff, ip = _profile_split(source)
    hidden = source.iloc[ih].reset_index(drop=True)
    frame = source.iloc[iff].reset_index(drop=True)
    pool = source.iloc[ip].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    analysis = pool.loc[_biased_keep(pool, rng)].reset_index(drop=True)
    return analysis[VISIBLE].copy(), frame[VISIBLE].copy(), hidden.copy()


def write_environment(source_csv: str | Path, app_dir: str | Path) -> None:
    app_dir = Path(app_dir); app_dir.mkdir(parents=True, exist_ok=True)
    analysis, frame, _ = build(source_csv)
    analysis.to_csv(app_dir / "analysis_sample.csv", index=False)
    frame.to_csv(app_dir / "frame_sample.csv", index=False)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    write_environment(here / "source_randhie.csv", here.parent / "environment" / "app")


