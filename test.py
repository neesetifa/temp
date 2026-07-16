The interval code in `app/solve.py` is incomplete. Implement:

```python
fit_interval_model(train_X, train_y, train_groups, calib_X, calib_y, calib_groups)
predict_interval(X, groups, params)
```

`predict_interval` returns two float arrays, `lower` and `upper`, each with shape `(N,)`.

The calibration split is uneven across groups. Some groups have plenty of rows, some have very few, and one evaluation group has no group-specific calibration rows. Group membership is useful, but a single fixed width for each group is not enough: residual spread also changes across the feature space, and the evaluation feature mix is not the same as the calibration mix. The current group-only approach misses noisy regions and wastes width on quieter rows.

Use the training split for the point model and the calibration split for interval calibration. The returned intervals should keep close to 90% coverage without solving the problem by making every interval wide. Lower and upper widths may differ.

Keep the implementation deterministic and self-contained. Do not use network access, external data, hidden files, or the public evaluation targets while fitting.

from __future__ import annotations
import numpy as np

N_GROUPS = 12
D = 12
GROUP_BIAS = np.array([-1.0, -0.55, 0.15, 0.65, -0.8, -0.2, 0.45, 1.0, -1.25, -0.35, 0.75, 1.35], dtype=float)
BASE_SIGMA = np.array([0.72, 0.95, 0.62, 1.10, 0.82, 1.18, 0.68, 1.30, 0.88, 1.42, 1.05, 1.55], dtype=float)
GROUP_CLUSTER = np.array([0,0,0,0,1,1,1,1,2,2,2,2], dtype=int)
GROUP_THRESHOLD = np.array([0.15,-0.15,0.35,-0.30,0.10,-0.20,0.30,-0.35,0.05,-0.25,0.25,-0.40])
GROUP_X_SHIFT = np.array([
    [-0.35, 0.20, 0.10], [-0.10, 0.10,-0.20], [0.20,-0.20,0.05], [0.40,0.15,-0.10],
    [-0.45,0.35,0.20], [-0.20,-0.30,0.15], [0.15,0.30,-0.25], [0.45,-0.15,0.05],
    [-0.55,-0.10,0.30], [-0.25,0.25,-0.30], [0.25,-0.35,0.20], [0.55,0.10,-0.15],
], dtype=float)
RISK_AXES = np.array([
    [ 0.95, -0.70, 0.35, 0.20],
    [-0.75,  0.30, 0.85,-0.25],
    [ 0.25,  0.90,-0.55, 0.45],
], dtype=float)


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def shared_signal(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (
        1.15*X[:,0] - 0.72*X[:,1] + 0.48*X[:,2] + 0.22*X[:,3]
        + 0.62*np.sin(1.1*X[:,4]) - 0.38*np.cos(0.9*X[:,5])
        + 0.32*X[:,0]*X[:,6] - 0.27*X[:,1]*X[:,7]
        + 0.21*X[:,2]*X[:,3] + 0.18*(X[:,8]**2 - 1.0)
        - 0.14*X[:,9]**2 + 0.12*np.sin(X[:,10]*X[:,11])
    )


def risk_score(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups, dtype=int)
    cl = GROUP_CLUSTER[groups]
    linear = np.sum(X[:,:4] * RISK_AXES[cl], axis=1)
    nonlinear = 0.50*np.sin(1.25*X[:,4]) + 0.36*X[:,5]*X[:,6] - 0.22*np.abs(X[:,7])
    return linear + nonlinear - GROUP_THRESHOLD[groups]


def local_scale(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    r = risk_score(X, groups)
    pocket = ((X[:,8] > 0.75) & (X[:,9] < -0.15)).astype(float)
    smooth = 0.48 + 1.42*_sigmoid(2.05*r) + 0.24*np.abs(X[:,7]) + 0.52*pocket
    return BASE_SIGMA[np.asarray(groups, dtype=int)] * smooth


def skew_direction(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    cl = GROUP_CLUSTER[np.asarray(groups, dtype=int)]
    s = X[:,10] + np.where(cl == 0, 0.45*X[:,1], np.where(cl == 1, -0.50*X[:,2], 0.40*X[:,3]))
    return np.where(s >= 0, 1.0, -1.0)


def sample_X(rng: np.random.Generator, groups: np.ndarray, split: str) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    X = rng.normal(0, 1, size=(len(groups), D))
    X[:,:3] += GROUP_X_SHIFT[groups]
    X[:,4] += 0.12*np.sin(groups)

    # Hidden evaluation changes the feature mix rather than the response rule.
    # Weak groups appear much more often in their high-risk region. Other groups
    # still contain a low/high mixture, so constant group widths waste budget.
    if split == 'hidden':
        cl = GROUP_CLUSTER[groups]
        axes = RISK_AXES[cl]
        weak = groups >= 8
        high_draw = rng.random(len(groups)) < np.where(weak, 0.78, 0.42)
        strength = np.where(weak, 1.10, 0.72) * high_draw
        X[:,:4] += axes * strength[:,None]
        # Preserve a sizeable low-risk slice for the width diagnostic.
        low_draw = (~high_draw) & (rng.random(len(groups)) < 0.42)
        X[:,:4] -= axes * (0.65*low_draw)[:,None]
    elif split == 'public':
        cl = GROUP_CLUSTER[groups]
        axes = RISK_AXES[cl]
        high_draw = rng.random(len(groups)) < np.where(groups >= 8, 0.33, 0.25)
        X[:,:4] += axes * (0.55*high_draw)[:,None]
    elif split == 'calib':
        cl = GROUP_CLUSTER[groups]
        axes = RISK_AXES[cl]
        high_draw = rng.random(len(groups)) < 0.22
        X[:,:4] += axes * (0.45*high_draw)[:,None]
    return X


def make_y(rng: np.random.Generator, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    sig = local_scale(X, groups)
    n = len(groups)
    normal = rng.normal(0, 1, n)
    t3 = rng.standard_t(df=3, size=n) / np.sqrt(3.0)
    one_sided = rng.exponential(1.0, n) - 0.72
    direction = skew_direction(X, groups)
    u = rng.random(n)
    eps = np.where(u < 0.80, normal, np.where(u < 0.93, 1.55*t3, 1.55*direction*one_sided))
    return shared_signal(X) + GROUP_BIAS[groups] + sig*eps


def sample_groups(rng, n, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return rng.choice(np.arange(N_GROUPS), size=n, p=p)


def dataset(seed=0, n_train=4200, n_calib=1300, n_public=900, n_hidden=2800):
    rng = np.random.default_rng(seed)
    p_train = np.array([.115,.105,.10,.095,.095,.085,.08,.07,.07,.065,.065,.055])
    p_calib = np.array([.145,.13,.12,.11,.105,.09,.08,.07,.055,.045,.025,0.0])
    p_public = np.array([.125,.115,.105,.10,.095,.085,.075,.07,.065,.055,.06,.05])
    p_hidden = np.array([.055,.055,.06,.06,.065,.065,.075,.08,.12,.13,.115,.12])

    tg = sample_groups(rng, n_train, p_train)
    cg = sample_groups(rng, n_calib, p_calib)
    pg = sample_groups(rng, n_public, p_public)
    hg = sample_groups(rng, n_hidden, p_hidden)

    train_X = sample_X(rng, tg, 'train')
    calib_X = sample_X(rng, cg, 'calib')
    public_X = sample_X(rng, pg, 'public')
    hidden_X = sample_X(rng, hg, 'hidden')

    return dict(
        train_X=train_X, train_y=make_y(rng, train_X, tg), train_groups=tg,
        calib_X=calib_X, calib_y=make_y(rng, calib_X, cg), calib_groups=cg,
        public_X=public_X, public_y=make_y(rng, public_X, pg), public_groups=pg,
        hidden_X=hidden_X, hidden_y=make_y(rng, hidden_X, hg), hidden_groups=hg,
    )


from __future__ import annotations

import json
import time
import numpy as np

from generators import dataset, risk_score, skew_direction


def _metrics(y, lower, upper, X, groups):
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups, dtype=int)

    finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    covered = finite & (lower <= y) & (y <= upper)
    width = upper - lower

    group_cov = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 40:
            group_cov[int(g)] = float(covered[m].mean())

    risk = risk_score(X, groups)
    high_risk = risk > 0.85
    low_risk = risk < -0.65
    weak = groups >= 8
    unseen = groups == 11
    direction = skew_direction(X, groups)
    upper_risk = (direction > 0) & (risk > 0.25)
    lower_risk = (direction < 0) & (risk > 0.25)

    return {
        "finite_monotone": bool(finite.all()),
        "overall_cov": float(covered.mean()),
        "worst_group_cov": float(min(group_cov.values())),
        "high_risk_cov": float(covered[high_risk].mean()),
        "weak_group_cov": float(covered[weak].mean()),
        "unseen_cov": float(covered[unseen].mean()),
        "upper_risk_cov": float(covered[upper_risk].mean()),
        "lower_risk_cov": float(covered[lower_risk].mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "p90_width": float(np.quantile(width, 0.90)),
        "low_risk_mean_width": float(width[low_risk].mean()),
    }


def _passes(m):
    return (
        m["finite_monotone"]
        and m["overall_cov"] >= 0.875
        and m["worst_group_cov"] >= 0.800
        and m["high_risk_cov"] >= 0.840
        and m["weak_group_cov"] >= 0.840
        and m["unseen_cov"] >= 0.820
        and m["upper_risk_cov"] >= 0.820
        and m["lower_risk_cov"] >= 0.820
        and m["mean_width"] <= 7.10
        and m["median_width"] <= 6.60
        and m["p90_width"] <= 13.20
        and m["low_risk_mean_width"] <= 4.15
    )


def _single_seed(solution_module, seed):
    d = dataset(seed)
    params = solution_module.fit_interval_model(
        d["train_X"], d["train_y"], d["train_groups"],
        d["calib_X"], d["calib_y"], d["calib_groups"],
    )
    lower, upper = solution_module.predict_interval(
        d["hidden_X"], d["hidden_groups"], params
    )
    m = _metrics(d["hidden_y"], lower, upper, d["hidden_X"], d["hidden_groups"])

    idx = np.arange(96)
    lo1, hi1 = solution_module.predict_interval(d["hidden_X"][idx], d["hidden_groups"][idx], params)
    lo2, hi2 = solution_module.predict_interval(d["hidden_X"][idx], d["hidden_groups"][idx], params)
    perm = np.array([17,3,44,0,5,29,12,7,70,9,50,1,81,25,33])
    lop, hip = solution_module.predict_interval(d["hidden_X"][idx][perm], d["hidden_groups"][idx][perm], params)
    stable = (
        np.allclose(lo1, lo2, atol=1e-12, rtol=0)
        and np.allclose(hi1, hi2, atol=1e-12, rtol=0)
        and np.allclose(lop, lo1[perm], atol=1e-12, rtol=0)
        and np.allclose(hip, hi1[perm], atol=1e-12, rtol=0)
    )
    m["deterministic_permutation"] = bool(stable)
    m["passed"] = bool(_passes(m) and stable)
    return m


def evaluate(solution_module, seeds=(101, 102, 103, 104, 105)):
    t0 = time.perf_counter()
    per_seed = {int(seed): _single_seed(solution_module, seed) for seed in seeds}
    keys = [
        "overall_cov", "worst_group_cov", "high_risk_cov", "weak_group_cov",
        "unseen_cov", "upper_risk_cov", "lower_risk_cov", "mean_width",
        "median_width", "p90_width", "low_risk_mean_width",
    ]
    aggregate = {k: float(np.mean([v[k] for v in per_seed.values()])) for k in keys}
    return {
        "passed": bool(all(v["passed"] for v in per_seed.values())),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "elapsed_seconds": time.perf_counter() - t0,
    }


if __name__ == "__main__":
    import importlib
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
    solution = importlib.import_module("solve")
    print(json.dumps(evaluate(solution), indent=2))
from __future__ import annotations
import numpy as np


def _ridge_fit(Phi, y, lam=1.0, weights=None):
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float))
        A = (Phi*w[:,None]).T @ (Phi*w[:,None])
        b = (Phi*w[:,None]).T @ (y*w)
    else:
        A = Phi.T @ Phi
        b = Phi.T @ y
    A = A + lam*np.eye(Phi.shape[1])
    A[0,0] -= lam
    return np.linalg.solve(A, b)


def _onehot(groups, n_groups):
    groups = np.asarray(groups, dtype=int)
    out = np.zeros((len(groups), n_groups), dtype=float)
    valid = (groups >= 0) & (groups < n_groups)
    out[np.arange(len(groups))[valid], groups[valid]] = 1.0
    return out


def _point_features(X, groups, n_groups):
    X = np.asarray(X, dtype=float)
    oh = _onehot(groups, n_groups)
    return np.concatenate([
        np.ones((len(X),1)), X, X**2,
        np.sin(X[:,:6]), np.cos(X[:,4:8]),
        X[:,[0,1,2,3]]*X[:,[6,7,8,9]],
        X[:,[0,1,2]]*X[:,[1,2,3]],
        oh,
    ], axis=1)


def _scale_features(X, groups, n_groups):
    X = np.asarray(X, dtype=float)
    oh = _onehot(groups, n_groups)
    global_parts = [
        np.ones((len(X),1)), X, np.abs(X), X**2,
        np.sin(X[:,:7]), np.cos(X[:,:7]),
        (X[:,0]*X[:,1])[:,None], (X[:,0]*X[:,2])[:,None],
        (X[:,1]*X[:,2])[:,None], (X[:,2]*X[:,3])[:,None],
        (X[:,4]*X[:,5])[:,None], (X[:,5]*X[:,6])[:,None],
        (X[:,8]*X[:,9])[:,None], (X[:,10]*X[:,1])[:,None],
        (X[:,10]*X[:,2])[:,None], (X[:,10]*X[:,3])[:,None],
    ]
    for j in range(6):
        for knot in (-1.0, -0.35, 0.35, 1.0):
            global_parts.append(np.maximum(X[:,j]-knot, 0.0)[:,None])
    # Let each group learn a different linear risk direction, but share nonlinear shape.
    interaction_base = np.concatenate([X[:,:6], np.abs(X[:,7:10])], axis=1)
    group_interactions = (oh[:,:,None] * interaction_base[:,None,:]).reshape(len(X), -1)
    global_parts.extend([oh, group_interactions])
    return np.concatenate(global_parts, axis=1)


def _quantile(vals, q, default):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)
    return float(np.quantile(vals, q, method='higher'))


def _crossfit_point_predictions(X, y, groups, n_groups, folds=5):
    pred = np.empty(len(y), dtype=float)
    idx = np.arange(len(y))
    for fold in range(folds):
        va = (idx % folds) == fold
        tr = ~va
        coef = _ridge_fit(_point_features(X[tr], groups[tr], n_groups), y[tr], lam=5.0)
        pred[va] = _point_features(X[va], groups[va], n_groups) @ coef
    return pred


def fit_interval_model(train_X, train_y, train_groups, calib_X, calib_y, calib_groups):
    train_X = np.asarray(train_X, dtype=float)
    calib_X = np.asarray(calib_X, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    calib_y = np.asarray(calib_y, dtype=float)
    train_groups = np.asarray(train_groups, dtype=int)
    calib_groups = np.asarray(calib_groups, dtype=int)
    n_groups = max(int(train_groups.max(initial=0)), int(calib_groups.max(initial=0))) + 1
    n_groups = max(n_groups, 12)

    oof_pred = _crossfit_point_predictions(train_X, train_y, train_groups, n_groups)
    point_coef = _ridge_fit(_point_features(train_X, train_groups, n_groups), train_y, lam=5.0)
    cal_pred = _point_features(calib_X, calib_groups, n_groups) @ point_coef
    train_res = train_y - oof_pred
    cal_res = calib_y - cal_pred

    Sf = _scale_features(train_X, train_groups, n_groups)
    # Separate lower/upper magnitude models capture feature-dependent skew. Values
    # on the opposite side receive a small floor rather than being discarded.
    floor = 0.08
    y_lo = np.log(np.maximum(-train_res, floor))
    y_hi = np.log(np.maximum(train_res, floor))
    w_lo = np.where(train_res < 0, 1.0, 0.12)
    w_hi = np.where(train_res > 0, 1.0, 0.12)
    lo_coef = _ridge_fit(Sf, y_lo, lam=38.0, weights=w_lo)
    hi_coef = _ridge_fit(Sf, y_hi, lam=38.0, weights=w_hi)

    Sc = _scale_features(calib_X, calib_groups, n_groups)
    lo_scale = np.exp(np.clip(Sc @ lo_coef, -2.4, 2.8))
    hi_scale = np.exp(np.clip(Sc @ hi_coef, -2.4, 2.8))
    lo_score = np.maximum(-cal_res, 0.0) / np.maximum(lo_scale, 1e-6)
    hi_score = np.maximum(cal_res, 0.0) / np.maximum(hi_scale, 1e-6)

    qtail = 0.952
    global_lo = _quantile(lo_score, qtail, 2.0)
    global_hi = _quantile(hi_score, qtail, 2.0)
    group_factors = {}
    for g in range(n_groups):
        m = calib_groups == g
        nc = int(m.sum())
        ql = _quantile(lo_score[m], qtail, global_lo)
        qh = _quantile(hi_score[m], qtail, global_hi)
        w = nc / (nc + 42.0)
        # A no-calibration group uses the feature model learned from OOF train
        # residuals, with only a small global guard rather than a blanket width.
        guard = 1.12 if nc == 0 else 1.065 if nc < 12 else 1.025 if nc < 35 else 1.0
        group_factors[g] = (
            float(((1-w)*global_lo + w*ql)*guard),
            float(((1-w)*global_hi + w*qh)*guard),
            nc,
        )

    return {
        'n_groups': n_groups,
        'point_coef': point_coef,
        'lo_coef': lo_coef,
        'hi_coef': hi_coef,
        'global_lo': float(global_lo),
        'global_hi': float(global_hi),
        'group_factors': group_factors,
    }


def predict_interval(X, groups, params):
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups, dtype=int)
    n_groups = int(params['n_groups'])
    center = _point_features(X, groups, n_groups) @ params['point_coef']
    S = _scale_features(X, groups, n_groups)
    lo_scale = np.exp(np.clip(S @ params['lo_coef'], -2.4, 2.8))
    hi_scale = np.exp(np.clip(S @ params['hi_coef'], -2.4, 2.8))
    lf = np.empty(len(groups)); hf = np.empty(len(groups))
    for i,g in enumerate(groups):
        if int(g) in params['group_factors']:
            lf[i], hf[i], _ = params['group_factors'][int(g)]
        else:
            lf[i] = params['global_lo']*1.12
            hf[i] = params['global_hi']*1.12
    return center - lo_scale*lf, center + hi_scale*hf


