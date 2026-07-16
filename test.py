# Build prediction intervals for grouped regression

I have a small tabular regression problem under `/app`. The goal is not just to
predict the center of `y`, but to return an interval for each new row.

The data is split into a training part and a calibration part. Groups are uneven:
some groups have many calibration rows, some have only a few, and evaluation can
include groups where group-specific calibration is weak. A single global interval
is usually too blunt, while fully separate intervals for every group can be noisy.

Please finish `app/solve.py`:

```python
def fit_interval_model(train_X, train_y, train_groups, calib_X, calib_y, calib_groups):
    ...

def predict_interval(X, groups, params):
    ...
```

`predict_interval` should return two one-dimensional arrays, `(lower, upper)`,
with one interval per input row.

Use the training data to fit the center prediction. Use the calibration data to
choose interval sizes. The group labels are meaningful categories, not an
ordered numeric variable. The evaluation checks coverage by group as well as the
average size of the intervals, so making every interval extremely wide is not a
good solution.

The code should be deterministic and self-contained. Do not use network access,
external data, hidden files, or public evaluation targets while fitting the
model.


from __future__ import annotations
import json, time
import numpy as np
from generators import dataset


def _metrics(y, lower, upper, groups):
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper); groups=np.asarray(groups,dtype=int)
    finite=np.isfinite(lower)&np.isfinite(upper)&(lower<=upper)
    cov=(lower<=y)&(y<=upper)&finite
    width=upper-lower
    group_cov={}
    for g in np.unique(groups):
        m=groups==g
        group_cov[int(g)]=float(cov[m].mean())
    tiny=np.isin(groups,[11,12,13])
    hi=np.isin(groups,[9,10,11,12,13])
    unseenish=(groups==13)
    return {
        'finite_monotone': bool(finite.all()),
        'overall_cov': float(cov.mean()),
        'worst_group_cov': float(min(v for g,v in group_cov.items() if (groups==g).sum()>=40)),
        'tiny_cov': float(cov[tiny].mean()),
        'hi_noise_cov': float(cov[hi].mean()),
        'unseen_cov': float(cov[unseenish].mean()),
        'mean_width': float(np.mean(width)),
        'median_width': float(np.median(width)),
        'p90_width': float(np.quantile(width,0.9)),
        'max_width': float(np.max(width)),
    }


def _passes(m):
    return (
        m['finite_monotone']
        and m['overall_cov'] >= 0.880
        and m['worst_group_cov'] >= 0.800
        and m['tiny_cov'] >= 0.820
        and m['hi_noise_cov'] >= 0.840
        and m['unseen_cov'] >= 0.800
        and m['mean_width'] <= 7.80
        and m['median_width'] <= 7.40
        and m['p90_width'] <= 16.20
    )


def _single_seed(solution_module, seed):
    d=dataset(seed)
    params=solution_module.fit_interval_model(
        d['train_X'], d['train_y'], d['train_groups'],
        d['calib_X'], d['calib_y'], d['calib_groups'],
    )
    lo,hi=solution_module.predict_interval(d['hidden_X'], d['hidden_groups'], params)
    m=_metrics(d['hidden_y'], lo, hi, d['hidden_groups'])

    # Determinism and permutation consistency on a nontrivial slice.
    idx=np.arange(80)
    lo2,hi2=solution_module.predict_interval(d['hidden_X'][idx], d['hidden_groups'][idx], params)
    lo3,hi3=solution_module.predict_interval(d['hidden_X'][idx], d['hidden_groups'][idx], params)
    perm=np.array([17,3,44,0,5,29,12,7,70,9,50,1])
    lop,hip=solution_module.predict_interval(d['hidden_X'][idx][perm], d['hidden_groups'][idx][perm], params)
    stable = np.allclose(lo2,lo3,atol=1e-12) and np.allclose(hi2,hi3,atol=1e-12)
    stable = stable and np.allclose(lop,lo2[perm],atol=1e-12) and np.allclose(hip,hi2[perm],atol=1e-12)
    m['deterministic_permutation'] = bool(stable)
    m['passed'] = bool(_passes(m) and stable)
    return m


def evaluate(solution_module, seeds=(101,102,103,104,105)):
    t0=time.perf_counter()
    per_seed={int(seed): _single_seed(solution_module, seed) for seed in seeds}
    keys=['overall_cov','worst_group_cov','tiny_cov','hi_noise_cov','unseen_cov','mean_width','median_width','p90_width']
    aggregate={k: float(np.mean([v[k] for v in per_seed.values()])) for k in keys}
    return {
        'passed': bool(all(v['passed'] for v in per_seed.values())),
        'aggregate': aggregate,
        'per_seed': per_seed,
        'elapsed_seconds': time.perf_counter()-t0,
    }


if __name__ == '__main__':
    import importlib, sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
    solution=importlib.import_module('solve')
    print(json.dumps(evaluate(solution), indent=2))

from __future__ import annotations
import numpy as np

N_GROUPS = 14
D = 10
GROUP_BIAS = np.array([-1.1, -0.7, -0.25, 0.15, 0.55, 0.95, -1.45, -0.95, -0.35, 0.25, 0.85, 1.35, -1.75, 1.75], dtype=float)
BASE_SIGMA = np.array([0.28, 0.36, 0.44, 0.55, 0.70, 0.85, 1.05, 1.28, 1.55, 1.95, 2.35, 2.85, 3.20, 3.65], dtype=float)
# Feature centers make groups have slightly different x distributions; this helps infer weak-calib groups from train.
GROUP_X_SHIFT = np.array([
    [-0.5,  0.2, 0.0], [-0.3,  0.1, 0.2], [-0.1, -0.1, 0.0], [0.1, -0.2, 0.1],
    [ 0.3,  0.1,-0.1], [ 0.5,  0.2, 0.0], [-0.7, 0.5, 0.3], [-0.4, 0.4,-0.2],
    [-0.2, -0.5,0.1], [0.2,-0.4,-0.2], [0.5,-0.3,0.3], [0.7,0.3,-0.3],
    [-0.9, -0.2,0.4], [0.9,0.2,-0.4]
], dtype=float)

def _sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def shared_signal(X: np.ndarray) -> np.ndarray:
    X=np.asarray(X)
    return (
        1.2*X[:,0] - 0.75*X[:,1] + 0.45*X[:,2]
        + 0.65*np.sin(1.2*X[:,3])
        - 0.45*np.cos(0.8*X[:,4])
        + 0.35*X[:,0]*X[:,5]
        - 0.25*X[:,6]*X[:,7]
        + 0.20*(X[:,8]**2 - 1.0)
        - 0.15*X[:,9]**2
    )

def sample_X(rng: np.random.Generator, groups: np.ndarray) -> np.ndarray:
    n=len(groups)
    X=rng.normal(0,1,size=(n,D))
    shifts = GROUP_X_SHIFT[groups]
    X[:,:3] += shifts
    # Group-specific mild feature shape; not enough to solve by features only.
    X[:,3] += 0.15*np.sin(groups)
    return X

def make_y(rng: np.random.Generator, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    groups=np.asarray(groups, dtype=int)
    sigma = BASE_SIGMA[groups] * (0.70 + 0.60*_sigmoid(1.0*X[:,0] - 0.6*X[:,1] + 0.3*X[:,2]))
    n=len(groups)
    # Mild skew + heavy tail. Centered enough that point prediction is still learnable.
    normal = rng.normal(0, 1, n)
    t3 = rng.standard_t(df=3, size=n) / np.sqrt(3.0)
    skew = rng.exponential(1.0, n) - 1.0
    choose = rng.random(n)
    eps = np.where(choose < 0.78, normal, np.where(choose < 0.92, t3*1.6, skew*1.35))
    return shared_signal(X) + GROUP_BIAS[groups] + sigma*eps

def sample_groups(rng: np.random.Generator, n: int, probs: np.ndarray) -> np.ndarray:
    return rng.choice(np.arange(N_GROUPS), size=n, p=np.asarray(probs, dtype=float)/np.sum(probs))

def dataset(seed=0, n_train=2800, n_calib=900, n_public=800, n_hidden=2200):
    rng=np.random.default_rng(seed)
    # Train has all groups, but high-noise rare groups have few samples.
    p_train=np.array([0.12,0.11,0.105,0.10,0.09,0.08,0.075,0.065,0.055,0.045,0.035,0.025,0.015,0.01])
    # Calibration is even more imbalanced: groups 12/13 are weak or missing.
    p_calib=np.array([0.15,0.13,0.12,0.11,0.095,0.08,0.065,0.055,0.045,0.035,0.025,0.015,0.006,0.0])
    p_public=np.array([0.13,0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.055,0.045,0.035,0.025,0.015,0.005])
    # Hidden stresses weak/high-noise groups more than calib/public.
    p_hidden=np.array([0.055,0.055,0.055,0.055,0.06,0.06,0.075,0.075,0.085,0.095,0.105,0.115,0.105,0.105])
    train_g=sample_groups(rng,n_train,p_train)
    calib_g=sample_groups(rng,n_calib,p_calib)
    public_g=sample_groups(rng,n_public,p_public)
    hidden_g=sample_groups(rng,n_hidden,p_hidden)
    train_X=sample_X(rng, train_g); calib_X=sample_X(rng, calib_g)
    public_X=sample_X(rng, public_g); hidden_X=sample_X(rng, hidden_g)
    train_y=make_y(rng,train_X,train_g); calib_y=make_y(rng,calib_X,calib_g)
    public_y=make_y(rng,public_X,public_g); hidden_y=make_y(rng,hidden_X,hidden_g)
    return dict(train_X=train_X, train_y=train_y, train_groups=train_g,
                calib_X=calib_X, calib_y=calib_y, calib_groups=calib_g,
                public_X=public_X, public_y=public_y, public_groups=public_g,
                hidden_X=hidden_X, hidden_y=hidden_y, hidden_groups=hidden_g)

if __name__=='__main__':
    data=dataset(42)
    for k,v in data.items(): print(k, v.shape if hasattr(v,'shape') else None)

