# Fix activation scoring

`app/solve.py` has the two stubs to fill in:

```python
fit_activation_model(train_X, train_observed, train_age_days, train_segment)
predict_activation_proba(X, age_days, segment, params)
```

Return one probability per row.

This is a trial activation model from a snapshot. Some rows are old enough that a missing activation is pretty meaningful. Some rows are new, so a missing activation is weaker evidence. Segments do not all settle at the same speed. The target is final activation probability, not just whether activation has already appeared in the snapshot.

The old model treated every missing activation as a normal negative and was badly low on newer rows and a few slow segments. Aim for useful final-label log loss, not just a high score on the snapshot labels.

Use only the arrays passed in. Keep it deterministic. No hidden files, network, outside data, or tuning on public eval answers.



import numpy as np
from solve import fit_activation_model, predict_activation_proba


def _logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))


def _load():
    train = np.load('train_data.npz')
    public = np.load('public_eval.npz')
    return train, public


def test_prediction_shape_range_and_determinism():
    train, public = _load()
    params = fit_activation_model(
        train['train_X'], train['train_observed'], train['train_age_days'], train['train_segment']
    )
    p1 = predict_activation_proba(public['public_X'], public['public_age_days'], public['public_segment'], params)
    p2 = predict_activation_proba(public['public_X'], public['public_age_days'], public['public_segment'], params)
    assert isinstance(p1, np.ndarray)
    assert p1.shape == public['public_final_label'].shape
    assert np.all(np.isfinite(p1))
    assert np.all(p1 >= 0.0) and np.all(p1 <= 1.0)
    np.testing.assert_allclose(p1, p2, atol=1e-12, rtol=1e-12)


def test_permutation_consistency_and_basic_public_quality():
    train, public = _load()
    params = fit_activation_model(
        train['train_X'], train['train_observed'], train['train_age_days'], train['train_segment']
    )
    X = public['public_X']; age = public['public_age_days']; seg = public['public_segment']; y = public['public_final_label']
    p = predict_activation_proba(X, age, seg, params)
    perm = np.arange(len(y))[::-1]
    pp = predict_activation_proba(X[perm], age[perm], seg[perm], params)
    np.testing.assert_allclose(pp, p[perm], atol=1e-12, rtol=1e-12)
    baseline = np.full_like(y, y.mean(), dtype=float)
    assert _logloss(y, p) < _logloss(y, baseline) - 0.025
    recent = age <= 3.0
    if recent.sum() > 20:
        assert np.mean(p[recent]) > 0.12



from __future__ import annotations
import time, json
import numpy as np
from generator import make_hidden


def _logloss(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1-1e-6)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))


def _brier(y, p):
    return float(np.mean((np.asarray(y)-np.asarray(p))**2))


def _auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos = y == 1
    neg = ~pos
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(p), dtype=float) + 1
    # average ties
    vals = p[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return float((ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg))


def _ece_by_segment(y, p, seg):
    y = np.asarray(y)
    p = np.asarray(p)
    seg = np.asarray(seg)
    errs = []
    weights = []
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < 30:
            continue
        errs.append(abs(float(y[m].mean() - p[m].mean())))
        weights.append(m.sum())
    if not errs:
        return 1.0
    return float(np.average(errs, weights=weights))


def _reward_decreasing(x, good, cutoff):
    if not np.isfinite(x) or x >= cutoff:
        return 0.0
    if x <= good:
        return 1.0
    return float((cutoff - x) / (cutoff - good))


def _reward_increasing(x, good, cutoff):
    if not np.isfinite(x) or x <= cutoff:
        return 0.0
    if x >= good:
        return 1.0
    return float((x - cutoff) / (good - cutoff))


def evaluate_one(fit_fn, predict_fn, train_data, seed):
    hidden = make_hidden(seed)
    params = fit_fn(
        train_data['train_X'], train_data['train_observed'],
        train_data['train_age_days'], train_data['train_segment']
    )
    p = predict_fn(hidden['X'], hidden['age_days'], hidden['segment'], params)
    p = np.asarray(p, dtype=np.float64)
    y = hidden['final_label']
    seg = hidden['segment']
    age = hidden['age_days']
    # Basic validity and permutation consistency.
    valid = (p.shape == y.shape and np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= 1))
    perm_ok = False
    if valid:
        rng = np.random.default_rng(991 + seed)
        perm = rng.permutation(len(y))
        p2 = np.asarray(predict_fn(hidden['X'][perm], hidden['age_days'][perm], hidden['segment'][perm], params), dtype=np.float64)
        perm_ok = p2.shape == p[perm].shape and np.allclose(p2, p[perm], atol=1e-12, rtol=1e-12)
    recent = age <= 3.0
    mature = age >= 14.0
    slow = np.isin(seg, [5,6,8,9])
    rare = seg == 8
    fast = np.isin(seg, [0,1])
    high_intent_recent = recent & (hidden['p_final'] >= np.quantile(hidden['p_final'], 0.65))
    metrics = {
        'valid': bool(valid),
        'permutation_consistent': bool(perm_ok),
        'overall_logloss': _logloss(y, p) if valid else np.inf,
        'brier': _brier(y, p) if valid else np.inf,
        'auc': _auc(y, p) if valid else 0.0,
        'recent_logloss': _logloss(y[recent], p[recent]) if valid and recent.sum() else np.inf,
        'high_intent_recent_logloss': _logloss(y[high_intent_recent], p[high_intent_recent]) if valid and high_intent_recent.sum() else np.inf,
        'slow_logloss': _logloss(y[slow], p[slow]) if valid and slow.sum() else np.inf,
        'rare_logloss': _logloss(y[rare], p[rare]) if valid and rare.sum() else np.inf,
        'mature_logloss': _logloss(y[mature], p[mature]) if valid and mature.sum() else np.inf,
        'fast_logloss': _logloss(y[fast], p[fast]) if valid and fast.sum() else np.inf,
        'segment_ece': _ece_by_segment(y, p, seg) if valid else np.inf,
        'mean_pred_recent': float(np.mean(p[recent])) if valid and recent.sum() else np.nan,
        'mean_pred_mature': float(np.mean(p[mature])) if valid and mature.sum() else np.nan,
    }
    hard_zero = (
        not valid or not perm_ok
        or metrics['overall_logloss'] > 0.78
        or metrics['recent_logloss'] > 0.86
        or metrics['slow_logloss'] > 0.86
        or metrics['segment_ece'] > 0.20
        or metrics['auc'] < 0.58
    )
    comps = {
        'overall': _reward_decreasing(metrics['overall_logloss'], 0.545, 0.70),
        'recent': _reward_decreasing(metrics['recent_logloss'], 0.525, 0.80),
        'high_intent_recent': _reward_decreasing(metrics['high_intent_recent_logloss'], 0.375, 0.78),
        'slow': _reward_decreasing(metrics['slow_logloss'], 0.520, 0.80),
        'rare': _reward_decreasing(metrics['rare_logloss'], 0.520, 0.88),
        'calibration': _reward_decreasing(metrics['segment_ece'], 0.035, 0.16),
        'auc': _reward_increasing(metrics['auc'], 0.785, 0.62),
        'brier': _reward_decreasing(metrics['brier'], 0.182, 0.245),
    }
    reward = 0.0 if hard_zero else float(
        0.24*comps['overall'] +
        0.16*comps['recent'] +
        0.14*comps['high_intent_recent'] +
        0.14*comps['slow'] +
        0.08*comps['rare'] +
        0.10*comps['calibration'] +
        0.08*comps['auc'] +
        0.06*comps['brier']
    )
    metrics['reward'] = reward
    metrics['component_rewards'] = comps
    metrics['hard_zero'] = bool(hard_zero)
    return metrics


def evaluate(fit_fn, predict_fn, train_npz_path='../app/train_data.npz', seeds=(1,2,3,4,5)):
    t0 = time.perf_counter()
    train_data = dict(np.load(train_npz_path))
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = evaluate_one(fit_fn, predict_fn, train_data, int(seed))
    keys = [k for k,v in next(iter(per_seed.values())).items() if isinstance(v, (float, int, np.floating, np.integer)) and k not in ['reward']]
    aggregate = {}
    for k in keys:
        vals = [per_seed[str(s)][k] for s in seeds]
        aggregate[k] = float(np.mean(vals))
    reward = float(np.mean([per_seed[str(s)]['reward'] for s in seeds]))
    # Keep a hard cutoff outside the continuous reward.
    if any(per_seed[str(s)]['hard_zero'] for s in seeds):
        reward = min(reward, 0.25)
    passed_cutoff = bool(reward > 0.0)
    return {
        'reward': reward,
        'passed_cutoff': passed_cutoff,
        'aggregate': aggregate,
        'per_seed': per_seed,
        'elapsed_seconds': time.perf_counter() - t0,
    }




from __future__ import annotations
import numpy as np


def _sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _features(X, segment, mu, sd, n_segments=10):
    Xs = (np.asarray(X, dtype=np.float64) - mu) / sd
    seg = np.asarray(segment, dtype=np.int64)
    n = Xs.shape[0]
    oh = np.zeros((n, n_segments), dtype=np.float64)
    ok = (seg >= 0) & (seg < n_segments)
    oh[np.arange(n)[ok], seg[ok]] = 1.0
    parts = [
        np.ones((n,1)),
        Xs,
        Xs[:, :6] ** 2,
        np.sin(Xs[:, :4]),
        np.cos(Xs[:, :4]),
        (Xs[:, [0]] * Xs[:, [1]]),
        (Xs[:, [2]] * Xs[:, [3]]),
        (Xs[:, [4]] * Xs[:, [5]]),
        oh,
        oh[:, 5:10] * Xs[:, [7]],
        oh[:, 0:2] * Xs[:, [8]],
    ]
    return np.hstack(parts)


def _logistic_ridge(Phi, y, sample_weight=None, lam=2.0, max_iter=40):
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = Phi.shape
    if sample_weight is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
    sw = np.clip(sw, 1e-4, 100.0)
    beta = np.zeros(d, dtype=np.float64)
    # intercept initialization
    ybar = np.average(np.clip(y, 1e-4, 1-1e-4), weights=sw)
    beta[0] = np.log(ybar / (1 - ybar))
    reg = np.ones(d) * lam
    reg[0] = 1e-6
    for _ in range(max_iter):
        z = Phi @ beta
        p = _sigmoid(z)
        r = (p - y) * sw
        grad = (Phi.T @ r) / max(1.0, sw.sum()) + reg * beta / n
        W = p * (1 - p) * sw
        H = (Phi.T * W) @ Phi / max(1.0, sw.sum())
        H.flat[::d+1] += reg / n + 1e-6
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # modest line search
        old_loss = _weighted_logloss(y, p, sw) + 0.5 * np.sum(reg * beta * beta) / n
        alpha = 1.0
        for _ls in range(12):
            nb = beta - alpha * step
            npred = _sigmoid(Phi @ nb)
            loss = _weighted_logloss(y, npred, sw) + 0.5 * np.sum(reg * nb * nb) / n
            if np.isfinite(loss) and loss <= old_loss + 1e-8:
                beta = nb
                break
            alpha *= 0.5
        if np.linalg.norm(alpha * step) < 1e-5:
            break
    return beta


def _weighted_logloss(y, p, w):
    p = np.clip(p, 1e-6, 1-1e-6)
    return float(np.sum(w * (-(y*np.log(p) + (1-y)*np.log(1-p)))) / max(1e-12, np.sum(w)))


def _estimate_arrival(age, observed, segment, max_seg=10):
    age = np.asarray(age, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.int64)
    mature = age >= 16.0
    global_mature = (observed[mature].mean() if mature.any() else observed.mean())
    global_mature = float(np.clip(global_mature, 0.05, 0.95))
    # observed rate by age bin divided by mature observed rate, smoothed.
    bins = np.array([0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0, 30.0])
    curves = np.zeros((max_seg, len(bins)-1), dtype=np.float64)
    counts = np.zeros_like(curves)
    global_curve = []
    for b in range(len(bins)-1):
        mask = (age >= bins[b]) & (age < bins[b+1])
        rate = observed[mask].mean() if mask.any() else global_mature
        global_curve.append(np.clip(rate / global_mature, 0.05, 1.0))
    global_curve = np.maximum.accumulate(np.asarray(global_curve))
    global_curve[-1] = 1.0
    for s in range(max_seg):
        sm = segment == s
        mat_s = sm & mature
        mature_rate = observed[mat_s].mean() if mat_s.sum() >= 25 else global_mature
        mature_rate = float(np.clip(0.65*mature_rate + 0.35*global_mature, 0.04, 0.95))
        vals = []
        for b in range(len(bins)-1):
            mask = sm & (age >= bins[b]) & (age < bins[b+1])
            counts[s,b] = mask.sum()
            if mask.sum() >= 20:
                val = observed[mask].mean() / mature_rate
                # More shrinkage for small bins.
                w = mask.sum() / (mask.sum() + 60.0)
                val = w * val + (1-w) * global_curve[b]
            else:
                val = global_curve[b]
            vals.append(np.clip(val, 0.04, 1.0))
        vals = np.maximum.accumulate(vals)
        vals[-1] = 1.0
        curves[s] = vals
    return {'bins': bins, 'curves': curves, 'global_curve': global_curve}


def _arrival_fraction(age, segment, arr):
    age = np.asarray(age, dtype=np.float64)
    seg = np.asarray(segment, dtype=np.int64)
    bins = arr['bins']
    idx = np.searchsorted(bins[1:], age, side='right')
    idx = np.clip(idx, 0, len(bins)-2)
    out = np.empty(len(age), dtype=np.float64)
    ok = (seg >= 0) & (seg < arr['curves'].shape[0])
    out[ok] = arr['curves'][seg[ok], idx[ok]]
    out[~ok] = arr['global_curve'][idx[~ok]]
    # Smooth interpolation within bin
    lo = bins[idx]
    hi = bins[idx+1]
    frac = np.clip((age - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    prev_idx = np.clip(idx-1, 0, len(bins)-2)
    base = np.empty_like(out)
    base[ok] = arr['curves'][seg[ok], prev_idx[ok]]
    base[~ok] = arr['global_curve'][prev_idx[~ok]]
    out = base * (1-frac) + out * frac
    return np.clip(out, 0.04, 1.0)


def fit_activation_model(train_X, train_observed, train_age_days, train_segment):
    X = np.asarray(train_X, dtype=np.float64)
    obs = np.asarray(train_observed, dtype=np.float64)
    age = np.asarray(train_age_days, dtype=np.float64)
    seg = np.asarray(train_segment, dtype=np.int64)
    mu, sd = _standardize_fit(X)
    Phi = _features(X, seg, mu, sd)
    arr = _estimate_arrival(age, obs, seg)
    F = _arrival_fraction(age, seg, arr)
    mature = age >= 15.0
    if mature.sum() < 300:
        mature = age >= np.quantile(age, 0.65)

    # EM-style fit for final activation.  A row with observed=0 is ambiguous;
    # the ambiguity is larger when the row is young or in a slow segment.
    # Start from a mostly-mature model, then repeatedly turn recent missing rows
    # into soft labels using the arrival curve.
    w = np.where(mature, 1.0, 0.18)
    beta = _logistic_ridge(Phi, obs, w, lam=4.0, max_iter=35)
    for it in range(5):
        p = _sigmoid(Phi @ beta)
        denom = np.clip(1.0 - p * F, 0.06, 1.0)
        y_soft = np.where(obs > 0.5, 1.0, p * (1.0 - F) / denom)
        y_soft = np.clip(y_soft, 0.01, 0.99)
        # Observed positives in low-arrival regions are rare but valuable evidence.
        pos_boost = np.where(obs > 0.5, np.clip(1.0 / np.sqrt(F), 1.0, 3.0), 1.0)
        sw = np.where(obs > 0.5, 1.0, 0.28 + 0.72 * F) * pos_boost
        sw *= np.where(mature, 1.05, 1.0)
        beta = _logistic_ridge(Phi, y_soft, sw, lam=2.0 + 0.2*it, max_iter=35)

    p1 = _sigmoid(Phi @ beta)
    # Segment-level mature calibration/shrinkage on the final model.
    offsets = np.zeros(10, dtype=np.float64)
    global_offset = 0.0
    if mature.sum() > 50:
        r0 = np.clip(obs[mature].mean(), 1e-3, 1-1e-3)
        q0 = np.clip(p1[mature].mean(), 1e-3, 1-1e-3)
        global_offset = np.log(r0/(1-r0)) - np.log(q0/(1-q0))
    for s in range(10):
        mask = mature & (seg == s)
        if mask.sum() >= 35:
            r = np.clip(obs[mask].mean(), 1e-3, 1-1e-3)
            q = np.clip(p1[mask].mean(), 1e-3, 1-1e-3)
            off = np.log(r/(1-r)) - np.log(q/(1-q))
            shrink = mask.sum() / (mask.sum() + 100.0)
            offsets[s] = shrink * off + (1-shrink) * global_offset
        else:
            offsets[s] = global_offset
    return {'mu': mu, 'sd': sd, 'beta': beta, 'offsets': offsets, 'n_segments': 10}

def predict_activation_proba(X, age_days, segment, params):
    X = np.asarray(X, dtype=np.float64)
    seg = np.asarray(segment, dtype=np.int64)
    Phi = _features(X, seg, params['mu'], params['sd'], params.get('n_segments', 10))
    z = Phi @ params['beta']
    off = np.zeros(len(seg), dtype=np.float64)
    ok = (seg >= 0) & (seg < len(params['offsets']))
    off[ok] = params['offsets'][seg[ok]]
    p = _sigmoid(z + 0.6 * off)
    return np.clip(p, 1e-5, 1 - 1e-5)



from __future__ import annotations
import numpy as np

SEGMENTS = 10
DIM = 12

SEG_BIAS = np.array([-0.55, -0.20, 0.15, 0.35, -0.05, 0.25, 0.55, -0.35, 0.65, 0.45])
# Larger means slower completion among eventual activations.
SEG_DELAY_SCALE = np.array([0.65, 0.85, 1.15, 1.35, 1.00, 1.85, 2.30, 1.60, 2.85, 2.15])
SEG_SHAPE = np.array([1.55, 1.45, 1.25, 1.15, 1.35, 1.05, 0.95, 1.10, 0.90, 1.00])


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _sample_segments(rng, n, mode):
    if mode == 'train':
        p = np.array([0.12,0.11,0.11,0.10,0.09,0.11,0.10,0.08,0.09,0.09])
    elif mode == 'public':
        p = np.array([0.14,0.12,0.13,0.12,0.11,0.10,0.08,0.08,0.03,0.09])
    elif mode == 'hidden':
        p = np.array([0.07,0.08,0.09,0.09,0.07,0.14,0.14,0.07,0.16,0.09])
    else:
        p = np.ones(SEGMENTS) / SEGMENTS
    p = p / p.sum()
    return rng.choice(SEGMENTS, size=n, p=p)


def _sample_X(rng, segments, mode):
    n = len(segments)
    X = rng.normal(0, 1, size=(n, DIM))
    # Segment-dependent feature shifts. Slow segments are slightly more enterprise-like.
    X[:, 7] += np.where(np.isin(segments, [5,6,8,9]), rng.normal(0.85, 0.35, n), rng.normal(-0.15, 0.25, n))
    X[:, 8] += np.where(np.isin(segments, [0,1]), rng.normal(0.55, 0.25, n), rng.normal(0.0, 0.35, n))
    X[:, 2] += 0.18 * (segments - 4.5) / 4.5
    X[:, 9] += rng.normal(0, 0.45, n)
    if mode == 'hidden':
        # More high-intent slow users and more recent-looking feature mix.
        mask = np.isin(segments, [5,6,8,9])
        X[mask, 0] += 0.35
        X[mask, 7] += 0.25
        X[:, 10] += rng.normal(0.35, 0.30, n)
    return X.astype(np.float64)


def final_propensity(X, segments):
    z = (
        -0.28
        + 0.78 * X[:, 0]
        - 0.58 * X[:, 1]
        + 0.36 * np.sin(1.15 * X[:, 2])
        + 0.30 * X[:, 3] * X[:, 4]
        - 0.24 * (X[:, 5] ** 2 - 1.0)
        + 0.32 * (X[:, 6] > 0.25)
        + 0.18 * X[:, 8]
        - 0.15 * X[:, 9]
        + 0.55 * X[:, 10]
        + SEG_BIAS[segments]
    )
    # Segment-specific intent interactions.
    z += np.where(np.isin(segments, [5,6,8,9]), 0.28 * X[:, 7] + 0.22 * X[:, 10] - 0.10 * X[:, 8], 0.05 * X[:, 8])
    z += np.where(np.isin(segments, [0,1]), 0.18 * X[:, 8], 0.0)
    return sigmoid(z)


def delay_scale(X, segments):
    # Enterprise-like users are often valuable but slower to activate.
    mult = np.exp(
        0.30 * X[:, 7]
        - 0.28 * X[:, 8]
        + 0.30 * np.maximum(X[:, 10], -1.5)
        + 0.12 * np.sin(X[:, 2])
    )
    return np.clip(SEG_DELAY_SCALE[segments] * mult, 0.25, 8.0)


def _sample_delay(rng, X, segments):
    # Weibull-like days, with a small long-tail bump in slow segments.
    scale = delay_scale(X, segments)
    shape = SEG_SHAPE[segments]
    u = np.clip(rng.random(len(segments)), 1e-6, 1-1e-6)
    delay = scale * (-np.log(1 - u)) ** (1.0 / shape)
    tail = (rng.random(len(segments)) < np.where(np.isin(segments, [6,8,9]), 0.12, 0.045))
    delay[tail] += rng.gamma(shape=2.0, scale=np.where(np.isin(segments[tail], [6,8,9]), 2.2, 1.0), size=tail.sum())
    return np.clip(delay, 0.05, 30.0)


def _sample_age(rng, segments, X, mode):
    n = len(segments)
    age = np.empty(n, dtype=np.float64)
    for s in range(SEGMENTS):
        idx = np.where(segments == s)[0]
        if idx.size == 0:
            continue
        m = idx.size
        # Cohort mix is not random: high-intent enterprise-like rows are more common
        # in newer snapshots, especially for slow segments.
        risk = 0.55 * X[idx, 7] + 0.30 * X[idx, 10] + 0.20 * X[idx, 0] - 0.20 * X[idx, 8]
        risk = 1.0 / (1.0 + np.exp(-risk))
        if mode == 'train':
            recent_base = 0.23 + (0.18 if s in [5,6,8,9] else 0.0)
            mature_base = 0.33 - (0.10 if s in [5,6,8] else 0.0)
            if s == 8:
                recent_base = 0.68
                mature_base = 0.025
        elif mode == 'public':
            recent_base = 0.16 + (0.08 if s in [5,6,8,9] else 0.0)
            mature_base = 0.46
        else:  # hidden
            recent_base = 0.39 + (0.13 if s in [5,6,8,9] else 0.0)
            mature_base = 0.20
            if s == 8:
                recent_base = 0.74
                mature_base = 0.08
        recent_prob = np.clip(recent_base + (0.23 if s in [5,6,8,9] else 0.13) * (risk - 0.5), 0.08, 0.78)
        mature_prob = np.clip(mature_base - (0.24 if s in [5,6,8,9] else 0.12) * (risk - 0.5), 0.08, 0.70)
        # Keep probabilities ordered.
        total = recent_prob + mature_prob
        over = total > 0.86
        if np.any(over):
            recent_prob[over] *= 0.86 / total[over]
            mature_prob[over] *= 0.86 / total[over]
        r = rng.random(m)
        a = np.empty(m)
        rec = r < recent_prob
        mat = r > 1 - mature_prob
        mid = ~(rec | mat)
        a[rec] = rng.uniform(0.2, 3.0, rec.sum())
        a[mid] = rng.uniform(3.0, 9.0, mid.sum())
        a[mat] = rng.uniform(12.0, 24.0, mat.sum())
        if s == 8 and mode in ['train','hidden']:
            a += rng.normal(-0.7 if mode == 'hidden' else -0.35, 0.25, m)
        age[idx] = np.clip(a, 0.05, 28.0)
    return age

def make_dataset(seed, n, mode='train'):
    rng = np.random.default_rng(seed)
    seg = _sample_segments(rng, n, mode)
    X = _sample_X(rng, seg, mode)
    p = final_propensity(X, seg)
    final = rng.random(n) < p
    delay = np.full(n, np.inf, dtype=np.float64)
    if final.any():
        delay[final] = _sample_delay(rng, X[final], seg[final])
    age = _sample_age(rng, seg, X, mode)
    observed = final & (delay <= age)
    return {
        'X': X.astype(np.float64),
        'segment': seg.astype(np.int64),
        'age_days': age.astype(np.float64),
        'observed': observed.astype(np.int64),
        'final_label': final.astype(np.int64),
        'delay_days': delay.astype(np.float64),
        'p_final': p.astype(np.float64),
    }


def make_visible_files(out_dir):
    out_dir = str(out_dir)
    train = make_dataset(1201, 3600, 'train')
    public = make_dataset(2201, 1000, 'public')
    np.savez(
        out_dir + '/train_data.npz',
        train_X=train['X'],
        train_observed=train['observed'],
        train_age_days=train['age_days'],
        train_segment=train['segment'],
    )
    np.savez(
        out_dir + '/public_eval.npz',
        public_X=public['X'],
        public_age_days=public['age_days'],
        public_segment=public['segment'],
        public_final_label=public['final_label'],
        public_observed=public['observed'],
    )


def make_hidden(seed):
    return make_dataset(5000 + int(seed), 5000, 'hidden')


