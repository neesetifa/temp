# Fix settlement logs

`app/solve.py` has two stubs:

```python
fit_settlement_model(
    train_event_time,
    train_event_code,
    train_event_amount,
    train_event_source,
    train_case_offsets,
    train_y,
)

predict_settlement(
    event_time,
    event_code,
    event_amount,
    event_source,
    case_offsets,
    params,
)
```

Return one settled value per case.

The input rows are packed case histories. A case can have several updates from different sources. The amount field is useful, but old rows are not all the same kind of signal: some rows are revisions, some are late updates, and some are repeated or stale.

The old version used the latest amount per case. It looked fine on short clean histories, but it missed corrections and cases where two sources disagreed.

Use the arrays passed in. No hidden files, network, outside data, or fitting on public eval answers.


  
import importlib
import sys
from pathlib import Path
import numpy as np
try:
    from private.generators import generate_dataset
    from private.model_utils import rmse
except Exception:
    from generators import generate_dataset
    from model_utils import rmse


def _load_train(train_path):
    d=np.load(train_path)
    return (
        d["train_event_time"], d["train_event_code"], d["train_event_amount"],
        d["train_event_source"], d["train_case_offsets"], d["train_y"]
    )


def _metric_reward(value, good, cutoff):
    value=float(value)
    if not np.isfinite(value) or value >= cutoff:
        return 0.0
    if value <= good:
        return 1.0
    return float(np.log(cutoff/value)/np.log(cutoff/good))


def _slice_rmse(y,p,mask):
    mask=np.asarray(mask,bool)
    if mask.sum()==0:
        return float('nan')
    return rmse(y[mask], p[mask])


def evaluate(fit_fn, predict_fn, train_path=None):
    if train_path is None:
        train_path = Path(__file__).resolve().parents[1] / "app" / "train_data.npz"
    train_args=_load_train(train_path)
    params=fit_fn(*train_args)
    seeds=[1101, 1102, 1103, 1104]
    profiles=['mixed','stress','mixed','stress']
    rows=[]
    for seed,profile in zip(seeds,profiles):
        d=generate_dataset(1100, seed=seed, split='hidden', profile=profile)
        pred=predict_fn(d['event_time'], d['event_code'], d['event_amount'], d['event_source'], d['case_offsets'], params)
        pred=np.asarray(pred,float)
        y=d['y']
        if pred.shape != y.shape or not np.all(np.isfinite(pred)):
            return {"reward":0.0,"passed_cutoff":False,"error":"bad prediction shape or non-finite"}
        rows.append((d,y,pred))
    y_all=np.concatenate([r[1] for r in rows])
    p_all=np.concatenate([r[2] for r in rows])
    meta={}
    for name in ['correction','duplicate','stale','rare_late','source_disagreement','short_history','long_history','high_value']:
        meta[name]=np.concatenate([r[0]['meta_'+name] for r in rows])
    m={
        'overall_rmse': rmse(y_all,p_all),
        'correction_rmse': _slice_rmse(y_all,p_all,meta['correction']),
        'duplicate_rmse': _slice_rmse(y_all,p_all,meta['duplicate']),
        'stale_rmse': _slice_rmse(y_all,p_all,meta['stale']),
        'rare_late_rmse': _slice_rmse(y_all,p_all,meta['rare_late']),
        'source_disagreement_rmse': _slice_rmse(y_all,p_all,meta['source_disagreement']),
        'short_history_rmse': _slice_rmse(y_all,p_all,meta['short_history']),
        'long_history_rmse': _slice_rmse(y_all,p_all,meta['long_history']),
        'high_value_rmse': _slice_rmse(y_all,p_all,meta['high_value']),
    }
    # hard cutoffs for bad/exploding solutions
    if m['overall_rmse'] > 22.0 or m['correction_rmse'] > 26.0 or m['rare_late_rmse'] > 28.0 or m['stale_rmse'] > 28.0:
        return {"reward":0.0,"passed_cutoff":False,"aggregate_metrics":m}
    specs={
        'overall_rmse': (9.25, 18.5),
        'correction_rmse': (9.40, 20.5),
        'duplicate_rmse': (9.20, 20.0),
        'stale_rmse': (8.50, 19.0),
        'rare_late_rmse': (9.75, 21.0),
        'source_disagreement_rmse': (9.25, 20.0),
        'long_history_rmse': (8.70, 19.0),
        'high_value_rmse': (11.10, 24.0),
    }
    weights={
        'overall_rmse':0.22,
        'correction_rmse':0.17,
        'duplicate_rmse':0.12,
        'stale_rmse':0.14,
        'rare_late_rmse':0.14,
        'source_disagreement_rmse':0.09,
        'long_history_rmse':0.07,
        'high_value_rmse':0.05,
    }
    comps={k:_metric_reward(m[k], *specs[k]) for k in specs}
    reward=sum(weights[k]*comps[k] for k in weights)
    # cap solutions that are fine on clean/short rows but fail lifecycle rows
    if m['short_history_rmse'] < 7.5 and (m['correction_rmse'] > 13.5 or m['stale_rmse'] > 13.5):
        reward=min(reward,0.70)
    if m['rare_late_rmse'] > 15.0:
        reward=min(reward,0.78)
    return {
        "reward": float(reward),
        "passed_cutoff": bool(reward >= 0.80),
        "aggregate_metrics": m,
        "aggregate_component_rewards": comps,
    }


def evaluate_solution(solution_module=None):
    app_dir=Path(__file__).resolve().parents[1] / 'app'
    train_path=app_dir / 'train_data.npz'
    if solution_module is None:
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        solution_module=importlib.import_module('solve')
    return evaluate(solution_module.fit_settlement_model, solution_module.predict_settlement, train_path=train_path)


import numpy as np
N_CODES = 8
N_SOURCES = 5


def case_iter(offsets):
    for i in range(len(offsets)-1):
        yield int(offsets[i]), int(offsets[i+1])


def ridge_fit(X, y, alpha=1.0):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig < 1e-8] = 1.0
    Xs = (X - mu) / sig
    A = Xs.T @ Xs + float(alpha) * np.eye(X.shape[1])
    b = Xs.T @ y
    coef = np.linalg.solve(A, b)
    intercept = float(y.mean())
    return {'coef': coef, 'intercept': intercept, 'mu': mu, 'sig': sig}


def ridge_predict(model, X):
    X = np.asarray(X, float)
    return model['intercept'] + ((X - model['mu']) / model['sig']) @ model['coef']


def _safe_stats(vals):
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(vals[-1]), float(np.sum(vals)), float(np.mean(vals)), float(np.max(vals)-np.min(vals))]


def basic_features(time, code, amount, source, offsets):
    rows=[]
    for a,b in case_iter(offsets):
        t=time[a:b]; c=code[a:b].astype(int); x=amount[a:b]; s=source[a:b].astype(int)
        if b<=a:
            rows.append(np.zeros(12)); continue
        order=np.argsort(t)
        x_ord=x[order]; c_ord=c[order]; s_ord=s[order]
        feats=[1.0, len(x), x_ord[-1], x_ord[0], np.mean(x), np.sum(x), np.min(x), np.max(x), np.std(x), np.ptp(t), c_ord[-1], s_ord[-1]]
        rows.append(feats)
    return np.asarray(rows, float)


def code_features(time, code, amount, source, offsets):
    rows=[]
    for a,b in case_iter(offsets):
        t=time[a:b]; c=code[a:b].astype(int); x=amount[a:b]
        order=np.argsort(t); t=t[order]; c=c[order]; x=x[order]
        feats=[1.0, len(x), x[-1] if len(x) else 0.0, x[0] if len(x) else 0.0, np.ptp(t) if len(t)>0 else 0.0]
        for k in range(N_CODES):
            vals=x[c==k]
            feats.append(float(vals.size))
            feats.extend(_safe_stats(vals))
        rows.append(feats)
    return np.asarray(rows, float)


def code_source_features(time, code, amount, source, offsets):
    rows=[]
    for a,b in case_iter(offsets):
        t=time[a:b]; c=code[a:b].astype(int); x=amount[a:b]; s=source[a:b].astype(int)
        order=np.argsort(t); c=c[order]; x=x[order]; s=s[order]
        feats=[1.0, len(x), x[-1] if len(x) else 0.0]
        for k in range(N_CODES):
            vals=x[c==k]
            feats.append(float(vals.size)); feats.append(float(np.sum(vals)) if vals.size else 0.0); feats.append(float(vals[-1]) if vals.size else 0.0)
        for src in range(N_SOURCES):
            vals=x[s==src]
            feats.append(float(vals.size)); feats.append(float(np.mean(vals)) if vals.size else 0.0); feats.append(float(vals[-1]) if vals.size else 0.0)
        for k in range(N_CODES):
            for src in range(N_SOURCES):
                vals=x[(c==k)&(s==src)]
                feats.append(float(vals.size)); feats.append(float(np.sum(vals)) if vals.size else 0.0); feats.append(float(vals[-1]) if vals.size else 0.0)
        rows.append(feats)
    return np.asarray(rows, float)


def semantic_features(time, code, amount, source, offsets):
    rows=[]
    for a,b in case_iter(offsets):
        t=time[a:b]; c=code[a:b].astype(int); x=amount[a:b]; s=source[a:b].astype(int)
        if b<=a:
            rows.append(np.zeros(80)); continue
        order=np.argsort(t); t=t[order]; c=c[order]; x=x[order]; s=s[order]
        # initial/base observation
        idx0=np.where(c==0)[0]
        base = float(x[idx0[0]]) if idx0.size else float(x[0])
        # Ledger reconstruction: code2 delta unless later replaced by code3 of same source; code3 is replacement delta.
        use2=np.ones(len(x), dtype=bool)
        code3_repl=[]
        corrected_old=[]
        for ii in np.where(c==3)[0]:
            prev=np.where((c[:ii]==2)&(s[:ii]==s[ii])&use2[:ii])[0]
            if prev.size:
                jj=prev[-1]
                use2[jj]=False
                corrected_old.append(float(x[jj]))
            code3_repl.append(float(x[ii]))
        sum_valid2=float(np.sum(x[(c==2)&use2]))
        sum_old2=float(np.sum(corrected_old)) if corrected_old else 0.0
        sum_repl3=float(np.sum(code3_repl)) if code3_repl else 0.0
        sum_rare6=float(np.sum(x[c==6]))
        ledger=base + sum_valid2 + sum_repl3 + sum_rare6
        naive_ledger=base + float(np.sum(x[c==2])) + float(np.sum(x[c==3])) + sum_rare6
        # absolute estimates
        latest_nonstale = None
        latest_close = None
        latest_abs = None
        for ii in range(len(x)):
            if c[ii] in (0,1,5,7): latest_abs=float(x[ii])
            if c[ii] in (0,1,7): latest_nonstale=float(x[ii])
            if c[ii]==7: latest_close=float(x[ii])
        if latest_nonstale is None: latest_nonstale=float(x[-1])
        if latest_abs is None: latest_abs=float(x[-1])
        if latest_close is None: latest_close=latest_nonstale
        abs_vals=x[np.isin(c, [0,1,7])]
        stale_vals=x[c==5]
        close_vals=x[c==7]
        # recency and disagreement features
        src_close=[]
        for src in range(N_SOURCES):
            vals=x[(c==7)&(s==src)]
            src_close.append(float(vals[-1]) if vals.size else 0.0)
        disagreement=float(np.std(close_vals)) if close_vals.size>=2 else 0.0
        last_code=int(c[-1]); last_source=int(s[-1])
        feats=[
            1.0, len(x), np.ptp(t), base, ledger, naive_ledger,
            sum_valid2, sum_old2, sum_repl3, sum_rare6,
            latest_nonstale, latest_abs, latest_close,
            float(x[-1]), float(x[0]), float(np.mean(x)), float(np.std(x)),
            float(np.sum(c==3)), float(np.sum(c==4)), float(np.sum(c==5)), float(np.sum(c==6)),
            disagreement, float(close_vals.size), float(stale_vals.size),
            latest_abs-latest_nonstale, ledger-latest_nonstale, ledger-latest_close,
        ]
        feats.extend(src_close)
        # code/source compact summaries
        for k in range(N_CODES):
            vals=x[c==k]
            feats.append(float(vals.size)); feats.append(float(np.sum(vals)) if vals.size else 0.0); feats.append(float(vals[-1]) if vals.size else 0.0)
        for src in range(N_SOURCES):
            vals=x[s==src]
            feats.append(float(vals.size)); feats.append(float(vals[-1]) if vals.size else 0.0)
        # one-hot last code/source
        feats.extend([1.0 if last_code==k else 0.0 for k in range(N_CODES)])
        feats.extend([1.0 if last_source==src else 0.0 for src in range(N_SOURCES)])
        rows.append(feats)
    # pad to rectangular in case we changed counts accidentally
    m=max(len(r) for r in rows)
    out=np.zeros((len(rows), m), float)
    for i,r in enumerate(rows): out[i,:len(r)]=r
    return out


def rmse(y, pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    return float(np.sqrt(np.mean((pred-y)**2)))


import numpy as np
try:
    from private.model_utils import semantic_features, ridge_fit, ridge_predict
except Exception:
    from model_utils import semantic_features, ridge_fit, ridge_predict


def fit_settlement_model(train_event_time, train_event_code, train_event_amount, train_event_source, train_case_offsets, train_y):
    X = semantic_features(train_event_time, train_event_code, train_event_amount, train_event_source, train_case_offsets)
    # modest ridge: semantic reconstruction has noise; train calibration handles source/code biases.
    model = ridge_fit(X, train_y, alpha=12.0)
    return {"model": model}


def predict_settlement(event_time, event_code, event_amount, event_source, case_offsets, params):
    X = semantic_features(event_time, event_code, event_amount, event_source, case_offsets)
    return ridge_predict(params["model"], X)



import numpy as np

N_CODES = 8
N_SOURCES = 5
ABS_CODES = (0, 1, 5, 7)

SRC_ABS_BIAS = np.array([0.0, 2.0, -3.0, 4.5, -0.8])
SRC_ABS_NOISE = np.array([7.0, 12.0, 11.0, 15.0, 6.0])
SRC_DELTA_NOISE = np.array([1.0, 2.0, 1.6, 2.6, 0.8])


def _rng(seed):
    return np.random.default_rng(int(seed))


def _choose_source(rng, kind='any'):
    if kind == 'close':
        return int(rng.choice(5, p=[0.26, 0.08, 0.10, 0.10, 0.46]))
    if kind == 'stale':
        return int(rng.choice(5, p=[0.05, 0.08, 0.78, 0.06, 0.03]))
    if kind == 'delta':
        return int(rng.choice(5, p=[0.34, 0.18, 0.17, 0.24, 0.07]))
    return int(rng.choice(5, p=[0.30, 0.22, 0.18, 0.18, 0.12]))


def _abs_amount(rng, value, code, source, early=0.0):
    code_bias = {0: -0.8, 1: 0.2, 5: -1.3, 7: 0.0}.get(int(code), 0.0)
    noise = SRC_ABS_NOISE[source]
    if code == 7 and source == 4:
        noise *= 0.85
    if code == 7 and source == 3:
        noise *= 1.25
    return float(value + SRC_ABS_BIAS[source] + code_bias + early + rng.normal(0, noise))


def _delta_amount(rng, value, source, scale=1.0):
    return float(value + rng.normal(0, SRC_DELTA_NOISE[source] * scale))


def generate_dataset(n_cases, seed, split='train', profile='mixed'):
    rng = _rng(seed)
    times=[]; codes=[]; amounts=[]; sources=[]; offsets=[0]; y=[]
    meta={k: [] for k in ['correction','duplicate','stale','rare_late','source_disagreement','short_history','long_history','high_value']}

    if split == 'train':
        p_corr, p_dup, p_stale, p_rare, p_dis = 0.44, 0.38, 0.32, 0.22, 0.26
    elif profile == 'stress':
        p_corr, p_dup, p_stale, p_rare, p_dis = 0.60, 0.54, 0.48, 0.34, 0.40
    else:
        p_corr, p_dup, p_stale, p_rare, p_dis = 0.52, 0.48, 0.42, 0.29, 0.35

    for ci in range(int(n_cases)):
        high = rng.random() < (0.16 if split=='train' else 0.22)
        base = 65 + 24 * rng.normal() + (70 + 28*rng.normal() if high else 0)
        base += 8*np.sin(rng.normal())
        current = float(base)
        past_values = [current]
        ev=[]
        t = 0.0
        corr = rng.random() < p_corr
        dup = rng.random() < p_dup
        stale = rng.random() < p_stale
        rare = rng.random() < p_rare
        dis = rng.random() < p_dis
        n_d = int(rng.integers(2, 6 + (1 if high else 0)))
        corr_idx = int(rng.integers(0, n_d)) if corr else -1

        # initial estimate of starting value
        s0 = _choose_source(rng)
        ev.append([t, 0, _abs_amount(rng, current, 0, s0, early=rng.normal(-1.0, 1.0)), s0])
        t += float(rng.uniform(0.5, 1.6))

        correction_done = False
        dup_done = False
        stale_value = current
        for j in range(n_d):
            s = _choose_source(rng, 'delta')
            d_true = float(rng.normal(0, 9.0 + 3.0*high) + rng.choice([-1,1])*rng.exponential(3.5))
            if corr and not correction_done and j == corr_idx:
                # A provisional delta that later gets replaced by code 3.
                old = d_true + rng.normal(22.0 * rng.choice([-1,1]), 8.0)
                ev.append([t, 2, _delta_amount(rng, old, s, 1.15), s])
                t += float(rng.uniform(0.3, 1.4))
                if dup and not dup_done and rng.random() < 0.45:
                    ev.append([t, 4, ev[-1][2] + rng.normal(0, 0.25), s])
                    dup_done = True
                    t += float(rng.uniform(0.2, 0.8))
                # Final ledger uses the replacement amount, not the provisional amount.
                newd = d_true + rng.normal(0, 1.2)
                ev.append([t, 3, _delta_amount(rng, newd, s, 0.95), s])
                current += newd
                correction_done = True
            else:
                ev.append([t, 2, _delta_amount(rng, d_true, s, 1.0), s])
                current += d_true
            past_values.append(current)
            t += float(rng.uniform(0.35, 1.8))
            if rng.random() < 0.25:
                ss = _choose_source(rng)
                ev.append([t, 1, _abs_amount(rng, current, 1, ss), ss])
                t += float(rng.uniform(0.2, 1.2))
            if dup and not dup_done and rng.random() < 0.35:
                # replay the most recent event; not part of ledger
                ev.append([t, 4, ev[-1][2] + rng.normal(0, 0.3), ev[-1][3]])
                dup_done = True
                t += float(rng.uniform(0.2, 0.9))

        if rare:
            s = 4 if rng.random() < 0.68 else _choose_source(rng, 'delta')
            late = float(rng.normal(0, 15.0 + 4.0*high) + rng.choice([-1,1])*rng.exponential(8.0))
            ev.append([t, 6, _delta_amount(rng, late, s, 0.9), s])
            current += late
            past_values.append(current)
            t += float(rng.uniform(0.35, 1.3))

        if stale:
            sv = float(rng.choice(past_values[:-1] if len(past_values)>1 else past_values))
            s = _choose_source(rng, 'stale')
            # Stale snapshots often arrive late.
            ev.append([t + rng.uniform(0.3, 1.7), 5, _abs_amount(rng, sv, 5, s), s])
            t += float(rng.uniform(0.1, 0.7))

        close_prob = 0.38 - 0.12*stale + 0.05*rare
        if rng.random() < close_prob:
            sc = _choose_source(rng, 'close')
            ev.append([t + rng.uniform(0.1, 1.0), 7, _abs_amount(rng, current, 7, sc), sc])
            if dis:
                sd = int(rng.choice([1,2,3]))
                ev.append([t + rng.uniform(0.2, 1.2), 7, _abs_amount(rng, current, 7, sd), sd])

        # Some histories receive a late stale row after closeout, killing latest-row rules.
        if stale and rng.random() < 0.42:
            sv = float(rng.choice(past_values))
            s = _choose_source(rng, 'stale')
            ev.append([t + rng.uniform(1.0, 2.4), 5, _abs_amount(rng, sv, 5, s), s])

        ev = sorted(ev, key=lambda x: x[0])
        final = float(current + rng.normal(0, 1.2 + 0.005*abs(current)))
        # Append arrays.
        for row in ev:
            times.append(row[0]); codes.append(row[1]); amounts.append(row[2]); sources.append(row[3])
        offsets.append(len(times)); y.append(final)
        length = len(ev)
        flags = {
            'correction': corr and correction_done,
            'duplicate': dup or dup_done,
            'stale': stale,
            'rare_late': rare,
            'source_disagreement': dis,
            'short_history': length <= 5,
            'long_history': length >= 10,
            'high_value': abs(final) > 125,
        }
        for k,v in flags.items(): meta[k].append(bool(v))

    arr = {
        'event_time': np.asarray(times, dtype=float),
        'event_code': np.asarray(codes, dtype=np.int64),
        'event_amount': np.asarray(amounts, dtype=float),
        'event_source': np.asarray(sources, dtype=np.int64),
        'case_offsets': np.asarray(offsets, dtype=np.int64),
        'y': np.asarray(y, dtype=float),
    }
    for k,v in meta.items(): arr['meta_'+k] = np.asarray(v, dtype=bool)
    return arr


def prefix_dataset(d, prefix):
    out = {}
    for k,v in d.items():
        if k.startswith('meta_'):
            out[k] = v
        else:
            out[prefix + k] = v
    return out

