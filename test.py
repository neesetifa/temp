`app/solve.py` has two stubs:

```python
fit_shift_model(train_X, train_y, train_batch)
predict_shift(X, batch, params)
```

Return one prediction per row.

Rows come from several batches. The old model looked good on a random split, but it was brittle when the batch mix changed. Some columns are very tempting on the common batches and not so helpful elsewhere.

Use the passed arrays only. No hidden files, network, outside data, or fitting on public eval answers. Aim to beat the plain pooled regression on RMSE.

  import numpy as np
from solve import fit_shift_model, predict_shift

def _rmse(y,p): return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))
def test_prediction_shape_range_and_public_rmse():
    tr=np.load('train_data.npz'); pu=np.load('public_eval.npz')
    params=fit_shift_model(tr['train_X'],tr['train_y'],tr['train_batch'])
    pred=predict_shift(pu['public_X'],pu['public_batch'],params)
    assert pred.shape==pu['public_y'].shape
    assert np.all(np.isfinite(pred))
    const=np.full_like(pu['public_y'],tr['train_y'].mean())
    assert _rmse(pu['public_y'],pred) < 0.88*_rmse(pu['public_y'],const)
def test_predict_is_deterministic_and_permutation_safe():
    tr=np.load('train_data.npz'); pu=np.load('public_eval.npz')
    params=fit_shift_model(tr['train_X'],tr['train_y'],tr['train_batch'])
    X=pu['public_X'][:250].copy(); b=pu['public_batch'][:250].copy(); p1=predict_shift(X,b,params); p2=predict_shift(X.copy(),b.copy(),params)
    assert np.allclose(p1,p2)
    rng=np.random.default_rng(0); perm=rng.permutation(len(X)); inv=np.empty_like(perm); inv[perm]=np.arange(len(perm))
    assert np.allclose(p1,predict_shift(X[perm].copy(),b[perm].copy(),params)[inv])
    assert np.allclose(X,pu['public_X'][:250]); assert np.array_equal(b,pu['public_batch'][:250])


from __future__ import annotations
import numpy as np
D_LAT=8; N_FEATURES=44; N_BATCHES=8
_rng_global=np.random.default_rng(192837)
A_STABLE=_rng_global.normal(size=(D_LAT,18)); A_STABLE/=np.linalg.norm(A_STABLE,axis=0,keepdims=True)
A_SHORT=_rng_global.normal(size=(D_LAT,12)); A_SHORT/=np.linalg.norm(A_SHORT,axis=0,keepdims=True)
PERM=np.array([19,3,41,7,25,11,0,34,15,28,5,42,21,9,37,13,1,31,17,40,23,6,33,12,29,2,38,20,10,35,16,43,26,4,39,22,8,30,14,36,24,18,32,27],dtype=int)
BATCH_MEAN=np.array([
 [ 0.25,-0.10, 0.05, 0.00, 0.10,-0.15, 0.05, 0.15],
 [-0.15, 0.20,-0.10, 0.10, 0.00, 0.05,-0.10, 0.05],
 [ 0.05, 0.10, 0.25,-0.15, 0.15, 0.00, 0.10,-0.10],
 [-0.10,-0.25, 0.10, 0.25,-0.05, 0.10, 0.15, 0.00],
 [ 0.20, 0.05,-0.20,-0.05,-0.25, 0.15,-0.15, 0.10],
 [-0.25, 0.28, 0.12,-0.12, 0.20,-0.10, 0.20,-0.18],
 [ 0.18,-0.28,-0.24, 0.18, 0.22, 0.12,-0.18, 0.24],
 [-0.22,-0.15, 0.25, 0.22,-0.18, 0.25, 0.05,-0.24],
],float)
SHORT_COEF=np.array([1.15,1.02,0.86,0.65,0.34,-0.55,-1.10,0.02],float)
SHORT_NOISE=np.array([0.20,0.22,0.25,0.32,0.44,0.50,0.42,0.58],float)

def _rng(seed): return np.random.default_rng(seed)

def stable_target(z):
    y=(0.92*z[:,0]*z[:,1] -0.82*z[:,2]*z[:,3] +0.70*z[:,4]*z[:,5]
       +0.58*(z[:,6]**2-1.0)*z[:,7] +0.38*np.sin(z[:,0]+z[:,2])*z[:,5])
    return 2.3*y

def _sample_batches(rng,n,split):
    if split=='train':
        batches=np.arange(6); probs=np.array([0.33,0.24,0.17,0.12,0.09,0.05])
    elif split=='public':
        batches=np.arange(6); probs=np.array([0.29,0.21,0.17,0.13,0.11,0.09])
    else:
        batches=np.arange(8); probs=np.array([0.12,0.09,0.08,0.08,0.08,0.12,0.28,0.15])
    return rng.choice(batches,size=n,p=probs/probs.sum()).astype(np.int64)

def _make_X(z,y,batch,rng):
    n=len(y); b=batch.astype(int)
    stable=z@A_STABLE + rng.normal(0,0.55,size=(n,18))
    # mild batch offsets, not enough to reveal target alone
    off=np.zeros((N_BATCHES,18))
    for bb in range(N_BATCHES): off[bb]=0.14*np.sin((bb+1)*(np.arange(18)+1.7))
    stable=stable+off[b]
    signs=np.array([1,-1,1,1,-1,1,-1,1,1,-1,1,-1],float)
    short=np.empty((n,12),float)
    for j in range(12):
        short[:,j]=signs[j]*SHORT_COEF[b]*y + 0.20*(z@A_SHORT[:,j]) + 0.4*np.sin(z[:,j%D_LAT])
        short[:,j]+=0.75*np.cos((b+1)*(j+1)) + rng.normal(0, SHORT_NOISE[b]*2.4, n)
    art=np.empty((n,8),float)
    for j in range(8): art[:,j]=1.0*np.sin(0.8*b+j)+0.25*z[:,(j+3)%D_LAT]+rng.normal(0,0.8,n)
    noise=rng.normal(0,1.0,size=(n,6))
    X=np.column_stack([stable,short,art,noise])
    assert X.shape[1]==N_FEATURES
    return X[:,PERM]

def make_dataset(seed=100,n=3500,split='train'):
    rng=_rng(seed); batch=_sample_batches(rng,n,split)
    z=rng.normal(0,1,size=(n,D_LAT))+BATCH_MEAN[batch]
    z[np.isin(batch,[5,6,7])] *= 1.15
    y_clean=stable_target(z)
    y=y_clean+rng.normal(0,0.38,n)
    X=_make_X(z,y_clean,batch,rng)
    return {'X':X.astype(np.float64),'y':y.astype(np.float64),'batch':batch.astype(np.int64),'y_clean':y_clean.astype(np.float64)}

STABLE_UNPERM=np.arange(18); SHORT_UNPERM=np.arange(18,30)
STABLE_COLS=np.array([int(np.where(PERM==j)[0][0]) for j in STABLE_UNPERM],int)
SHORTCUT_COLS=np.array([int(np.where(PERM==j)[0][0]) for j in SHORT_UNPERM],int)


from __future__ import annotations
import numpy as np

PAIR_I, PAIR_J = np.triu_indices(44, k=1)

def _standardize_fit(X):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-8] = 1.0
    return mu, sd

def _standardize_apply(X, mu, sd):
    return (X - mu) / sd

def _fit_ridge(Phi, y, lam=1.0):
    Phi = np.asarray(Phi, float); y = np.asarray(y, float)
    mu = Phi.mean(axis=0); sd = Phi.std(axis=0); sd[sd < 1e-8] = 1.0
    Z = (Phi - mu) / sd; Z[:, 0] = 1.0
    reg = lam * np.eye(Z.shape[1]); reg[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return {"coef": coef, "mu": mu, "sd": sd}

def _pred_ridge(model, Phi):
    Z = (np.asarray(Phi, float) - model["mu"]) / model["sd"]
    Z[:, 0] = 1.0
    return Z @ model["coef"]

def _expanded(X, mu=None, sd=None, selected_pairs=None, include_linear=True):
    X = np.asarray(X, float)
    if mu is None:
        mu, sd = _standardize_fit(X)
    Z = _standardize_apply(X, mu, sd)
    parts = [np.ones((len(X), 1))]
    if include_linear:
        parts += [Z, Z ** 2]
    if selected_pairs is None:
        P = Z[:, PAIR_I] * Z[:, PAIR_J]
    else:
        P = Z[:, PAIR_I[selected_pairs]] * Z[:, PAIR_J[selected_pairs]]
    parts.append(P)
    return np.column_stack(parts), mu, sd

def _feature_slopes(F, y, batch):
    batches = np.unique(batch)
    slopes = np.zeros((len(batches), F.shape[1])); counts = np.zeros(len(batches))
    for i, b in enumerate(batches):
        m = batch == b; counts[i] = m.sum()
        Fb = F[m]; yb = y[m]
        yc = yb - yb.mean(); Fc = Fb - Fb.mean(axis=0)
        slopes[i] = np.mean(Fc * yc[:, None], axis=0) / (np.mean(Fc * Fc, axis=0) + 1e-6)
    return slopes, counts, batches

def _stability_score(F, y, batch):
    slopes, counts, batches = _feature_slopes(F, y, batch)
    w = np.sqrt(counts); w = w / w.sum()
    w = 0.5 * w + 0.5 * np.ones_like(w) / len(w)
    mean = np.sum(slopes * w[:, None], axis=0)
    sd = np.sqrt(np.sum(w[:, None] * (slopes - mean) ** 2, axis=0))
    sign_agree = np.mean(np.sign(slopes) == np.sign(mean)[None, :], axis=0)
    return np.abs(mean) * (0.20 + sign_agree) / (sd + 0.08), slopes

def _choose_pairs(X, y, batch, k=130):
    mu, sd = _standardize_fit(X); Z = _standardize_apply(X, mu, sd)
    P = Z[:, PAIR_I] * Z[:, PAIR_J]
    score, _ = _stability_score(P, y, batch)
    order = np.argsort(score)[::-1]
    chosen = []; counts = np.zeros(X.shape[1], int)
    for idx in order:
        i, j = int(PAIR_I[idx]), int(PAIR_J[idx])
        if counts[i] >= 16 or counts[j] >= 16:
            continue
        chosen.append(int(idx)); counts[i] += 1; counts[j] += 1
        if len(chosen) >= k:
            break
    return np.array(chosen, dtype=int), mu, sd

def _batch_res(pred, y, batch):
    out = {}; g = float(np.mean(y - pred))
    for b in np.unique(batch):
        m = batch == b; n = int(m.sum()); alpha = n / (n + 250.0)
        out[int(b)] = float(alpha * np.mean(y[m] - pred[m]) + (1 - alpha) * g)
    return out

def fit_shift_model(train_X, train_y, train_batch):
    X = np.asarray(train_X, float); y = np.asarray(train_y, float); batch = np.asarray(train_batch, int)
    pairs, mu, sd = _choose_pairs(X, y, batch, k=130)
    Phi, _, _ = _expanded(X, mu, sd, selected_pairs=pairs, include_linear=False)
    model = _fit_ridge(Phi, y, lam=40.0)
    train_pred = _pred_ridge(model, Phi)
    return {"pairs": pairs, "mu": mu, "sd": sd, "model": model, "batch_corr": _batch_res(train_pred, y, batch)}

def predict_shift(X, batch, params):
    X = np.asarray(X, float); batch = np.asarray(batch, int)
    Phi, _, _ = _expanded(X, params["mu"], params["sd"], selected_pairs=params["pairs"], include_linear=False)
    pred = _pred_ridge(params["model"], Phi)
    pred += np.array([params["batch_corr"].get(int(b), 0.0) for b in batch], float)
    return np.asarray(pred, float)

from __future__ import annotations
import numpy as np, time
from pathlib import Path
from generators import make_dataset

def _rmse(y,p): return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))
def _dec(v,good,cut):
    v=float(v)
    if not np.isfinite(v) or v>=cut: return 0.0
    if v<=good: return 1.0
    return float((cut-v)/(cut-good))
def _metrics(y,p,batch):
    b=np.asarray(batch); masks={'overall':np.ones(len(y),bool),'seen_common':np.isin(b,[0,1,2,3]),'weak_batch':b==4,'rare_batch':b==5,'unseen_flip':b==6,'unseen_null':b==7,'shifted_all':np.isin(b,[5,6,7])}
    return {k+'_rmse':_rmse(y[m],p[m]) for k,m in masks.items() if np.any(m)}
def _score(m):
    if m['overall_rmse']>8.0 or m['unseen_flip_rmse']>11.0 or m['shifted_all_rmse']>9.5:
        return 0.0,{k:0.0 for k in ['overall','unseen_flip','unseen_null','rare_batch','weak_batch','shifted_all','seen_common']}
    comps={'overall':_dec(m['overall_rmse'],3.45,8.0),'unseen_flip':_dec(m['unseen_flip_rmse'],3.70,11.0),'unseen_null':_dec(m['unseen_null_rmse'],4.10,10.0),'rare_batch':_dec(m['rare_batch_rmse'],3.60,9.0),'weak_batch':_dec(m['weak_batch_rmse'],3.00,8.5),'shifted_all':_dec(m['shifted_all_rmse'],3.75,9.5),'seen_common':_dec(m['seen_common_rmse'],3.10,7.5)}
    r=0.26*comps['overall']+0.20*comps['unseen_flip']+0.16*comps['unseen_null']+0.12*comps['rare_batch']+0.08*comps['weak_batch']+0.13*comps['shifted_all']+0.05*comps['seen_common']
    return float(np.clip(r,0,1)), comps

def evaluate(fit_fn,pred_fn,train_path=None,seeds=(601,602,603,604)):
    t0=time.perf_counter(); train_path=train_path or Path(__file__).resolve().parents[1]/'app'/'train_data.npz'
    tr=np.load(train_path); params=fit_fn(tr['train_X'],tr['train_y'],tr['train_batch'])
    per={}; rewards=[]
    for seed in seeds:
        data=make_dataset(seed=seed,n=1500,split='hidden'); X=data['X'].copy(); b=data['batch'].copy()
        p1=np.asarray(pred_fn(X,b,params),float); perm=np.random.default_rng(seed+777).permutation(len(X)); inv=np.empty_like(perm); inv[perm]=np.arange(len(perm))
        p2=np.asarray(pred_fn(X[perm].copy(),b[perm].copy(),params),float)[inv]
        valid=p1.shape==data['y'].shape and np.all(np.isfinite(p1)) and np.allclose(p1,p2,atol=1e-9,rtol=1e-9) and np.allclose(X,data['X']) and np.array_equal(b,data['batch']) and np.std(p1)>0.05
        if not valid:
            metrics={'overall_rmse':float('inf'),'seen_common_rmse':float('inf'),'weak_batch_rmse':float('inf'),'rare_batch_rmse':float('inf'),'unseen_flip_rmse':float('inf'),'unseen_null_rmse':float('inf'),'shifted_all_rmse':float('inf'),'invalid':1.0}; r,comps=0.0,{}
        else:
            metrics=_metrics(data['y'],p1,data['batch']); r,comps=_score(metrics)
        rewards.append(r); per[str(seed)]={'reward':float(r),'metrics':{k:float(v) for k,v in metrics.items()},'component_rewards':{k:float(v) for k,v in comps.items()}}
    keys=[k for k in per[str(seeds[0])]['metrics'] if k!='invalid']; agg={k:float(np.nanmean([per[str(s)]['metrics'].get(k,np.nan) for s in seeds])) for k in keys}
    ar,ac=_score(agg); final=float(0.7*ar+0.3*np.mean(rewards))
    return {'reward':final,'passed_cutoff':bool(final>0),'aggregate_metrics':agg,'aggregate_component_rewards':ac,'per_seed':per,'elapsed_seconds':time.perf_counter()-t0}

