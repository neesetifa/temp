# Batch wait model

There are two functions in `app/solve.py` to fill in:

```python
fit_wait_model(
    train_job_X,
    train_job_code,
    train_batch_X,
    train_batch_offsets,
    train_y,
)

predict_wait_time(
    job_X,
    job_code,
    batch_X,
    batch_offsets,
    params,
)
```

The job rows are packed by batch. For batch `i`, the rows are

```python
batch_offsets[i] : batch_offsets[i + 1]
```

Return one predicted wait value for each job row.

The old model scored each job row by itself. It was fine on quiet batches, but
it missed cases where the same-looking job landed very differently depending on
what arrived with it. Some batches have rows that mostly get in each other's way.
Some have a lot of busy-looking rows that do not matter much for a given job.
Some have a few urgent rows mixed into a slower group.

Use the training batches and their wait values to build the model. At prediction
time, use the rows and batch offsets passed into `predict_wait_time`.

Keep the input arrays unchanged. Do not read hidden files or use outside data.


solve
  """Starter implementation for shared_capacity_waits.

This deliberately uses only the training mean. It is here to make the public
checks import and run; it is not intended to score well on hidden data.
"""

from __future__ import annotations

import numpy as np


def fit_wait_model(train_job_X, train_job_code, train_batch_X, train_batch_offsets, train_y):
    y = np.asarray(train_y, dtype=float)
    return {"mean": float(np.mean(y)) if y.size else 0.0}


def predict_wait_time(job_X, job_code, batch_X, batch_offsets, params):
    n = int(np.asarray(job_X).shape[0])
    return np.full(n, float(params.get("mean", 0.0)), dtype=float)


reference
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def softplus(x): return np.log1p(np.exp(np.clip(x,-30,30)))

def onehot(x,n):
    out=np.zeros((len(x),n),float); out[np.arange(len(x)),x.astype(int)]=1.0; return out

def _proxy_demands(X,C):
    size,comp,prio,slack,frag,decoy,arr = X.T
    fam,st,hand,prd = C.T.astype(int)
    g=fam%4
    d0=0.8*(g==0)+0.45*np.isin(st,[0,1,6])+0.55*np.log1p(size)+0.25*comp-0.45*(hand==5)+0.08*(prd%4)
    d1=0.85*(g==1)+0.50*np.isin(st,[1,2,7])+0.35*size+0.35*frag+0.15*(hand==1)-0.45*(hand==5)
    d2=0.85*(g==2)+0.55*np.isin(st,[3,4,6])+0.32*comp+0.65*(hand==5)+0.18*(st%3==0)
    d3=0.75*(g==3)+0.50*np.isin(st,[5,7])+0.35*frag+0.24*prio+0.15*(hand==4)-0.35*(hand==5)
    D=0.35+softplus(np.vstack([d0,d1,d2,d3]).T)
    D *= (0.7+0.45*np.log1p(size))[:,None]
    return D

def reference_features(job_X, job_code, batch_X, offsets):
    X=job_X; C=job_code.astype(int); B=batch_X
    n=len(X); fam,st,hand,prd=C.T
    D=_proxy_demands(X,C)
    dnorm=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-6)
    feats=[]
    base=[X, C/np.array([11,7,5,47.0]), onehot(fam%4,4), onehot(st,8), onehot(hand,6)]
    feats.extend(base)
    # containers
    batch_feats=np.zeros((n, 0))
    rel=np.zeros((n,4)); stress=np.zeros((n,4)); sums=np.zeros((n,4)); means=np.zeros((n,4)); same=np.zeros((n,8)); misc=[]
    bf_list=[]
    for bi in range(len(offsets)-1):
        a,b=offsets[bi],offsets[bi+1]
        idx=slice(a,b); m=b-a
        Xi=X[idx]; Ci=C[idx]; Di=D[idx]; dni=dnorm[idx]
        fami,sti,handi,prdi=Ci.T
        pr=Xi[:,2]; size=Xi[:,0]; comp=Xi[:,1]
        total=Di.sum(axis=0); mean=Di.mean(axis=0)
        # capacity proxy from batch_X
        bx=B[bi]
        cap=np.array([18.0,17.0,16.0,15.5])*(0.65+0.85*bx[0]) + np.array([5*bx[1],2.2*bx[2],4.5*bx[2],3*bx[1]])
        cap *= (1.0 - 0.22*bx[4]*np.array([0.2,1.0,0.55,0.4]))
        cap_shift=cap*np.array([1.25-0.30*bx[2],0.78+0.20*bx[1],1.12,0.85+0.25*bx[2]])
        # use both normal and shifted stress variants
        for ii in range(m):
            dot=dni @ dni[ii]
            compat=0.30+0.70*dot + 0.28*(sti==sti[ii]) + 0.18*((fami%4)==(fami[ii]%4)) + 0.15*(handi==handi[ii])
            pf=0.58 + 0.85*np.maximum(pr-pr[ii],0) + 0.20*(pr>0.82) + 0.16*(pr[ii]<0.25)
            weights=compat*pf; weights[ii]=0.35
            rel[a+ii]=(Di*weights[:,None]).sum(axis=0)
            sums[a+ii]=total; means[a+ii]=mean
            stress[a+ii]=(rel[a+ii]-cap*(0.45+0.015*m))/(cap*(0.55+0.01*m))
            same[a+ii]=[
                m, (fami==fami[ii]).sum()-1, (sti==sti[ii]).sum()-1, (handi==handi[ii]).sum()-1,
                ((fami%4)==(fami[ii]%4)).sum()-1, (pr>pr[ii]+0.25).sum(), ((sti==sti[ii]) & (pr>pr[ii]+0.25)).sum(), (handi==5).sum()
            ]
        # batch feat matrix
        # family group and station counts repeated per row
        fg=np.bincount(fami%4,minlength=4)/max(m,1)
        hc=np.bincount(handi,minlength=6)/max(m,1)
        sc=np.bincount(sti,minlength=8)/max(m,1)
        rep=np.tile(np.r_[B[bi], m, total, total/(cap+1e-6), total/(cap_shift+1e-6), fg,hc,sc], (m,1))
        bf_list.append(rep)
    bf=np.vstack(bf_list)
    feats.extend([D, dnorm, sums, means, rel, softplus(stress), same, bf])
    # interactions own demand*stress
    feats.append(D*softplus(stress))
    feats.append(dnorm*rel/(np.maximum(sums,1e-6)))
    return np.hstack(feats).astype(float)



def fit_wait_model(train_job_X, train_job_code, train_batch_X, train_batch_offsets, train_y):
    X = reference_features(
        np.asarray(train_job_X, dtype=float),
        np.asarray(train_job_code, dtype=int),
        np.asarray(train_batch_X, dtype=float),
        np.asarray(train_batch_offsets, dtype=int),
    )
    y = np.asarray(train_y, dtype=float)
    model = HistGradientBoostingRegressor(
        max_iter=280,
        learning_rate=0.045,
        max_leaf_nodes=43,
        l2_regularization=0.03,
        random_state=17,
    )
    model.fit(X, y)
    return {"model": model}


def predict_wait_time(job_X, job_code, batch_X, batch_offsets, params):
    X = reference_features(
        np.asarray(job_X, dtype=float),
        np.asarray(job_code, dtype=int),
        np.asarray(batch_X, dtype=float),
        np.asarray(batch_offsets, dtype=int),
    )
    pred = params["model"].predict(X)
    return np.asarray(pred, dtype=float)

test_main
from hidden_eval import evaluate


def test_hidden_eval_runs():
    reward, metrics = evaluate()
    assert 0.0 <= reward <= 1.0
    assert metrics["overall_rmse"] >= 0.0

generator
import numpy as np

N_FAMILY=12; N_STATION=8; N_HANDLING=6; N_PRODUCT=48; N_RES=4
SLICE_NAMES = np.array([
    'sparse_clean','crowded_same_family','crowded_mixed_family','same_count_different_mix',
    'rare_family','priority_conflict','capacity_shift','decoy_load','long_tail_batch_size'
])

def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30, 30)))

def _static():
    rng=np.random.default_rng(71031)
    family = rng.normal(0,0.5,(N_FAMILY,N_RES))
    # make family groups have similar patterns
    base_groups=np.array([[1.2,0.2,-0.5,0.1],[-0.2,1.1,0.2,-0.3],[0.1,-0.4,1.2,0.3],[0.6,0.4,-0.2,1.0]])
    for f in range(N_FAMILY):
        family[f]+=base_groups[f%4]+0.18*rng.normal(size=N_RES)
    station=rng.normal(0,0.35,(N_STATION,N_RES))
    station += np.array([[0.7,0,0,0],[0.3,0.5,0,0],[0,0.8,0.1,0],[0,0.2,0.7,0],[0,0,0.8,0.2],[0.2,0,0,0.8],[0.5,0,0.4,0],[0,0.6,0,0.5]])
    handling=rng.normal(0,0.25,(N_HANDLING,N_RES))
    handling[5] += np.array([-1.0,-1.1,0.6,-0.7])  # decoy: high visible but resource only r2
    product=rng.normal(0,0.22,(N_PRODUCT,N_RES))
    prod_family=np.arange(N_PRODUCT)%N_FAMILY
    prod_station=(np.arange(N_PRODUCT)*3 + prod_family)%N_STATION
    pair = 0.7 + 0.25*rng.normal(size=(N_FAMILY,N_FAMILY))
    for i in range(N_FAMILY):
        for j in range(N_FAMILY):
            if i%4 == j%4: pair[i,j]+=0.45
            if i==j: pair[i,j]+=0.35
            if {i%4,j%4}=={1,3}: pair[i,j]+=0.25
    pair=np.clip(pair,0.25,1.8)
    return family,station,handling,product,prod_family,prod_station,pair

def _sample_slice(rng, split):
    if split == 'hidden':
        p=np.array([0.10,0.12,0.13,0.13,0.13,0.12,0.12,0.13,0.12])
    else:
        p=np.array([0.22,0.11,0.11,0.10,0.08,0.10,0.10,0.09,0.09])
    p=p/p.sum()
    return int(rng.choice(len(SLICE_NAMES),p=p))

def _choose_family(rng, slice_id, n):
    base=np.array([0.12,0.10,0.11,0.09,0.10,0.09,0.08,0.08,0.07,0.06,0.05,0.05])
    if slice_id==1: # crowded same
        f0=int(rng.choice(10, p=np.array([.13,.11,.11,.1,.1,.09,.09,.08,.1,.09])))
        out=np.where(rng.random(n)<0.78, f0, rng.choice(N_FAMILY,size=n,p=base/base.sum()))
    elif slice_id==4:
        p=base.copy(); p[10:]=0.18; p[:10]*=0.65; p=p/p.sum(); out=rng.choice(N_FAMILY,size=n,p=p)
    elif slice_id==3: # same count different mix: two groups overrepresented
        groups=[rng.integers(0,4), rng.integers(0,4)]
        fams=[g+4*rng.integers(0,3) for g in groups]
        out=np.where(rng.random(n)<0.55, fams[0], fams[1])
        mask=rng.random(n)<0.25; out[mask]=rng.choice(N_FAMILY,size=mask.sum(),p=base/base.sum())
    else:
        out=rng.choice(N_FAMILY,size=n,p=base/base.sum())
    return out.astype(np.int64)

def generate_dataset(n_batches, seed, split='train'):
    rng=np.random.default_rng(seed)
    family,station,handling,product,prod_family,prod_station,pair=_static()
    job_Xs=[]; job_codes=[]; batch_X=[]; ys=[]; offsets=[0]; slices=[]
    for b in range(n_batches):
        sid=_sample_slice(rng,split)
        # batch size
        if sid==0: n=int(rng.integers(4,9))
        elif sid in (1,2,3,5,7): n=int(rng.integers(14,29))
        elif sid==8: n=int(rng.choice([rng.integers(2,5), rng.integers(28,47)]))
        else: n=int(rng.integers(9,22))
        staff=rng.uniform(0.35,1.0)
        equip_a=rng.integers(0,3)/2
        equip_b=rng.integers(0,3)/2
        shift=rng.uniform(0,1)
        maint=(rng.random()< (0.28 if sid==6 else 0.10))*1.0
        local_load_hint=rng.normal(0,1)
        day=rng.integers(0,7)/6
        bx=np.array([staff,equip_a,equip_b,shift,maint,local_load_hint,day],float)
        batch_X.append(bx)
        fam=_choose_family(rng,sid,n)
        # stations with noise around product default/family
        st=(fam*2 + rng.integers(0,4,size=n))%N_STATION
        if sid==2:
            st=rng.choice(N_STATION,size=n)
        hand=rng.choice(N_HANDLING,size=n,p=np.array([.21,.18,.18,.17,.16,.10]))
        if sid==7:
            # many decoys
            hand=np.where(rng.random(n)<0.48,5,hand)
        prd=(fam + N_FAMILY*rng.integers(0,4,size=n))%N_PRODUCT
        # features size, complexity, priority, slack, fragility, decoy_flagish, arrival_pos
        size=rng.lognormal(mean=0.0,sigma=0.42,size=n)
        if sid==7:
            size=np.where(hand==5, size*rng.uniform(1.8,2.8,size=n), size)
        complexity=np.clip(rng.beta(2.0,2.3,size=n)+0.12*(fam%4==2),0,1.3)
        if sid==5:
            priority=np.where(rng.random(n)<0.35,rng.uniform(0.78,1.0,size=n),rng.uniform(0.0,0.38,size=n))
        else:
            priority=rng.beta(1.6,2.5,size=n)
        slack=rng.beta(2,2,size=n)
        fragility=np.clip(rng.beta(1.7,3.2,size=n)+0.2*(hand==4),0,1.2)
        visible_decoy=(hand==5).astype(float)+rng.normal(0,0.05,size=n)
        arr=(np.arange(n)+rng.normal(0,0.3,size=n))/max(n-1,1)
        X=np.stack([size,complexity,priority,slack,fragility,visible_decoy,arr],axis=1).astype(np.float64)
        codes=np.stack([fam,st,hand,prd],axis=1).astype(np.int64)
        # true demand
        z=family[fam]+station[st]+handling[hand]+product[prd]
        z += np.stack([
            0.65*np.log1p(size)+0.40*complexity-0.15*slack,
            0.45*size+0.35*fragility+0.25*(hand==1),
            0.30*complexity+0.55*(hand==5)+0.25*(st%3==0),
            0.40*fragility+0.25*priority+0.25*(fam%4==3),
        ],axis=1)
        dem=0.38+_softplus(z)
        dem *= (0.65+0.55*np.log1p(size))[:,None]
        # decoys have high visible size but reduced on main shared resources, not r2
        dec=(hand==5)
        dem[dec] *= np.array([0.28,0.25,1.15,0.35])
        # capacity, shift changes which resources are scarce
        cap=np.array([18.0,17.0,16.0,15.5])*(0.65+0.85*staff)
        cap += np.array([5.0*equip_a,2.2*equip_b,4.5*equip_b,3.0*equip_a])
        cap *= (1.0 - 0.22*maint*np.array([0.2,1.0,0.55,0.4]))
        if sid==6:
            # visible capacity shift: a resource swap / bottleneck depending on equip/shift
            cap *= np.array([1.25-0.30*equip_b,0.78+0.20*equip_a,1.12,0.85+0.25*equip_b])
        # total load and job-specific related load
        # normalized demand vectors
        dnorm=dem/(np.linalg.norm(dem,axis=1,keepdims=True)+1e-6)
        related=np.zeros((n,N_RES))
        for i in range(n):
            compat=(0.25+0.75*(dnorm @ dnorm[i]))
            compat *= pair[fam[i],fam]
            compat *= (1.0 + 0.30*(st==st[i]) + 0.18*(hand==hand[i]))
            # priority: higher-priority neighbors hurt low-priority jobs more; low priority neighbors barely hurt high priority
            pf=0.55 + 0.95*np.maximum(priority-priority[i],0) + 0.22*(priority>0.82) + 0.18*(priority[i]<0.25)
            pf *= (1.0 - 0.20*np.maximum(priority[i]-priority,0))
            weights=compat*pf
            weights[i]=0.35  # own service load still counts some
            related[i]=(dem*weights[:,None]).sum(axis=0)
        # nonlinear stress: capacity shared, adjust for n
        stress=(related - cap[None,:]*(0.45+0.015*n))/(cap[None,:]*(0.55+0.01*n))
        sens=0.30+dnorm+0.25*fragility[:,None]*np.array([0.2,0.3,0.15,0.6])
        cong=(sens*_softplus(1.2*stress)).sum(axis=1)
        same_family_count=np.array([(fam==fam[i]).sum()-1 for i in range(n)])
        high_prio_same_station=np.array([((st==st[i]) & (priority>priority[i]+0.25)).sum() for i in range(n)])
        base=18+10*np.log1p(size)+7*complexity+5*fragility-5*priority+3*(1-slack)
        y=base + 13.5*cong + 0.9*np.sqrt(np.maximum(same_family_count,0))*((fam%4==1)+0.4) + 1.5*high_prio_same_station
        # extra hidden interactions for sustained/capacity/rare
        if sid==4:
            y += 4.0*((fam>=10).astype(float))*(0.5+cong)
        if sid==2:
            y += 1.5*np.maximum(0, (dem[:,1]-dem[:,0]))*_softplus(stress[:,1])
        if sid==3:
            y += 3.0*((fam%4)==(fam[0]%4))*(cong>1.2)
        if sid==8:
            y += 2.5*(n>25)*_softplus(stress[:,3])
        if sid==7:
            # decoys delay only jobs with resource 2 overlap, not all jobs
            dec_load=dem[dec,2].sum() if dec.any() else 0.0
            y += 2.0*(dnorm[:,2]>0.55)*_softplus((dec_load-cap[2]*0.25)/(cap[2]*0.4))
        noise=rng.normal(0,1.2+0.25*np.sqrt(n),size=n)
        y=y+noise
        job_Xs.append(X); job_codes.append(codes); ys.append(y.astype(np.float64)); slices.extend([sid]*n); offsets.append(offsets[-1]+n)
    return {
        'job_X':np.vstack(job_Xs).astype(np.float64),
        'job_code':np.vstack(job_codes).astype(np.int64),
        'batch_X':np.vstack(batch_X).astype(np.float64),
        'batch_offsets':np.array(offsets,dtype=np.int64),
        'y':np.concatenate(ys).astype(np.float64),
        'slice_id':np.array(slices,dtype=np.int64),
        'slice_names':SLICE_NAMES,
    }

hidden_eval
"""Hidden verifier for shared_capacity_waits."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from generator import SLICE_NAMES, generate_dataset

HIDDEN_SEED = 8041
HIDDEN_N_BATCHES = 250
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))

# Slice RMSE anchors. These are intentionally verifier-side: public tests only
# check the interface, while the hidden run uses per-slice continuous scoring.
SLICE_GOOD = np.array([3.0, 9.5, 5.2, 8.1, 7.6, 6.7, 5.0, 6.9, 7.6], dtype=float)
SLICE_BAD = np.array([10.0, 18.0, 12.0, 16.0, 15.0, 14.0, 12.0, 14.0, 16.0], dtype=float)
OVERALL_GOOD = 7.4
OVERALL_BAD = 15.5


def _load_train(app_dir: Path) -> tuple[Any, ...]:
    data = np.load(app_dir / "train_data.npz")
    return (
        data["train_job_X"],
        data["train_job_code"],
        data["train_batch_X"],
        data["train_batch_offsets"],
        data["train_y"],
    )


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solve module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_in_process(solve_path: Path, payload: dict[str, Any]) -> list[np.ndarray]:
    module = _load_module(solve_path)
    params = module.fit_wait_model(*payload["train"])
    preds = []
    for args in payload["inputs"]:
        preds.append(np.asarray(module.predict_wait_time(*args, params), dtype=float))
    return preds


def _run_candidate(solve_path: Path, payload: dict[str, Any]) -> list[np.ndarray]:
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)

    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get("preds") if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, dtype=float) for p in preds]


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2)))


def _score_rmse(rmse: float, good: float, bad: float) -> float:
    if not math.isfinite(rmse):
        return 0.0
    return float(np.clip((bad - rmse) / (bad - good), 0.0, 1.0))


def _validate_predictions(pred: np.ndarray, expected_n: int) -> None:
    if pred.shape != (expected_n,):
        raise ValueError(f"prediction shape must be ({expected_n},), got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions must be finite")


def _compute_metrics(y: np.ndarray, pred: np.ndarray, slice_id: np.ndarray) -> dict[str, Any]:
    overall_rmse = _rmse(y, pred)
    overall_score = _score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    per_slice: dict[str, Any] = {}
    slice_scores = []
    for sid, name in enumerate(SLICE_NAMES):
        mask = slice_id == sid
        if not bool(mask.any()):
            continue
        rmse = _rmse(y[mask], pred[mask])
        score = _score_rmse(rmse, float(SLICE_GOOD[sid]), float(SLICE_BAD[sid]))
        per_slice[str(name)] = {"rmse": rmse, "score": score, "n": int(mask.sum())}
        slice_scores.append(score)
    slice_mean = float(np.mean(slice_scores)) if slice_scores else 0.0
    reward = float(0.35 * overall_score + 0.65 * slice_mean)
    return {
        "reward": reward,
        "overall_rmse": overall_rmse,
        "overall_score": overall_score,
        "slice_mean": slice_mean,
        "per_slice": per_slice,
    }


def evaluate(solve_path: str | Path | None = None, app_dir: str | Path | None = None, log_dir: str | Path | None = None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / "solve.py"

    train = _load_train(app)
    hidden = generate_dataset(HIDDEN_N_BATCHES, HIDDEN_SEED, "hidden")
    inputs = (
        hidden["job_X"],
        hidden["job_code"],
        hidden["batch_X"],
        hidden["batch_offsets"],
    )
    payload = {"train": train, "inputs": [inputs, inputs]}

    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")
    pred0, pred1 = preds
    _validate_predictions(pred0, len(hidden["y"]))
    _validate_predictions(pred1, len(hidden["y"]))
    if not np.allclose(pred0, pred1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")

    metrics = _compute_metrics(hidden["y"], pred0, hidden["slice_id"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    reward, metrics = evaluate()
    print(json.dumps({"reward": reward, "overall_rmse": metrics["overall_rmse"]}, sort_keys=True))

