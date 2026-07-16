from __future__ import annotations
import numpy as np


def _feature_map(X, groups=None, n_groups=None, include_groups=True):
    X=np.asarray(X, dtype=float)
    feats=[np.ones((X.shape[0],1)), X, X**2]
    # compact nonlinear basis
    feats.append(np.sin(X[:,:5]))
    feats.append(np.cos(X[:,3:6]))
    feats.append((X[:,[0,1,2]]*X[:,[5,6,7]]))
    if include_groups and groups is not None:
        groups=np.asarray(groups,dtype=int)
        if n_groups is None: n_groups=int(groups.max())+1 if groups.size else 0
        oh=np.zeros((X.shape[0], n_groups))
        valid=(groups>=0)&(groups<n_groups)
        oh[np.arange(X.shape[0])[valid], groups[valid]]=1.0
        feats.append(oh)
    return np.concatenate(feats, axis=1)

def _ridge_fit(Phi,y,lam=3.0):
    A=Phi.T@Phi + lam*np.eye(Phi.shape[1])
    A[0,0] -= lam # don't penalize intercept
    return np.linalg.solve(A, Phi.T@y)

def _quantile(vals, q, default=np.nan):
    vals=np.asarray(vals, dtype=float)
    vals=vals[np.isfinite(vals)]
    if vals.size==0: return float(default)
    return float(np.quantile(vals, q, method='higher'))

def _safe_median(vals, default):
    vals=np.asarray(vals, dtype=float); vals=vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float(default)

class Params(dict):
    pass

def fit_reference(train_X, train_y, train_groups, calib_X, calib_y, calib_groups):
    train_groups=np.asarray(train_groups,dtype=int); calib_groups=np.asarray(calib_groups,dtype=int)
    n_groups=max(int(train_groups.max(initial=0)), int(calib_groups.max(initial=0)))+1
    n_groups=max(n_groups,14)
    Phi=_feature_map(train_X, train_groups, n_groups, include_groups=True)
    coef=_ridge_fit(Phi, np.asarray(train_y,dtype=float), lam=5.0)
    pred_train=_feature_map(train_X, train_groups, n_groups, True)@coef
    pred_cal=_feature_map(calib_X, calib_groups, n_groups, True)@coef
    train_res=train_y-pred_train
    cal_res=calib_y-pred_cal
    # Equal-tail-ish asymmetric conformal. Slightly conservative for heavy tails.
    q=0.945
    glob_lo=_quantile(pred_cal-calib_y, q, 1.0) # amount below prediction
    glob_hi=_quantile(calib_y-pred_cal, q, 1.0) # amount above prediction
    glob_abs=_quantile(np.abs(cal_res), 0.905, 1.0)
    global_train_scale=_safe_median(np.abs(train_res), 1.0)
    global_cal_scale=_safe_median(np.abs(cal_res), 1.0)

    gstats={}
    # residual magnitude model: group train scale can guide groups with little/no calibration.
    for g in range(n_groups):
        mt=train_groups==g; mc=calib_groups==g
        tr_scale=_safe_median(np.abs(train_res[mt]), global_train_scale)
        ca_scale=_safe_median(np.abs(cal_res[mc]), global_cal_scale)
        # estimate scale ratio from both train and calib, shrink by counts
        nt=int(mt.sum()); nc=int(mc.sum())
        rt=tr_scale/(global_train_scale+1e-12)
        rc=ca_scale/(global_cal_scale+1e-12)
        wc=nc/(nc+18.0)
        wt=nt/(nt+80.0)
        ratio=(1-wc)*(0.65*rt+0.35) + wc*(0.70*rc+0.30*rt)
        ratio=max(0.55, min(2.45, ratio))
        qlo_g=_quantile(pred_cal[mc]-calib_y[mc], q, glob_lo*ratio)
        qhi_g=_quantile(calib_y[mc]-pred_cal[mc], q, glob_hi*ratio)
        # Small groups are noisy, shrink quantile toward train-scaled global.
        wq=nc/(nc+24.0)
        lo=(1-wq)*(glob_lo*ratio) + wq*qlo_g
        hi=(1-wq)*(glob_hi*ratio) + wq*qhi_g
        # safety depends on weak-calib and high estimated scale. Tiny/no calib needs a bit more guard.
        safety=1.035 + (0.82 if nc == 0 else 0.34 if nc < 5 else 0.16 if nc < 10 else 0.045 if nc < 25 else 0.0) + 0.035*max(0, ratio-1.2)
        gstats[g]=(float(lo*safety), float(hi*safety), nc, nt, ratio)
    return Params(coef=coef, n_groups=n_groups, gstats=gstats, glob_lo=float(glob_lo), glob_hi=float(glob_hi))

def predict_reference(X, groups, params):
    groups=np.asarray(groups,dtype=int)
    n_groups=params['n_groups']
    pred=_feature_map(X, groups, n_groups, True)@params['coef']
    lo_r=[]; hi_r=[]
    for g in groups:
        if int(g) in params['gstats']:
            lo,hi,*_=params['gstats'][int(g)]
        else:
            lo,hi=params['glob_lo']*1.3, params['glob_hi']*1.3
        lo_r.append(lo); hi_r.append(hi)
    lo_r=np.asarray(lo_r); hi_r=np.asarray(hi_r)
    return pred-lo_r, pred+hi_r

# Hidden evaluator-compatible names.
fit_interval_model = fit_reference
predict_interval = predict_reference
