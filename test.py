# Fix routed labels

`app/solve.py` has two stubs:

```python
fit_label_model(train_X, train_observed_label, train_route, train_item, train_checked)
predict_label_proba(X, observed_label, route, item, params)
```

Return class probabilities for each row.

Each row has a label that came through one route. `train_checked` marks the training rows where that label was checked later; the other routed labels are still useful, but they are not always the final class. The same item can also show up more than once when it went through more than one route.

The old model trained straight on the routed label. It looked fine on easy rows, but it was overconfident when routes disagreed or when a rare route showed up.

Use the arrays passed in. No hidden files, network, outside data, or fitting on public eval answers.

  
from __future__ import annotations
import numpy as np
K = 5
D = 14
N_ROUTES = 6

def _rng(seed): return np.random.default_rng(seed)

def softmax(S):
    S = S - np.max(S, axis=-1, keepdims=True); E = np.exp(S); return E/E.sum(axis=-1, keepdims=True)

W = np.array([
 [ 1.20,-0.80, 0.55, 0.15,-0.10, 0.35,-0.25, 0.15, 0.20,-0.15, 0.10, 0.05,-0.05, 0.00],
 [-0.35, 1.10,-0.60, 0.90, 0.10,-0.30, 0.30,-0.15, 0.10, 0.20,-0.10, 0.10, 0.00,-0.05],
 [ 0.10,-0.45, 1.15,-0.85, 0.70, 0.10, 0.25, 0.35,-0.30, 0.10, 0.20,-0.05, 0.05, 0.10],
 [-0.60, 0.20,-0.15, 0.50,-0.95, 1.05,-0.35, 0.25, 0.10,-0.20, 0.05, 0.20,-0.10, 0.05],
 [ 0.05,-0.20,-0.55, 0.10, 1.10,-0.45, 0.80,-0.25, 0.15, 0.20,-0.10,-0.15, 0.10, 0.05],
], dtype=float)
BIAS = np.array([0.10,-0.20,0.00,-0.35,-0.10])

def true_proba_from_x(X):
    X = np.asarray(X, dtype=float); S = X @ W.T + BIAS
    S[:,0] += 0.55*X[:,0]*X[:,2] - 0.30*X[:,6]**2
    S[:,1] += 0.45*X[:,1]*X[:,3] + 0.20*np.sin(X[:,5])
    S[:,2] += 0.50*X[:,4]*X[:,7] - 0.25*X[:,0]*X[:,9]
    S[:,3] += 0.60*X[:,5]*X[:,8] + 0.30*np.cos(X[:,2])
    S[:,4] += 0.55*X[:,6]*X[:,10] - 0.30*X[:,3]*X[:,11]
    return softmax(S)

def sample_items(seed=1, n_items=3000, split='train'):
    rng = _rng(seed); X = rng.normal(0,1,size=(n_items,D)); mix = rng.uniform(size=n_items)
    if split == 'hidden':
        X[mix<0.23,0]+=1.25; X[mix<0.23,2]+=0.95
        m=(mix>=0.23)&(mix<0.41); X[m,4]-=1.15; X[m,7]+=1.05
        m=(mix>=0.41)&(mix<0.57); X[m,6]+=1.20; X[m,10]+=0.90
    else:
        X[mix<0.17,0]+=1.05; X[mix<0.17,2]+=0.80
        m=(mix>=0.17)&(mix<0.31); X[m,4]-=1.00; X[m,7]+=0.90
        m=(mix>=0.31)&(mix<0.43); X[m,6]+=1.00; X[m,10]+=0.75
    P = true_proba_from_x(X)
    y = np.array([rng.choice(K, p=P[i]) for i in range(n_items)], dtype=np.int64)
    return X.astype(np.float64), y

def _region(X):
    X=np.asarray(X); r1=X[:,0]+0.55*X[:,2]-0.25*X[:,6]; r2=X[:,4]-0.70*X[:,7]+0.20*X[:,9]; r3=X[:,6]+0.45*X[:,10]-0.30*X[:,3]
    out=np.zeros(len(X), dtype=np.int64); out[r1>1.25]=1; out[(r2<-1.10)&(out==0)]=2; out[(r3>1.30)&(out==0)]=3; return out

def _decoy(y, reg):
    if reg==1: return np.array([1,2,3,4,0], dtype=np.int64)[y]
    if reg==2: return np.array([0,2,2,3,3], dtype=np.int64)[y]
    if reg==3: return np.array([4,0,1,2,3], dtype=np.int64)[y]
    return (y+1)%K

def _kind(rng,x,y,split):
    reg=int(_region(x[None,:])[0])
    if split=='hidden':
        if reg==1 and rng.random()<0.84: return 'conflict'
        if reg==2 and rng.random()<0.78: return 'old_taxonomy'
        if reg==3 and rng.random()<0.84: return 'rare_route'
        if y in (3,4) and rng.random()<0.34: return 'minority'
        return rng.choice(['single_easy','mixed'], p=[0.48,0.52])
    else:
        if reg==1 and rng.random()<0.58: return 'conflict'
        if reg==2 and rng.random()<0.54: return 'old_taxonomy'
        if reg==3 and rng.random()<0.58: return 'rare_route'
        if y in (3,4) and rng.random()<0.22: return 'minority'
        return rng.choice(['single_easy','mixed'], p=[0.58,0.42])

def _route_probs(kind):
    if kind=='single_easy': return np.array([0.55,0.22,0.06,0.04,0.11,0.02])
    if kind=='conflict': return np.array([0.12,0.10,0.36,0.10,0.08,0.24])
    if kind=='rare_route': return np.array([0.12,0.08,0.14,0.06,0.08,0.52])
    if kind=='old_taxonomy': return np.array([0.12,0.10,0.22,0.40,0.10,0.06])
    if kind=='minority': return np.array([0.20,0.12,0.28,0.10,0.10,0.20])
    return np.ones(N_ROUTES)/N_ROUTES

def _routes(rng, kind):
    if kind=='single_easy': n=int(rng.choice([1,2],p=[0.78,0.22]))
    elif kind in ('conflict','rare_route','old_taxonomy'): n=int(rng.choice([2,3,4],p=[0.20,0.56,0.24]))
    else: n=int(rng.choice([1,2,3],p=[0.34,0.42,0.24]))
    p=_route_probs(kind); return rng.choice(np.arange(N_ROUTES),size=min(n,N_ROUTES),replace=False,p=p/p.sum())

def _obs(rng,y,r,x,split):
    reg=int(_region(x[None,:])[0]); d=int(_decoy(int(y), reg)); hard=1.0 if split=='hidden' else 0.0; r=int(r)
    if r==0:
        if rng.random()<0.90-0.16*(reg==1)-0.08*(reg==3): return int(y)
        return d if rng.random()<0.65 else int(rng.integers(K))
    if r==1:
        if rng.random()<0.70-0.06*(reg==2): return int(y)
        return int((y+rng.choice([-1,1]))%K)
    if r==2:
        if reg==1 and rng.random()<0.90+0.03*hard: return d
        if rng.random()<0.62: return int(y)
        return d if rng.random()<0.70 else int(rng.integers(K))
    if r==3:
        if reg==2 and y in (1,2,4) and rng.random()<0.90: return int(_decoy(int(y),2))
        if rng.random()<0.55: return int(y)
        return int(_decoy(int(y),2))
    if r==4:
        if rng.random()<0.93-0.08*(reg==3): return int(y)
        return d
    if r==5:
        if reg==3 and rng.random()<0.91+0.03*hard: return d
        if rng.random()<0.58: return int(y)
        return d if rng.random()<0.75 else int(rng.integers(K))
    return int(y)

def make_dataset(seed=123, n_items=3200, split='train'):
    rng=_rng(seed); X_item,y_item=sample_items(seed+19,n_items,split=split); regs=_region(X_item)
    rows=[]; obs=[]; route=[]; item=[]; true=[]; kinds=[]; checked=[]
    for iid in range(n_items):
        x=X_item[iid]; y=int(y_item[iid]); kind=_kind(rng,x,y,split); rs=_routes(rng,kind)
        if kind=='conflict' and rng.random()<0.62: rs=np.array([2,5,rng.choice([0,1,4])],dtype=np.int64)
        if kind=='old_taxonomy' and rng.random()<0.54: rs=np.array([3,2,rng.choice([0,4])],dtype=np.int64)
        # A small checked/audited signal. It is not present for every item and is sparser in hard routes.
        base_check = 0.18 if split=='train' else 0.0
        if kind in ('conflict','rare_route','old_taxonomy'): base_check = 0.13 if split=='train' else 0.0
        if kind=='minority': base_check = 0.16 if split=='train' else 0.0
        has_checked = (split=='train' and rng.random() < base_check)
        checked_route_index = int(rng.integers(len(rs))) if has_checked else -1
        for j,r in enumerate(rs):
            rows.append(x + rng.normal(0,0.025,size=D)); route.append(int(r)); item.append(iid); true.append(y)
            lab = _obs(rng,y,int(r),x,split)
            if has_checked and j==checked_route_index:
                lab = y; checked.append(True)
            else:
                checked.append(False)
            obs.append(lab); kinds.append(kind if kind!='mixed' else ('bias_region' if regs[iid]>0 else 'mixed'))
    return {'X':np.asarray(rows,dtype=np.float64), 'observed_label':np.asarray(obs,dtype=np.int64), 'route':np.asarray(route,dtype=np.int64), 'item':np.asarray(item,dtype=np.int64), 'checked':np.asarray(checked,dtype=bool), 'true_label':np.asarray(true,dtype=np.int64), 'kind':np.asarray(kinds), 'region':np.asarray([regs[i] for i in item],dtype=np.int64)}


from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from generators import make_dataset, K

def _normalize(P): P=np.maximum(np.asarray(P,float),1e-12); return P/P.sum(axis=1,keepdims=True)
def _logloss(y,P): P=_normalize(P); return float(-np.mean(np.log(np.clip(P[np.arange(len(y)),y],1e-12,1))))
def _brier(y,P):
    Y=np.zeros_like(P); Y[np.arange(len(y)),y]=1; return float(np.mean(np.sum((P-Y)**2,axis=1)))
def _ece(y,P,bins=10):
    conf=P.max(axis=1); pred=P.argmax(axis=1); acc=(pred==y).astype(float); out=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins; m=(conf>=lo)&((conf<hi) if b<bins-1 else (conf<=hi))
        if np.any(m): out += np.mean(m)*abs(float(acc[m].mean())-float(conf[m].mean()))
    return float(out)
def _dec(v,good,cut):
    if not np.isfinite(v) or v>=cut: return 0.0
    if v<=good: return 1.0
    return float((cut-v)/(cut-good))
def _metrics(y,P,kind,route):
    kind=np.asarray(kind); route=np.asarray(route); masks={'overall':np.ones(len(y),bool),'single':kind=='single_easy','conflict':kind=='conflict','rare_route':(kind=='rare_route')|(route==5),'old_taxonomy':kind=='old_taxonomy','bias_region':np.isin(kind,['conflict','old_taxonomy','rare_route','bias_region']),'minority':np.isin(y,[3,4])}
    out={}
    for name,m in masks.items(): out[name+'_logloss']=_logloss(y[m],P[m]) if np.any(m) else np.nan
    out['brier']=_brier(y,P); out['ece']=_ece(y,P); return out
def _score(m):
    if m['overall_logloss']>2.25 or m['conflict_logloss']>2.70 or m['bias_region_logloss']>2.75 or m['minority_logloss']>2.75:
        return 0.0,{k:0.0 for k in ['overall','conflict','rare_route','old_taxonomy','bias_region','minority','single','calibration']}
    comps={'overall':_dec(m['overall_logloss'],0.48,1.60),'conflict':_dec(m['conflict_logloss'],0.56,2.05),'rare_route':_dec(m['rare_route_logloss'],0.62,2.10),'old_taxonomy':_dec(m['old_taxonomy_logloss'],0.60,2.10),'bias_region':_dec(m['bias_region_logloss'],0.60,2.05),'minority':_dec(m['minority_logloss'],0.62,2.10),'single':_dec(m['single_logloss'],0.50,1.55),'calibration':0.55*_dec(m['brier'],0.28,0.72)+0.45*_dec(m['ece'],0.055,0.28)}
    reward=0.26*comps['overall']+0.17*comps['conflict']+0.12*comps['rare_route']+0.11*comps['old_taxonomy']+0.13*comps['bias_region']+0.12*comps['minority']+0.04*comps['single']+0.05*comps['calibration']
    if m['single_logloss']<0.80 and m['conflict_logloss']>1.25: reward=min(reward,0.62)
    if m['single_logloss']<0.80 and m['bias_region_logloss']>1.25: reward=min(reward,0.62)
    if m['rare_route_logloss']>1.45 and m['old_taxonomy_logloss']>1.45: reward=min(reward,0.58)
    return float(np.clip(reward,0,1)), comps
def evaluate(fit_fn,predict_fn,train_path=None,seeds=(551,552,553,554)):
    t0=time.perf_counter();
    if train_path is None: train_path=Path(__file__).resolve().parents[1]/'app'/'train_data.npz'
    tr=np.load(train_path); params=fit_fn(tr['train_X'],tr['train_observed_label'],tr['train_route'],tr['train_item'],tr['train_checked'])
    per={}; rewards=[]
    for seed in seeds:
        data=make_dataset(seed=seed,n_items=1500,split='hidden'); X=data['X'].copy(); obs=data['observed_label'].copy(); route=data['route'].copy(); item=data['item'].copy()
        P1=np.asarray(predict_fn(X,obs,route,item,params),float); rng=np.random.default_rng(seed+999); perm=rng.permutation(len(X)); inv=np.empty_like(perm); inv[perm]=np.arange(len(perm)); P2=np.asarray(predict_fn(X[perm].copy(),obs[perm].copy(),route[perm].copy(),item[perm].copy(),params),float)[inv]
        valid=(P1.shape==(len(X),K) and np.all(np.isfinite(P1)) and np.all(P1>=-1e-9) and np.allclose(P1.sum(axis=1),1,atol=1e-6) and np.allclose(P1,P2,atol=1e-8,rtol=1e-8) and np.allclose(X,data['X']) and np.array_equal(obs,data['observed_label']) and np.array_equal(route,data['route']) and np.array_equal(item,data['item']) and np.mean(P1.max(axis=1))<0.985 and len(np.unique(P1.argmax(axis=1)))>=4)
        if not valid:
            metrics={k:float('inf') for k in ['overall_logloss','single_logloss','conflict_logloss','rare_route_logloss','old_taxonomy_logloss','bias_region_logloss','minority_logloss','brier','ece']}; metrics['invalid']=1.0; reward,comps=0.0,{}
        else:
            metrics=_metrics(data['true_label'],P1,data['kind'],route); reward,comps=_score(metrics)
        rewards.append(reward); per[str(seed)]={'reward':float(reward),'metrics':{k:float(v) for k,v in metrics.items()},'component_rewards':{k:float(v) for k,v in comps.items()}}
    keys=[k for k in per[str(seeds[0])]['metrics'].keys() if k!='invalid']; agg={k:float(np.nanmean([per[str(s)]['metrics'].get(k,np.nan) for s in seeds])) for k in keys}; ar,ac=_score(agg); final=float(0.70*ar+0.30*np.mean(rewards)); return {'reward':final,'passed_cutoff':bool(final>0),'aggregate_metrics':agg,'aggregate_component_rewards':ac,'per_seed':per,'elapsed_seconds':time.perf_counter()-t0}
def evaluate_solution(solution_module=None):
    import importlib, sys
    app_dir=Path(__file__).resolve().parents[1]/'app'
    if solution_module is None:
        if str(app_dir) not in sys.path: sys.path.insert(0,str(app_dir))
        solution_module=importlib.import_module('solve')
    return evaluate(solution_module.fit_label_model, solution_module.predict_label_proba, train_path=app_dir/'train_data.npz')


from __future__ import annotations
import numpy as np
K=5; N_ROUTES=6

def _softmax(S):
    S=S-np.max(S,axis=1,keepdims=True); E=np.exp(S); return E/E.sum(axis=1,keepdims=True)
def _normalize(P):
    P=np.maximum(np.asarray(P,dtype=float),1e-12); return P/P.sum(axis=1,keepdims=True)
def _onehot(y,k=K,smooth=0.03):
    y=np.asarray(y,dtype=int); P=np.full((len(y),k),smooth/k); P[np.arange(len(y)),y]+=1-smooth; return _normalize(P)
def _region(X):
    X=np.asarray(X); r1=X[:,0]+0.55*X[:,2]-0.25*X[:,6]; r2=X[:,4]-0.70*X[:,7]+0.20*X[:,9]; r3=X[:,6]+0.45*X[:,10]-0.30*X[:,3]
    out=np.zeros(len(X),dtype=np.int64); out[r1>1.25]=1; out[(r2<-1.10)&(out==0)]=2; out[(r3>1.30)&(out==0)]=3; return out
def _feature_map(X):
    X=np.asarray(X,dtype=float); pairs=[(0,2),(1,3),(4,7),(5,8),(6,10),(0,9),(3,11),(2,6),(4,12),(7,13)]
    inter=np.column_stack([X[:,i]*X[:,j] for i,j in pairs])
    return np.column_stack([np.ones(len(X)),X,X**2,inter,np.sin(np.clip(X[:,[2,5,8]],-4,4)),(X[:,0]+0.55*X[:,2]-0.25*X[:,6])[:,None],(X[:,4]-0.70*X[:,7]+0.20*X[:,9])[:,None],(X[:,6]+0.45*X[:,10]-0.30*X[:,3])[:,None]])
def _fit(Phi,Y,lam=2.0):
    Phi=np.asarray(Phi,float); Y=np.asarray(Y,float); mu=Phi.mean(axis=0); sd=Phi.std(axis=0); sd[sd<1e-8]=1; Z=(Phi-mu)/sd; Z[:,0]=1
    A=Z.T@Z; reg=lam*np.eye(A.shape[0]); reg[0,0]=0; W=np.linalg.solve(A+reg,Z.T@Y); return {'W':W,'mu':mu,'sd':sd}
def _pred(model,Phi,temp=1.0):
    Z=(Phi-model['mu'])/model['sd']; Z[:,0]=1; return _softmax((Z@model['W'])/temp)
def _groups(item):
    item=np.asarray(item); order=np.argsort(item,kind='mergesort'); out=[]; s=0
    while s<len(order):
        e=s+1
        while e<len(order) and item[order[e]]==item[order[s]]: e+=1
        out.append(order[s:e]); s=e
    return out

def _initial_soft(X, obs, route, item, checked):
    groups=_groups(item); soft=np.zeros((len(X),K))+0.08
    route_w=np.array([1.15,0.72,0.40,0.34,1.45,0.32])
    # first, checked labels are trusted
    for idx in groups:
        q=np.ones(K)*0.10
        checked_idx=[j for j in idx if checked[j]]
        if checked_idx:
            for j in checked_idx: q[int(obs[j])]+=4.0
        else:
            for j in idx: q[int(obs[j])]+=route_w[int(route[j])]
        q=q/q.sum(); soft[idx]=q
    return soft

def _fit_tables(X, obs, route, pseudo, checked=None):
    reg=_region(X); table=np.ones((N_ROUTES,K,4,K))*0.12; glob=np.ones((N_ROUTES,K,K))*0.20; counts=np.zeros((N_ROUTES,K,4))
    for i in range(len(X)):
        r,l,g=int(route[i]),int(obs[i]),int(reg[i]); weight=2.5 if (checked is not None and checked[i]) else 1.0
        table[r,l,g]+=weight*pseudo[i]; glob[r,l]+=weight*pseudo[i]; counts[r,l,g]+=weight
    table=table/table.sum(axis=-1,keepdims=True); glob=glob/glob.sum(axis=-1,keepdims=True); alpha=counts/(counts+25.0)
    table=alpha[...,None]*table+(1-alpha[...,None])*glob[:,:,None,:]
    return table, glob

def _combine(X,obs,route,item,feat,table,prior,evidence=0.90):
    reg=_region(X); out=np.zeros_like(feat)
    for idx in _groups(item):
        base=np.mean(np.log(np.maximum(feat[idx],1e-12)),axis=0); lab=np.zeros(K)
        for j in idx:
            lab += evidence*np.log(np.maximum(table[int(route[j]),int(obs[j]),int(reg[j])],1e-12))
        q=np.exp(base+lab-np.max(base+lab)); q=q/q.sum(); q=0.94*q+0.04*prior+0.02/K; out[idx]=q
    return _normalize(out)

def fit_label_model(train_X, train_observed_label, train_route, train_item, train_checked):
    X=np.asarray(train_X,float); obs=np.asarray(train_observed_label,int); route=np.asarray(train_route,int); item=np.asarray(train_item,int); checked=np.asarray(train_checked,bool)
    soft=_initial_soft(X,obs,route,item,checked)
    # use checked labels directly when present, but do not ignore the large weak-labeled pool
    if checked.any():
        soft[checked]=_onehot(obs[checked],K,smooth=0.015)
    for it in range(4):
        model=_fit(_feature_map(X),soft,lam=3.8)
        feat=_pred(model,_feature_map(X),temp=1.04)
        pseudo=_normalize(0.64*feat+0.36*soft)
        if checked.any(): pseudo[checked]=_onehot(obs[checked],K,smooth=0.01)
        table,glob=_fit_tables(X,obs,route,pseudo,checked)
        soft=_combine(X,obs,route,item,feat,table,pseudo.mean(axis=0),evidence=0.78)
        if checked.any(): soft[checked]=_onehot(obs[checked],K,smooth=0.01)
    model=_fit(_feature_map(X),soft,lam=3.0)
    feat=_pred(model,_feature_map(X),temp=1.02); pseudo=_normalize(0.55*feat+0.45*soft)
    if checked.any(): pseudo[checked]=_onehot(obs[checked],K,smooth=0.01)
    table,glob=_fit_tables(X,obs,route,pseudo,checked)
    return {'model':model,'table':table,'prior':pseudo.mean(axis=0)}

def predict_label_proba(X, observed_label, route, item, params):
    X=np.asarray(X,float); obs=np.asarray(observed_label,int); route=np.asarray(route,int); item=np.asarray(item,int)
    feat=_pred(params['model'],_feature_map(X),temp=1.04)
    return _combine(X,obs,route,item,feat,params['table'],params['prior'],evidence=0.82)


