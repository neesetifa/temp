# Dependency bundle validation

The current model in `app/solve.py` needs to be replaced.

The rows come from Python package dependency bundles. Each node is an installed package distribution and each directed edge is a dependency declaration from one package to another. The training target contains one validation-workload value for every package row.

The previous model mostly treated packages independently. That misses cases where two packages with similar local metadata sit in very different dependency positions. A package with many downstream users does not behave like one with only outgoing dependencies, and optional or conditional declarations do not always act like ordinary dependencies.

Implement these two functions in `app/solve.py`:

```python
def fit_link_model(
    train_node_X,
    train_node_code,
    train_edge_src,
    train_edge_dst,
    train_edge_code,
    train_graph_X,
    train_node_offsets,
    train_edge_offsets,
    train_y,
):
    ...
```

```python
def predict_link_score(
    node_X,
    node_code,
    edge_src,
    edge_dst,
    edge_code,
    graph_X,
    node_offsets,
    edge_offsets,
    params,
):
    ...
```

The package rows for graph `i` are:

```python
node_offsets[i] : node_offsets[i + 1]
```

The edges for graph `i` are:

```python
edge_offsets[i] : edge_offsets[i + 1]
```

`edge_src[j] -> edge_dst[j]` means that the source package declares a dependency on the destination package. Node indices are global indices into the packed node arrays passed to the function.

`node_X` has these columns:

1. installed size in MB
2. installed file count
3. package metadata size in KB
4. number of non-extra requirements declared by the package
5. number of optional-extra requirements
6. entry-point count
7. package-classifier count
8. package major version

`node_code` has three categorical columns:

1. license-family code
2. development-status code
3. minimum-Python-version bucket

For the third column, `0` is unknown, `1` is Python 3.6 or earlier, then `2` through `7` correspond to Python 3.7 through Python 3.12 or later.

`edge_code` describes the dependency declaration:

- `0`: ordinary/unconditional dependency
- `1`: Python-version conditional dependency
- `2`: platform-conditional dependency
- `3`: optional-extra dependency
- `4`: another conditional dependency

`graph_X` contains bundle-level fields derived from the same package snapshot:

1. root package installed size in MB
2. total installed size of packages in the bundle, in MB
3. root package reverse-dependency count in the snapshot
4. root package dependency count in the snapshot
5. fraction of bundle edges that are optional-extra declarations
6. fraction of bundle edges that are not ordinary/unconditional declarations

Use the dependency structure as well as the package metadata. Direction, dependency type, and packages reached through nearby dependency chains can all carry useful information. A single degree count or one undirected neighbor average is not enough on all of the training examples.

`predict_link_score` must return one finite value per package row, in the same order as `node_X`.

Do not modify the input arrays. Do not read hidden files or use files outside the provided environment. Do not use external data or network access. The result must be deterministic for the same inputs.


solve

"""Starter implementation. Replace the two functions with a learned model."""
from __future__ import annotations
import numpy as np


def fit_link_model(train_node_X, train_node_code, train_edge_src, train_edge_dst, train_edge_code,
                   train_graph_X, train_node_offsets, train_edge_offsets, train_y):
    y = np.asarray(train_y, dtype=float)
    return {"mean": float(np.mean(y)) if y.size else 0.0}


def predict_link_score(node_X, node_code, edge_src, edge_dst, edge_code, graph_X, node_offsets, edge_offsets, params):
    n = int(np.asarray(node_X).shape[0])
    return np.full(n, float(params.get("mean", 0.0)), dtype=float)

hidden
"""Hidden verifier for the package-dependency realism build."""
from __future__ import annotations
import importlib.util,json,math,os
from pathlib import Path
from typing import Any
import numpy as np
from generator import SLICE_NAMES,generate_dataset

HIDDEN_SEED=81173
HIDDEN_N_GRAPHS=320
LOG_DIR=Path(os.environ.get('VERIFIER_LOG_DIR','/logs/verifier'))
APP_DIR=Path(os.environ.get('APP_DIR','/app'))
SLICE_GOOD=np.array([0.086,0.085,0.086,0.089,0.086,0.082,0.088,0.080,0.090,0.088],float)
SLICE_BAD =np.array([0.234,0.378,0.413,0.400,0.590,0.264,0.316,0.292,0.361,0.411],float)
OVERALL_GOOD=0.087
OVERALL_BAD=0.416


def _load_train(app):
 d=np.load(app/'train_data.npz');return (d['train_node_X'],d['train_node_code'],d['train_edge_src'],d['train_edge_dst'],d['train_edge_code'],d['train_graph_X'],d['train_node_offsets'],d['train_edge_offsets'],d['train_y'])

def _load_module(path):
 spec=importlib.util.spec_from_file_location('candidate_solve',path)
 if spec is None or spec.loader is None:raise RuntimeError(f'cannot load solve module at {path}')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def _run_in_process(solve_path,payload):
 mod=_load_module(solve_path);params=mod.fit_link_model(*payload['train']);return [np.asarray(mod.predict_link_score(*args,params),float) for args in payload['inputs']]

def _run_candidate(solve_path,payload):
 try:import sandbox_util # type: ignore
 except Exception:return _run_in_process(solve_path,payload)
 out=sandbox_util.run_eval(str(solve_path),payload);preds=out.get('preds') if isinstance(out,dict) else None
 if preds is None:raise RuntimeError("sandbox result did not contain 'preds'")
 return [np.asarray(p,float) for p in preds]

def _rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))
def _score(v,g,b):
 if not math.isfinite(v):return 0.0
 return float(np.clip((b-v)/(b-g),0,1))
def _validate(p,n):
 if p.shape!=(n,):raise ValueError(f'prediction shape must be ({n},), got {p.shape}')
 if not np.isfinite(p).all():raise ValueError('predictions must be finite')
def _metrics(y,p,sid):
 overall=_rmse(y,p);oscore=_score(overall,OVERALL_GOOD,OVERALL_BAD);per={};ss=[]
 for i,name in enumerate(SLICE_NAMES):
  m=sid==i
  if not np.any(m):continue
  r=_rmse(y[m],p[m]);s=_score(r,float(SLICE_GOOD[i]),float(SLICE_BAD[i]));per[str(name)]={'rmse':r,'score':s,'n':int(m.sum())};ss.append(s)
 slice_mean=float(np.mean(ss)) if ss else 0.0;reward=float(.35*oscore+.65*slice_mean)
 if overall>=OVERALL_BAD:reward=0.0
 return {'reward':reward,'overall_rmse':overall,'overall_score':oscore,'slice_mean':slice_mean,'per_slice':per}

def evaluate(solve_path=None,app_dir=None,log_dir=None):
 app=Path(app_dir) if app_dir is not None else APP_DIR;logs=Path(log_dir) if log_dir is not None else LOG_DIR;solve=Path(solve_path) if solve_path is not None else app/'solve.py'
 train=_load_train(app);hidden=generate_dataset(HIDDEN_N_GRAPHS,HIDDEN_SEED,'hidden');inputs=(hidden['node_X'],hidden['node_code'],hidden['edge_src'],hidden['edge_dst'],hidden['edge_code'],hidden['graph_X'],hidden['node_offsets'],hidden['edge_offsets'])
 payload={'train':train,'inputs':[inputs,inputs]};preds=_run_candidate(solve,payload)
 if len(preds)!=2:raise ValueError('candidate must return predictions for both verifier inputs')
 p0,p1=preds;_validate(p0,len(hidden['y']));_validate(p1,len(hidden['y']))
 if not np.allclose(p0,p1,rtol=0.0,atol=1e-8):raise ValueError('predictions must be deterministic for repeated inputs')
 metrics=_metrics(hidden['y'],p0,hidden['node_slice']);logs.mkdir(parents=True,exist_ok=True);(logs/'reward.txt').write_text(f"{metrics['reward']:.12f}\n");(logs/'metrics.json').write_text(json.dumps(metrics,sort_keys=True)+'\n');return metrics['reward'],metrics

if __name__=='__main__':
 reward,metrics=evaluate();print(json.dumps({'reward':reward,'overall_rmse':metrics['overall_rmse']},sort_keys=True))

from hidden_eval import evaluate


def test_hidden_eval_runs():
    reward, metrics = evaluate()
    assert 0.0 <= reward <= 1.0
    assert metrics["overall_rmse"] >= 0.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import numpy as np

N_EDGE_TYPE=5
SLICE_NAMES=np.array([
    'sparse_clean','direction_sensitive','edge_type_mix','deep_dependency_chain','hub_package',
    'conditional_dependency','matched_degree_context','large_bundle_context','rare_metadata_code','dense_bundle'
])


def _seed_file():
    here=Path(__file__).resolve().parent
    for p in [here/'real_dependency_seed.npz',here.parent/'tests'/'real_dependency_seed.npz']:
        if p.exists():return p
    raise FileNotFoundError('real_dependency_seed.npz not found')


@lru_cache(maxsize=1)
def _seed():
    d=np.load(_seed_file());s={k:d[k] for k in d.files};n=len(s['node_X_base'])
    out=[[] for _ in range(n)];inn=[[] for _ in range(n)]
    for e,(u,v) in enumerate(zip(s['edge_src'],s['edge_dst'])):
        out[int(u)].append(e);inn[int(v)].append(e)
    s['out_edges']=out;s['in_edges']=inn
    s['global_in_degree']=np.bincount(s['edge_dst'].astype(int),minlength=n)
    s['global_out_degree']=np.bincount(s['edge_src'].astype(int),minlength=n)
    combos=[tuple(map(int,r)) for r in s['node_code_base']];counts={}
    for c in combos:counts[c]=counts.get(c,0)+1
    s['code_combo_count']=np.array([counts[c] for c in combos],int)
    return s


def _split_buckets(split):
    if split=='train':return set(range(7))
    if split=='public':return {7}
    if split=='calibration':return {8}
    if split=='hidden':return {9}
    raise ValueError(split)


def _root_pool(s,split):
    degree=s['global_in_degree']+s['global_out_degree'];b=s['split_bucket']
    return np.where(np.isin(b,list(_split_buckets(split)))&(degree>=2))[0]


def _build_bundle(s,rng,split):
    pool=_root_pool(s,split);root=int(rng.choice(pool))
    root_in=int(s['global_in_degree'][root]);root_out=int(s['global_out_degree'][root])
    # Extraction radius/cap are sampling controls only; node/edge content always
    # comes from the observed dependency graph. Larger real roots naturally get
    # larger bundle caps.
    depth=int(rng.choice([2,3,4],p=[.26,.56,.18]))
    cap=int(np.clip(14+3*root_out+2*math.sqrt(root_in)+rng.integers(-4,9),14,110))
    chosen={root};front=[(root,0)];q=0
    while q<len(front) and len(chosen)<cap:
        u,d=front[q];q+=1
        if d>=depth:continue
        outs=list(s['out_edges'][u]);rng.shuffle(outs)
        for e in outs:
            v=int(s['edge_dst'][e])
            if v not in chosen and len(chosen)<cap:
                chosen.add(v);front.append((v,d+1))
        # Include a bounded sample of real reverse dependents so a bundle can
        # represent update/validation impact as well as install dependencies.
        if d<=1 and len(chosen)<cap:
            ins=list(s['in_edges'][u]);rng.shuffle(ins)
            take=min(len(ins),max(0,int(round(math.sqrt(len(ins))))))
            for e in ins[:take]:
                v=int(s['edge_src'][e])
                if v not in chosen and len(chosen)<cap:
                    chosen.add(v);front.append((v,d+1))
    nodes=np.array(sorted(chosen),int);loc={int(g):i for i,g in enumerate(nodes)}
    es=[];ed=[];ec=[]
    for u,v,t in zip(s['edge_src'],s['edge_dst'],s['edge_code']):
        u=int(u);v=int(v)
        if u in loc and v in loc:
            es.append(loc[u]);ed.append(loc[v]);ec.append(int(t))
    if len(nodes)<7 or len(es)<6:return _build_bundle(s,rng,split)
    X=s['node_X_base'][nodes].astype(float,copy=True);C=s['node_code_base'][nodes].astype(int,copy=True)
    ec_arr=np.asarray(ec,int)
    gx=np.array([
        float(s['node_X_base'][root,0]),
        float(np.sum(X[:,0])),
        float(root_in),
        float(root_out),
        float(np.mean(ec_arr==3)) if len(ec_arr) else 0.0,
        float(np.mean(ec_arr!=0)) if len(ec_arr) else 0.0,
    ],float)
    return nodes,X,C,gx,np.asarray(es,int),np.asarray(ed,int),ec_arr


def _z(X):
    X=np.asarray(X,float);return np.log1p(np.maximum(X,0.0))


def _target_for_bundle(X,C,gx,src,dst,etype,rng):
    n=len(X);Z=_z(X);size,files,meta,base_req,opt_req,entry,classifiers,major=Z.T
    lic,dev,pyfloor=C.T;unstable=np.isin(dev,[1,2,3,4]).astype(float);mature=np.isin(dev,[5,6]).astype(float)
    root_size,total_size,root_in,root_out,optional_frac,nonbase_frac=gx
    own=(.94+.36*size+.105*files+.055*meta+.09*base_req+.04*entry+.035*classifiers+.045*major
         +.11*unstable-.045*mature+.032*pyfloor+.012*(lic==0))
    out_w=np.array([1.00,.70,.62,.34,.50]);in_w=np.array([1.00,.78,.70,.40,.55])
    dep_work=.28+.23*size+.055*files+.065*entry+.045*base_req
    rev_work=.24+.17*size+.050*files+.075*entry+.040*base_req
    out_sum=np.zeros(n);in_sum=np.zeros(n);in_type=np.zeros((n,5));out_type=np.zeros((n,5))
    incoming=[[] for _ in range(n)];outgoing=[[] for _ in range(n)]
    for e,(u0,v0,t0) in enumerate(zip(src,dst,etype)):
        u=int(u0);v=int(v0);t=int(t0);outgoing[u].append(e);incoming[v].append(e)
        maturity_gate=1+.10*unstable[v]+.06*(pyfloor[v]>pyfloor[u])
        reverse_gate=1+.08*unstable[u]+.035*(pyfloor[u]>pyfloor[v])
        ov=out_w[t]*dep_work[v]*maturity_gate;iv=in_w[t]*rev_work[u]*reverse_gate
        out_sum[u]+=ov;in_sum[v]+=iv;out_type[u,t]+=ov;in_type[v,t]+=iv
    pair=np.array([[.34,.22,.20,.11,.16],[.25,.18,.16,.09,.13],[.23,.16,.15,.08,.12],[.15,.11,.10,.055,.08],[.19,.14,.12,.07,.10]])
    f2=np.zeros(n);r2=np.zeros(n);two_count=np.zeros(n,int)
    for mid in range(n):
        if not incoming[mid] or not outgoing[mid]:continue
        mid_gate=.86+.08*unstable[mid]+.04*base_req[mid]
        for e1 in incoming[mid]:
            a=int(src[e1]);t1=int(etype[e1])
            for e2 in outgoing[mid]:
                c=int(dst[e2]);t2=int(etype[e2])
                if a==c:continue
                w=pair[t1,t2]*mid_gate
                f2[a]+=w*(.34+.16*dep_work[c]);r2[c]+=w*(.32+.14*rev_work[a]);two_count[a]+=1;two_count[c]+=1
    indeg=np.bincount(dst,minlength=n).astype(float);outdeg=np.bincount(src,minlength=n).astype(float)
    imbalance=np.tanh((indeg-outdeg)/(1+np.sqrt(indeg+outdeg)))
    typed=np.tanh((in_type[:,0]+.6*in_type[:,1]-out_type[:,0]-.4*out_type[:,3])/2)
    bundle_scale=.91+.034*np.log1p(total_size)+.11*optional_frac+.055*nonbase_frac+.018*np.log1p(root_in)
    rel=(.88*np.log1p(out_sum)+1.10*np.log1p(in_sum)+.67*np.log1p(f2)+.86*np.log1p(r2))
    y=(own+bundle_scale*rel+.16*imbalance+.13*typed+.055*np.log1p(indeg*outdeg)+.045*optional_frac*np.log1p(out_type[:,3]+in_type[:,3]))
    noise_sd=.055+.018*np.sqrt(np.maximum(y,0))+.018*(unstable>0)
    y=y+rng.normal(0,noise_sd,n)
    in_type_count=np.stack([np.bincount(dst[etype==t],minlength=n) if np.any(etype==t) else np.zeros(n) for t in range(5)],axis=1)
    out_type_count=np.stack([np.bincount(src[etype==t],minlength=n) if np.any(etype==t) else np.zeros(n) for t in range(5)],axis=1)
    return y,{'in_deg':indeg,'out_deg':outdeg,'in_type_count':in_type_count,'out_type_count':out_type_count,'two_count':two_count}


def _slice_ids(s,global_nodes,gx,src,dst,etype,aux):
    n=len(global_nodes);ind=aux['in_deg'];out=aux['out_deg'];deg=ind+out;two=aux['two_count']
    incident=np.sum(aux['in_type_count'][:,1:]+aux['out_type_count'][:,1:],axis=1)
    type_count=np.sum((aux['in_type_count']+aux['out_type_count'])>0,axis=1)
    density=len(src)/max(n,1);rare=s['code_combo_count'][global_nodes]<=2;hub=(s['global_in_degree'][global_nodes]>=18)|(ind>=7)
    large=(n>=45) or (gx[1]>=250) or (gx[2]>=20)
    sid=np.zeros(n,int)
    for i in range(n):
        if rare[i]:sid[i]=8
        elif hub[i]:sid[i]=4
        elif type_count[i]>=3 and incident[i]>=2:sid[i]=2
        elif density>=3.0 and deg[i]>=2:sid[i]=9
        elif two[i]>=10:sid[i]=3
        elif abs(ind[i]-out[i])>=3 and deg[i]>=3:sid[i]=1
        elif 2<=deg[i]<=6 and two[i]>=4:sid[i]=6
        elif large and deg[i]>=1:sid[i]=7
        elif incident[i]>=1:sid[i]=5
        else:sid[i]=0
    return sid


def generate_dataset(n_graphs,seed_value,split='train'):
    s=_seed();rng=np.random.default_rng(seed_value);Xs=[];Cs=[];Gs=[];Ss=[];Ds=[];Es=[];Ys=[];Slices=[];no=[0];eo=[0]
    for _ in range(int(n_graphs)):
        gnodes,X,C,gx,src,dst,et=_build_bundle(s,rng,split);y,aux=_target_for_bundle(X,C,gx,src,dst,et,rng);sid=_slice_ids(s,gnodes,gx,src,dst,et,aux)
        base=no[-1];Xs.append(X);Cs.append(C);Gs.append(gx);Ss.append(src+base);Ds.append(dst+base);Es.append(et);Ys.append(y);Slices.append(sid);no.append(base+len(X));eo.append(eo[-1]+len(src))
    return {'node_X':np.concatenate(Xs).astype(float),'node_code':np.concatenate(Cs).astype(int),'edge_src':np.concatenate(Ss).astype(int),'edge_dst':np.concatenate(Ds).astype(int),'edge_code':np.concatenate(Es).astype(int),'graph_X':np.asarray(Gs,float),'node_offsets':np.asarray(no,int),'edge_offsets':np.asarray(eo,int),'y':np.concatenate(Ys).astype(float),'node_slice':np.concatenate(Slices).astype(int)}

reference
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge

N_EDGE_TYPE=5;N_LICENSE=9;N_DEV=8;N_PYFLOOR=8

def _onehot(v,n):
 v=np.asarray(v,int);o=np.zeros((len(v),n),float);ok=(v>=0)&(v<n);o[np.arange(len(v))[ok],v[ok]]=1;return o

def _z(X):return np.log1p(np.maximum(np.asarray(X,float),0.0))

def _mechanism_features(X,C,src_all,dst_all,et_all,G,noff,eoff):
 X=np.asarray(X,float);C=np.asarray(C,int);src_all=np.asarray(src_all,int);dst_all=np.asarray(dst_all,int);et_all=np.asarray(et_all,int);G=np.asarray(G,float);noff=np.asarray(noff,int);eoff=np.asarray(eoff,int);Z=_z(X);rows=[]
 for gi in range(len(noff)-1):
  a,b=int(noff[gi]),int(noff[gi+1]);ea,eb=int(eoff[gi]),int(eoff[gi+1]);n=b-a;Zi=Z[a:b];Ci=C[a:b];lic,dev,pyfloor=Ci.T
  src=src_all[ea:eb]-a;dst=dst_all[ea:eb]-a;et=np.clip(et_all[ea:eb],0,4);gx=G[gi];root_size,total_size,root_in,root_out,optional_frac,nonbase_frac=gx
  size,files,meta,base_req,opt_req,entry,classifiers,major=Zi.T;unstable=np.isin(dev,[1,2,3,4]).astype(float);mature=np.isin(dev,[5,6]).astype(float)
  own=np.column_stack([size,files,meta,base_req,opt_req,entry,classifiers,major,unstable,mature,pyfloor,(lic==0).astype(float)])
  out_w=np.array([1.00,.70,.62,.34,.50]);in_w=np.array([1.00,.78,.70,.40,.55]);dep=.28+.23*size+.055*files+.065*entry+.045*base_req;rev=.24+.17*size+.050*files+.075*entry+.040*base_req
  os=np.zeros(n);ins=np.zeros(n);it=np.zeros((n,5));ot=np.zeros((n,5));incoming=[[] for _ in range(n)];outgoing=[[] for _ in range(n)]
  for e,(u0,v0,t0) in enumerate(zip(src,dst,et)):
   u=int(u0);v=int(v0);t=int(t0);outgoing[u].append(e);incoming[v].append(e);mg=1+.10*unstable[v]+.06*(pyfloor[v]>pyfloor[u]);rg=1+.08*unstable[u]+.035*(pyfloor[u]>pyfloor[v]);ov=out_w[t]*dep[v]*mg;iv=in_w[t]*rev[u]*rg;os[u]+=ov;ins[v]+=iv;ot[u,t]+=ov;it[v,t]+=iv
  pair=np.array([[.34,.22,.20,.11,.16],[.25,.18,.16,.09,.13],[.23,.16,.15,.08,.12],[.15,.11,.10,.055,.08],[.19,.14,.12,.07,.10]]);f2=np.zeros(n);r2=np.zeros(n)
  for mid in range(n):
   if not incoming[mid] or not outgoing[mid]:continue
   mg=.86+.08*unstable[mid]+.04*base_req[mid]
   for e1 in incoming[mid]:
    aa=int(src[e1]);t1=int(et[e1])
    for e2 in outgoing[mid]:
     cc=int(dst[e2]);t2=int(et[e2])
     if aa==cc:continue
     w=pair[t1,t2]*mg;f2[aa]+=w*(.34+.16*dep[cc]);r2[cc]+=w*(.32+.14*rev[aa])
  indeg=np.bincount(dst,minlength=n).astype(float);outdeg=np.bincount(src,minlength=n).astype(float);imb=np.tanh((indeg-outdeg)/(1+np.sqrt(indeg+outdeg)));typed=np.tanh((it[:,0]+.6*it[:,1]-ot[:,0]-.4*ot[:,3])/2);scale=.91+.034*np.log1p(total_size)+.11*optional_frac+.055*nonbase_frac+.018*np.log1p(root_in)
  rel=np.column_stack([scale*np.log1p(os),scale*np.log1p(ins),scale*np.log1p(f2),scale*np.log1p(r2),imb,typed,np.log1p(indeg*outdeg),optional_frac*np.log1p(ot[:,3]+it[:,3]),np.log1p(indeg),np.log1p(outdeg)])
  cat=np.concatenate([_onehot(lic,N_LICENSE),_onehot(dev,N_DEV),_onehot(pyfloor,N_PYFLOOR)],axis=1);g=np.repeat(np.array([[np.log1p(root_size),np.log1p(total_size),np.log1p(root_in),np.log1p(root_out),optional_frac,nonbase_frac]],float),n,axis=0);rows.append(np.concatenate([own,rel,cat,g],axis=1))
 return np.concatenate(rows,axis=0)

def fit_link_model(train_node_X,train_node_code,train_edge_src,train_edge_dst,train_edge_code,train_graph_X,train_node_offsets,train_edge_offsets,train_y):
 F=_mechanism_features(train_node_X,train_node_code,train_edge_src,train_edge_dst,train_edge_code,train_graph_X,train_node_offsets,train_edge_offsets);m=Ridge(alpha=.18);m.fit(F,np.asarray(train_y,float));return {'model':m}

def predict_link_score(node_X,node_code,edge_src,edge_dst,edge_code,graph_X,node_offsets,edge_offsets,params):
 F=_mechanism_features(node_X,node_code,edge_src,edge_dst,edge_code,graph_X,node_offsets,edge_offsets);return np.asarray(params['model'].predict(F),float)
