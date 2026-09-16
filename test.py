# Conditional population response

Build a reusable regression model for heterogeneous population assays.

Each case contains a variable number of object/cell measurements (`cells`, 12 numeric channels), a 5-dimensional treatment/condition vector, and two case-level metadata fields. The target is one continuous sample-level response.

Your model must use the labeled training data in `/app/train.npz` and implement `/app/model.py` with:

```python
fit_model(train_path: str, model_path: str) -> None
predict(cells, offsets, treatment, case_meta, model_path: str) -> numpy.ndarray
```

`offsets` has length `n_cases + 1`; cells for case `i` are `cells[offsets[i]:offsets[i+1]]`. `predict` must return one finite float per case, in input order.

The hidden evaluator uses fresh independent cases from the same assay process. Population sizes are irregular. The treatment vector is part of the prediction query: the usefulness of a cell phenotype may depend on the treatment condition, so do not assume that a single treatment-independent population summary is sufficient.

The primary objective is RMSE on hidden cases. The evaluator also checks broad, naturally occurring cohorts (smaller populations, stronger treatment conditions, and rarer case families) so a method should work across the assay distribution rather than only its easiest region.

You may use NumPy, SciPy, pandas, scikit-learn, and PyTorch. Do not modify the training data or evaluator files.

## Data statement

This is a grounded synthetic benchmark modeling a heterogeneous population assay. Individual measurements, population composition, treatment descriptors, nuisance variation, and responses are synthetic. Fixtures are designed to resemble irregular population-measurement data with correlated channels, long-tailed subpopulations, variable population sizes, sample-level measurement shifts, heteroscedasticity, sparse channel dropout, and heterogeneous treatment response. The task does not claim to reproduce a particular clinical assay or deployed instrument.

reference
import numpy as np, torch

class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phi=torch.nn.Sequential(torch.nn.Linear(22,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,48),torch.nn.SiLU())
        self.rho=torch.nn.Sequential(torch.nn.Linear(55,64),torch.nn.SiLU(),torch.nn.Linear(64,32),torch.nn.SiLU(),torch.nn.Linear(32,1))
    def forward(self,x,q,m):
        qq=q[:,None,:].expand(-1,x.shape[1],-1)
        z=self.phi(torch.cat([x,qq,x[:,:,:5]*qq],-1)).mean(1)
        return self.rho(torch.cat([z,q,m],-1)).squeeze(-1)

def _pack(d,k=96):
    X,o=d['cells'],d['offsets']; out=[]
    for i in range(len(d['treatment'])):
        a=X[o[i]:o[i+1]]; idx=np.linspace(0,len(a)-1,k).round().astype(int); out.append(a[idx])
    return np.asarray(out,np.float32)

def fit_model(train_path,model_path):
    torch.set_num_threads(4); torch.manual_seed(17); np.random.seed(17)
    z=np.load(train_path); d={k:z[k] for k in z.files}; X=_pack(d)
    mu=X.reshape(-1,12).mean(0); sd=X.reshape(-1,12).std(0)+1e-5; X=(X-mu)/sd
    Q=d['treatment'].astype('float32'); M=d['case_meta'].astype('float32'); Y=d['y'].astype('float32')
    net=_Net(); opt=torch.optim.AdamW(net.parameters(),lr=1.5e-3,weight_decay=2e-4)
    X,Q,M,Y=map(torch.tensor,(X,Q,M,Y))
    best=None; bestloss=1e9
    # deterministic 90/10 internal split; no hidden tuning
    nv=max(240,len(Y)//10); ti=torch.arange(len(Y)-nv); vi=torch.arange(len(Y)-nv,len(Y))
    for ep in range(22):
        perm=ti[torch.randperm(len(ti))]; net.train()
        for s in range(0,len(perm),64):
            ii=perm[s:s+64]; pred=net(X[ii],Q[ii],M[ii]); loss=((pred-Y[ii])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl=float(((net(X[vi],Q[vi],M[vi])-Y[vi])**2).mean())
        if vl<bestloss: bestloss=vl; best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    torch.save({'state':best,'mu':mu,'sd':sd},model_path)

def predict(cells,offsets,treatment,case_meta,model_path):
    ck=torch.load(model_path,map_location='cpu',weights_only=False); net=_Net(); net.load_state_dict(ck['state']); net.eval()
    d={'cells':np.asarray(cells),'offsets':np.asarray(offsets),'treatment':np.asarray(treatment),'case_meta':np.asarray(case_meta)}
    X=(_pack(d)-ck['mu'])/ck['sd']; Q=np.asarray(treatment,np.float32); M=np.asarray(case_meta,np.float32)
    out=[]
    with torch.no_grad():
        for s in range(0,len(Q),256): out.append(net(torch.tensor(X[s:s+256]),torch.tensor(Q[s:s+256]),torch.tensor(M[s:s+256])).numpy())
    return np.concatenate(out).astype(float)

generator
import numpy as np

# Grounded synthetic heterogeneous-population assay.
# All cases are independent specimens; hidden cases use fresh RNG seeds.

def _softplus(x): return np.logaddexp(0.0, x)

def generate(n_cases=2500, seed=0, labeled=True):
    rng=np.random.default_rng(seed)
    K,D,Q=9,12,5
    # fixed population-level assay physics; not exposed as parameters to agents
    gr=np.random.default_rng(73191)
    centers=gr.normal(0,1,(K,D)); centers[:,0]+=np.linspace(-1.7,1.7,K)
    centers[:,1]=0.55*centers[:,0]+gr.normal(0,.65,K)
    load=gr.normal(0,.34,(K,D,3))
    qW=gr.normal(0,.8,(D,Q)); qW/=np.sqrt(D)
    q2=gr.normal(0,.45,(D,Q)); q2/=np.sqrt(D)
    typeq=gr.normal(0,.75,(K,Q)); typeq-=typeq.mean(0,keepdims=True)
    base=gr.normal(0,.35,K)
    cells=[]; offs=[0]; qs=[]; ys=[]; meta=[]
    for i in range(n_cases):
        # long-tailed specimen composition with correlated production/biological family
        fam=int(rng.choice(5,p=[.43,.25,.16,.10,.06]))
        alpha=np.exp(gr.normal(-.25,.65,K))*(0.65+0.18*fam)
        alpha[(fam+np.arange(2))%K]*=3.2
        p=rng.dirichlet(alpha)
        n=int(np.clip(np.exp(rng.normal(np.log(115),.48)),35,430))
        z=rng.choice(K,n,p=p)
        nuisance=rng.normal(0,.32,3)
        X=centers[z]+np.einsum('ndk,k->nd',load[z],nuisance)
        # non-Gaussian marker channels, bounded/saturated assay channels, heteroscedasticity
        X += rng.normal(0, .25+0.05*np.abs(X), X.shape)
        X[:,2]=np.sign(X[:,2])*np.log1p(np.abs(X[:,2])*1.8)
        X[:,3]=2.4/(1+np.exp(-X[:,3]))-1.2
        X[:,4]=np.maximum(-1.35,X[:,4])
        # sparse channel dropout, encoded as sentinel 0 after specimen-wise centering-like measurement
        miss=rng.random(X.shape)<(0.006+0.012*(z[:,None]==7))
        X[miss]=0.0
        q=rng.normal(0,1,Q); q[0]+=rng.choice([-1,0,1],p=[.18,.64,.18])*.8
        q=np.clip(q,-2.6,2.6)
        # true aggregate response depends on treatment-conditioned interpretation of each cell
        s=(X@qW)@q/np.sqrt(Q) + 0.38*((X*X)@q2)@np.tanh(q)/np.sqrt(Q)
        s += (typeq[z]@q)/np.sqrt(Q)+base[z]
        # heterogeneous activation: both positive and negative responding subpopulations
        r=np.tanh(s*.72)+0.22*np.sin(s*1.45)+0.10*np.tanh(X[:,0]*q[1]-X[:,5]*q[3])
        y=0.82*r.mean()+0.10*np.quantile(r,.82)-0.07*np.quantile(r,.18)
        y += 0.045*nuisance[0]-0.025*nuisance[1]+rng.normal(0,.018+0.010/np.sqrt(n/100))
        cells.append(X.astype(np.float32)); offs.append(offs[-1]+n); qs.append(q.astype(np.float32)); meta.append([fam,np.log1p(n)])
        ys.append(y)
    out={'cells':np.concatenate(cells), 'offsets':np.asarray(offs,np.int64), 'treatment':np.asarray(qs,np.float32), 'case_meta':np.asarray(meta,np.float32)}
    if labeled: out['y']=np.asarray(ys,np.float32)
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=2500); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--unlabeled',action='store_true'); a=ap.parse_args()
    np.savez_compressed(a.out,**generate(a.n,a.seed,not a.unlabeled))


hidden
import importlib.util, json, math, os, tempfile, numpy as np
from pathlib import Path
from generator import generate

def load_model(path='/app/model.py'):
 spec=importlib.util.spec_from_file_location('agent_model',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def score_rmse(r):
 # continuous anchors from frozen v0.1 calibration; valid weak models retain gradient
 BAD,GOOD=.225,.105
 return float(np.clip((BAD-r)/(BAD-GOOD),0,1))
def evaluate(model_py='/app/model.py'):
 m=load_model(model_py); model_path='/tmp/conditional_population_model.bin'
 m.fit_model('/app/train.npz',model_path)
 d=generate(1200,882341,labeled=True)
 p=np.asarray(m.predict(d['cells'],d['offsets'],d['treatment'],d['case_meta'],model_path),float)
 if p.shape!=(len(d['y']),) or not np.all(np.isfinite(p)): return {'reward':0.0,'error':'invalid predictions'}
 err=p-d['y']; overall=float(np.sqrt(np.mean(err**2)))
 sizes=np.diff(d['offsets']); qnorm=np.linalg.norm(d['treatment'],axis=1); fam=d['case_meta'][:,0]
 masks={'small_population':sizes<=np.quantile(sizes,.30),'strong_treatment':qnorm>=np.quantile(qnorm,.70),'rare_family':fam>=3}
 metrics={'rmse':overall}; parts=[score_rmse(overall)]; weights=[.70]
 for k,mask in masks.items():
  r=float(np.sqrt(np.mean(err[mask]**2))); metrics[k+'_rmse']=r; metrics[k+'_n']=int(mask.sum()); parts.append(score_rmse(r)); weights.append(.10)
 reward=float(np.dot(parts,weights))
 return {'reward':reward,**metrics}
if __name__=='__main__': print(json.dumps(evaluate(),indent=2))


test main
import numpy as np, sys
from generator import generate

def test_generator_separation_and_shapes():
 a=generate(50,1); b=generate(50,2)
 assert a['offsets'][0]==0 and a['offsets'][-1]==len(a['cells'])
 assert len(a['y'])==50 and a['treatment'].shape==(50,5)
 assert not np.array_equal(a['cells'][:100],b['cells'][:100])
 assert np.min(np.diff(a['offsets']))>=35 and np.max(np.diff(a['offsets']))<=430

def test_realism_fixture_properties():
 d=generate(800,9); n=np.diff(d['offsets']); fam=d['case_meta'][:,0].astype(int)
 assert np.std(n)>30 and len(np.unique(n))>100
 counts=np.bincount(fam,minlength=5); assert counts.max()>2*counts.min()
 assert np.mean(d['cells']==0)>0.003
 assert np.all(np.isfinite(d['y'])) and np.std(d['y'])>.1

  
