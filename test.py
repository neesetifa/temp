# Speed up the packed lattice scorer

The old scorer only returned the final lattice scores. We now also need row
posteriors for diagnostics.

Please implement `score_packed_lattices` in `app/solve.py`.

The input stores multiple variable-length lattices in packed arrays. Candidate
rows are grouped into frames with `frame_offsets`; frames are grouped into
sequences with `sequence_offsets`.

The decoder state is a pair of `(context, label)`. This applies at the start
and end too: `start_scores` and `end_scores` are tables with shape
`(num_contexts, num_labels)`.

For a candidate row with context `c` and label `y`, a previous component
`(pc, py)` contributes

`previous_score[pc, py] + transition_scores[pc, c, py, y] + candidate_scores[row]`

so the transition table is indexed as previous context, current context,
previous label, current label.

Rows with the same label are combined to decide which labels fit in the frame
budget. Forward values combine with `logsumexp`, best-path values combine with
`max`, and a label is pinned if any of its rows is pinned. Reachable pinned
labels stay in the frame without using the ordinary budget. Rank the remaining
labels by their combined forward score; exact ties go to the smaller label id.

The budget decision is at the label level, but the recurrence state is still at
the `(context, label)` level. If a label survives, carry every reachable context
for that label into the next frame. Do not collapse the label down to a single
context.

Use the labels selected by that forward ranking for both returned recurrences.
Do not run a separate beam for the best-path score. An empty frame, or a frame
where nothing survives, makes that sequence impossible, but it should not affect
the other sequences in the packed input.

Return three float64 arrays:

`log_partitions` with shape `(batch,)`
`best_scores` with shape `(batch,)`
`row_log_marginals` with shape equal to `candidate_scores.shape`

`row_log_marginals[row]` is the log posterior probability that the pruned
lattice path used that proposal row. Rows removed by pruning, unreachable rows,
and rows in impossible sequences should get `-inf`.

The posterior is for the proposal row, not just for its label. If several rows
merge into the same `(context, label)` component, they split that component's
posterior according to their log contribution to the component.

Long sequences need to remain numerically stable. The large cases are meant to
be processed as packed data, so simply enumerating all paths is not feasible.
Avoid allocating a tensor covering every frame and every possible
state-to-state transition.


from __future__ import annotations
import numpy as np

def pack_sequences(frames_per_sequence):
    scores, labels, contexts, pins = [], [], [], []
    frame_offsets=[0]; sequence_offsets=[0]; budgets=[]
    for seq in frames_per_sequence:
        for frame in seq:
            budgets.append(int(frame['budget']))
            scores.extend(frame['scores']); labels.extend(frame['labels']); contexts.extend(frame['contexts']); pins.extend(frame['pinned'])
            frame_offsets.append(len(scores))
        sequence_offsets.append(len(budgets))
    return (np.asarray(scores,dtype=np.float64), np.asarray(labels,dtype=np.int64), np.asarray(contexts,dtype=np.int64), np.asarray(pins,dtype=bool), np.asarray(frame_offsets,dtype=np.int64), np.asarray(sequence_offsets,dtype=np.int64), np.asarray(budgets,dtype=np.int64))

def random_case(seed:int,batch:int=4,n_contexts:int=3,n_labels:int=8,min_frames:int=2,max_frames:int=6,min_rows:int=4,max_rows:int=14,allow_empty:bool=False):
    rng=np.random.default_rng(seed)
    transition=rng.normal(0,0.7,size=(n_contexts,n_contexts,n_labels,n_labels)); transition[rng.random(transition.shape)<0.08]=-np.inf
    start=rng.normal(0,0.4,size=(n_contexts,n_labels)); start[rng.random(start.shape)<0.1]=-np.inf
    end=rng.normal(0,0.4,size=(n_contexts,n_labels)); seqs=[]
    for _ in range(batch):
        seq=[]
        for _ in range(rng.integers(min_frames,max_frames+1)):
            rows=0 if (allow_empty and rng.random()<0.05) else int(rng.integers(min_rows,max_rows+1))
            labels=rng.integers(0,n_labels,size=rows); contexts=rng.integers(0,n_contexts,size=rows)
            if rows>=6:
                labels[-3:]=labels[0]; contexts[-3]=contexts[0]; contexts[-2]=(contexts[0]+1)%n_contexts; contexts[-1]=contexts[0]
            seq.append({'scores':rng.normal(0,1.0,size=rows).tolist(),'labels':labels.tolist(),'contexts':contexts.tolist(),'pinned':(rng.random(rows)<0.1).tolist(),'budget':int(rng.integers(0,min(5,n_labels)+1))})
        seqs.append(seq)
    return (*pack_sequences(seqs), transition, start, end)

def large_case(seed=99,batch=8,frames=50,n_contexts=4,n_labels=80,rows=64):
    rng=np.random.default_rng(seed)
    transition=rng.normal(0,0.35,size=(n_contexts,n_contexts,n_labels,n_labels)); transition[rng.random(transition.shape)<0.025]=-np.inf
    start=rng.normal(0,0.25,size=(n_contexts,n_labels)); end=rng.normal(0,0.25,size=(n_contexts,n_labels)); seqs=[]
    for _ in range(batch):
        seq=[]
        for _ in range(frames):
            labels=rng.integers(0,n_labels,size=rows); contexts=rng.integers(0,n_contexts,size=rows)
            labels[-8:]=labels[:8]; contexts[-4:]=contexts[:4]
            seq.append({'scores':rng.normal(0,0.7,size=rows).tolist(),'labels':labels.tolist(),'contexts':contexts.tolist(),'pinned':(rng.random(rows)<0.03).tolist(),'budget':14})
        seqs.append(seq)
    return (*pack_sequences(seqs), transition, start, end)



from __future__ import annotations
import json, time, numpy as np
from generators import pack_sequences, random_case, large_case
from reference_solution import score_packed_lattices as oracle

def _same(got, exp, atol=1e-9):
    return np.allclose(got[0], exp[0], rtol=1e-9, atol=atol, equal_nan=True) and np.allclose(got[1], exp[1], rtol=1e-9, atol=atol, equal_nan=True) and np.allclose(got[2], exp[2], rtol=1e-9, atol=atol, equal_nan=True)

def _check(name, solution, args, results):
    exp=oracle(*args); got=solution(*args)
    results[name]=bool(isinstance(got,tuple) and len(got)==3 and got[0].shape==exp[0].shape and got[1].shape==exp[1].shape and got[2].shape==exp[2].shape and _same(got,exp))

def _finite_difference_check(solution,args,rows=(0,1,3,6),eps=1e-5):
    z,b,m=solution(*args); ok=True
    for r in rows:
        if r>=len(args[0]) or np.isneginf(m[r]): continue
        plus=[np.array(x,copy=True) if isinstance(x,np.ndarray) else x for x in args]
        minus=[np.array(x,copy=True) if isinstance(x,np.ndarray) else x for x in args]
        plus[0][r]+=eps; minus[0][r]-=eps
        deriv=(solution(*plus)[0][0]-solution(*minus)[0][0])/(2*eps)
        if not np.allclose(deriv,np.exp(m[r]),rtol=5e-4,atol=5e-5): ok=False
    return bool(ok)

def evaluate(solution, run_large=True):
    results={}
    seq=[[{'scores':[0.0,-0.7,0.2],'labels':[0,0,1],'contexts':[0,1,0],'pinned':[False,False,False],'budget':1},{'scores':[0.0,0.1],'labels':[1,1],'contexts':[0,1],'pinned':[False,False],'budget':1}]]
    T=np.zeros((2,2,2,2)); T[1,1,0,1]=3.0; start=np.zeros((2,2)); end=np.zeros((2,2))
    _check('multi_context_carry_future_transition', solution, (*pack_sequences(seq),T,start,end), results)
    seq=[[{'scores':[0.0,-0.1,2.0],'labels':[0,0,1],'contexts':[0,1,0],'pinned':[True,False,False],'budget':0},{'scores':[0.0],'labels':[1],'contexts':[1],'pinned':[False],'budget':1}]]
    T=np.zeros((2,2,2,2)); T[1,1,0,1]=4.0
    _check('pinned_label_carries_all_contexts', solution, (*pack_sequences(seq),T,start,end), results)
    seq=[[{'scores':[0.0,np.log(5.0),0.3],'labels':[0,0,1],'contexts':[0,0,0],'pinned':[False,False,False],'budget':1}]]
    _check('duplicate_component_row_posterior_split', solution, (*pack_sequences(seq),np.zeros((1,1,2,2)),np.zeros((1,2)),np.zeros((1,2))), results)
    seq=[[{'scores':[0.0,-0.2],'labels':[0,0],'contexts':[0,1],'pinned':[False,False],'budget':1}]]
    _check('context_specific_end_scores', solution, (*pack_sequences(seq),np.zeros((2,2,1,1)),np.zeros((2,1)),np.array([[0.0],[3.0]])), results)
    seqs=[[{'scores':[0.0],'labels':[0],'contexts':[0],'pinned':[False],'budget':1},{'scores':[],'labels':[],'contexts':[],'pinned':[],'budget':1}],[{'scores':[0.2],'labels':[1],'contexts':[0],'pinned':[False],'budget':1}]]
    _check('impossible_sequence_isolation', solution, (*pack_sequences(seqs),np.zeros((1,1,2,2)),np.zeros((1,2)),np.zeros((1,2))), results)
    for seed in [3,9,17,28,44]: _check(f'random_{seed}', solution, random_case(seed), results)
    fd_args=random_case(123,batch=1,min_frames=3,max_frames=3,allow_empty=False)
    results['finite_difference_row_marginals']=_finite_difference_check(solution,fd_args)
    long_seq=[[]]
    for _ in range(650): long_seq[0].append({'scores':[12.0,11.7,11.5],'labels':[0,0,1],'contexts':[0,1,0],'pinned':[False,False,False],'budget':1})
    _check('long_sequence_stability', solution, (*pack_sequences(long_seq),np.zeros((2,2,2,2)),np.zeros((2,2)),np.zeros((2,2))), results)
    runtime=None
    if run_large:
        args=large_case(); exp=oracle(*args); t0=time.perf_counter(); got=solution(*args); runtime=time.perf_counter()-t0
        results['large_correct']=_same(got,exp,atol=1e-9); results['large_runtime_under_8s']=bool(runtime<8.0)
    return {'passed':bool(all(results.values())),'results':results,'large_runtime_seconds':runtime}

if __name__=='__main__':
    from reference_solution import score_packed_lattices
    print(json.dumps(evaluate(score_packed_lattices),indent=2))


from __future__ import annotations
import numpy as np

def _logsumexp(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return -np.inf
    m = np.max(vals)
    if np.isneginf(m):
        return -np.inf
    return float(m + np.log(np.exp(vals - m).sum()))

def _logsumexp_axis(x, axis):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] == 0:
        shape = list(x.shape); del shape[axis]
        return np.full(shape, -np.inf, dtype=np.float64)
    m = np.max(x, axis=axis)
    out = np.full_like(m, -np.inf, dtype=np.float64)
    good = ~np.isneginf(m)
    expm = np.expand_dims(m, axis)
    with np.errstate(invalid='ignore'):
        s = np.sum(np.exp(x - expm), axis=axis)
    out[good] = m[good] + np.log(s[good])
    return out

def _merge_rows_for_frame(row_indices, labels, contexts, row_f, row_v, pinned):
    component = {}; row_component_key = {}
    for idx, y, c, rf, rv, pin in zip(row_indices, labels, contexts, row_f, row_v, pinned):
        key = (int(c), int(y)); row_component_key[int(idx)] = key
        if key not in component:
            component[key] = {'f': [], 'v': [], 'pinned': bool(pin), 'rows': []}
        component[key]['f'].append(float(rf)); component[key]['v'].append(float(rv))
        component[key]['pinned'] = component[key]['pinned'] or bool(pin)
        component[key]['rows'].append(int(idx))
    comp_items = []
    for (c, y), data in component.items():
        cf = _logsumexp(data['f'])
        if np.isneginf(cf):
            continue
        cv = max(data['v'])
        comp_items.append((c, y, cf, cv, data['pinned'], data['rows']))
    label_to_comps = {}
    for j, (c, y, cf, cv, pin, rows) in enumerate(comp_items):
        label_to_comps.setdefault(y, []).append(j)
    label_items = []
    for y, js in label_to_comps.items():
        lf = _logsumexp([comp_items[j][2] for j in js])
        lv = max(comp_items[j][3] for j in js)
        lp = any(comp_items[j][4] for j in js)
        if not np.isneginf(lf):
            label_items.append((y, lf, lv, lp))
    return comp_items, label_items, row_component_key

def score_packed_lattices(candidate_scores, candidate_labels, candidate_contexts, candidate_pinned, frame_offsets, sequence_offsets, frame_budgets, transition_scores, start_scores, end_scores):
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    candidate_contexts = np.asarray(candidate_contexts, dtype=np.int64)
    candidate_pinned = np.asarray(candidate_pinned, dtype=bool)
    frame_offsets = np.asarray(frame_offsets, dtype=np.int64)
    sequence_offsets = np.asarray(sequence_offsets, dtype=np.int64)
    frame_budgets = np.asarray(frame_budgets, dtype=np.int64)
    transition_scores = np.asarray(transition_scores, dtype=np.float64)
    start_scores = np.asarray(start_scores, dtype=np.float64)
    end_scores = np.asarray(end_scores, dtype=np.float64)
    B = len(sequence_offsets) - 1
    logz = np.full(B, -np.inf, dtype=np.float64)
    best = np.full(B, -np.inf, dtype=np.float64)
    row_marg = np.full(candidate_scores.shape, -np.inf, dtype=np.float64)
    for b in range(B):
        f0, f1 = int(sequence_offsets[b]), int(sequence_offsets[b+1])
        pc, py = np.nonzero(~np.isneginf(start_scores))
        pf = start_scores[pc, py].astype(np.float64)
        pv = pf.copy(); impossible = (pf.size == 0); frames = []
        for f in range(f0, f1):
            lo, hi = int(frame_offsets[f]), int(frame_offsets[f+1])
            if impossible or lo == hi:
                impossible = True; frames.append(None); continue
            rows = np.arange(lo, hi, dtype=np.int64)
            cy = candidate_labels[lo:hi]; cc = candidate_contexts[lo:hi]
            scores = candidate_scores[lo:hi]; pins = candidate_pinned[lo:hi]
            trans = transition_scores[pc[None, :], cc[:, None], py[None, :], cy[:, None]]
            row_f = scores + _logsumexp_axis(pf[None, :] + trans, axis=1)
            row_v = scores + np.max(pv[None, :] + trans, axis=1)
            comp_items, label_items, row_component_key = _merge_rows_for_frame(rows, cy, cc, row_f, row_v, pins)
            if not label_items:
                impossible = True; frames.append(None); continue
            pinned_labels = [x for x in label_items if x[3]]
            ordinary_labels = [x for x in label_items if not x[3]]
            ordinary_labels.sort(key=lambda x: (-x[1], x[0]))
            keep_labels = {x[0] for x in pinned_labels}
            keep_labels.update(x[0] for x in ordinary_labels[:max(0, int(frame_budgets[f]))])
            if not keep_labels:
                impossible = True; frames.append(None); continue
            kept_comps = [x for x in comp_items if x[1] in keep_labels]
            kept_comps.sort(key=lambda x: (x[0], x[1]))
            if not kept_comps:
                impossible = True; frames.append(None); continue
            comp_index = {(c, y): j for j, (c, y, *_rest) in enumerate(kept_comps)}
            kept_rows=[]; row_to_comp={}; row_forward_abs={}
            for r, key in row_component_key.items():
                if key in comp_index:
                    kept_rows.append(r); row_to_comp[r]=comp_index[key]; row_forward_abs[r]=float(row_f[r-lo])
            frames.append({'rows':np.asarray(kept_rows,dtype=np.int64),'row_to_comp':row_to_comp,'row_forward':row_forward_abs,'comp_contexts':np.asarray([x[0] for x in kept_comps],dtype=np.int64),'comp_labels':np.asarray([x[1] for x in kept_comps],dtype=np.int64),'comp_forward':np.asarray([x[2] for x in kept_comps],dtype=np.float64),'comp_viterbi':np.asarray([x[3] for x in kept_comps],dtype=np.float64)})
            pc=frames[-1]['comp_contexts']; py=frames[-1]['comp_labels']; pf=frames[-1]['comp_forward']; pv=frames[-1]['comp_viterbi']
        if impossible or not frames or frames[-1] is None:
            continue
        last=frames[-1]
        finals=last['comp_forward']+end_scores[last['comp_contexts'],last['comp_labels']]
        logz_b=_logsumexp(finals); logz[b]=logz_b
        best[b]=np.max(last['comp_viterbi']+end_scores[last['comp_contexts'],last['comp_labels']])
        beta=end_scores[last['comp_contexts'],last['comp_labels']].astype(np.float64)
        for local_t in range(len(frames)-1,-1,-1):
            fr=frames[local_t]
            if fr is None: break
            for r in fr['rows']:
                ci=fr['row_to_comp'][int(r)]
                row_marg[int(r)] = fr['row_forward'][int(r)] + beta[ci] - logz_b
            if local_t==0: continue
            prev_fr=frames[local_t-1]
            if prev_fr is None: break
            prev_c=prev_fr['comp_contexts']; prev_y=prev_fr['comp_labels']
            rows=fr['rows']; row_c=candidate_contexts[rows]; row_y=candidate_labels[rows]
            row_scores=candidate_scores[rows]
            row_beta=np.array([beta[fr['row_to_comp'][int(r)]] for r in rows])
            trans=transition_scores[prev_c[:,None], row_c[None,:], prev_y[:,None], row_y[None,:]]
            beta=_logsumexp_axis(row_scores[None,:]+trans+row_beta[None,:], axis=1)
    return logz, best, row_marg
