# Speed up the packed lattice scorer

The old scorer handled one sequence at a time. The packed batches used now are larger, so please complete `score_packed_lattices` in `solve.py`.

The input stores candidate rows by frame using `frame_offsets`, and frames by sequence using `sequence_offsets`. Each row is a proposal for a logical label, but it also has a context id. Use that row's current context and label, together with the previous surviving context and label, when applying `transition_scores`.

Rows with the same label are combined for deciding which labels fit in the frame budget. Their forward scores combine with logsumexp, their best-path scores combine with max, and a label is pinned when any of its proposals is pinned. Reachable pinned labels stay in the frame without using the ordinary budget. Rank the other labels by their combined forward score; exact ties go to the smaller label id.

A label-level decision is not the same thing as throwing away the row contexts. When a label survives the budget step, every reachable context for that surviving label is still part of the decoder state for the next frame. The context is not just decoration: later transitions and the final end score can depend on it.

Use the labels selected by the forward ranking for both returned recurrences. Do not run a separate beam for the best-path score. An empty frame, or a frame where nothing survives, makes that sequence impossible, but it should not affect other sequences in the packed input.

Return two float64 arrays shaped `(batch,)`: the final log-partition value and the final best-path score after applying `end_scores`. Long sequences should stay numerically stable. Avoid building an array covering every frame and every possible context-label to context-label transition.


from __future__ import annotations
import json, time, numpy as np
from generators import pack_sequences, random_case, large_case
from reference_solution import score_packed_lattices as oracle


def _same(got, exp, atol=1e-10):
    return (np.allclose(got[0], exp[0], rtol=1e-10, atol=atol, equal_nan=True)
            and np.allclose(got[1], exp[1], rtol=1e-10, atol=atol, equal_nan=True))


def _case_multi_context_future():
    # Label 0 survives; context 0 has higher immediate mass, context 1 is needed next.
    seq=[[{'scores':[0.4,0.0,-0.1], 'labels':[0,0,1], 'contexts':[0,1,0], 'pinned':[False,False,False], 'budget':1},
          {'scores':[0.0], 'labels':[2], 'contexts':[0], 'pinned':[False], 'budget':1}]]
    T=np.full((2,2,3,3), -8.0); T[:]= -8.0
    T[:,0,:,0]=0.0; T[:,1,:,0]=0.0; T[1,0,0,2]=4.0; T[0,0,0,2]=-4.0
    start=np.zeros((2,3)); end=np.zeros((2,3))
    return (*pack_sequences(seq), T, start, end)


def _case_label_aggregate_beats_component():
    seq=[[{'scores':[0.0,0.0,0.5], 'labels':[0,0,1], 'contexts':[0,1,0], 'pinned':[False,False,False], 'budget':1}]]
    T=np.zeros((2,2,2,2)); start=np.zeros((2,2)); end=np.array([[0.0,3.0],[0.0,3.0]])
    # label 0 aggregate logsumexp(2 components) beats label 1, even though each component alone does not.
    return (*pack_sequences(seq), T, start, end)


def _case_pinned_carries_unpinned_context():
    seq=[[{'scores':[0.0,0.1,0.9], 'labels':[0,0,1], 'contexts':[0,1,0], 'pinned':[True,False,False], 'budget':0},
          {'scores':[0.0], 'labels':[2], 'contexts':[0], 'pinned':[False], 'budget':1}]]
    T=np.full((2,2,3,3), -6.0); T[:, :, :, :] = -6.0
    T[:,0,:,0]=0.0; T[:,1,:,0]=0.0; T[1,0,0,2]=5.0; T[0,0,0,2]=-5.0
    start=np.zeros((2,3)); end=np.zeros((2,3))
    return (*pack_sequences(seq), T, start, end)


def _case_context_end_score():
    seq=[[{'scores':[0.3,0.0], 'labels':[0,0], 'contexts':[0,1], 'pinned':[False,False], 'budget':1}]]
    T=np.zeros((2,2,1,1)); start=np.zeros((2,1)); end=np.array([[-4.0],[3.0]])
    return (*pack_sequences(seq), T, start, end)


def _case_duplicate_same_component():
    seq=[[{'scores':[0.0,0.0,0.2], 'labels':[0,0,1], 'contexts':[0,0,0], 'pinned':[False,False,False], 'budget':1}]]
    T=np.zeros((1,1,2,2)); start=np.zeros((1,2)); end=np.array([[0.0,2.0]])
    return (*pack_sequences(seq), T, start, end)


def _case_impossible_isolation():
    seqs=[[
        {'scores':[0.0], 'labels':[0], 'contexts':[0], 'pinned':[False], 'budget':1},
        {'scores':[], 'labels':[], 'contexts':[], 'pinned':[], 'budget':1}],
          [{'scores':[0.1], 'labels':[1], 'contexts':[0], 'pinned':[False], 'budget':1},
           {'scores':[0.2], 'labels':[1], 'contexts':[1], 'pinned':[False], 'budget':1}]]
    T=np.zeros((2,2,2,2)); start=np.zeros((2,2)); end=np.zeros((2,2))
    return (*pack_sequences(seqs), T, start, end)


def _case_long_norm():
    seq=[[]]
    for _ in range(650):
        seq[0].append({'scores':[12.0,11.8,11.6,11.4], 'labels':[0,0,1,1], 'contexts':[0,1,0,1], 'pinned':[False,False,False,False], 'budget':2})
    T=np.zeros((2,2,2,2)); start=np.zeros((2,2)); end=np.zeros((2,2))
    return (*pack_sequences(seq), T, start, end)


def evaluate(solution, run_large=True):
    cases={
        'multi_context_survives_future': _case_multi_context_future(),
        'label_aggregate_not_component': _case_label_aggregate_beats_component(),
        'pinned_label_carries_all_contexts': _case_pinned_carries_unpinned_context(),
        'context_specific_end_score': _case_context_end_score(),
        'duplicate_same_context_merge': _case_duplicate_same_component(),
        'impossible_sequence_isolated': _case_impossible_isolation(),
        'long_sequence_normalization': _case_long_norm(),
    }
    for seed in [3,11,17,29,53,101]:
        cases[f'random_{seed}']=random_case(seed)
    results={}
    for name,args in cases.items():
        exp=oracle(*args); got=solution(*args); results[name]=bool(_same(got, exp))
    runtime=None
    if run_large:
        args=large_case()
        exp=oracle(*args)
        t0=time.perf_counter(); got=solution(*args); runtime=time.perf_counter()-t0
        results['large_correct']=bool(_same(got, exp, atol=1e-9))
        # Conservative relative-ish limit for this prototype; tune on target env if needed.
        results['large_runtime_under_8s']=bool(runtime < 8.0)
    return {'passed': bool(all(results.values())), 'results': results, 'large_runtime_seconds': runtime}

if __name__=='__main__':
    from reference_solution import score_packed_lattices
    print(json.dumps(evaluate(score_packed_lattices), indent=2))


from __future__ import annotations
import numpy as np


def pack_sequences(seqs):
    scores=[]; labels=[]; contexts=[]; pinned=[]; frame_offsets=[0]; seq_offsets=[0]; budgets=[]
    for seq in seqs:
        for frame in seq:
            scores.extend(frame['scores']); labels.extend(frame['labels']); contexts.extend(frame['contexts']); pinned.extend(frame['pinned'])
            budgets.append(int(frame['budget'])); frame_offsets.append(len(scores))
        seq_offsets.append(len(budgets))
    return (np.asarray(scores, float), np.asarray(labels, np.int64), np.asarray(contexts, np.int64),
            np.asarray(pinned, bool), np.asarray(frame_offsets, np.int64), np.asarray(seq_offsets, np.int64),
            np.asarray(budgets, np.int64))


def random_case(seed=0, batch=4, n_contexts=4, n_labels=9, min_frames=2, max_frames=8, rows_range=(3,20), allow_empty=True):
    rng = np.random.default_rng(seed)
    T = rng.normal(0, 0.7, size=(n_contexts, n_contexts, n_labels, n_labels))
    T[rng.random(T.shape) < 0.08] = -np.inf
    start = rng.normal(0, 0.4, size=(n_contexts, n_labels))
    start[rng.random(start.shape) < 0.08] = -np.inf
    end = rng.normal(0, 0.4, size=(n_contexts, n_labels))
    seqs=[]
    for _ in range(batch):
        seq=[]
        for _ in range(int(rng.integers(min_frames, max_frames+1))):
            if allow_empty and rng.random() < 0.04:
                rows=0
            else:
                rows=int(rng.integers(rows_range[0], rows_range[1]+1))
            labs=rng.integers(0,n_labels,size=rows)
            ctx=rng.integers(0,n_contexts,size=rows)
            if rows>=6 and rng.random()<0.8:
                labs[-3:]=labs[:3]
            if rows>=8 and rng.random()<0.5:
                ctx[-2:]=ctx[:2]
            seq.append({'scores':rng.normal(0,1.0,size=rows).tolist(),
                        'labels':labs.tolist(), 'contexts':ctx.tolist(),
                        'pinned':(rng.random(rows)<0.1).tolist(),
                        'budget':int(rng.integers(0,min(5,n_labels)+1))})
        seqs.append(seq)
    return (*pack_sequences(seqs), T, start, end)


def large_case(seed=99, batch=12, frames=45, n_contexts=8, n_labels=64, rows=56):
    rng=np.random.default_rng(seed)
    T=rng.normal(0,0.35,size=(n_contexts,n_contexts,n_labels,n_labels))
    T[rng.random(T.shape)<0.02] = -np.inf
    start=rng.normal(0,0.25,size=(n_contexts,n_labels))
    end=rng.normal(0,0.25,size=(n_contexts,n_labels))
    seqs=[]
    for _ in range(batch):
        seq=[]
        for _ in range(frames):
            labs=rng.integers(0,n_labels,size=rows); ctx=rng.integers(0,n_contexts,size=rows)
            labs[-12:]=labs[:12]
            seq.append({'scores':rng.normal(0,0.7,size=rows).tolist(),
                        'labels':labs.tolist(), 'contexts':ctx.tolist(),
                        'pinned':(rng.random(rows)<0.03).tolist(), 'budget':12})
        seqs.append(seq)
    return (*pack_sequences(seqs), T, start, end)

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


def score_packed_lattices(candidate_scores, candidate_labels, candidate_contexts, candidate_pinned,
                          frame_offsets, sequence_offsets, frame_budgets,
                          transition_scores, start_scores, end_scores):
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

    batch = len(sequence_offsets) - 1
    n_contexts, n_labels = start_scores.shape
    out_z = np.full(batch, -np.inf, dtype=np.float64)
    out_b = np.full(batch, -np.inf, dtype=np.float64)

    for b in range(batch):
        f0, f1 = int(sequence_offsets[b]), int(sequence_offsets[b + 1])
        comps = []
        alpha = []
        beta = []
        for c in range(n_contexts):
            for l in range(n_labels):
                a = float(start_scores[c, l])
                if not np.isneginf(a):
                    comps.append((c, l))
                    alpha.append(a)
                    beta.append(a)
        alpha = np.asarray(alpha, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        # normalize initial forward vector
        scale = 0.0
        if alpha.size:
            m = np.max(alpha)
            alpha = alpha - m
            scale = float(m)
        impossible = not comps

        for f in range(f0, f1):
            lo, hi = int(frame_offsets[f]), int(frame_offsets[f + 1])
            if impossible or lo == hi:
                impossible = True
                continue

            row_records = []  # (ctx, label, forward_abs, best, pinned)
            for r in range(lo, hi):
                cc = int(candidate_contexts[r])
                ll = int(candidate_labels[r])
                emission = float(candidate_scores[r])
                f_terms = []
                b_terms = []
                for j, (pc, pl) in enumerate(comps):
                    tr = float(transition_scores[pc, cc, pl, ll])
                    f_terms.append(alpha[j] + scale + tr)
                    b_terms.append(beta[j] + tr)
                rf = emission + _logsumexp(f_terms)
                rb = emission + (max(b_terms) if b_terms else -np.inf)
                row_records.append((cc, ll, rf, rb, bool(candidate_pinned[r])))

            # label-level aggregate for pruning
            labels = sorted(set(x[1] for x in row_records))
            label_info = []
            for lab in labels:
                rows = [x for x in row_records if x[1] == lab]
                lf = _logsumexp([x[2] for x in rows])
                if np.isneginf(lf):
                    continue
                lb = max(x[3] for x in rows)
                lp = any(x[4] for x in rows)
                label_info.append((lab, lf, lb, lp))
            if not label_info:
                impossible = True
                continue

            pinned_labels = [x for x in label_info if x[3]]
            ordinary = [x for x in label_info if not x[3]]
            ordinary.sort(key=lambda x: (-x[1], x[0]))
            keep_labels = {x[0] for x in pinned_labels + ordinary[:max(0, int(frame_budgets[f]))]}
            if not keep_labels:
                impossible = True
                continue

            # carry all reachable contexts/components for surviving labels.
            comp_keys = sorted(set((x[0], x[1]) for x in row_records if x[1] in keep_labels))
            new_comps = []
            new_alpha_abs = []
            new_beta = []
            for key in comp_keys:
                rows = [x for x in row_records if (x[0], x[1]) == key]
                af = _logsumexp([x[2] for x in rows])
                if np.isneginf(af):
                    continue
                ab = max(x[3] for x in rows)
                new_comps.append(key)
                new_alpha_abs.append(af)
                new_beta.append(ab)
            if not new_comps:
                impossible = True
                continue
            comps = new_comps
            alpha_abs = np.asarray(new_alpha_abs, dtype=np.float64)
            beta = np.asarray(new_beta, dtype=np.float64)
            m = np.max(alpha_abs)
            alpha = alpha_abs - m
            scale = float(m)

        if impossible or not comps:
            continue
        final_terms = [alpha[j] + scale + float(end_scores[c, l]) for j, (c, l) in enumerate(comps)]
        out_z[b] = _logsumexp(final_terms)
        out_b[b] = max(beta[j] + float(end_scores[c, l]) for j, (c, l) in enumerate(comps))
    return out_z, out_b
