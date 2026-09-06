hidden
"""Hidden verifier for typed_dependency_graphs v0.3a."""
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np
from generator import SLICE_NAMES, generate_dataset

HIDDEN_SEED = 81173
HIDDEN_N_GRAPHS = 320
LOG_DIR = Path(os.environ.get('VERIFIER_LOG_DIR', '/logs/verifier'))
APP_DIR = Path(os.environ.get('APP_DIR', '/app'))
SLICE_GOOD = np.array([.107,.109,.105,.111,.108,.100,.101,.097,.111,.108], float)
SLICE_BAD  = np.array([.249,.378,.245,.279,.255,.209,.278,.413,.260,.269], float)
SLICE_WEIGHTS = np.array([.14,.10,.07,.08,.13,.09,.09,.06,.10,.14], float)
OVERALL_GOOD = .108
OVERALL_BAD = .270


def _load_train(app):
    d = np.load(app / 'train_data.npz')
    return (
        d['train_node_X'], d['train_node_code'], d['train_edge_src'], d['train_edge_dst'], d['train_edge_code'],
        d['train_graph_X'], d['train_node_offsets'], d['train_edge_offsets'], d['train_y'],
    )


def _load_module(path):
    spec = importlib.util.spec_from_file_location('candidate_solve', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load solve module at {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_in_process(solve_path, payload):
    mod = _load_module(solve_path)
    params = mod.fit_link_model(*payload['train'])
    return [np.asarray(mod.predict_link_score(*args, params), float) for args in payload['inputs']]


def _run_candidate(solve_path, payload):
    try:
        import sandbox_util  # type: ignore
    except Exception:
        return _run_in_process(solve_path, payload)
    out = sandbox_util.run_eval(str(solve_path), payload)
    preds = out.get('preds') if isinstance(out, dict) else None
    if preds is None:
        raise RuntimeError("sandbox result did not contain 'preds'")
    return [np.asarray(p, float) for p in preds]


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _score(v, good, bad):
    if not math.isfinite(v):
        return 0.0
    return float(np.clip((bad - v) / (bad - good), 0.0, 1.0))


def _validate(p, n):
    if p.shape != (n,):
        raise ValueError(f'prediction shape must be ({n},), got {p.shape}')
    if not np.isfinite(p).all():
        raise ValueError('predictions must be finite')


def _metrics(y, p, sid):
    overall = _rmse(y, p)
    oscore = _score(overall, OVERALL_GOOD, OVERALL_BAD)
    per = {}
    ss = []
    ww = []
    for i, name in enumerate(SLICE_NAMES):
        m = sid == i
        if not np.any(m):
            continue
        r = _rmse(y[m], p[m])
        s = _score(r, float(SLICE_GOOD[i]), float(SLICE_BAD[i]))
        per[str(name)] = {'rmse': r, 'score': s, 'n': int(m.sum())}
        ss.append(s)
        ww.append(float(SLICE_WEIGHTS[i]))
    slice_mean = float(np.average(ss, weights=ww)) if ss else 0.0
    reward = float(.35 * oscore + .65 * slice_mean)
    if overall >= OVERALL_BAD:
        reward = 0.0
    return {'reward': reward, 'overall_rmse': overall, 'overall_score': oscore, 'slice_mean': slice_mean, 'per_slice': per}


def evaluate(solve_path=None, app_dir=None, log_dir=None):
    app = Path(app_dir) if app_dir is not None else APP_DIR
    logs = Path(log_dir) if log_dir is not None else LOG_DIR
    solve = Path(solve_path) if solve_path is not None else app / 'solve.py'
    train = _load_train(app)
    hidden = generate_dataset(HIDDEN_N_GRAPHS, HIDDEN_SEED, 'hidden')
    inputs = (
        hidden['node_X'], hidden['node_code'], hidden['edge_src'], hidden['edge_dst'], hidden['edge_code'],
        hidden['graph_X'], hidden['node_offsets'], hidden['edge_offsets'],
    )
    payload = {'train': train, 'inputs': [inputs, inputs]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError('candidate must return predictions for both verifier inputs')
    p0, p1 = preds
    _validate(p0, len(hidden['y']))
    _validate(p1, len(hidden['y']))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8):
        raise ValueError('predictions must be deterministic for repeated inputs')
    metrics = _metrics(hidden['y'], p0, hidden['node_slice'])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'reward.txt').write_text(f"{metrics['reward']:.12f}\n")
    (logs / 'metrics.json').write_text(json.dumps(metrics, sort_keys=True) + '\n')
    return metrics['reward'], metrics


if __name__ == '__main__':
    reward, metrics = evaluate()
    print(json.dumps({'reward': reward, 'overall_rmse': metrics['overall_rmse']}, sort_keys=True))


generator
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import numpy as np

N_EDGE_TYPE = 5
SLICE_NAMES = np.array([
    'sparse_clean',
    'high_degree_low_effect',
    'low_degree_high_effect',
    'conditional_context_shift',
    'optional_context_shift',
    'incoming_dominant',
    'outgoing_dominant',
    'long_range_sensitive',
    'mixed_edge_bundle',
    'rare_metadata_regime',
])


def _seed_file():
    here = Path(__file__).resolve().parent
    for p in (here / 'real_dependency_seed.npz', here.parent / 'tests' / 'real_dependency_seed.npz'):
        if p.exists():
            return p
    raise FileNotFoundError('real_dependency_seed.npz not found')


@lru_cache(maxsize=1)
def _seed():
    d = np.load(_seed_file())
    s = {k: d[k] for k in d.files}
    n = len(s['node_X_base'])
    out = [[] for _ in range(n)]
    inn = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(s['edge_src'], s['edge_dst'])):
        out[int(u)].append(e)
        inn[int(v)].append(e)
    s['out_edges'] = out
    s['in_edges'] = inn
    s['global_in_degree'] = np.bincount(s['edge_dst'].astype(int), minlength=n)
    s['global_out_degree'] = np.bincount(s['edge_src'].astype(int), minlength=n)
    combos = [tuple(map(int, r)) for r in s['node_code_base']]
    counts = {}
    for c in combos:
        counts[c] = counts.get(c, 0) + 1
    s['code_combo_count'] = np.array([counts[c] for c in combos], int)
    return s


def _split_buckets(split):
    if split == 'train':
        return set(range(7))
    if split == 'public':
        return {7}
    if split == 'calibration':
        return {8}
    if split == 'hidden':
        return {9}
    raise ValueError(split)


def _root_pool(s, split):
    degree = s['global_in_degree'] + s['global_out_degree']
    b = s['split_bucket']
    return np.where(np.isin(b, list(_split_buckets(split))) & (degree >= 2))[0]


def _build_bundle(s, rng, split):
    pool = _root_pool(s, split)
    root = int(rng.choice(pool))
    root_in = int(s['global_in_degree'][root])
    root_out = int(s['global_out_degree'][root])
    depth = int(rng.choice([2, 3, 4], p=[.26, .56, .18]))
    cap = int(np.clip(14 + 3 * root_out + 2 * math.sqrt(root_in) + rng.integers(-4, 9), 14, 110))
    chosen = {root}
    front = [(root, 0)]
    q = 0
    while q < len(front) and len(chosen) < cap:
        u, d = front[q]
        q += 1
        if d >= depth:
            continue
        outs = list(s['out_edges'][u])
        rng.shuffle(outs)
        for e in outs:
            v = int(s['edge_dst'][e])
            if v not in chosen and len(chosen) < cap:
                chosen.add(v)
                front.append((v, d + 1))
        if d <= 1 and len(chosen) < cap:
            ins = list(s['in_edges'][u])
            rng.shuffle(ins)
            take = min(len(ins), max(0, int(round(math.sqrt(len(ins))))))
            for e in ins[:take]:
                v = int(s['edge_src'][e])
                if v not in chosen and len(chosen) < cap:
                    chosen.add(v)
                    front.append((v, d + 1))
    nodes = np.array(sorted(chosen), int)
    loc = {int(g): i for i, g in enumerate(nodes)}
    es, ed, ec = [], [], []
    for u, v, t in zip(s['edge_src'], s['edge_dst'], s['edge_code']):
        u, v = int(u), int(v)
        if u in loc and v in loc:
            es.append(loc[u])
            ed.append(loc[v])
            ec.append(int(t))
    if len(nodes) < 7 or len(es) < 6:
        return _build_bundle(s, rng, split)
    X = s['node_X_base'][nodes].astype(float, copy=True)
    C = s['node_code_base'][nodes].astype(int, copy=True)
    ec_arr = np.asarray(ec, int)
    gx = np.array([
        float(s['node_X_base'][root, 0]),
        float(np.sum(X[:, 0])),
        float(root_in),
        float(root_out),
        float(np.mean(ec_arr == 3)) if len(ec_arr) else 0.0,
        float(np.mean(ec_arr != 0)) if len(ec_arr) else 0.0,
    ], float)
    return nodes, X, C, gx, np.asarray(es, int), np.asarray(ed, int), ec_arr


def _z(X):
    return np.log1p(np.maximum(np.asarray(X, float), 0.0))


def _sigmoid(x):
    x = np.clip(np.asarray(x, float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mechanism_basis(X, C, gx, src, dst, etype):
    """Visible-data-only relational basis used to produce the workload target."""
    n = len(X)
    Z = _z(X)
    size, files, meta, base_req, opt_req, entry, classifiers, major = Z.T
    lic, dev, pyfloor = np.asarray(C, int).T
    unstable = np.isin(dev, [1, 2, 3, 4]).astype(float)
    mature = np.isin(dev, [5, 6]).astype(float)
    known_py = (pyfloor > 0).astype(float)
    root_size, total_size, root_in, root_out, optional_frac, nonbase_frac = map(float, gx)
    ltotal = np.log1p(max(total_size, 0.0))
    lroot_in = np.log1p(max(root_in, 0.0))

    size_center = size - np.median(size)
    req_center = base_req - np.median(base_req)
    node_gate = _sigmoid(.76 * size_center + .34 * req_center + .55 * unstable - .24 * mature - .11 * (pyfloor - 4.0))
    graph_mix = float(_sigmoid(3.4 * (nonbase_frac - .60) + 2.2 * (optional_frac - .54) + .15 * (ltotal - 6.0)))
    graph_large = float(_sigmoid(.72 * (ltotal - 6.0) + .18 * lroot_in))
    dir_gate = _sigmoid(-.30 + 1.08 * unstable - .42 * mature + .30 * (entry > 0) - .28 * size_center + .60 * graph_mix)
    long_gate = _sigmoid(-.48 + .56 * size_center + .32 * req_center + .60 * graph_large - .46 * opt_req / np.maximum(base_req + opt_req, .25))
    risk = (.60 * unstable - .34 * mature + .28 * np.tanh(size_center) + .22 * np.tanh(req_center)
            + .14 * known_py * np.tanh((pyfloor - 4.0) / 2.0) + .12 * (entry > 0) + .08 * np.tanh(major - 1.0))

    indeg = np.bincount(dst, minlength=n).astype(float)
    outdeg = np.bincount(src, minlength=n).astype(float)

    out_a = np.array([1.00, .61, .53, .19, .40])
    out_b = np.array([.63, .93, .80, .74, .68])
    in_a = np.array([.96, .71, .62, .27, .47])
    in_b = np.array([.69, .96, .84, .68, .72])

    out_sum = np.zeros(n); in_sum = np.zeros(n)
    out_sq = np.zeros(n); in_sq = np.zeros(n)
    out_abs = np.zeros(n); in_abs = np.zeros(n)
    out_max = np.zeros(n); in_max = np.zeros(n)
    out_type_sum = np.zeros((n, 5)); in_type_sum = np.zeros((n, 5))
    out_type_cnt = np.zeros((n, 5)); in_type_cnt = np.zeros((n, 5))
    edge_fw = np.zeros(len(src)); edge_rv = np.zeros(len(src))
    edge_trans = np.zeros(len(src))
    incoming = [[] for _ in range(n)]; outgoing = [[] for _ in range(n)]

    for e, (u0, v0, t0) in enumerate(zip(src, dst, etype)):
        u, v, t = int(u0), int(v0), int(np.clip(t0, 0, 4))
        incoming[v].append(e); outgoing[u].append(e)
        bo = (1.0 - graph_mix) * out_a[t] + graph_mix * out_b[t]
        bi = (1.0 - graph_mix) * in_a[t] + graph_mix * in_b[t]
        py_gap = np.tanh((pyfloor[v] - pyfloor[u]) / 2.0)
        size_gap = np.tanh(size[v] - size[u])
        req_gap = np.tanh(base_req[v] - base_req[u])
        compat = np.tanh(1.00 * (risk[v] - .58 * risk[u]) + .24 * py_gap + .17 * size_gap + .10 * req_gap)
        reverse = np.tanh(.92 * (risk[u] - .52 * risk[v]) - .18 * py_gap - .13 * size_gap + .08 * req_gap)
        wo = bo * (.72 + .48 * node_gate[v])
        wi = bi * (.70 + .50 * node_gate[u])
        so = wo * compat
        si = wi * reverse
        edge_fw[e] = so; edge_rv[e] = si
        edge_trans[e] = max(.04, .52 * abs(so) + .48 * abs(si))
        out_sum[u] += so; in_sum[v] += si
        out_sq[u] += so * so; in_sq[v] += si * si
        out_abs[u] += abs(so); in_abs[v] += abs(si)
        out_max[u] = max(out_max[u], abs(so)); in_max[v] = max(in_max[v], abs(si))
        out_type_sum[u, t] += so; in_type_sum[v, t] += si
        out_type_cnt[u, t] += 1.0; in_type_cnt[v, t] += 1.0

    out_mean = out_sum / np.maximum(outdeg, 1.0)
    in_mean = in_sum / np.maximum(indeg, 1.0)
    out_rms = np.sqrt(out_sq / np.maximum(outdeg, 1.0))
    in_rms = np.sqrt(in_sq / np.maximum(indeg, 1.0))
    out_hetero = np.maximum(out_rms - np.abs(out_mean), 0.0)
    in_hetero = np.maximum(in_rms - np.abs(in_mean), 0.0)

    pair_a = np.array([
        [.35, .16, .13, -.08, .09],
        [.17, .14, .11, -.06, .08],
        [.15, .12, .10, -.05, .07],
        [-.07, -.05, -.04, .09, -.03],
        [.10, .08, .07, -.03, .06],
    ])
    pair_b = np.array([
        [.12, .24, .20, .16, .18],
        [.23, .22, .19, .14, .18],
        [.21, .19, .18, .13, .17],
        [.15, .17, .15, .18, .14],
        [.18, .19, .17, .13, .16],
    ])
    pair = (1.0 - graph_mix) * pair_a + graph_mix * pair_b
    f2_sum = np.zeros(n); r2_sum = np.zeros(n)
    f2_sq = np.zeros(n); r2_sq = np.zeros(n)
    f2_cnt = np.zeros(n); r2_cnt = np.zeros(n)
    two_count = np.zeros(n, int)
    for mid in range(n):
        if not incoming[mid] or not outgoing[mid]:
            continue
        mid_gate = .58 + .70 * node_gate[mid]
        for e1 in incoming[mid]:
            a = int(src[e1]); t1 = int(etype[e1])
            for e2 in outgoing[mid]:
                c = int(dst[e2]); t2 = int(etype[e2])
                if a == c:
                    continue
                path = pair[t1, t2] * mid_gate * edge_fw[e1] * edge_fw[e2]
                revp = pair[t2, t1] * mid_gate * edge_rv[e2] * edge_rv[e1]
                f2_sum[a] += path; f2_sq[a] += path * path; f2_cnt[a] += 1.0
                r2_sum[c] += revp; r2_sq[c] += revp * revp; r2_cnt[c] += 1.0
                two_count[a] += 1; two_count[c] += 1
    f2_mean = f2_sum / np.maximum(f2_cnt, 1.0)
    r2_mean = r2_sum / np.maximum(r2_cnt, 1.0)
    f2_rms = np.sqrt(f2_sq / np.maximum(f2_cnt, 1.0))
    r2_rms = np.sqrt(r2_sq / np.maximum(r2_cnt, 1.0))

    # Degree-normalized signed propagation. Node-specific attenuation is visible-state dependent,
    # preventing one global decay constant from being sufficient.
    alpha_f = .14 + .68 * _sigmoid(-.30 + .76 * long_gate + .38 * graph_large - .28 * dir_gate)
    alpha_r = .13 + .70 * _sigmoid(-.25 + .60 * long_gate + .36 * graph_mix + .24 * dir_gate)
    fw = risk.copy(); rv = risk.copy()
    prop_f = np.zeros(n); prop_r = np.zeros(n)
    out_norm = np.maximum(np.bincount(src, weights=edge_trans, minlength=n), .35)
    in_norm = np.maximum(np.bincount(dst, weights=edge_trans, minlength=n), .35)
    for k in range(3):
        nfw = np.zeros(n); nrv = np.zeros(n)
        if len(src):
            np.add.at(nfw, dst, edge_trans * fw[src] / out_norm[src])
            np.add.at(nrv, src, edge_trans * rv[dst] / in_norm[dst])
        fw = nfw * alpha_f; rv = nrv * alpha_r
        prop_f += fw / (1.0 + .35 * k); prop_r += rv / (1.0 + .31 * k)

    direct_in = (.58 + .92 * dir_gate) * np.tanh(1.35 * in_mean)
    direct_out = (1.22 - .62 * dir_gate) * np.tanh(1.30 * out_mean)
    hetero_in = (.38 + .44 * node_gate) * np.tanh(1.55 * in_hetero)
    hetero_out = (.42 + .40 * (1.0 - node_gate)) * np.tanh(1.55 * out_hetero)
    edge_peak = np.tanh(.95 * (in_max - out_max)) * (2.0 * dir_gate - 1.0)
    two_forward = (.34 + .58 * (1.0 - dir_gate)) * np.tanh(1.8 * f2_mean)
    two_reverse = (.28 + .72 * dir_gate) * np.tanh(1.8 * r2_mean)
    two_spread = np.tanh(1.25 * (f2_rms - r2_rms)) * (2.0 * graph_mix - 1.0)
    long_forward = (.28 + .70 * long_gate) * np.tanh(1.35 * prop_f)
    long_reverse = (.30 + .66 * (1.0 - long_gate) + .18 * graph_mix) * np.tanh(1.35 * prop_r)

    opt_mean = (out_type_sum[:, 3] + in_type_sum[:, 3]) / np.maximum(out_type_cnt[:, 3] + in_type_cnt[:, 3], 1.0)
    cond_sum = np.sum(out_type_sum[:, [1, 2, 4]] + in_type_sum[:, [1, 2, 4]], axis=1)
    cond_cnt = np.sum(out_type_cnt[:, [1, 2, 4]] + in_type_cnt[:, [1, 2, 4]], axis=1)
    cond_mean = cond_sum / np.maximum(cond_cnt, 1.0)
    opt_ctx = np.tanh(1.4 * opt_mean) * (.28 + .98 * graph_mix) * (.44 + .56 * node_gate)
    cond_ctx = np.tanh(1.35 * cond_mean) * (1.08 - .48 * graph_mix) * (.52 + .48 * (1.0 - node_gate))
    cross = np.tanh(1.15 * in_mean * out_mean) * (2.0 * graph_mix - 1.0)
    asym = np.tanh(1.25 * (in_mean - out_mean)) * (2.0 * dir_gate - 1.0)
    weak_degree = np.tanh(np.log1p(indeg + outdeg) / 2.5) * (.30 + .25 * np.abs(2.0 * node_gate - 1.0))

    own1 = size; own2 = files; own3 = meta; own4 = base_req
    own5 = entry; own6 = classifiers; own7 = major; own8 = unstable; own9 = mature
    own10 = known_py * np.tanh((pyfloor - 4.0) / 2.0)
    own11 = np.tanh(base_req - opt_req)
    own12 = node_gate * np.tanh(size_center)

    basis = np.column_stack([
        own1, own2, own3, own4, own5, own6, own7, own8, own9, own10, own11, own12,
        direct_in, direct_out, hetero_in, hetero_out, edge_peak,
        two_forward, two_reverse, two_spread, long_forward, long_reverse,
        opt_ctx, cond_ctx, cross, asym, weak_degree,
        graph_mix * hetero_in, (1.0 - graph_mix) * hetero_out,
        node_gate * long_forward, (1.0 - node_gate) * long_reverse,
    ])
    coef = np.array([
        .18, .050, .035, .055, .048, .020, .032, .075, -.034, .040, .050, .046,
        .78, .72, .58, .54, .35,
        .64, .70, .32, .60, .64,
        .24, .22, .30, .27, .08,
        .20, .18, .18, .17,
    ])
    base = 1.34
    y_clean = base + basis @ coef
    aux = {
        'in_deg': indeg, 'out_deg': outdeg,
        'in_load': in_mean, 'out_load': out_mean,
        'direct_strength': np.abs(.78 * direct_in + .72 * direct_out),
        'long_strength': np.abs(.60 * long_forward + .64 * long_reverse),
        'graph_mix': np.full(n, graph_mix),
        'optional_incident': (out_type_cnt[:, 3] + in_type_cnt[:, 3]) > 0,
        'conditional_incident': np.sum(out_type_cnt[:, [1, 2, 4]] + in_type_cnt[:, [1, 2, 4]], axis=1) > 0,
        'two_count': two_count,
    }
    return basis, y_clean, aux

def _target_for_bundle(X, C, gx, src, dst, etype, rng):
    basis, clean, aux = _mechanism_basis(X, C, gx, src, dst, etype)
    Z = _z(X)
    unstable = np.isin(np.asarray(C, int)[:, 1], [1, 2, 3, 4]).astype(float)
    # Realistic heteroscedastic measurement/operational variation without changing the visible mechanism.
    noise_sd = .080 + .018 * np.sqrt(np.maximum(clean, 0.0)) + .014 * unstable + .010 * np.tanh(Z[:, 0])
    y = clean + rng.normal(0.0, noise_sd, len(clean))
    return y, aux


def _slice_ids(s, global_nodes, gx, src, dst, etype, aux):
    n = len(global_nodes)
    indeg = aux['in_deg']
    outdeg = aux['out_deg']
    deg = indeg + outdeg
    direct = aux['direct_strength']
    long_strength = aux['long_strength']
    rare = s['code_combo_count'][global_nodes] <= 2
    optional_incident = aux['optional_incident']
    conditional_incident = aux['conditional_incident']
    graph_mix = float(aux['graph_mix'][0]) if n else 0.0
    optional_frac = float(gx[4])
    nonbase_frac = float(gx[5])
    sid = np.zeros(n, int)
    for i in range(n):
        if rare[i]:
            sid[i] = 9
        elif graph_mix > .72 and deg[i] >= 2:
            sid[i] = 8
        elif long_strength[i] > .34 and aux['two_count'][i] >= 10:
            sid[i] = 7
        elif indeg[i] >= max(2.0, 1.8 * outdeg[i]) and direct[i] > .45:
            sid[i] = 5
        elif outdeg[i] >= max(2.0, 1.8 * indeg[i]) and direct[i] > .45:
            sid[i] = 6
        elif optional_incident[i] and optional_frac > .58:
            sid[i] = 4
        elif conditional_incident[i] and nonbase_frac > .48:
            sid[i] = 3
        elif deg[i] >= 8 and direct[i] < .70:
            sid[i] = 1
        elif deg[i] <= 4 and direct[i] > .42:
            sid[i] = 2
        else:
            sid[i] = 0
    return sid


def generate_dataset(n_graphs, seed_value, split='train'):
    s = _seed()
    rng = np.random.default_rng(seed_value)
    Xs, Cs, Gs, Ss, Ds, Es, Ys, Slices = [], [], [], [], [], [], [], []
    no, eo = [0], [0]
    for _ in range(int(n_graphs)):
        gnodes, X, C, gx, src, dst, et = _build_bundle(s, rng, split)
        y, aux = _target_for_bundle(X, C, gx, src, dst, et, rng)
        sid = _slice_ids(s, gnodes, gx, src, dst, et, aux)
        base = no[-1]
        Xs.append(X); Cs.append(C); Gs.append(gx)
        Ss.append(src + base); Ds.append(dst + base); Es.append(et)
        Ys.append(y); Slices.append(sid)
        no.append(base + len(X)); eo.append(eo[-1] + len(src))
    return {
        'node_X': np.concatenate(Xs).astype(float),
        'node_code': np.concatenate(Cs).astype(int),
        'edge_src': np.concatenate(Ss).astype(int),
        'edge_dst': np.concatenate(Ds).astype(int),
        'edge_code': np.concatenate(Es).astype(int),
        'graph_X': np.asarray(Gs, float),
        'node_offsets': np.asarray(no, int),
        'edge_offsets': np.asarray(eo, int),
        'y': np.concatenate(Ys).astype(float),
        'node_slice': np.concatenate(Slices).astype(int),
    }


