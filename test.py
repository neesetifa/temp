generator
from __future__ import annotations

import numpy as np

N_FAMILY = 24
N_ROLE = 9
N_VARIANT = 72
N_EDGE_TYPE = 7

SLICE_NAMES = np.array([
    "sparse_clean",
    "direction_flip",
    "edge_type_swap",
    "two_hop_chain",
    "decoy_hub",
    "receiver_gate",
    "matched_degree_motif",
    "graph_mode_shift",
    "rare_node_code",
    "dense_mixed",
])


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30.0, 30.0)))


def _static():
    rng = np.random.default_rng(881203)
    # Family behavior shares four coarse groups; variants share family/role behavior.
    group = rng.normal(0.0, 0.25, (6, 4))
    family_emb = np.empty((N_FAMILY, 4))
    for f in range(N_FAMILY):
        family_emb[f] = group[f % 6] + rng.normal(0.0, 0.10, 4)
        family_emb[f, f % 4] += 0.55

    role_emb = rng.normal(0.0, 0.24, (N_ROLE, 4))
    for r in range(N_ROLE):
        role_emb[r, r % 4] += 0.60
        role_emb[r, (r + 1) % 4] += 0.18

    # Directed family/role affinities are low-rank plus a few structured exceptions.
    fam_aff = 0.35 + 0.18 * (family_emb @ family_emb.T)
    role_aff = 0.45 + 0.16 * (role_emb @ role_emb.T)
    for i in range(N_FAMILY):
        fam_aff[i, (i + 5) % N_FAMILY] += 0.22
        fam_aff[i, (i + 11) % N_FAMILY] -= 0.18
    for i in range(N_ROLE):
        role_aff[i, (i + 2) % N_ROLE] += 0.20
        role_aff[i, (i + 5) % N_ROLE] -= 0.15
    fam_aff = np.clip(fam_aff, 0.08, 1.25)
    role_aff = np.clip(role_aff, 0.10, 1.20)

    # Edge type behavior differs for incoming/outgoing messages and changes with graph mode.
    edge_in = np.array([0.90, 0.62, -0.42, 0.35, 0.78, -0.18, 0.08])
    edge_out = np.array([-0.16, 0.52, 0.28, -0.40, 0.18, 0.66, 0.03])
    mode_delta = np.array([
        [ 0.18, -0.12,  0.05,  0.15, -0.20,  0.08,  0.00],
        [-0.10,  0.16, -0.12,  0.06,  0.20, -0.15,  0.00],
        [ 0.05,  0.02,  0.18, -0.20,  0.05,  0.16,  0.00],
    ])

    # Ordered two-hop edge-type pair effects. Only a subset are strong.
    path = rng.normal(0.0, 0.035, (N_EDGE_TYPE, N_EDGE_TYPE))
    for a, b, val in [
        (0, 4, 0.42), (1, 0, 0.31), (4, 1, 0.36), (5, 3, 0.28),
        (3, 5, -0.24), (2, 0, -0.20), (0, 2, 0.19), (1, 5, 0.22),
    ]:
        path[a, b] += val
    path_role = 0.72 + 0.18 * rng.normal(size=(N_ROLE, N_ROLE))
    for r in range(N_ROLE):
        path_role[r, r] += 0.28
        path_role[r, (r + 2) % N_ROLE] += 0.20
        path_role[r, (r + 5) % N_ROLE] -= 0.18
    path_role = np.clip(path_role, 0.24, 1.34)
    return family_emb, role_emb, fam_aff, role_aff, edge_in, edge_out, mode_delta, path, path_role


def _sample_slice(rng, split):
    if split == "hidden":
        p = np.array([0.08, 0.11, 0.10, 0.13, 0.11, 0.10, 0.13, 0.10, 0.08, 0.06])
    elif split == "public":
        p = np.array([0.24, 0.08, 0.08, 0.09, 0.09, 0.10, 0.08, 0.08, 0.07, 0.09])
    else:
        p = np.array([0.28, 0.07, 0.07, 0.08, 0.08, 0.09, 0.07, 0.08, 0.07, 0.11])
    return int(rng.choice(len(SLICE_NAMES), p=p / p.sum()))


def _sample_codes(rng, n, sid):
    fam_p = np.linspace(1.45, 0.45, N_FAMILY)
    fam_p = fam_p / fam_p.sum()
    role_p = np.array([0.16, 0.14, 0.13, 0.12, 0.115, 0.105, 0.09, 0.07, 0.07])
    if sid == 8:
        fam_p = fam_p.copy()
        fam_p[18:] *= 3.3
        fam_p[:10] *= 0.62
        fam_p /= fam_p.sum()
    fam = rng.choice(N_FAMILY, size=n, p=fam_p)
    role = rng.choice(N_ROLE, size=n, p=role_p / role_p.sum())
    # Variants are long-tailed; behavior mostly shares family and role structure.
    vg = rng.choice(3, size=n, p=[0.62, 0.27, 0.11])
    variant = (fam + 24 * vg) % N_VARIANT
    if sid == 8:
        mask = rng.random(n) < 0.45
        variant[mask] = (fam[mask] + 48) % N_VARIANT
    return np.stack([fam, role, variant], axis=1).astype(np.int64)


def _node_features(rng, codes, sid):
    n = len(codes)
    fam, role, variant = codes.T
    load = rng.lognormal(mean=-0.08, sigma=0.48, size=n)
    urgency = rng.beta(1.7, 2.0, size=n)
    stability = rng.beta(2.3, 1.9, size=n)
    reserve = rng.beta(2.0, 2.1, size=n)
    signal = np.clip(rng.normal(0.48 + 0.09 * (role % 3 == 0), 0.22, size=n), 0, 1.2)
    gate = rng.beta(1.9, 1.9, size=n)
    if sid == 5:
        # Receiver-gate slice deliberately has matched visible loads with split receiver states.
        gate = np.where(rng.random(n) < 0.5, rng.uniform(0.02, 0.28, n), rng.uniform(0.72, 0.98, n))
        load = np.clip(rng.normal(1.0, 0.18, n), 0.45, 1.65)
    if sid == 4:
        signal = np.clip(signal + rng.normal(0.08, 0.12, n), 0, 1.2)
    return np.stack([load, urgency, stability, reserve, signal, gate], axis=1).astype(float)


def _graph_features(rng, sid):
    mode = rng.uniform(0, 1)
    support_a = rng.beta(2.0, 2.0)
    support_b = rng.beta(2.0, 2.0)
    phase = rng.uniform(0, 1)
    if sid == 7:
        mode = rng.choice([rng.uniform(0.00, 0.12), rng.uniform(0.88, 1.00)])
        phase = rng.choice([rng.uniform(0.0, 0.15), rng.uniform(0.85, 1.0)])
    return np.array([mode, support_a, support_b, phase], dtype=float)


def _add_edge(src, dst, typ, u, v, t):
    if u == v:
        return
    src.append(int(u)); dst.append(int(v)); typ.append(int(t))


def _make_edges(rng, n, codes, X, sid):
    fam, role, variant = codes.T
    src, dst, typ = [], [], []
    if sid == 0:
        target_m = int(rng.integers(max(n - 2, 1), max(n + 3, 2)))
    elif sid == 9:
        target_m = int(rng.integers(3 * n, 5 * n + 1))
    else:
        target_m = int(rng.integers(2 * n, 3 * n + 1))

    # Base graph with role/family-dependent preferences but broad variation.
    tries = 0
    seen = set()
    while len(src) < target_m and tries < target_m * 30 + 100:
        tries += 1
        u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
        if u == v or (u, v) in seen:
            continue
        affinity = 0.14 + 0.18 * (fam[u] % 6 == fam[v] % 6) + 0.13 * (role[u] == role[v])
        affinity += 0.10 * (role[u] % 3 == (role[v] + 1) % 3)
        if rng.random() > min(0.82, affinity + 0.20):
            continue
        t = int((2 * role[u] + role[v] + (fam[u] % 3) + rng.integers(0, 3)) % N_EDGE_TYPE)
        _add_edge(src, dst, typ, u, v, t)
        seen.add((u, v))

    # Inject motifs that attack coarse graph summaries.
    if sid == 1:  # direction_flip
        order = np.argsort(X[:, 1] + 0.15 * X[:, 0])
        for k in range(min(n - 1, max(4, n // 2))):
            low = int(order[k]); high = int(order[-(k + 1)])
            if k % 2 == 0:
                _add_edge(src, dst, typ, high, low, 0)
            else:
                _add_edge(src, dst, typ, low, high, 0)
    elif sid == 2:  # edge_type_swap
        for k in range(min(n, 10)):
            u = k % n; v = (3 * k + 2) % n
            if u != v:
                _add_edge(src, dst, typ, u, v, 4 if k % 2 == 0 else 2)
    elif sid == 3:  # two_hop_chain
        perm = rng.permutation(n)
        for k in range(0, n - 2, 3):
            a, b, c = map(int, perm[k:k+3])
            pair = [(0, 4), (1, 0), (4, 1), (5, 3)][(k // 3) % 4]
            _add_edge(src, dst, typ, a, b, pair[0])
            _add_edge(src, dst, typ, b, c, pair[1])
    elif sid == 4:  # decoy_hub
        hub = int(np.argmax(X[:, 0] + X[:, 4]))
        for v in range(n):
            if v != hub and rng.random() < 0.82:
                _add_edge(src, dst, typ, hub, v, 6)
                if rng.random() < 0.45:
                    _add_edge(src, dst, typ, v, hub, 6)
    elif sid == 6:  # matched_degree_motif
        # Directed cycles and cross-links preserve low-order degree statistics but change motif order.
        perm = rng.permutation(n)
        for k in range(n):
            u = int(perm[k]); v = int(perm[(k + 1) % n])
            _add_edge(src, dst, typ, u, v, [0, 4, 1, 5][k % 4])
        for k in range(0, n, 2):
            u = int(perm[k]); v = int(perm[(k + max(2, n // 3)) % n])
            _add_edge(src, dst, typ, u, v, [1, 0, 5, 3][(k // 2) % 4])

    # Deduplicate exactly repeated directed typed edges while allowing same pair with different type.
    packed = {}
    for u, v, t in zip(src, dst, typ):
        packed[(u, v, t)] = None
    if not packed:
        packed[(0, 1 if n > 1 else 0, 0)] = None
    arr = np.array(list(packed.keys()), dtype=np.int64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _target_for_graph(X, codes, bx, src, dst, etype, rng):
    family_emb, role_emb, fam_aff, role_aff, edge_in, edge_out, mode_delta, path, path_role = _static()
    fam, role, variant = codes.T
    n = len(X)
    mode_bin = 0 if bx[0] < 0.33 else (2 if bx[0] > 0.67 else 1)
    in_w = edge_in + mode_delta[mode_bin]
    out_w = edge_out - 0.55 * mode_delta[mode_bin]

    load, urgency, stability, reserve, signal, gate = X.T
    own_lat = family_emb[fam] + role_emb[role]
    own = (
        2.2
        + 1.00 * np.log1p(load)
        + 0.82 * urgency
        - 0.54 * stability
        - 0.35 * reserve
        + 0.30 * signal
        + 0.24 * own_lat[:, 0]
        - 0.18 * own_lat[:, 2]
        + 0.10 * (variant // 24)
    )

    in_msg = np.zeros(n)
    out_msg = np.zeros(n)
    in_abs = np.zeros(n)
    reciprocity = np.zeros(n)
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    edge_lookup = set()
    for e, (u, v, t) in enumerate(zip(src, dst, etype)):
        incoming[int(v)].append(e)
        outgoing[int(u)].append(e)
        edge_lookup.add((int(u), int(v)))

    sender_strength = _softplus(0.72 * load + 0.58 * signal + 0.38 * urgency - 0.22 * reserve)
    receiver_gate = 0.34 + 0.90 * _sigmoid(2.2 * (gate - 0.5) + 0.60 * urgency - 0.35 * stability)

    for u, v, t in zip(src, dst, etype):
        u = int(u); v = int(v); t = int(t)
        aff = fam_aff[fam[u], fam[v]] * role_aff[role[u], role[v]]
        state_match = 0.72 + 0.26 * _sigmoid(2.0 * (signal[u] - gate[v]))
        variant_share = 1.0 + 0.12 * ((variant[u] // 24) == (variant[v] // 24))
        val = sender_strength[u] * aff * state_match * variant_share
        in_msg[v] += in_w[t] * val * receiver_gate[v]
        out_msg[u] += out_w[t] * (0.55 + 0.45 * urgency[v]) * aff * (0.65 + 0.35 * signal[u])
        in_abs[v] += abs(in_w[t]) * val
        if (v, u) in edge_lookup:
            reciprocity[v] += 0.10 * (t in (0, 1, 4)) * aff

    # Ordered two-hop paths ending at each receiver. This cannot be recovered from degree/count alone.
    two = np.zeros(n)
    for mid in range(n):
        if not incoming[mid] or not outgoing[mid]:
            continue
        mid_gate = 0.45 + 0.55 * _sigmoid(2.0 * (signal[mid] + stability[mid] - 0.9))
        for e1 in incoming[mid]:
            a = int(src[e1]); t1 = int(etype[e1])
            upstream = 0.55 + 0.45 * sender_strength[a]
            for e2 in outgoing[mid]:
                c = int(dst[e2]); t2 = int(etype[e2])
                if a == c:
                    continue
                pair = path[t1, t2]
                if abs(pair) < 0.06:
                    continue
                end_aff = 0.42 + 0.58 * fam_aff[fam[a], fam[c]]
                role_gate = path_role[role[mid], role[c]]
                mode_gate = 1.0 + 0.16 * ((mode_bin == 0 and t1 in (0, 4)) or (mode_bin == 2 and t2 in (1, 5)))
                two[c] += pair * upstream * mid_gate * end_aff * role_gate * mode_gate * (0.54 + 0.62 * receiver_gate[c])

    # Graph metadata changes how local messages translate into final outcomes.
    graph_scale = 0.86 + 0.30 * bx[1] - 0.18 * bx[2] + 0.16 * np.sin(np.pi * bx[3])
    local_pressure = _softplus(0.72 * in_msg + 0.18 * in_abs - 0.65 - 0.28 * bx[1])
    # Relational structure carries most of the predictive variation. Own-node fields still
    # matter, but a row-only model should leave substantial error.
    y = (0.56 * own + 3.45 * graph_scale * local_pressure
         + 1.25 * np.tanh(out_msg) + 3.05 * np.tanh(two) + 1.55 * reciprocity)
    # Heteroskedastic but bounded noise; reference is not expected to be perfect.
    noise_sd = 0.30 + 0.09 * (len(src) / max(n, 1) > 3.0) + 0.06 * (gate < 0.15)
    y += rng.normal(0.0, noise_sd)
    return y.astype(float)


def generate_dataset(n_graphs: int, seed: int, split: str = "train"):
    rng = np.random.default_rng(seed)
    node_Xs, node_codes, graph_Xs, ys = [], [], [], []
    edge_src_all, edge_dst_all, edge_type_all = [], [], []
    node_offsets = [0]
    edge_offsets = [0]
    graph_slices = []
    node_slices = []

    node_base = 0
    for _ in range(int(n_graphs)):
        sid = _sample_slice(rng, split)
        if sid == 0:
            n = int(rng.integers(6, 12))
        elif sid == 9:
            n = int(rng.integers(18, 31))
        elif sid in (3, 4, 6):
            n = int(rng.integers(12, 25))
        else:
            n = int(rng.integers(8, 22))
        codes = _sample_codes(rng, n, sid)
        X = _node_features(rng, codes, sid)
        bx = _graph_features(rng, sid)
        src, dst, et = _make_edges(rng, n, codes, X, sid)
        y = _target_for_graph(X, codes, bx, src, dst, et, rng)

        node_Xs.append(X)
        node_codes.append(codes)
        graph_Xs.append(bx)
        ys.append(y)
        edge_src_all.append(src + node_base)
        edge_dst_all.append(dst + node_base)
        edge_type_all.append(et)
        node_base += n
        node_offsets.append(node_base)
        edge_offsets.append(edge_offsets[-1] + len(src))
        graph_slices.append(sid)
        node_slices.extend([sid] * n)

    return {
        "node_X": np.vstack(node_Xs).astype(np.float64),
        "node_code": np.vstack(node_codes).astype(np.int64),
        "edge_src": np.concatenate(edge_src_all).astype(np.int64),
        "edge_dst": np.concatenate(edge_dst_all).astype(np.int64),
        "edge_code": np.concatenate(edge_type_all).astype(np.int64),
        "graph_X": np.vstack(graph_Xs).astype(np.float64),
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "y": np.concatenate(ys).astype(np.float64),
        "graph_slice": np.asarray(graph_slices, dtype=np.int64),
        "node_slice": np.asarray(node_slices, dtype=np.int64),
    }

hidden
"""Hidden verifier for typed_dependency_graphs."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any
import numpy as np

from generator import SLICE_NAMES, generate_dataset

HIDDEN_SEED = 81173
HIDDEN_N_GRAPHS = 320
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))

# Fixed RMSE anchors calibrated on a disjoint seed. Each slice gets its own
# continuous scale so a model cannot hide a structural failure in the average.
SLICE_GOOD = np.array([0.29, 0.31, 0.35, 0.33, 0.36, 0.28, 0.34, 0.29, 0.29, 0.35], dtype=float)
SLICE_BAD = np.array([0.82, 0.78, 0.80, 0.94, 0.78, 0.76, 1.34, 0.77, 0.78, 1.23], dtype=float)
OVERALL_GOOD = 0.34
OVERALL_BAD = 0.92


def _load_train(app_dir: Path) -> tuple[Any, ...]:
    d = np.load(app_dir / "train_data.npz")
    return (
        d["train_node_X"], d["train_node_code"], d["train_edge_src"], d["train_edge_dst"],
        d["train_edge_code"], d["train_graph_X"], d["train_node_offsets"], d["train_edge_offsets"], d["train_y"],
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
    params = module.fit_link_model(*payload["train"])
    return [np.asarray(module.predict_link_score(*args, params), dtype=float) for args in payload["inputs"]]


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


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _score_rmse(value: float, good: float, bad: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def _validate(pred: np.ndarray, n: int):
    if pred.shape != (n,):
        raise ValueError(f"prediction shape must be ({n},), got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions must be finite")


def _metrics(y, pred, slice_id):
    overall_rmse = _rmse(y, pred)
    overall_score = _score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    per_slice = {}
    ss = []
    for sid, name in enumerate(SLICE_NAMES):
        m = slice_id == sid
        if not bool(np.any(m)):
            continue
        r = _rmse(y[m], pred[m])
        s = _score_rmse(r, float(SLICE_GOOD[sid]), float(SLICE_BAD[sid]))
        per_slice[str(name)] = {"rmse": r, "score": s, "n": int(m.sum())}
        ss.append(s)
    slice_mean = float(np.mean(ss)) if ss else 0.0
    reward = float(0.35 * overall_score + 0.65 * slice_mean)
    # Predictions at or beyond the no-skill error scale are treated as grossly broken.
    if overall_rmse >= OVERALL_BAD:
        reward = 0.0
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
    hidden = generate_dataset(HIDDEN_N_GRAPHS, HIDDEN_SEED, "hidden")
    inputs = (
        hidden["node_X"], hidden["node_code"], hidden["edge_src"], hidden["edge_dst"], hidden["edge_code"],
        hidden["graph_X"], hidden["node_offsets"], hidden["edge_offsets"],
    )
    payload = {"train": train, "inputs": [inputs, inputs]}
    preds = _run_candidate(solve, payload)
    if len(preds) != 2:
        raise ValueError("candidate must return predictions for both verifier inputs")
    p0, p1 = preds
    _validate(p0, len(hidden["y"])); _validate(p1, len(hidden["y"]))
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-8):
        raise ValueError("predictions must be deterministic for repeated inputs")
    metrics = _metrics(hidden["y"], p0, hidden["node_slice"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{metrics['reward']:.12f}\n")
    (logs / "metrics.json").write_text(json.dumps(metrics, sort_keys=True) + "\n")
    return metrics["reward"], metrics


if __name__ == "__main__":
    reward, metrics = evaluate()
    print(json.dumps({"reward": reward, "overall_rmse": metrics["overall_rmse"]}, sort_keys=True))

test main
from hidden_eval import evaluate


def test_hidden_eval_runs():
    reward, metrics = evaluate()
    assert 0.0 <= reward <= 1.0
    assert metrics["overall_rmse"] >= 0.0

reference
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

N_FAMILY = 24
N_ROLE = 9
N_VARIANT = 72
N_EDGE_TYPE = 7


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -30.0, 30.0)))


def _static():
    rng = np.random.default_rng(881203)
    group = rng.normal(0.0, 0.25, (6, 4))
    family_emb = np.empty((N_FAMILY, 4))
    for f in range(N_FAMILY):
        family_emb[f] = group[f % 6] + rng.normal(0.0, 0.10, 4)
        family_emb[f, f % 4] += 0.55
    role_emb = rng.normal(0.0, 0.24, (N_ROLE, 4))
    for r in range(N_ROLE):
        role_emb[r, r % 4] += 0.60
        role_emb[r, (r + 1) % 4] += 0.18
    fam_aff = 0.35 + 0.18 * (family_emb @ family_emb.T)
    role_aff = 0.45 + 0.16 * (role_emb @ role_emb.T)
    for i in range(N_FAMILY):
        fam_aff[i, (i + 5) % N_FAMILY] += 0.22
        fam_aff[i, (i + 11) % N_FAMILY] -= 0.18
    for i in range(N_ROLE):
        role_aff[i, (i + 2) % N_ROLE] += 0.20
        role_aff[i, (i + 5) % N_ROLE] -= 0.15
    fam_aff = np.clip(fam_aff, 0.08, 1.25)
    role_aff = np.clip(role_aff, 0.10, 1.20)
    edge_in = np.array([0.90, 0.62, -0.42, 0.35, 0.78, -0.18, 0.08])
    edge_out = np.array([-0.16, 0.52, 0.28, -0.40, 0.18, 0.66, 0.03])
    mode_delta = np.array([
        [ 0.18, -0.12,  0.05,  0.15, -0.20,  0.08,  0.00],
        [-0.10,  0.16, -0.12,  0.06,  0.20, -0.15,  0.00],
        [ 0.05,  0.02,  0.18, -0.20,  0.05,  0.16,  0.00],
    ])
    path = rng.normal(0.0, 0.035, (N_EDGE_TYPE, N_EDGE_TYPE))
    for a, b, val in [
        (0, 4, 0.42), (1, 0, 0.31), (4, 1, 0.36), (5, 3, 0.28),
        (3, 5, -0.24), (2, 0, -0.20), (0, 2, 0.19), (1, 5, 0.22),
    ]:
        path[a, b] += val
    path_role = 0.72 + 0.18 * rng.normal(size=(N_ROLE, N_ROLE))
    for r in range(N_ROLE):
        path_role[r, r] += 0.28
        path_role[r, (r + 2) % N_ROLE] += 0.20
        path_role[r, (r + 5) % N_ROLE] -= 0.18
    path_role = np.clip(path_role, 0.24, 1.34)
    return family_emb, role_emb, fam_aff, role_aff, edge_in, edge_out, mode_delta, path, path_role


def _mechanism_features(node_X, node_code, edge_src, edge_dst, edge_code, graph_X, node_offsets, edge_offsets):
    X = np.asarray(node_X, dtype=float)
    C = np.asarray(node_code, dtype=int)
    src_all = np.asarray(edge_src, dtype=int)
    dst_all = np.asarray(edge_dst, dtype=int)
    et_all = np.asarray(edge_code, dtype=int)
    G = np.asarray(graph_X, dtype=float)
    noff = np.asarray(node_offsets, dtype=int)
    eoff = np.asarray(edge_offsets, dtype=int)
    family_emb, role_emb, fam_aff, role_aff, edge_in, edge_out, mode_delta, path, path_role = _static()

    out = np.zeros((len(X), 12), dtype=float)
    fam_all, role_all, var_all = C.T
    for gi in range(len(noff) - 1):
        a, b = int(noff[gi]), int(noff[gi + 1])
        ea, eb = int(eoff[gi]), int(eoff[gi + 1])
        if b <= a:
            continue
        Xi = X[a:b]
        fam = np.clip(fam_all[a:b], 0, N_FAMILY - 1)
        role = np.clip(role_all[a:b], 0, N_ROLE - 1)
        variant = np.clip(var_all[a:b], 0, N_VARIANT - 1)
        src = src_all[ea:eb] - a
        dst = dst_all[ea:eb] - a
        etype = np.clip(et_all[ea:eb], 0, N_EDGE_TYPE - 1)
        bx = G[gi]
        n = b - a
        mode_bin = 0 if bx[0] < 0.33 else (2 if bx[0] > 0.67 else 1)
        in_w = edge_in + mode_delta[mode_bin]
        out_w = edge_out - 0.55 * mode_delta[mode_bin]
        load, urgency, stability, reserve, signal, gate = Xi.T
        own_lat = family_emb[fam] + role_emb[role]
        own = (2.2 + 1.00 * np.log1p(load) + 0.82 * urgency - 0.54 * stability
               - 0.35 * reserve + 0.30 * signal + 0.24 * own_lat[:, 0]
               - 0.18 * own_lat[:, 2] + 0.10 * (variant // 24))
        in_msg = np.zeros(n); out_msg = np.zeros(n); in_abs = np.zeros(n); reciprocity = np.zeros(n)
        incoming = [[] for _ in range(n)]; outgoing = [[] for _ in range(n)]
        edge_lookup = set()
        for e, (u, v, t) in enumerate(zip(src, dst, etype)):
            u = int(u); v = int(v)
            if 0 <= u < n and 0 <= v < n:
                incoming[v].append(e); outgoing[u].append(e); edge_lookup.add((u, v))
        sender_strength = _softplus(0.72 * load + 0.58 * signal + 0.38 * urgency - 0.22 * reserve)
        receiver_gate = 0.34 + 0.90 * _sigmoid(2.2 * (gate - 0.5) + 0.60 * urgency - 0.35 * stability)
        for u, v, t in zip(src, dst, etype):
            u = int(u); v = int(v); t = int(t)
            if not (0 <= u < n and 0 <= v < n):
                continue
            aff = fam_aff[fam[u], fam[v]] * role_aff[role[u], role[v]]
            state_match = 0.72 + 0.26 * _sigmoid(2.0 * (signal[u] - gate[v]))
            variant_share = 1.0 + 0.12 * ((variant[u] // 24) == (variant[v] // 24))
            val = sender_strength[u] * aff * state_match * variant_share
            in_msg[v] += in_w[t] * val * receiver_gate[v]
            out_msg[u] += out_w[t] * (0.55 + 0.45 * urgency[v]) * aff * (0.65 + 0.35 * signal[u])
            in_abs[v] += abs(in_w[t]) * val
            if (v, u) in edge_lookup:
                reciprocity[v] += 0.10 * (t in (0, 1, 4)) * aff
        two = np.zeros(n)
        for mid in range(n):
            if not incoming[mid] or not outgoing[mid]:
                continue
            mid_gate = 0.45 + 0.55 * _sigmoid(2.0 * (signal[mid] + stability[mid] - 0.9))
            for e1 in incoming[mid]:
                aa = int(src[e1]); t1 = int(etype[e1])
                upstream = 0.55 + 0.45 * sender_strength[aa]
                for e2 in outgoing[mid]:
                    cc = int(dst[e2]); t2 = int(etype[e2])
                    if aa == cc:
                        continue
                    pair = path[t1, t2]
                    if abs(pair) < 0.06:
                        continue
                    end_aff = 0.42 + 0.58 * fam_aff[fam[aa], fam[cc]]
                    role_gate = path_role[role[mid], role[cc]]
                    mode_gate = 1.0 + 0.16 * ((mode_bin == 0 and t1 in (0, 4)) or (mode_bin == 2 and t2 in (1, 5)))
                    two[cc] += pair * upstream * mid_gate * end_aff * role_gate * mode_gate * (0.54 + 0.62 * receiver_gate[cc])
        graph_scale = 0.86 + 0.30 * bx[1] - 0.18 * bx[2] + 0.16 * np.sin(np.pi * bx[3])
        local_pressure = _softplus(0.72 * in_msg + 0.18 * in_abs - 0.65 - 0.28 * bx[1])
        deg_in = np.bincount(dst.astype(int), minlength=n) if len(dst) else np.zeros(n)
        deg_out = np.bincount(src.astype(int), minlength=n) if len(src) else np.zeros(n)
        out[a:b] = np.column_stack([
            own,
            graph_scale * local_pressure,
            np.tanh(out_msg),
            np.tanh(two),
            reciprocity,
            local_pressure,
            in_msg,
            in_abs,
            sender_strength,
            receiver_gate,
            deg_in,
            deg_out,
        ])
    return out


def fit_link_model(train_node_X, train_node_code, train_edge_src, train_edge_dst, train_edge_code,
                   train_graph_X, train_node_offsets, train_edge_offsets, train_y):
    F = _mechanism_features(train_node_X, train_node_code, train_edge_src, train_edge_dst, train_edge_code,
                            train_graph_X, train_node_offsets, train_edge_offsets)
    y = np.asarray(train_y, dtype=float)
    model = Ridge(alpha=0.35)
    model.fit(F, y)
    return {"model": model}


def predict_link_score(node_X, node_code, edge_src, edge_dst, edge_code, graph_X, node_offsets, edge_offsets, params):
    F = _mechanism_features(node_X, node_code, edge_src, edge_dst, edge_code, graph_X, node_offsets, edge_offsets)
    return np.asarray(params["model"].predict(F), dtype=float)

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

