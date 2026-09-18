"""Reference selector for margin-adaptive manifold coverage.

The downstream learner is a frozen k=7 uniform-weight kNN.  The selector therefore
needs more than geometric coverage: a locally difficult query can fail whenever its
4th same-class prototype is farther away than the competing-class vote support.

The reference uses only agent-visible pool features/labels.  It:
  1. removes locally inconsistent rows,
  2. builds a global + margin-adaptive coverage skeleton,
  3. repairs local kNN vote-support deficits, and
  4. chooses among a small, pre-declared family of robust settings using deterministic
     visible-pool proxy shards (global-support and low-margin support).

No generator constants, latent curve coordinates, or hidden evaluation rows are used.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

POOL_CSV = os.environ.get("KNN_POOL", "/app/data/pool.csv")
CONFIG_JSON = os.environ.get("KNN_CONFIG", "/app/data/config.json")
OUTPUT_CSV = os.environ.get("KNN_OUTPUT", "/output/selected_indices.csv")

DENOISE_K = 7
DENOISE_THR = 0.5
VOTE_NEED = 4  # majority support needed by a k=7 uniform kNN

# These are deliberately a small robust family, not hidden-tuned constants.  The final
# choice is made from the visible pool itself via _proxy_score().
CANDIDATE_SETTINGS = (
    (0.90, 0.70),
    (1.00, 0.60),
    (1.10, 0.60),
    (1.00, 0.70),
)


def _loo_agreement(X, y, k=DENOISE_K):
    """Leave-one-out neighbour label agreement, deterministic and order-invariant."""
    n = len(y)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    out = np.empty(n, dtype=float)
    for i in range(n):
        row = idx[i]
        row = row[row != i][:k]
        out[i] = np.mean(y[row] == y[i]) if len(row) else 0.0
    return out


def _farthest_point_cover(Xc, take):
    """Deterministic greedy k-center coverage over actual pool rows."""
    if take <= 0 or len(Xc) == 0:
        return np.array([], dtype=int)
    order = np.lexsort(Xc.T[::-1])
    Xs = Xc[order]
    n = len(Xs)
    if take >= n:
        return order
    c0 = int(np.argmin(((Xs - Xs.mean(0)) ** 2).sum(1)))
    sel = [c0]
    mind = ((Xs - Xs[c0]) ** 2).sum(1)
    for _ in range(take - 1):
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        mind = np.minimum(mind, ((Xs - Xs[nxt]) ** 2).sum(1))
    return order[np.asarray(sel, dtype=int)]


def _cross_class_k_margin(X, y, clean, k=1):
    """Distance to the k-th nearest clean different-class survivor.

    Using the k-th rather than only the nearest opponent matches the support requirement
    of the frozen kNN and is less sensitive to a single stray survivor.
    """
    margin = np.full(len(y), np.inf, dtype=float)
    clean_idx = np.where(clean)[0]
    for c in np.unique(y):
        same = clean_idx[y[clean_idx] == c]
        other = clean_idx[y[clean_idx] != c]
        if len(same) == 0 or len(other) == 0:
            continue
        kk = min(k, len(other))
        nn = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(X[other])
        d, _ = nn.kneighbors(X[same])
        margin[same] = d[:, -1]
    return margin


def _initial_cover(Xc, margin, take, beta):
    """Preserve global support, then spend more resolution at small margins."""
    if take <= 0 or len(Xc) == 0:
        return np.array([], dtype=int)
    if take >= len(Xc):
        return np.arange(len(Xc), dtype=int)

    # A substantial uniform skeleton prevents a low-margin-only collapse.
    seed_take = max(4, min(take, int(round(0.55 * take))))
    sel = _farthest_point_cover(Xc, seed_take).tolist()

    qlo, qhi = np.quantile(margin, [0.05, 0.85])
    scaled = np.clip((margin - qlo) / (qhi - qlo + 1e-12), 0.0, 1.0)
    weight = (0.15 + scaled) ** (-beta)

    mind = np.full(len(Xc), np.inf, dtype=float)
    for j in sel:
        mind = np.minimum(mind, ((Xc - Xc[j]) ** 2).sum(1))

    while len(sel) < take:
        score = np.sqrt(mind + 1e-12) * weight
        score[np.asarray(sel, dtype=int)] = -np.inf
        j = int(np.argmax(score))
        sel.append(j)
        mind = np.minimum(mind, ((Xc - Xc[j]) ** 2).sum(1))
    return np.asarray(sel, dtype=int)


def _update_k4(arr, mask, dist):
    if not np.any(mask):
        return
    tmp = np.concatenate([arr[mask], dist[mask, None]], axis=1)
    arr[mask] = np.sort(tmp, axis=1)[:, :VOTE_NEED]


def _vote_support_repair(X, y, clean, selected, budget):
    """Use remaining budget on locations where the frozen kNN lacks own-class votes.

    For each clean pool point, track the 4th-nearest selected same-class and different-
    class prototypes.  If same-class support is farther away, that point is at risk of
    losing a k=7 majority vote even when its raw cross-class margin is not minimal.
    """
    selected = [int(v) for v in selected]
    used = set(selected)
    clean_idx = np.where(clean)[0]
    Xc = X[clean_idx]
    yc = y[clean_idx]

    own = np.full((len(clean_idx), VOTE_NEED), np.inf, dtype=float)
    foreign = np.full((len(clean_idx), VOTE_NEED), np.inf, dtype=float)

    for s in selected:
        d = np.sqrt(((Xc - X[s]) ** 2).sum(1))
        same = yc == y[s]
        _update_k4(own, same, d)
        _update_k4(foreign, ~same, d)

    while len(selected) < budget:
        # ratio > 1 means the fourth own vote arrives after the fourth foreign support.
        risk = np.maximum(own[:, -1] / (foreign[:, -1] + 1e-6) - 1.0, 0.0)
        if used:
            risk[np.isin(clean_idx, np.fromiter(used, dtype=int))] = -np.inf
        q = int(np.argmax(risk))

        # Once explicit vote deficits are repaired, fall back to the least-supported
        # clean location so the last few rows still improve global coverage.
        if (not np.isfinite(risk[q])) or risk[q] <= 0.0:
            fallback = own[:, 0].copy()
            if used:
                fallback[np.isin(clean_idx, np.fromiter(used, dtype=int))] = -np.inf
            q = int(np.argmax(fallback))

        s = int(clean_idx[q])
        selected.append(s)
        used.add(s)
        d = np.sqrt(((Xc - X[s]) ** 2).sum(1))
        same = yc == y[s]
        _update_k4(own, same, d)
        _update_k4(foreign, ~same, d)

    return np.asarray(selected[:budget], dtype=int)


def _candidate_select(X, y, budget, clean, margin, beta, initial_frac):
    classes = np.unique(y)
    per = budget // len(classes)
    initial_per = max(VOTE_NEED, int(round(per * initial_frac)))
    chosen = []

    for c in classes:
        cand = np.where(clean & (y == c))[0]
        if len(cand) == 0:
            cand = np.where(y == c)[0]
        mc = margin[cand].copy()
        finite = np.isfinite(mc)
        fill = np.max(mc[finite]) if finite.any() else 1.0
        mc = np.where(finite, mc, fill)
        loc = _initial_cover(X[cand], mc, min(initial_per, len(cand)), beta)
        chosen.extend(cand[loc].tolist())

    return _vote_support_repair(X, y, clean, chosen, budget)


def _build_proxy_sets(X, y, clean, margin):
    """Deterministic visible-pool stand-ins for the two deployment concerns.

    The global proxy de-densifies the pool with per-class farthest-point probes.  The
    difficulty proxy takes low-margin survivors and de-densifies those too.  These are
    used only to choose among the small robust setting family above.
    """
    global_probe, difficult_probe = [], []
    for c in np.unique(y):
        cand = np.where(clean & (y == c))[0]
        if len(cand) == 0:
            continue
        g = _farthest_point_cover(X[cand], min(180, len(cand)))
        global_probe.extend(cand[g].tolist())

        mc = margin[cand]
        finite = np.isfinite(mc)
        if not finite.any():
            continue
        threshold = np.quantile(mc[finite], 0.22)
        hard = cand[mc <= threshold]
        if len(hard):
            h = _farthest_point_cover(X[hard], min(100, len(hard)))
            difficult_probe.extend(hard[h].tolist())

    return np.asarray(global_probe, dtype=int), np.asarray(difficult_probe, dtype=int)


def _proxy_score(X, y, selected, proxy_sets):
    model = KNeighborsClassifier(n_neighbors=7, metric="euclidean", weights="uniform")
    model.fit(X[selected], y[selected])
    vals = []
    for ids in proxy_sets:
        ids = ids[~np.isin(ids, selected)]
        if len(ids) == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.mean(model.predict(X[ids]) == y[ids])))
    return min(vals), sum(vals) / len(vals)


def select_positions(X, y, B):
    agree = _loo_agreement(X, y, DENOISE_K)
    clean = agree >= DENOISE_THR
    margin = _cross_class_k_margin(X, y, clean, 1)
    proxy_sets = _build_proxy_sets(X, y, clean, margin)

    best = None
    for beta, initial_frac in CANDIDATE_SETTINGS:
        sel = _candidate_select(X, y, B, clean, margin, beta, initial_frac)
        proxy = _proxy_score(X, y, sel, proxy_sets)
        item = (proxy[0], proxy[1], -beta, -initial_frac, sel)
        if best is None or item[:4] > best[:4]:
            best = item
    return best[-1]


def _load_pool(feat_cols, label_col, id_col):
    feats, ys, ids = [], [], []
    with open(POOL_CSV) as f:
        for row in csv.DictReader(f):
            feats.append([float(row[c]) for c in feat_cols])
            ys.append(int(float(row[label_col])))
            ids.append(int(float(row[id_col])))
    return np.asarray(feats, float), np.asarray(ys, int), np.asarray(ids, int)


def main():
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)
    X, y, ids = _load_pool(cfg["feature_cols"], cfg["label_col"], cfg["id_col"])
    B = int(cfg["budget"])
    pos = select_positions(X, y, B)
    selected_idx = np.unique(ids[pos])[:B]
    out = Path(OUTPUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("idx\n")
        for v in selected_idx:
            f.write(f"{int(v)}\n")
    print(f"pool {len(y)}x{X.shape[1]} -> selected {len(selected_idx)}/{B} -> {out}")


if __name__ == "__main__":
