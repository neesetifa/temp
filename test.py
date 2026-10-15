# ===== BEGIN FILE: tmp/environment/Dockerfile =====
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 pytest==8.3.5

WORKDIR /app
COPY app/ /app/
COPY tests/ /app/tests/

# ===== END FILE: tmp/environment/Dockerfile =====

# ===== BEGIN FILE: tmp/environment/app/build_index.py =====
"""Starter index builder: plain product quantization (4 sub-spaces x 256 centroids = 32 bits),
k-means trained on the catalog embeddings.  This is what the old offline job did.

    python3 /app/build_index.py            # writes /app/index.npz
"""
import numpy as np, os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.npz")
M, K = 4, 256


def kmeans(X, K, iters, rng):
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = ((C * C).sum(1)[None] - 2 * X @ C.T).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                C[k] = X[m].mean(0)
    return C


def main():
    rng = np.random.default_rng(0)
    X = np.load(os.path.join(DATA, "catalog", "item_embeddings.npy")).astype(np.float32)
    n, d = X.shape
    ds = d // M
    codebooks = np.zeros((M, K, d), np.float32)
    codes = np.zeros((n, M), np.uint8)
    for m in range(M):
        sl = slice(m * ds, (m + 1) * ds)
        C = kmeans(X[:, sl], K, 10, rng)
        codebooks[m, :, sl] = C
        codes[:, m] = ((C * C).sum(1)[None] - 2 * X[:, sl] @ C.T).argmin(1)
    np.savez(OUT, codebooks_0=codebooks, codes_0=codes)      # one tier, 32 bits per item
    print("wrote", OUT)


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/environment/app/build_index.py =====

# ===== BEGIN FILE: tmp/environment/app/tools/replay.py =====
"""Offline replay harness: replay logged requests against an index and report retrieval recall.

    python3 /app/tools/replay.py                                  # /app/index.npz vs the whole log
    python3 /app/tools/replay.py --since 2026-09-14 --model-version u2i-v14 --max-queries 5000

Ground truth is the exact top-10 by full-precision inner product over active catalog items.
Reports stage-1 recall@10 (ADC order) and recall@10 after the full-precision rerank of the
ADC top-20, i.e. the same two numbers the online dashboard shows.
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serving_sim as ss

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(APP, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(APP, "index.npz"))
    ap.add_argument("--since", default=None, help="only requests at/after this ISO date")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X = np.load(os.path.join(DATA, "catalog", "item_embeddings.npy")).astype(np.float64)
    with open(os.path.join(DATA, "catalog", "items.csv")) as f:
        active = np.array([r["status"] == "active" for r in csv.DictReader(f)])
    tier, tiers, bits = ss.load_index(a.index, len(X))
    Xh = ss.decode(tier, tiers)

    Q = np.load(os.path.join(DATA, "query_log", "query_embeddings.npy")).astype(np.float64)
    with open(os.path.join(DATA, "query_log", "requests.csv")) as f:
        req = list(csv.DictReader(f))
    keep = np.isfinite(Q).all(1)
    if a.since:
        keep &= np.array([r["ts"] >= a.since for r in req])
    if a.model_version:
        keep &= np.array([r["model_version"] == a.model_version for r in req])
    idx = np.flatnonzero(keep)
    if a.max_queries and len(idx) > a.max_queries:
        idx = np.sort(np.random.default_rng(a.seed).choice(idx, a.max_queries, replace=False))
    Q = Q[idx]
    thr = ss.exact_threshold(Q, X, active)
    r10, r100 = ss.replay(Q, X, active, Xh, thr)
    desc = ", ".join(f"tier {t}: {int((tier == t).sum())} items {cb.shape[0]}x{cb.shape[1]}" for t, (cb, _) in enumerate(tiers))
    print(f"index: {desc}; {bits:.2f} bits/item; queries={len(Q)}")
    print(f"stage-1 recall@10          {r10.mean():.4f}")
    print(f"recall@10 after rerank@20  {r100.mean():.4f}")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/environment/app/tools/replay.py =====

# ===== BEGIN FILE: tmp/environment/app/tools/serving_sim.py =====
"""Offline copy of the serving path: how the serving tier parses an index, and the two-stage
retrieval replay used for the dashboard metrics.

Index contract (/app/index.npz), a tiered additive-codebook index:
  tier          uint8 (N,)               tier of each catalog row; optional if there is one tier
  codebooks_t   float16/float32 (M_t, K_t, 64)   for t = 0..T-1 (T <= 4)
  codes_t       uint8 (N_t, M_t)          codes of the rows with tier == t, in catalog order
  1 <= M_t <= 32, K_t a power of two in [2, 256]
  memory = sum_t N_t*M_t*log2(K_t) + N*ceil(log2 T)  bits  <=  32 * N
  reconstruction of item i in tier t:  sum_m codebooks_t[m, codes_t[row, m]]
"""
import numpy as np, os, stat, zipfile

D = 64
BITS_PER_ITEM = 32
MAX_TIERS = 4
MAX_M = 32
MAX_FILE = 64 * 2**20
SHORTLIST = 20
TOPK = 10


class IndexError_(ValueError):
    pass


def _pow2(k):
    return 2 <= k <= 256 and (k & (k - 1)) == 0


ALLOWED = {"tier.npy"} | {f"codebooks_{t}.npy" for t in range(MAX_TIERS)} | {f"codes_{t}.npy" for t in range(MAX_TIERS)}


def _read_members(path):
    st = os.lstat(path)                                   # refuse symlinks / special files
    if not stat.S_ISREG(st.st_mode):
        raise IndexError_("index.npz is not a regular file")
    if st.st_size > MAX_FILE:
        raise IndexError_(f"index.npz too large ({st.st_size} bytes)")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        try:
            zf = zipfile.ZipFile(f)
            infos = zf.infolist()
        except (zipfile.BadZipFile, OSError, ValueError):
            raise IndexError_("index.npz is not a valid npz")
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or not set(names) <= ALLOWED:
            raise IndexError_(f"unexpected arrays in index.npz: {sorted(names)}")
        arr = {}
        for info in infos:
            key = info.filename[:-4]
            try:
                with zf.open(info) as fp:
                    # validate the .npy header before anything is allocated
                    version = np.lib.format.read_magic(fp)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
                    else:
                        raise IndexError_(f"{key}: unsupported .npy version {version}")
                    if dtype.hasobject or dtype.kind not in "fu" or dtype.itemsize > 8:
                        raise IndexError_(f"{key}: dtype {dtype} not allowed")
                    if len(shape) > 3 or any(int(x) < 0 for x in shape):
                        raise IndexError_(f"{key}: bad shape {shape}")
                    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    if nbytes > MAX_FILE:
                        raise IndexError_(f"{key}: {nbytes} bytes declared, too large")
                    data = fp.read(nbytes + 1)
                    if len(data) != nbytes:
                        raise IndexError_(f"{key}: data size does not match header")
                    a = np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                    arr[key] = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("="), copy=True)
            except IndexError_:
                raise
            except Exception as e:      # corrupt member, bad CRC, malformed header, ...
                raise IndexError_(f"{key}: unreadable ({type(e).__name__}: {e})")
    return arr


def load_index(path, n_items):
    """returns (tier (N,) int array, [(codebooks float64, codes uint8)] per tier, bits per item)."""
    arr = _read_members(path)
    T = 0
    while f"codebooks_{T}" in arr or f"codes_{T}" in arr:
        T += 1
    if T == 0:
        raise IndexError_("index.npz needs codebooks_0 and codes_0")
    if set(arr) - {"tier"} != {f"codebooks_{t}" for t in range(T)} | {f"codes_{t}" for t in range(T)}:
        raise IndexError_(f"tiers must be numbered 0..T-1 with both codebooks_t and codes_t, got {sorted(arr)}")
    if "tier" in arr:
        tier = arr["tier"]
        if tier.dtype.kind != "u" or tier.dtype.itemsize != 1 or tier.shape != (n_items,):
            raise IndexError_(f"tier must be uint8 ({n_items},), got {tier.dtype} {tier.shape}")
        if int(tier.max(initial=0)) >= T:
            raise IndexError_("tier value without matching codebooks")
    elif T == 1:
        tier = np.zeros(n_items, np.uint8)
    else:
        raise IndexError_("more than one tier requires the tier array")
    tier = tier.astype(np.int64)
    tiers, bits = [], 0
    for t in range(T):
        cb, codes = arr[f"codebooks_{t}"], arr[f"codes_{t}"]
        if cb.dtype.kind != "f" or cb.dtype.itemsize not in (2, 4) or cb.ndim != 3 or cb.shape[2] != D:
            raise IndexError_(f"codebooks_{t} must be float16/float32 (M, K, {D}), got {cb.dtype} {cb.shape}")
        M, K = cb.shape[0], cb.shape[1]
        if not (1 <= M <= MAX_M) or not _pow2(K):
            raise IndexError_(f"tier {t}: need 1<=M<={MAX_M} and K a power of two in [2,256], got M={M} K={K}")
        n_t = int((tier == t).sum())
        if codes.dtype.kind != "u" or codes.dtype.itemsize != 1 or codes.shape != (n_t, M):
            raise IndexError_(f"codes_{t} must be uint8 ({n_t}, {M}), got {codes.dtype} {codes.shape}")
        if n_t and int(codes.max()) >= K:
            raise IndexError_(f"codes_{t}: code value >= K")
        cb = cb.astype(np.float64)
        if not np.isfinite(cb).all():
            raise IndexError_(f"codebooks_{t}: non-finite values")
        bits += n_t * M * int(np.log2(K))
        tiers.append((cb, codes))
    bits += n_items * int(np.ceil(np.log2(T))) if T > 1 else 0
    if bits > BITS_PER_ITEM * n_items:
        raise IndexError_(f"index uses {bits / n_items:.3f} bits per item on average, budget is {BITS_PER_ITEM}")
    return tier, tiers, bits / n_items


def decode(tier, tiers):
    X = np.zeros((len(tier), D))
    for t, (cb, codes) in enumerate(tiers):
        rows = np.flatnonzero(tier == t)
        Xt = np.zeros((len(rows), D))
        for m in range(cb.shape[0]):
            Xt += cb[m][codes[:, m]]
        X[rows] = Xt
    return X


def replay(Q, X, active, Xh, thr, chunk=500):
    """stage 1: ADC top-20 over active items; stage 2: exact rerank -> top-10.
    returns per-query tie-aware recall@10 of stage 1 and recall@10 after the rerank."""
    ai = np.flatnonzero(active)
    Xa, Xha = X[ai], Xh[ai]
    r10 = np.empty(len(Q)); r100 = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        A = q @ Xha.T
        A[~np.isfinite(A)] = -np.inf
        top = np.argpartition(-A, SHORTLIST - 1, 1)[:, :SHORTLIST]
        o = np.argsort(-np.take_along_axis(A, top, 1), 1, kind="stable")
        top = np.take_along_axis(top, o, 1)
        ex = np.einsum("nd,nkd->nk", q, Xa[top])
        t = thr[s:s + chunk, None]
        hit = ex >= t - 1e-6 * np.abs(t)
        r10[s:s + chunk] = np.minimum(hit[:, :TOPK].sum(1), TOPK) / TOPK
        r100[s:s + chunk] = np.minimum(hit.sum(1), TOPK) / TOPK
    return r10, r100


def exact_threshold(Q, X, active, chunk=500):
    Xa = X[active]
    thr = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ Xa.T
        thr[s:s + chunk] = -np.partition(-S, TOPK - 1, 1)[:, TOPK - 1]
    return thr


def group_score(miss, miss0, miss1):
    """log-scale normalisation of a miss rate between the 0-anchor and the 1-anchor."""
    miss = max(miss, 1e-6)
    return float(np.clip(np.log(miss0 / miss) / np.log(miss0 / miss1), 0.0, 1.0))

# ===== END FILE: tmp/environment/app/tools/serving_sim.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Build the 4-byte retrieval index for the Sep 27 catalog snapshot

You are on the retrieval team of an e-commerce app. Candidate generation uses a two-tower model,
`u2i-v14`:

- the **item tower** is run offline over the whole catalog every Sunday night and its 64-d
  embeddings are exported;
- the **user tower** runs online for every request, on the user's most recent clicks, and
  produces a 64-d query vector;
- the **retrieval stage** returns the catalog items with the largest inner product
  `<query, item>` (maximum inner product search).

The catalog (~112M items) is split by item-id hash into 1,024 shards; every replica of the
serving tier keeps all shards in memory, searches each shard and merges the per-shard top-k.
Retrieval is moving onto the shared in-memory tier, which gives it a hard budget of
**32 bits (4 bytes) per catalog item** for the item codes, about 450 MB per replica (the
float32 embeddings would be 256 bytes per item, ~29 GB), and only understands one index layout:
additive codebooks scored with lookup tables (ADC). Each shard's index is rebuilt every Sunday night from that week's
catalog snapshot and then serves the following **two weeks** of traffic.

Your job: build the index for **shard 0417** from the snapshot taken **Sunday 2026-09-27**.
(The recipe will later be rolled out to the other shards; here only this shard's index is
evaluated.) It will serve production traffic from **Mon 2026-09-28 to Sun 2026-10-11**. That window includes the October holiday
(Oct 1–7) and the holiday sale, and marketing is running a user acquisition campaign alongside it.
The request logs you have end at the snapshot; none of the logged weeks contains a holiday.

## What you have (`/app/data`, read-only inputs)

- `catalog/items.csv` – the shard's catalog snapshot: `item_id, category_id, subcategory_id,
  brand_id, launched_at, status` (`active` or `delisted`). Row *r* corresponds to row *r* of
  `catalog/item_embeddings.npy` (float16, shape `(N, 64)`, the item-tower export).
- `query_log/requests.csv` + `query_log/query_embeddings.npy` – a ~1% sample of retrieval
  requests from 2026-08-03 to 2026-09-27: `request_id, ts, user_id (hashed), model_version`,
  and the query vector the user tower produced (float16, same row order). Every request is sent
  to every shard, so the same log serves all shards. This is the raw export from the logging
  pipeline; nobody has cleaned it.
- `tools/serving_sim.py` – an offline copy of the serving path: how the serving tier parses an
  index and scores it. `tools/replay.py` replays logged requests against an index and prints the
  two dashboard metrics described below.
- `build_index.py` – the old offline job (plain product quantization, 4 sub-spaces x 256
  centroids, trained on the catalog). Run it to get a valid baseline index.

## What to deliver: `/app/index.npz`

The serving tier supports a **tiered** layout: catalog rows can be split into up to 4 tiers, each
with its own additive codebooks and code length (for example, longer codes for items you expect
to be retrieved a lot and shorter ones for the long tail). Memory is accounted per item.

| array | dtype | shape | meaning |
|---|---|---|---|
| `codebooks_t` | float16 or float32 | `(M_t, K_t, 64)` | tier *t*'s additive codebooks, *t* = 0..T-1, T ≤ 4 |
| `codes_t` | uint8 | `(N_t, M_t)` | codes of the rows in tier *t*, in catalog row order |
| `tier` | uint8 | `(N,)` | tier of every catalog row (optional when T = 1) |

- `1 <= M_t <= 32`; `K_t` is a power of two between 2 and 256.
- Memory: `Σ_t N_t · M_t · log2(K_t)` bits for the codes, plus `N · ⌈log2 T⌉` bits for the tier
  map (free with a single tier), must be **≤ 32 · N** bits, i.e. 4 bytes per item on average.
  Codebooks are not counted.
- Item *i* in tier *t* (the *j*-th row of the catalog with `tier == t`) is reconstructed as
  `x̂_i = Σ_m codebooks_t[m, codes_t[j, m]]`; the serving tier scores `<query, x̂_i>` with lookup
  tables. This layout covers PQ, OPQ (fold the rotation into the codebooks), residual and
  additive quantization.
- Every catalog row needs a code, delisted items included (they are filtered at serving time).
- At most 64 MB, all values finite, a plain `.npz` (`np.savez` / `np.savez_compressed`), no other
  arrays.

How you build it is up to you. Only `/app/index.npz` is read; nothing you leave elsewhere is run.
The machine has 4 CPU cores and 12 GB RAM and no GPU.

## How it is evaluated

When the two weeks are over, the replay team takes a sample of the production requests from
those two weeks and replays them against your index, exactly as `tools/serving_sim.py` does.
As on the online dashboards, the replay set contains only real-user traffic served by
`u2i-v14`: scraper accounts, other model versions and broken log rows are excluded.

1. **Stage 1 (your index):** ADC scores over the shard's *active* catalog items; the shard keeps
   its top 20 (with 1,024 shards, the rerank service can afford 20 candidates per shard).
2. **Stage 2 (rerank):** the 20 candidates are rescored with the full-precision embeddings and
   the top 10 are sent on to the merge.

Ground truth for each request is the exact top 10 within this shard by full-precision inner
product over the active items (ties at the 10th score count as hits). Two metrics are computed
separately for each of the two serving weeks:

- **stage-1 recall@10**: fraction of the ground-truth top 10 in the ADC top 10;
- **recall@10 after rerank**: fraction of the ground-truth top 10 in the ADC top 20.

Each of the four numbers is turned into a score in [0, 1] on a log scale of the miss rate
(`1 - recall`), between a reference index built with standard methods (score 0) and a target
level (score 1). The reward is a weighted average: 0.4 for stage-1 recall@10 and 0.6 for recall
after rerank, split equally between the two weeks. An index that is missing or violates the
format gets 0.

`python3 /app/tools/replay.py` gives you the same two metrics on the logged requests (see
`--help` for filters).

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
"""Public format check for /app/index.npz (the hidden replay applies the same parser).

    python3 -m pytest -q /app/tests/test_public.py     # or: python3 /app/tests/test_public.py
"""
import os, sys
import numpy as np

sys.path.insert(0, "/app/tools")
import serving_sim as ss

INDEX = "/app/index.npz"
N = np.load("/app/data/catalog/item_embeddings.npy", mmap_mode="r").shape[0]


def test_index_exists():
    assert os.path.isfile(INDEX) and not os.path.islink(INDEX), "write /app/index.npz (a regular file)"


def test_index_format_and_budget():
    tier, tiers, bits = ss.load_index(INDEX, N)
    assert bits <= ss.BITS_PER_ITEM
    X = ss.decode(tier, tiers)
    assert X.shape == (N, ss.D) and np.isfinite(X).all()


if __name__ == "__main__":
    test_index_exists(); test_index_format_and_budget(); print("ok")

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/solve.sh =====
#!/bin/bash
# Privileged oracle: a tiered index built with knowledge of the actual Sep 28 - Oct 11 traffic.
cp /solution/index.npz /app/index.npz

# ===== END FILE: tmp/solution/solve.sh =====

# ===== BEGIN FILE: tmp/tests/Dockerfile =====
# Verifier image.  Only parses /app/index.npz as data (no candidate code is executed), so numpy is all it needs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN pip install numpy==2.2.6 pytest==8.3.5

# ===== END FILE: tmp/tests/Dockerfile =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
"""Verifier: replay the hidden serving-window traffic against /app/index.npz and write the reward.

The delivered index is untrusted *data*; nothing from /app is executed or imported.  The catalog
embeddings / status used for ground truth and rerank come from /tests (the agent could have
edited /app/data).
"""
import json, os, sys, stat, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring as sc

APP = os.environ.get("RIB_APP_DIR", "/app")
LOG = os.environ.get("RIB_LOG_DIR", "/logs/verifier")


def prepare_log_dir():
    os.makedirs(LOG, exist_ok=True)
    try:
        os.chown(LOG, 0, 0)
        os.chmod(LOG, 0o755)
    except PermissionError:
        pass
    for name in ("reward.txt", "metrics.json"):
        p = os.path.join(LOG, name)
        if os.path.lexists(p):
            os.unlink(p)


def write_out(reward, metrics):
    for name, text in (("metrics.json", json.dumps(metrics, indent=2)), ("reward.txt", f"{reward:.6f}\n")):
        p = os.path.join(LOG, name)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(text)


def main():
    t0 = time.time()
    prepare_log_dir()
    H = np.load(os.path.join(HERE, "hidden", "hidden_traffic.npz"), allow_pickle=False)
    A = json.load(open(os.path.join(HERE, "hidden", "anchors.json")))
    X = H["catalog_emb"].astype(np.float64)
    active = H["active"].astype(bool)
    Q = H["eval_q"].astype(np.float64)
    week = H["eval_week"]
    metrics = {"n_items": int(len(X)), "n_queries": int(len(Q))}
    try:
        tier, tiers, bits = sc.load_index(os.path.join(APP, "index.npz"), len(X))
    except Exception as e:  # any failure to parse the untrusted file scores 0
        metrics["error"] = f"invalid index: {e}"
        write_out(0.0, metrics)
        print(metrics["error"])
        return
    metrics["index"] = {"bits_per_item": float(bits),
                        "tiers": [{"items": int((tier == t).sum()), "M": int(cb.shape[0]), "K": int(cb.shape[1])}
                                  for t, (cb, _) in enumerate(tiers)]}
    Xh = sc.decode(tier, tiers)
    thr = H["eval_thr"].astype(np.float64)
    r10, r100 = sc.replay(Q, X, active, Xh, thr)
    groups = {}
    reward = 0.0
    for g, spec in A["groups"].items():
        r = r10 if spec["metric"] == "stage1_recall10" else r100
        m = week == spec["week"]
        miss = float(1.0 - r[m].mean())
        s = sc.group_score(miss, spec["miss0"], spec["miss1"])
        groups[g] = {"miss": miss, "recall": 1 - miss, "score": s, "weight": spec["weight"]}
        reward += spec["weight"] * s
    metrics["groups"] = groups
    metrics["reward"] = reward
    metrics["seconds"] = time.time() - t0
    write_out(reward, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/rescore.py =====
"""Recompute a reward offline from a run's metrics.json with the current (or a modified) anchors.json.

    python3 rescore.py metrics.json [anchors.json]
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring as sc

m = json.load(open(sys.argv[1]))
A = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "hidden", "anchors.json")))
if "groups" not in m:
    print("run had no valid index:", m.get("error")); print("reward 0.0"); sys.exit(0)
tot = 0.0
for g, spec in A["groups"].items():
    s = sc.group_score(m["groups"][g]["miss"], spec["miss0"], spec["miss1"])
    tot += spec["weight"] * s
    print(f"{g:14s} miss {m['groups'][g]['miss']:.5f}  score {s:.3f}")
print(f"reward {tot:.4f}")

# ===== END FILE: tmp/tests/rescore.py =====

# ===== BEGIN FILE: tmp/tests/scoring.py =====
"""Shared scoring code: parse the delivered index as untrusted data, replay traffic through the
two-stage retrieval path.

Index contract (/app/index.npz), a tiered additive-codebook index:
  tier          uint8 (N,)               tier of each catalog row; optional if there is one tier
  codebooks_t   float16/float32 (M_t, K_t, 64)   for t = 0..T-1 (T <= 4)
  codes_t       uint8 (N_t, M_t)          codes of the rows with tier == t, in catalog order
  1 <= M_t <= 32, K_t a power of two in [2, 256]
  memory = sum_t N_t*M_t*log2(K_t) + N*ceil(log2 T)  bits  <=  32 * N
  reconstruction of item i in tier t:  sum_m codebooks_t[m, codes_t[row, m]]
"""
import numpy as np, os, stat, zipfile

D = 64
BITS_PER_ITEM = 32
MAX_TIERS = 4
MAX_M = 32
MAX_FILE = 64 * 2**20
SHORTLIST = 20
TOPK = 10


class IndexError_(ValueError):
    pass


def _pow2(k):
    return 2 <= k <= 256 and (k & (k - 1)) == 0


ALLOWED = {"tier.npy"} | {f"codebooks_{t}.npy" for t in range(MAX_TIERS)} | {f"codes_{t}.npy" for t in range(MAX_TIERS)}


def _read_members(path):
    st = os.lstat(path)                                   # refuse symlinks / special files
    if not stat.S_ISREG(st.st_mode):
        raise IndexError_("index.npz is not a regular file")
    if st.st_size > MAX_FILE:
        raise IndexError_(f"index.npz too large ({st.st_size} bytes)")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        try:
            zf = zipfile.ZipFile(f)
            infos = zf.infolist()
        except (zipfile.BadZipFile, OSError, ValueError):
            raise IndexError_("index.npz is not a valid npz")
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or not set(names) <= ALLOWED:
            raise IndexError_(f"unexpected arrays in index.npz: {sorted(names)}")
        arr = {}
        for info in infos:
            key = info.filename[:-4]
            try:
                with zf.open(info) as fp:
                    # validate the .npy header before anything is allocated
                    version = np.lib.format.read_magic(fp)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
                    else:
                        raise IndexError_(f"{key}: unsupported .npy version {version}")
                    if dtype.hasobject or dtype.kind not in "fu" or dtype.itemsize > 8:
                        raise IndexError_(f"{key}: dtype {dtype} not allowed")
                    if len(shape) > 3 or any(int(x) < 0 for x in shape):
                        raise IndexError_(f"{key}: bad shape {shape}")
                    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    if nbytes > MAX_FILE:
                        raise IndexError_(f"{key}: {nbytes} bytes declared, too large")
                    data = fp.read(nbytes + 1)
                    if len(data) != nbytes:
                        raise IndexError_(f"{key}: data size does not match header")
                    a = np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                    arr[key] = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("="), copy=True)
            except IndexError_:
                raise
            except Exception as e:      # corrupt member, bad CRC, malformed header, ...
                raise IndexError_(f"{key}: unreadable ({type(e).__name__}: {e})")
    return arr


def load_index(path, n_items):
    """returns (tier (N,) int array, [(codebooks float64, codes uint8)] per tier, bits per item)."""
    arr = _read_members(path)
    T = 0
    while f"codebooks_{T}" in arr or f"codes_{T}" in arr:
        T += 1
    if T == 0:
        raise IndexError_("index.npz needs codebooks_0 and codes_0")
    if set(arr) - {"tier"} != {f"codebooks_{t}" for t in range(T)} | {f"codes_{t}" for t in range(T)}:
        raise IndexError_(f"tiers must be numbered 0..T-1 with both codebooks_t and codes_t, got {sorted(arr)}")
    if "tier" in arr:
        tier = arr["tier"]
        if tier.dtype.kind != "u" or tier.dtype.itemsize != 1 or tier.shape != (n_items,):
            raise IndexError_(f"tier must be uint8 ({n_items},), got {tier.dtype} {tier.shape}")
        if int(tier.max(initial=0)) >= T:
            raise IndexError_("tier value without matching codebooks")
    elif T == 1:
        tier = np.zeros(n_items, np.uint8)
    else:
        raise IndexError_("more than one tier requires the tier array")
    tier = tier.astype(np.int64)
    tiers, bits = [], 0
    for t in range(T):
        cb, codes = arr[f"codebooks_{t}"], arr[f"codes_{t}"]
        if cb.dtype.kind != "f" or cb.dtype.itemsize not in (2, 4) or cb.ndim != 3 or cb.shape[2] != D:
            raise IndexError_(f"codebooks_{t} must be float16/float32 (M, K, {D}), got {cb.dtype} {cb.shape}")
        M, K = cb.shape[0], cb.shape[1]
        if not (1 <= M <= MAX_M) or not _pow2(K):
            raise IndexError_(f"tier {t}: need 1<=M<={MAX_M} and K a power of two in [2,256], got M={M} K={K}")
        n_t = int((tier == t).sum())
        if codes.dtype.kind != "u" or codes.dtype.itemsize != 1 or codes.shape != (n_t, M):
            raise IndexError_(f"codes_{t} must be uint8 ({n_t}, {M}), got {codes.dtype} {codes.shape}")
        if n_t and int(codes.max()) >= K:
            raise IndexError_(f"codes_{t}: code value >= K")
        cb = cb.astype(np.float64)
        if not np.isfinite(cb).all():
            raise IndexError_(f"codebooks_{t}: non-finite values")
        bits += n_t * M * int(np.log2(K))
        tiers.append((cb, codes))
    bits += n_items * int(np.ceil(np.log2(T))) if T > 1 else 0
    if bits > BITS_PER_ITEM * n_items:
        raise IndexError_(f"index uses {bits / n_items:.3f} bits per item on average, budget is {BITS_PER_ITEM}")
    return tier, tiers, bits / n_items


def decode(tier, tiers):
    X = np.zeros((len(tier), D))
    for t, (cb, codes) in enumerate(tiers):
        rows = np.flatnonzero(tier == t)
        Xt = np.zeros((len(rows), D))
        for m in range(cb.shape[0]):
            Xt += cb[m][codes[:, m]]
        X[rows] = Xt
    return X


def replay(Q, X, active, Xh, thr, chunk=500):
    """stage 1: ADC top-20 over active items; stage 2: exact rerank -> top-10.
    returns per-query tie-aware recall@10 of stage 1 and recall@10 after the rerank."""
    ai = np.flatnonzero(active)
    Xa, Xha = X[ai], Xh[ai]
    r10 = np.empty(len(Q)); r100 = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        A = q @ Xha.T
        A[~np.isfinite(A)] = -np.inf
        top = np.argpartition(-A, SHORTLIST - 1, 1)[:, :SHORTLIST]
        o = np.argsort(-np.take_along_axis(A, top, 1), 1, kind="stable")
        top = np.take_along_axis(top, o, 1)
        ex = np.einsum("nd,nkd->nk", q, Xa[top])
        t = thr[s:s + chunk, None]
        hit = ex >= t - 1e-6 * np.abs(t)
        r10[s:s + chunk] = np.minimum(hit[:, :TOPK].sum(1), TOPK) / TOPK
        r100[s:s + chunk] = np.minimum(hit.sum(1), TOPK) / TOPK
    return r10, r100


def exact_threshold(Q, X, active, chunk=500):
    Xa = X[active]
    thr = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ Xa.T
        thr[s:s + chunk] = -np.partition(-S, TOPK - 1, 1)[:, TOPK - 1]
    return thr


def group_score(miss, miss0, miss1):
    """log-scale normalisation of a miss rate between the 0-anchor and the 1-anchor."""
    miss = max(miss, 1e-6)
    return float(np.clip(np.log(miss0 / miss) / np.log(miss0 / miss1), 0.0, 1.0))

# ===== END FILE: tmp/tests/scoring.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
# Verifier entry point: replays the hidden Sep 28 - Oct 11 traffic against /app/index.npz and writes
# /logs/verifier/reward.txt (+ metrics.json).  The index is parsed as data; nothing from /app runs.
set -u
mkdir -p /logs/verifier
chmod -R go-rwx /tests 2>/dev/null || true
cd /
python3 -I /tests/hidden_eval.py
if [ ! -s /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
exit 0

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_security.py =====
"""Regression tests for the verifier's handling of hostile /app/index.npz files.

Run as root inside the verifier image:  python3 /tests/test_security.py
Every hostile index must yield reward 0 with an "invalid index" error (never a crash, never a
read of hidden data); a valid index must still score; and the reverse control shows the symlink
case is caught by the symlink check itself.
"""
import io, json, os, shutil, subprocess, sys, tempfile, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring as sc

N = int(np.load(os.path.join(HERE, "hidden", "hidden_traffic.npz"))["active"].shape[0])
D = sc.D
FAIL = []


def run_eval(app):
    log = tempfile.mkdtemp()
    env = dict(os.environ, RIB_APP_DIR=app, RIB_LOG_DIR=log)
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "hidden_eval.py")], env=env,
                       capture_output=True, text=True, timeout=900)
    rew = open(os.path.join(log, "reward.txt")).read().strip() if os.path.exists(os.path.join(log, "reward.txt")) else None
    met = json.load(open(os.path.join(log, "metrics.json"))) if os.path.exists(os.path.join(log, "metrics.json")) else {}
    shutil.rmtree(log)
    return p.returncode, rew, met, p.stderr[-2000:]


def valid_arrays(M=4, K=256, seed=0):
    r = np.random.default_rng(seed)
    return {"codebooks_0": r.normal(0, 0.5, (M, K, D)).astype(np.float32),
            "codes_0": r.integers(0, K, (N, M)).astype(np.uint8)}


def two_tiers(n_hot=1000, hot=(8, 256), cold=(3, 256), tier_dtype=np.uint8, drop_tier=False, bump=0):
    r = np.random.default_rng(1)
    tier = np.ones(N, tier_dtype); tier[:n_hot] = 0
    out = {"codebooks_0": r.normal(0, .5, (hot[0], hot[1], D)).astype(np.float32),
           "codes_0": r.integers(0, hot[1], (n_hot + bump, hot[0])).astype(np.uint8),
           "codebooks_1": r.normal(0, .5, (cold[0], cold[1], D)).astype(np.float32),
           "codes_1": r.integers(0, cold[1], (N - n_hot, cold[0])).astype(np.uint8)}
    if not drop_tier:
        out["tier"] = tier
    return out


def write_npz(path, arrays):
    np.savez(path, **arrays)


def raw_npz(path, members):
    """members: name -> raw bytes of a .npy file"""
    with zipfile.ZipFile(path, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)


def npy_bytes(a):
    b = io.BytesIO(); np.save(b, a, allow_pickle=True); return b.getvalue()


def expect_zero(name, build):
    app = tempfile.mkdtemp(); os.chmod(app, 0o755)
    try:
        build(os.path.join(app, "index.npz"))
        rc, rew, met, err = run_eval(app)
        ok = rc == 0 and rew is not None and float(rew) == 0.0 and "invalid index" in met.get("error", "")
        print(f"[{'ok' if ok else 'FAIL'}] {name}: rc={rc} reward={rew} err={met.get('error', '')[:90]!r}")
        if not ok:
            FAIL.append(name); print(err)
    finally:
        shutil.rmtree(app)


def huge_header(path):
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (100000000, 256, 64), }"
    hdr = hdr + " " * (63 - (len(hdr) + 10) % 64) + "\n"
    blob = b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode() + b"\0" * 64
    raw_npz(path, {"codebooks_0.npy": blob, "codes_0.npy": npy_bytes(valid_arrays()["codes_0"])})


def truncated(path):
    a = valid_arrays(); b = npy_bytes(a["codebooks_0"])[:-1000]
    raw_npz(path, {"codebooks_0.npy": b, "codes_0.npy": npy_bytes(a["codes_0"])})


def zip_bomb(path):
    a = valid_arrays()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("codebooks_0.npy", npy_bytes(a["codebooks_0"]))
        z.writestr("codes_0.npy", npy_bytes(a["codes_0"]))
        z.writestr("codes_0.npy", b"\0" * (300 * 2**20))       # duplicate member, 300 MB of zeros


def hidden_symlink(path):
    os.symlink(os.path.join(HERE, "hidden", "hidden_traffic.npz"), path)


def valid_symlink(path):
    tgt = path + ".real.npz"; write_npz(tgt, valid_arrays()); os.symlink(tgt, path)


V = valid_arrays()
cases = {
    "symlink to hidden data": hidden_symlink,
    "symlink to a valid index": valid_symlink,
    "directory instead of file": lambda p: os.mkdir(p),
    "not a zip": lambda p: open(p, "wb").write(b"hello"),
    "pickled object array": lambda p: raw_npz(p, {"codebooks_0.npy": npy_bytes(np.array([{"a": 1}], dtype=object)),
                                                   "codes_0.npy": npy_bytes(V["codes_0"])}),
    "header declares 6.5 TB": huge_header,
    "truncated member": truncated,
    "zip bomb / duplicate member": zip_bomb,
    "extra member": lambda p: write_npz(p, {**V, "note": np.zeros(3, np.float32)}),
    "missing codes": lambda p: write_npz(p, {"codebooks_0": V["codebooks_0"]}),
    "single tier over budget (5x256)": lambda p: write_npz(p, valid_arrays(M=5)),
    "tier map pushes over budget (4x256 + 4x256)": lambda p: write_npz(p, two_tiers(hot=(4, 256), cold=(4, 256))),
    "two tiers without tier array": lambda p: write_npz(p, two_tiers(drop_tier=True)),
    "tier numbering gap": lambda p: write_npz(p, {**V, "codebooks_2": V["codebooks_0"], "codes_2": V["codes_0"][:0]}),
    "tier value without codebooks": lambda p: write_npz(p, {**V, "tier": np.full(N, 1, np.uint8)}),
    "codes rows != tier size": lambda p: write_npz(p, two_tiers(bump=1)),
    "tier array int64": lambda p: write_npz(p, two_tiers(tier_dtype=np.int64)),
    "M > 32": lambda p: write_npz(p, {"codebooks_0": np.zeros((33, 2, D), np.float32), "codes_0": np.zeros((N, 33), np.uint8),
                                      "tier": np.zeros(N, np.uint8)}),
    "K not a power of two": lambda p: write_npz(p, {"codebooks_0": np.zeros((4, 200, D), np.float32),
                                                     "codes_0": np.zeros((N, 4), np.uint8)}),
    "code >= K": lambda p: write_npz(p, {"codebooks_0": np.zeros((8, 16, D), np.float32),
                                         "codes_0": np.full((N, 8), 16, np.uint8)}),
    "codes int64": lambda p: write_npz(p, {"codebooks_0": V["codebooks_0"], "codes_0": V["codes_0"].astype(np.int64)}),
    "wrong row count": lambda p: write_npz(p, {"codebooks_0": V["codebooks_0"], "codes_0": V["codes_0"][:-1]}),
    "NaN in codebooks": lambda p: write_npz(p, {**V, "codebooks_0": np.full((4, 256, D), np.nan, np.float32)}),
    "float64 codebooks": lambda p: write_npz(p, {**V, "codebooks_0": V["codebooks_0"].astype(np.float64)}),
    "missing file": lambda p: None,
}

if __name__ == "__main__":
    for name, build in cases.items():
        expect_zero(name, build)
    # positive control: a valid (random) index is parsed and scored
    app = tempfile.mkdtemp(); os.chmod(app, 0o755)
    write_npz(os.path.join(app, "index.npz"), two_tiers())       # 2 tiers, 1000 x 64 + rest x 24 bits + map
    rc, rew, met, err = run_eval(app)
    ok = rc == 0 and rew is not None and "groups" in met and "error" not in met
    print(f"[{'ok' if ok else 'FAIL'}] positive control: valid index scored, reward={rew}")
    if not ok:
        FAIL.append("positive control"); print(err)
    shutil.rmtree(app)
    # reverse control: without the lstat/O_NOFOLLOW check the symlinked valid index would load
    app = tempfile.mkdtemp()
    p = os.path.join(app, "index.npz"); valid_symlink(p)
    with open(p, "rb") as f:              # following the link works ...
        z = np.load(f); loaded = set(z.files)
    caught = False
    try:
        sc.load_index(p, N)
    except sc.IndexError_:
        caught = True                    # ... so it is the symlink check that rejects it
    ok = loaded == {"codebooks_0", "codes_0"} and caught
    print(f"[{'ok' if ok else 'FAIL'}] reverse control: link target readable={loaded == {'codebooks_0', 'codes_0'}}, rejected by load_index={caught}")
    if not ok:
        FAIL.append("reverse control")
    shutil.rmtree(app)
    print("SECURITY TESTS", "FAILED: " + ", ".join(FAIL) if FAIL else "PASSED")
    sys.exit(1 if FAIL else 0)

# ===== END FILE: tmp/tests/test_security.py =====

