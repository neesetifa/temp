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
additive codebooks scored with lookup tables (ADC). Each shard's index is rebuilt every Sunday
night from that week's catalog snapshot and then serves the following **two weeks** of traffic.

Your job: build the index for **shard 0417** from the snapshot taken **Sunday 2026-09-27**.
This shard is the first to switch over; the recipe will be rolled out to the other shards
afterwards. The index will serve production traffic from **Mon 2026-09-28 to Sun 2026-10-11**.
That window includes the October holiday (Oct 1–7) and the holiday sale, and marketing is
running a user acquisition campaign alongside it. The request logs you have end at the
snapshot; none of the logged weeks contains a holiday.

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

How you build it is up to you. The Sunday deploy job picks up `/app/index.npz` and nothing else
from `/app`. The build machine has 4 CPU cores, 12 GB RAM and no GPU.

## Serving path and metrics

The index goes live on Mon 2026-09-28 and serves this shard for two weeks. Every request goes
through two stages, exactly as in `tools/serving_sim.py`:

1. **Stage 1 (this index):** ADC scores over the shard's *active* catalog items; the shard keeps
   its top 20 (with 1,024 shards, the rerank service can afford 20 candidates per shard).
2. **Stage 2 (rerank):** the 20 candidates are rescored with the full-precision embeddings and
   the top 10 are sent on to the merge.

The retrieval dashboards track two recall numbers per serving week, on real-user traffic served
by `u2i-v14` (scraper accounts, other model versions and broken log rows are excluded).
Recall is measured against the exact top 10 within the shard by full-precision inner product
over the active items (ties at the 10th score count as hits):

- **recall@10 after rerank**: fraction of the exact top 10 that is in the ADC top 20. This is
  what the merge and the ranking stages actually receive, so it is the team's headline number.
- **stage-1 recall@10**: fraction of the exact top 10 that is in the ADC top 10. When the rerank
  service sheds load, the shard's ADC top 10 is sent on directly, so this one matters too.

Both have to hold up in both weeks, holiday week included. The team reads them as miss rates
(`1 - recall`): going from 99.0% to 99.5% recall halves the misses and counts for far more than
going from 90.0% to 90.5%. The serving tier validates the layout and the memory budget when it
loads an index and refuses to load one that does not conform.

`python3 /app/tools/replay.py` computes the same two metrics on the logged requests (see
`--help` for filters).

serving_sim
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


test_public
"""Format check for /app/index.npz with the serving tier's own parser (tools/serving_sim.py).

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



