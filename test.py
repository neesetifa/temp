# ===== BEGIN FILE: tmp/environment/Dockerfile =====
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

RUN pip install --no-cache-dir \
      numpy==2.2.6 \
      scipy==1.15.3 \
      scikit-learn==1.6.1 \
      pytest==8.3.5 \
      pytest-json-report==1.5.0

WORKDIR /app
COPY app/ /app/
COPY tests/ /app/tests/

# ===== END FILE: tmp/environment/Dockerfile =====

# ===== BEGIN FILE: tmp/environment/README.md =====
# kv_cache_eviction

Edit `app/policy.py` (and optionally add your own precomputed files under `app/`).

Files:

- `app/model.npz`: weights of the pretrained 4-layer decoder (vocab 256, d=128, 4 heads, RoPE).
- `app/kvmodel.py`: numpy inference code with the chunked KV-eviction loop used by the verifier.
- `app/data/train_tokens.npy`: int16 array (800, 512) of training-corpus sequences.
- `app/policy.py`: starter policy (attention sinks + recent window).
- `app/evaluate_local.py`: local KL check for your policy.

Run the public tests from the `environment` directory with:

```bash
python -m pytest tests
```

# ===== END FILE: tmp/environment/README.md =====

# ===== BEGIN FILE: tmp/environment/app/evaluate_local.py =====
"""Local evaluation helper.

Runs /app/policy.py in-process on sequences from a token file and reports the
mean per-position KL(full cache || your policy) over positions after the first
chunk. The hidden verifier uses the same inference code, chunk size and budget,
but different sequences, a separate process for the policy, and a weighted
breakdown of positions (see instruction.md).

    python evaluate_local.py --tokens data/train_tokens.npy --n 20
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np

from kvmodel import TinyLM, kl_full_vs

APP = Path(__file__).resolve().parent
CHUNK = 32
BUDGET = 96


def load_policy(path):
    spec = importlib.util.spec_from_file_location("policy", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EvictionPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=str(APP / "data" / "train_tokens.npy"))
    ap.add_argument("--policy", default=str(APP / "policy.py"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ns = ap.parse_args()
    model = TinyLM(APP / "model.npz")
    toks = np.load(ns.tokens)[ns.offset:ns.offset + ns.n]
    cfg = {"d": model.d, "heads": model.H, "layers": model.L, "head_dim": model.hd, "vocab": model.V,
           "chunk": CHUNK, "seq_len": toks.shape[1], "budget": BUDGET, "app_dir": str(APP)}
    policy = load_policy(ns.policy)(cfg)
    t0 = time.time()
    kls, top1 = [], []
    for t in toks:
        full = model.run(t)
        test = model.run(t, policy=policy, budget=BUDGET, chunk=CHUNK)
        kls.append(kl_full_vs(full, test)[CHUNK:-1])
        top1.append((full.argmax(-1) == test.argmax(-1))[CHUNK:-1])
    kl = np.concatenate(kls)
    print(f"sequences={len(toks)}  mean_KL={kl.mean():.5f}  p99_KL={np.percentile(kl, 99):.4f}  "
          f"max_KL={kl.max():.3f}  top1_agree={np.concatenate(top1).mean():.4f}  sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/environment/app/evaluate_local.py =====

# ===== BEGIN FILE: tmp/environment/app/kvmodel.py =====
"""Chunked numpy inference for the tiny decoder, with a KV-eviction hook.

This file is shipped to the agent (as /app/kvmodel.py) and the verifier uses an
identical copy. Inference is float64 numpy.

Processing model
----------------
The sequence is processed in chunks of `chunk` tokens. For every chunk and every
layer (in order 0..L-1):

  1. the chunk's queries attend to [kept cache entries of this layer/head] plus
     the chunk's own entries (causal inside the chunk);
  2. the chunk's keys/values are appended to the cache;
  3. the eviction policy is called with a `LayerView` and returns, per head,
     the indices (into that head's current cache arrays) to KEEP.

After the last layer of a chunk the total number of cache entries over all
layers and heads must be <= budget. Heads may hold different numbers of
entries and layers may hold different totals.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EPS = 1e-6
_C = np.sqrt(2.0 / np.pi)


class BudgetViolation(RuntimeError):
    pass


class PolicyContractError(RuntimeError):
    pass


def load_weights(path):
    with np.load(path, allow_pickle=False) as z:
        w = {k: np.asarray(z[k], dtype=np.float64) for k in z.files if not k.startswith("cfg_")}
        cfg = {k[4:]: int(z[k]) for k in z.files if k.startswith("cfg_")}
    return w, cfg


def _rms(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * g


def _gelu(u):
    return 0.5 * u * (1 + np.tanh(_C * (u + 0.044715 * u * u * u)))


def _rope_cs(pos, hd, base=10000.0):
    half = hd // 2
    inv = base ** (-np.arange(half) / half)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, pos):
    """Rotate pre-RoPE vectors x[..., n, hd] to absolute positions pos[n]."""
    cos, sin = _rope_cs(pos, x.shape[-1])
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


@dataclass
class LayerView:
    """Everything the policy may look at for one (chunk, layer) call.

    Lists are indexed by head. For head h, cache entries are aligned across
    positions[h], keys[h], keys_rot[h], values[h] and the columns of attn[h].
    The last `chunk_end - chunk_start` entries of every head are the current
    chunk's own entries (positions chunk_start..chunk_end-1, in order).
    """
    layer: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    seq_len: int
    positions: list      # [H] int64 (n_h,)
    keys: list           # [H] float64 (n_h, hd)   pre-RoPE keys
    keys_rot: list       # [H] float64 (n_h, hd)   keys after RoPE at their position
    values: list         # [H] float64 (n_h, hd)
    queries: list        # [H] float64 (C, hd)     pre-RoPE queries of the chunk tokens
    attn: list           # [H] float64 (C, n_h)    softmax attention of chunk queries
    budget: int          # global entry budget (all layers, all heads)
    held: np.ndarray     # (L,) entries currently held per layer *before* this call
                         #  (layers < layer already evicted for this chunk; this
                         #   layer's count includes the new chunk)


class TinyLM:
    def __init__(self, weights_path):
        self.w, self.cfg = load_weights(weights_path)
        self.d = self.cfg["d"]
        self.H = self.cfg["heads"]
        self.L = self.cfg["layers"]
        self.hd = self.d // self.H
        self.V = self.cfg["vocab"]

    # ------------------------------------------------------------------
    def run(self, tokens, policy=None, budget=None, chunk=32, collect=None):
        """Return log-probs [T, V]. policy=None means full cache (no eviction).

        collect: optional callback(layer, chunk_start, chunk_end, head, positions, attn)
        used by analysis code to observe attention without evicting.
        """
        w, H, L, hd = self.w, self.H, self.L, self.hd
        tokens = np.asarray(tokens, dtype=np.int64)
        T = tokens.shape[0]
        scale = 1.0 / np.sqrt(hd)
        cpos = [[np.zeros(0, np.int64) for _ in range(H)] for _ in range(L)]
        ck = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        ckr = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        cv = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        out = np.zeros((T, self.V))
        if policy is not None and hasattr(policy, "begin_sequence"):
            policy.begin_sequence(T)
        for ci, s in enumerate(range(0, T, chunk)):
            e = min(T, s + chunk)
            C = e - s
            pos = np.arange(s, e)
            h = w["emb"][tokens[s:e]]
            intra = np.triu(np.full((C, C), -np.inf), 1)
            for l in range(L):
                a = _rms(h, w[f"g1_{l}"])
                q = (a @ w[f"Wq_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                k = (a @ w[f"Wk_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                v = (a @ w[f"Wv_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                qr = apply_rope(q, pos)
                kr = apply_rope(k, pos)
                o = np.zeros((C, H, hd))
                attn_all = []
                for hh in range(H):
                    K = np.concatenate([ckr[l][hh], kr[hh]], 0)
                    Vv = np.concatenate([cv[l][hh], v[hh]], 0)
                    n_old = ckr[l][hh].shape[0]
                    sc = (qr[hh] @ K.T) * scale
                    sc[:, n_old:] += intra
                    sc -= sc.max(-1, keepdims=True)
                    P = np.exp(sc)
                    P /= P.sum(-1, keepdims=True)
                    o[:, hh] = P @ Vv
                    attn_all.append(P)
                    cpos[l][hh] = np.concatenate([cpos[l][hh], pos])
                    ck[l][hh] = np.concatenate([ck[l][hh], k[hh]], 0)
                    ckr[l][hh] = K
                    cv[l][hh] = Vv
                    if collect is not None:
                        collect(l, s, e, hh, cpos[l][hh], P)
                h = h + o.reshape(C, self.d) @ w[f"Wo_{l}"]
                m = _rms(h, w[f"g2_{l}"])
                h = h + _gelu(m @ w[f"W1_{l}"]) @ w[f"W2_{l}"]
                if policy is not None:
                    held = np.array([sum(len(cpos[ll][hh]) for hh in range(H)) for ll in range(L)], dtype=np.int64)
                    view = LayerView(
                        layer=l, chunk_index=ci, chunk_start=s, chunk_end=e, seq_len=T,
                        positions=[cpos[l][hh].copy() for hh in range(H)],
                        keys=[ck[l][hh].copy() for hh in range(H)],
                        keys_rot=[ckr[l][hh].copy() for hh in range(H)],
                        values=[cv[l][hh].copy() for hh in range(H)],
                        queries=[q[hh].copy() for hh in range(H)],
                        attn=[attn_all[hh].copy() for hh in range(H)],
                        budget=int(budget), held=held,
                    )
                    keep = policy.select(view)
                    keep = validate_keep(keep, [len(cpos[l][hh]) for hh in range(H)])
                    for hh in range(H):
                        idx = keep[hh]
                        cpos[l][hh] = cpos[l][hh][idx]
                        ck[l][hh] = ck[l][hh][idx]
                        ckr[l][hh] = ckr[l][hh][idx]
                        cv[l][hh] = cv[l][hh][idx]
            if policy is not None:
                total = sum(len(cpos[ll][hh]) for ll in range(L) for hh in range(H))
                if total > budget:
                    raise BudgetViolation(f"chunk {ci}: {total} cache entries > budget {budget}")
            f = _rms(h, w["gf"])
            logits = f @ w["Wu"]
            logits -= logits.max(-1, keepdims=True)
            out[s:e] = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        return out


def validate_keep(keep, sizes):
    if not isinstance(keep, (list, tuple)) or len(keep) != len(sizes):
        raise PolicyContractError("select() must return one index array per head")
    res = []
    for hh, (idx, n) in enumerate(zip(keep, sizes)):
        idx = np.asarray(idx)
        if idx.dtype == bool:
            if idx.shape != (n,):
                raise PolicyContractError(f"head {hh}: boolean mask has wrong shape")
            idx = np.flatnonzero(idx)
        if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
            raise PolicyContractError(f"head {hh}: indices must be a 1-D integer array")
        idx = idx.astype(np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise PolicyContractError(f"head {hh}: index out of range")
        idx = np.unique(idx)
        res.append(idx)
    return res


def kl_full_vs(full_logp, test_logp):
    """Per-position KL(full || test) in nats."""
    pf = np.exp(full_logp)
    return np.sum(pf * (full_logp - test_logp), axis=-1)

# ===== END FILE: tmp/environment/app/kvmodel.py =====

# ===== BEGIN FILE: tmp/environment/app/policy.py =====
"""Starter KV-cache eviction policy: attention sinks + recent window.

Replace with your own policy. The verifier imports this file, instantiates
EvictionPolicy(config) once, calls begin_sequence(seq_len) before every
sequence, and select(view) after every (chunk, layer) step.
"""
from __future__ import annotations

import numpy as np


class EvictionPolicy:
    def __init__(self, config: dict):
        # config keys: d, heads, layers, head_dim, vocab, chunk, seq_len, budget, app_dir
        self.L = int(config["layers"])
        self.H = int(config["heads"])
        self.budget = int(config["budget"])
        self.n_sink = 4

    def begin_sequence(self, seq_len: int) -> None:
        pass

    def select(self, view):
        cap = self.budget // (self.L * self.H)
        keep = []
        for pos in view.positions:
            score = pos.astype(np.float64) + 1e9 * (pos < self.n_sink)
            k = min(cap, len(pos))
            keep.append(np.sort(np.argsort(-score, kind="stable")[:k]))
        return keep

# ===== END FILE: tmp/environment/app/policy.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Task

`/app/model.npz` is a small pretrained decoder-only language model (4 layers, 4 heads per layer, head dim 32, RoPE). `/app/kvmodel.py` is its numpy inference code. `/app/data/train_tokens.npy` holds 800 token sequences (512 tokens each) from the private corpus the model was trained on.

Write a KV-cache eviction policy in `/app/policy.py` so that the model behaves as closely as possible to its full-cache self under a small cache budget.

## How the policy is run

`TinyLM.run(tokens, policy, budget, chunk)` in `kvmodel.py` is exactly the loop the verifier uses:

- sequences are processed in chunks of **32** tokens;
- for each chunk and each layer (0..3), the chunk's queries attend to the entries still in that layer's cache plus the chunk itself (causal), the chunk's keys/values are appended, and then `policy.select(view)` returns, for every head, the indices of the entries to **keep**;
- after the last layer of every chunk, the total number of entries over all layers and heads must be **at most 96** (the full cache reaches 8192). Heads and layers may hold different numbers of entries.

The policy only selects entries; it cannot modify, merge or add keys/values. `view` is a `kvmodel.LayerView`, and its docstring lists every field. The policy never receives token ids.

Required interface (see the starter `/app/policy.py`):

```python
class EvictionPolicy:
    def __init__(self, config: dict): ...        # d, heads, layers, head_dim, vocab, chunk, seq_len, budget, app_dir
    def begin_sequence(self, seq_len: int): ...  # optional
    def select(self, view) -> list[np.ndarray]:  # one int index array (or bool mask) per head
```

## Scoring

The verifier runs hidden sequences twice, once with the full cache and once with your policy. On every position after the first chunk it compares the next-token distributions using KL(full || yours), plus top-1 agreement on a subset of positions.

Positions are grouped by the role their token plays in the sequence, and the groups are weighted separately. Each group's KL is scored on a log scale, from score 0 (roughly where generic attention-statistics policies such as heavy-hitter or observation-window eviction land) to score 1 (a strong reference policy). Part of the hidden set is drawn from a workload whose statistics differ from the training corpus.

The reward is continuous in [0, 1]. `python /app/evaluate_local.py` gives a quick unweighted local estimate.

## Rules

- The verifier runs your policy in a separate, unprivileged process with no network. `select` sees only what is in `view`.
- Your policy may read any files you place under `/app` (for example, tables or parameters you precompute), but must not write files at evaluation time.
- The total policy compute for the hidden evaluation (about 7,700 `select` calls) must stay under 20 minutes on 4 CPU cores.
- Exceeding the budget, returning invalid indices, crashing or timing out gives reward 0.

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" if (ROOT / "app").exists() else ROOT
import sys
sys.path.insert(0, str(APP))
from kvmodel import TinyLM  # noqa: E402

CHUNK, BUDGET = 32, 96


def _policy():
    spec = importlib.util.spec_from_file_location("policy", str(APP / "policy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EvictionPolicy


def _cfg(model, T):
    return {"d": model.d, "heads": model.H, "layers": model.L, "head_dim": model.hd, "vocab": model.V,
            "chunk": CHUNK, "seq_len": T, "budget": BUDGET, "app_dir": str(APP)}


def test_policy_runs_within_budget():
    model = TinyLM(APP / "model.npz")
    toks = np.load(APP / "data" / "train_tokens.npy")[:2]
    pol = _policy()(_cfg(model, toks.shape[1]))
    for t in toks:
        lp = model.run(t, policy=pol, budget=BUDGET, chunk=CHUNK)
        assert lp.shape == (toks.shape[1], model.V)
        assert np.all(np.isfinite(lp))


def test_keep_all_matches_full_cache():
    model = TinyLM(APP / "model.npz")
    t = np.load(APP / "data" / "train_tokens.npy")[0]

    class KeepAll:
        def select(self, view):
            return [np.arange(len(p)) for p in view.positions]

    a = model.run(t)
    b = model.run(t, policy=KeepAll(), budget=10 ** 9, chunk=CHUNK)
    assert np.allclose(a, b, atol=1e-10)

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/README.md =====
# kv_cache_eviction: reference and calibration notes (verifier side)

**Task one-liner:** Under a 96-entry global KV budget (about 1.2% of the full 8192-entry cache), write a chunked KV-cache eviction policy for a private pretrained 4-layer decoder. The policy is scored by KL(full || compressed) per position-role group on hidden in-distribution and shifted workloads. Generic attention-statistics eviction (H2O, SnapKV, Ada-KV, PyramidKV) scores 0. Reaching the reference needs model-specific reverse engineering plus learned, budget-aware prioritization.

## What makes it hard (all discoverable from /app only)

1. **Heterogeneous heads.** Layer-1 heads 0/1/2 are associative-retrieval heads that attend to fact values. Layer-1 head 3 tracks the latest topic marker. Several layer-2/3 heads sink onto DEF markers, and the rest average diffusely over filler. Every head type needs a different kind of entry.
2. **Future importance is not current attention.** Fact values receive almost no attention until they are asked for, so heavy-hitter and observation-window scores evict exactly the entries that matter.
3. **Tight budget.** 96 entries cannot hold every fact in every retrieval head and still serve the diffuse heads. The policy must predict which facts will be asked later. Pinned facts get asked late, unpinned facts are asked soon or never, some keys are asked more often, and facts already asked may be asked again. It then has to trade fact retention against filler and other context.
4. **Shift.** The hidden shifted shard has more facts, more pins, longer ask gaps and longer topic segments.
5. **No token ids.** The policy never receives token ids, but layer-0 pre-RoPE keys are a deterministic function of the token, so tokens can be decoded exactly from `view.keys` using the shipped weights.

## Reference (`policy.py` + `ref_assets.npz`)

The reference uses only the shipped weights and training corpus:

- exact token decoding and an incremental grammar parse;
- a gradient-boosted predictor of future max attention for every (layer, head, entry), with trees exported to numpy;
- head roles detected from training attention;
- a gradient-boosted P(fact asked later) model;
- a hybrid in-layer top-k, with layer shares, head weights and fact weight/bias tuned by coordinate search on held-out training sequences.

The tuning objective is the mean over recovered position groups of log mean KL. Runtime is about 1.2 s per sequence.

Assets were produced on 2 CPU cores, about 1.5 h in total:

```
python build_reference.py --app /app --out base512.npz --budget 512 --n-train 80 --n-tune 8           # fit t_* trees
python build_reference.py --app /app --out v1_160.npz --budget 160 --n-tune 8 --reuse base512.npz      # shares @160
python build_reference.py --app /app --out v3_160.npz --budget 160 --n-tune 8 --reuse v1_160.npz --warm
python build_reference.py --app /app --out v3_96.npz  --budget 96  --n-tune 8 --reuse v3_160.npz --warm
python build_hybrid.py    --app /app --base v3_96.npz --budget 96 --out ref_assets.npz                 # final
```

`build_planned.py` holds helpers used by `build_hybrid.py`: head-role detection, key ask rates and fact rows.

## Calibration ladder (hidden shards, 60 sequences each, budget 96)

| policy | reward | in_dist answer_kl / top1 / filler_kl / other_kl | shifted answer_kl / top1 / filler_kl / other_kl |
|---|---:|---|---|
| starter: sinks + recent window | 0.000 | 7.41 / 0.12 / 2.47 / 1.39 | 7.95 / 0.06 / 2.19 / 1.26 |
| H2O | 0.000 | 7.18 / 0.13 / 0.77 / 0.66 | 7.83 / 0.07 / 0.59 / 0.67 |
| Pyramid + Ada-SnapKV (EMA) | 0.000 | about 7.3 / 0.16 / 0.74 / 0.71 | about 8.2 / 0.08 / 1.02 / 0.86 |
| strong-agent probe A: token decode, fact heads keep all facts, topic head, H2O elsewhere | 0.667 | 0.24 / 0.97 / 0.046 / 0.45 | 0.50 / 0.93 / 0.073 / 0.50 |
| strong-agent probe B: probe A + pinned priority, 60% fact budget | 0.807 | - | - |
| reference | 1.000 | 0.085 / 0.989 / 0.032 / 0.341 | 0.324 / 0.957 / 0.055 / 0.440 |
| cheating future-attention oracle (budget 160, not a policy) | - | 0.013 / 0.996 / 0.030 / 0.128 | 0.010 / 0.994 / 0.048 / 0.240 |

Reward = 0.4 × in_dist + 0.6 × shifted. Each shard score is
0.45·answer_kl + 0.20·answer_top1 + 0.20·filler_kl + 0.15·other_kl.
Each KL component is interpolated in log space between its anchors; top-1 is interpolated linearly. The anchors and how they were chosen are recorded in `tests/hidden_eval.py`.

## Verifier notes

- The model weights used for scoring are `tests/model.npz`, the verifier's own copy. `/app/model.npz` is never trusted.
- The policy runs as `nobody` in `tests/policy_server.py` and receives one framed `.npz` (`allow_pickle=False`) per (chunk, layer) step. It never sees token ids, future positions or verifier objects. `test_security.py` checks this, and `/tests` is locked down with `chmod 700` before scoring.
- Hard zero on: budget violation, invalid indices, crash, or a total policy time above 20 min.
- Runtime: starter about 60 s and reference about 150 s for the whole `test.sh`, measured on 2 cores.
- Design change during development: the quote/re-quote copy mechanism did not train at this model size, so its positions are folded into `other` and quotes act only as distractor spans.

# ===== END FILE: tmp/solution/README.md =====

# ===== BEGIN FILE: tmp/solution/build_hybrid.py =====
"""Builder for the final (hybrid) reference assets. Agent-visible data only.

    python build_hybrid.py --app /app --base <assets with t_* trees and tuned shares> --budget 96 --out ref_assets.npz

Combines the general learned importance model (t_* trees, from build_reference.py)
with head roles + fact-query model (from build_planned.py helpers): inside every
layer all heads compete in one top-k; in fact heads the fact-value entries are
scored by fact_w * P(asked later) + fact_b; the topic heads always keep the
latest topic marker. All allocation knobs are tuned by coordinate search on
held-out training sequences with the group-balanced log-KL objective.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import EvictionPolicy  # noqa: E402
from build_reference import export_hgb, position_groups  # noqa: E402
from build_planned import head_roles, key_rates, fact_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="/app")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--n-tune", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ns = ap.parse_args()
    app = Path(ns.app)
    sys.path.insert(0, str(app))
    from kvmodel import TinyLM, kl_full_vs
    model = TinyLM(app / "model.npz")
    toks = np.load(app / "data" / "train_tokens.npy").astype(np.int64)
    chunk, budget, L, H = 32, ns.budget, model.L, model.H
    t0 = time.time()
    with np.load(ns.base, allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    assets = {k: v for k, v in base.items() if k.startswith("t_")}
    fh, th, _, _ = head_roles(model, toks)
    kr = key_rates(toks[:600])
    X, y = fact_rows(toks[:400], kr, chunk)
    q = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                                      min_samples_leaf=50, random_state=0).fit(X, y)
    assets.update({"q_" + k[2:]: v for k, v in export_hgb(q).items()})
    assets.update({"fact_heads": fh, "topic_heads": th, "key_rate": kr})
    print("heads", fh.tolist(), th.tolist(), f"{time.time() - t0:.0f}s", flush=True)

    tune = toks[600:600 + ns.n_tune]
    fulls = [model.run(t) for t in tune]
    groups = [position_groups(t)[chunk:-1] for t in tune]
    cfg = {"d": model.d, "heads": H, "layers": L, "head_dim": model.hd, "vocab": model.V,
           "chunk": chunk, "seq_len": toks.shape[1], "budget": budget, "app_dir": str(app)}

    def objective(p):
        a = dict(assets)
        a.update({"layer_frac": p["layer_frac"], "head_w": p["head_w"], "min_recent": np.asarray(p["min_recent"]),
                  "fact_w": np.asarray(p["fact_w"]), "fact_b": np.asarray(p["fact_b"])})
        pol = EvictionPolicy(cfg, assets=a)
        kls = [kl_full_vs(f, model.run(t, policy=pol, budget=budget, chunk=chunk))[chunk:-1]
               for t, f in zip(tune, fulls)]
        kl, g = np.concatenate(kls), np.concatenate(groups)
        return float(np.mean([np.log(kl[g == k].mean() + 1e-6) for k in (1, 3, 0)]))

    p = {"layer_frac": np.asarray(base.get("layer_frac", np.full(L, 1 / L)), float),
         "head_w": np.asarray(base.get("head_w", np.ones((L, H))), float),
         "min_recent": int(base.get("min_recent", 1)), "fact_w": 1.0, "fact_b": 0.0}
    best = objective(p)
    print("start", round(best, 4), flush=True)

    def try_(q, tag):
        nonlocal best, p
        v = objective(q)
        if v < best:
            best, p = v, q
            print(tag, round(v, 4), f"{time.time() - t0:.0f}s", flush=True)
            return True
        return False

    for it in range(ns.passes):
        for m in (0.25, 0.5, 2.0, 4.0):
            try_({**p, "fact_w": p["fact_w"] * m}, f"fact_w {p['fact_w'] * m}")
        for d in (-0.3, -0.1, 0.1, 0.3):
            try_({**p, "fact_b": p["fact_b"] + d}, f"fact_b {p['fact_b'] + d}")
        for l in range(L):
            for d in (-0.12, -0.05, 0.05, 0.12):
                fr = p["layer_frac"].copy()
                fr[l] = max(0.02, fr[l] + d)
                try_({**p, "layer_frac": fr / fr.sum()}, f"layer_frac {np.round(fr / fr.sum(), 3)}")
        for l in range(L):
            for h in range(H):
                for m in (0.5, 2.0):
                    hw = p["head_w"].copy()
                    hw[l, h] *= m
                    try_({**p, "head_w": hw}, f"head_w {l} {h} {hw[l, h]}")
        for mr in (0, 1, 2, 4):
            if mr != p["min_recent"]:
                try_({**p, "min_recent": mr}, f"min_recent {mr}")
    assets.update({"layer_frac": p["layer_frac"], "head_w": p["head_w"], "min_recent": np.asarray(p["min_recent"]),
                   "fact_w": np.asarray(p["fact_w"]), "fact_b": np.asarray(p["fact_b"])})
    np.savez_compressed(ns.out, **assets)
    print("saved", ns.out, {k: np.round(v, 3).tolist() if hasattr(v, "tolist") else v for k, v in p.items()},
          "obj", round(best, 4), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/solution/build_hybrid.py =====

# ===== BEGIN FILE: tmp/solution/build_planned.py =====
"""Builder for the planned-mode reference assets (agent-visible data only).

    python build_planned.py --app /app --base ref_assets_trees.npz --budget 96 --out ref_assets.npz

1. Head roles from full-cache attention on training sequences:
   fact heads  = heads that put >0.3 attention on the fact value at "ASK key" positions;
   topic heads = heads that put >0.5 attention on the latest topic marker at filler positions.
2. Per-key ask rate estimated on the training corpus (smoothed).
3. Fact-query model: P(fact is asked at or after chunk end e | fact-level features),
   gradient boosted, exported to numpy trees (prefix q_).
4. Budget plan tuned by coordinate search on held-out training sequences with the
   group-balanced objective from build_reference.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import (SeqState, EvictionPolicy, fact_features, ROLE_FACT_VAL, ROLE_ASK_KEY,  # noqa: E402
                    ROLE_FILLER, ROLE_TOPIC)
from build_reference import export_hgb, full_attention, position_groups  # noqa: E402


def head_roles(model, toks, n=30):
    L, H = model.L, model.H
    fact = np.zeros((L, H)); nf = 0
    topic = np.zeros((L, H)); nt = 0
    for t in toks[:n]:
        st = SeqState(len(t)); st.push(t)
        _, A = full_attention(model, t)
        for q in np.flatnonzero(st.role == ROLE_ASK_KEY):
            k = int(t[q])
            fp = [v for kk, v in st.fact_pos.items() if kk == k]
            if fp and 0 <= fp[0][2] < q:
                fact += A[:, :, q, fp[0][2]]; nf += 1
        tp = np.flatnonzero(st.role == ROLE_TOPIC)
        for q in np.flatnonzero(st.role == ROLE_FILLER):
            prev = tp[tp < q]
            if prev.size and q - prev.max() > 8:
                topic += A[:, :, q, prev.max()]; nt += 1
    fact /= max(nf, 1); topic /= max(nt, 1)
    fh = np.argwhere(fact > 0.3)
    th = np.argwhere((topic > 0.5) & (fact <= 0.3))
    return fh, th, fact, topic


def key_rates(toks):
    asked = np.zeros(64); defined = np.zeros(64)
    for t in toks:
        st = SeqState(len(t)); st.push(t)
        for k in st.fact_pos:
            defined[k - 80] += 1
            asked[k - 80] += bool(st.asks.get(k))
    return (asked + 2.0) / (defined + 4.0)


def fact_rows(toks, key_rate, chunk):
    X, y = [], []
    for t in toks:
        st = SeqState(len(t)); st.push(t)
        for e in range(chunk, len(t), chunk):
            vpos, F = fact_features(st, e, key_rate)
            if not vpos.size:
                continue
            lab = [any(x >= e for x in st.asks.get(int(st.key_of[p]), [])) for p in vpos]
            X.append(F); y.append(np.asarray(lab, float))
    return np.concatenate(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="/app")
    ap.add_argument("--base", required=True, help="assets file with the general importance trees (t_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--n-tune", type=int, default=8)
    ap.add_argument("--passes", type=int, default=2)
    ns = ap.parse_args()
    app = Path(ns.app)
    sys.path.insert(0, str(app))
    from kvmodel import TinyLM, kl_full_vs
    model = TinyLM(app / "model.npz")
    toks = np.load(app / "data" / "train_tokens.npy").astype(np.int64)
    chunk, budget = 32, ns.budget
    L, H = model.L, model.H
    t0 = time.time()
    with np.load(ns.base, allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    assets = {k: v for k, v in base.items() if k.startswith("t_")}

    fh, th, fact_att, topic_att = head_roles(model, toks)
    print("fact heads", fh.tolist(), "topic heads", th.tolist(), f"{time.time() - t0:.0f}s", flush=True)
    kr = key_rates(toks[:600])
    X, y = fact_rows(toks[:400], kr, chunk)
    q = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                                      min_samples_leaf=50, random_state=0).fit(X, y)
    qa = export_hgb(q)
    print(f"fact model rows={len(y)} pos_rate={y.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    assets.update({"q_" + k[2:]: v for k, v in qa.items()})
    assets.update({"fact_heads": fh, "topic_heads": th, "key_rate": kr})

    tune = toks[600:600 + ns.n_tune]
    fulls = [model.run(t) for t in tune]
    groups = [position_groups(t)[chunk:-1] for t in tune]
    cfg = {"d": model.d, "heads": H, "layers": L, "head_dim": model.hd, "vocab": model.V,
           "chunk": chunk, "seq_len": toks.shape[1], "budget": budget, "app_dir": str(app)}

    def objective(p):
        a = dict(assets)
        a.update({"fact_cap": p["fact_cap"], "layer_frac": p["layer_frac"], "head_w": p["head_w"],
                  "min_recent": np.asarray(p["min_recent"])})
        pol = EvictionPolicy(cfg, assets=a)
        kls = [kl_full_vs(f, model.run(t, policy=pol, budget=budget, chunk=chunk))[chunk:-1]
               for t, f in zip(tune, fulls)]
        kl, g = np.concatenate(kls), np.concatenate(groups)
        return float(np.mean([np.log(kl[g == k].mean() + 1e-6) for k in (1, 3, 0)]))

    special = {tuple(x) for x in fh.tolist()} | {tuple(x) for x in th.tolist()}
    p = {"fact_cap": np.full(len(fh), 0.2), "layer_frac": np.full(L, 1.0 / L),
         "head_w": np.ones((L, H)), "min_recent": 1}
    best = objective(p)
    print("start", round(best, 4), flush=True)

    def try_(q, tag):
        nonlocal best, p
        v = objective(q)
        if v < best:
            best, p = v, q
            print(tag, round(v, 4), f"{time.time() - t0:.0f}s", flush=True)

    for it in range(ns.passes):
        for i in range(len(fh)):
            for d in (-0.08, -0.04, 0.04, 0.08):
                q = {**p, "fact_cap": p["fact_cap"].copy()}
                q["fact_cap"][i] = max(0.01, q["fact_cap"][i] + d)
                if (q["fact_cap"].sum() * budget + 2 * len(th)) < budget - 8:
                    try_(q, f"fact_cap {np.round(q['fact_cap'], 3)}")
        for l in range(L):
            for d in (-0.12, -0.05, 0.05, 0.12):
                q = {**p, "layer_frac": p["layer_frac"].copy()}
                q["layer_frac"][l] = max(0.02, q["layer_frac"][l] + d)
                q["layer_frac"] /= q["layer_frac"].sum()
                try_(q, f"layer_frac {np.round(q['layer_frac'], 3)}")
        for l in range(L):
            for h in range(H):
                if (l, h) in special:
                    continue
                for m in (0.5, 2.0):
                    q = {**p, "head_w": p["head_w"].copy()}
                    q["head_w"][l, h] *= m
                    try_(q, f"head_w {l} {h} {q['head_w'][l, h]}")
        for mr in (0, 1, 2, 4):
            if mr != p["min_recent"]:
                try_({**p, "min_recent": mr}, f"min_recent {mr}")
    assets.update({"fact_cap": p["fact_cap"], "layer_frac": p["layer_frac"], "head_w": p["head_w"],
                   "min_recent": np.asarray(p["min_recent"])})
    np.savez_compressed(ns.out, **assets)
    print("saved", ns.out, {k: np.round(v, 3).tolist() if hasattr(v, "tolist") else v for k, v in p.items()},
          "obj", round(best, 4), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/solution/build_planned.py =====

# ===== BEGIN FILE: tmp/solution/build_reference.py =====
"""Offline builder for the reference policy assets (uses agent-visible data only).

    python build_reference.py --app /app --out /solution/ref_assets.npz

Steps
  1. run the full-cache model on training sequences and record attention;
  2. at every chunk boundary, featurize each (layer, head, past position) with the
     same features the online policy computes, target = max attention future
     queries pay to that entry;
  3. fit a gradient-boosted regressor, export its trees to plain numpy arrays;
  4. tune per-layer budget shares by direct search on training sequences
     (objective: unweighted mean KL over all positions).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import SeqState, EvictionPolicy, N_ATT_FEATS  # noqa: E402


def full_attention(model, tokens):
    L, H, T = model.L, model.H, len(tokens)
    A = np.zeros((L, H, T, T))

    def cb(l, s, e, hh, pos, P):
        A[l, hh, s:e][:, pos] = P

    lp = model.run(tokens, collect=cb, chunk=T)
    return lp, A


def make_rows(model, tokens, chunk, rng, filler_keep=0.25):
    T = len(tokens)
    L, H = model.L, model.H
    _, A = full_attention(model, tokens)
    st = SeqState(T)
    X, y = [], []
    for e in range(chunk, T, chunk):
        st.push(tokens[e - chunk:e])
        pos_all = np.arange(e)
        role = st.role[:e]
        keep = (role != 1) | (pos_all >= e - 48) | (rng.random(e) < filler_keep)
        pos = pos_all[keep]
        F = st.features(pos, e)
        for l in range(L):
            for h in range(H):
                Ach = A[l, h, e - chunk:e][:, pos]          # chunk queries -> candidates
                acc = A[l, h, :e][:, pos].sum(0)
                att = np.stack([Ach.mean(0), Ach.max(0), Ach[-4:].mean(0),
                                acc / np.maximum(1, e - pos)], 1)
                lh = np.full((len(pos), 2), [l, l * H + h], dtype=float)
                X.append(np.concatenate([F, att, lh], 1))
                y.append(A[l, h, e:, :][:, pos].max(0))
    return np.concatenate(X), np.concatenate(y)


def export_hgb(model):
    feats, thr, left, right, leaf, val, mgl, offs = [], [], [], [], [], [], [], [0]
    for pred in model._predictors:
        nodes = pred[0].nodes
        feats.append(nodes["feature_idx"].astype(np.int64))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int64))
        right.append(nodes["right"].astype(np.int64))
        leaf.append(nodes["is_leaf"].astype(np.int64))
        val.append(nodes["value"].astype(np.float64))
        mgl.append(nodes["missing_go_to_left"].astype(np.int64))
        offs.append(offs[-1] + len(nodes))
    return {
        "t_feat": np.concatenate(feats), "t_thr": np.concatenate(thr),
        "t_left": np.concatenate(left), "t_right": np.concatenate(right),
        "t_leaf": np.concatenate(leaf), "t_val": np.concatenate(val),
        "t_mgl": np.concatenate(mgl), "t_off": np.asarray(offs, dtype=np.int64),
        "t_base": np.asarray(float(np.ravel(model._baseline_prediction)[0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="/app")
    ap.add_argument("--out", default=str(HERE / "ref_assets.npz"))
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-tune", type=int, default=12)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--reuse", default=None, help="reuse trees from an existing assets file")
    ap.add_argument("--warm", action="store_true", help="also start tuning from the reused shares/weights")
    ns = ap.parse_args()
    app = Path(ns.app)
    sys.path.insert(0, str(app))
    from kvmodel import TinyLM, kl_full_vs

    model = TinyLM(app / "model.npz")
    toks = np.load(app / "data" / "train_tokens.npy").astype(np.int64)
    chunk, budget = 32, ns.budget
    rng = np.random.default_rng(0)
    t0 = time.time()
    if ns.reuse:
        with np.load(ns.reuse, allow_pickle=False) as z:
            keep = ("t_", "layer_frac", "head_w", "min_recent") if ns.warm else ("t_",)
            assets = {k: z[k] for k in z.files if k.startswith(keep)}
    else:
        assets = fit_trees(model, toks, chunk, rng, ns.n_train, t0)
    tune_layers(model, toks, chunk, budget, assets, ns, app, kl_full_vs, t0)


def fit_trees(model, toks, chunk, rng, n_train, t0):
    Xs, ys = [], []
    for i in range(n_train):
        X, y = make_rows(model, toks[i], chunk, rng)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    print(f"rows={len(y)}  featurize {time.time() - t0:.0f}s", flush=True)
    gbm = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=63,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    gbm.fit(X, y)
    assets = export_hgb(gbm)
    # sanity: exported trees reproduce sklearn predictions
    from policy import predict_trees
    sub = X[:5000]
    assert np.allclose(predict_trees(assets, sub), gbm.predict(sub), atol=1e-8)
    print(f"gbm fit done {time.time() - t0:.0f}s", flush=True)
    return assets


def position_groups(tokens):
    """Group predicting positions by recovered role: 1 answer, 3 filler, 0 other."""
    st = SeqState(len(tokens))
    st.push(tokens)
    r = st.role
    g = np.zeros(len(tokens), dtype=np.int64)
    g[:-1][r[:-1] == 6] = 1                                  # key in "ASK key" -> predicts the value
    g[:-1][(r[:-1] == 1) & (r[1:] == 1)] = 3                 # filler -> filler
    return g


def tune_layers(model, toks, chunk, budget, assets, ns, app, kl_full_vs, t0):
    tune = toks[ns.n_train:ns.n_train + ns.n_tune]
    fulls = [model.run(t) for t in tune]
    groups = [position_groups(t)[chunk:-1] for t in tune]
    cfg = {"d": model.d, "heads": model.H, "layers": model.L, "head_dim": model.hd,
           "vocab": model.V, "chunk": chunk, "seq_len": toks.shape[1], "budget": budget,
           "app_dir": str(app)}
    L, H = model.L, model.H

    def objective(frac, hw, min_recent):
        a = dict(assets)
        a["layer_frac"] = np.asarray(frac, dtype=float)
        a["head_w"] = np.asarray(hw, dtype=float)
        a["min_recent"] = np.asarray(min_recent)
        pol = EvictionPolicy(cfg, assets=a)
        kls = []
        for t, f in zip(tune, fulls):
            kls.append(kl_full_vs(f, model.run(t, policy=pol, budget=budget, chunk=chunk))[chunk:-1])
        kl, g = np.concatenate(kls), np.concatenate(groups)
        return float(np.mean([np.log(kl[g == k].mean() + 1e-6) for k in (1, 3, 0)]))

    frac = np.full(L, 1.0 / L)
    hw = np.ones((L, H))
    mr = 1
    if "layer_frac" in assets and getattr(ns, "warm", False):
        frac = np.asarray(assets["layer_frac"], dtype=float)
        hw = np.asarray(assets.get("head_w", hw), dtype=float)
        mr = int(assets.get("min_recent", mr))
    best_val = objective(frac, hw, mr)
    print("start", best_val, flush=True)
    for it in range(2):
        for l in range(L):
            for delta in (-0.12, -0.05, 0.05, 0.12):
                f = frac.copy()
                f[l] += delta
                if f[l] < 0.02:
                    continue
                f = f / f.sum()
                v = objective(f, hw, mr)
                if v < best_val:
                    best_val, frac = v, f
                    print("frac", np.round(f, 3), round(v, 4), f"{time.time() - t0:.0f}s", flush=True)
        for l in range(L):
            for h in range(H):
                for mult in (0.5, 2.0):
                    w = hw.copy()
                    w[l, h] *= mult
                    v = objective(frac, w, mr)
                    if v < best_val:
                        best_val, hw = v, w
                        print("head_w", l, h, w[l, h], round(v, 4), f"{time.time() - t0:.0f}s", flush=True)
        for m in (0, 1, 2, 4):
            v = objective(frac, hw, m)
            if v < best_val:
                best_val, mr = v, m
    assets = dict(assets)
    assets["layer_frac"] = frac
    assets["head_w"] = hw
    assets["min_recent"] = np.asarray(mr)
    np.savez_compressed(ns.out, **assets)
    print("saved", ns.out, "layer_frac", np.round(frac, 3), "min_recent", mr, "obj", best_val,
          "\nhead_w", np.round(hw, 2).tolist(), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/solution/build_reference.py =====

# ===== BEGIN FILE: tmp/solution/policy.py =====
"""Reference eviction policy for kv_cache_eviction.

Everything here is derived from agent-visible material only: the model weights
(/app/model.npz) and the training corpus (/app/data/train_tokens.npy).

1. Token recovery. Layer-0 pre-RoPE keys are a deterministic function of the
   token id (RMSNorm(emb[tok]) @ Wk_0), so the policy decodes each new chunk's
   tokens exactly from view.keys at layer 0 and keeps its own token history.
2. Role features. From the recovered history it derives the role of every
   cached position in the corpus grammar (fact value, pinned fact, key asked
   already, quote word / quote length / already re-quoted, topic marker and
   whether it was superseded, ...), plus recency and attention statistics.
3. Learned future importance. A gradient-boosted model trained offline on the
   training corpus predicts, for (layer, head, position), the maximum
   attention that *future* queries will pay to the entry under the full cache.
4. Allocation. Per-layer budget shares and per-head importance multipliers
   are tuned offline (objective: mean over position groups -- fact answers,
   filler words, everything else, as recovered by the same parse -- of log
   mean KL on training sequences); inside a layer the entries of all heads
   compete in a single top-k on weighted predicted importance.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

EPS = 1e-6

# token classes (recovered by inspecting the training corpus)
C_BOS, C_DEF, C_PIN, C_ASK, C_QO, C_QC, C_RE, C_TOPIC, C_WORD, C_KEY, C_VAL, C_OTHER = range(12)

ROLE_OTHER, ROLE_FILLER, ROLE_FACT_DEF, ROLE_FACT_KEY, ROLE_FACT_VAL, ROLE_ASK, ROLE_ASK_KEY, \
    ROLE_ASK_VAL, ROLE_QUOTE, ROLE_REQUOTE, ROLE_TOPIC, ROLE_BOS, ROLE_PIN, ROLE_QMARK = range(14)


def tok_class(t):
    t = np.asarray(t)
    c = np.full(t.shape, C_OTHER, dtype=np.int64)
    c[t == 0] = C_BOS
    c[t == 1] = C_DEF
    c[t == 2] = C_PIN
    c[t == 3] = C_ASK
    c[t == 4] = C_QO
    c[t == 5] = C_QC
    c[t == 6] = C_RE
    c[(t >= 7) & (t <= 10)] = C_TOPIC
    c[(t >= 16) & (t < 80)] = C_WORD
    c[(t >= 80) & (t < 144)] = C_KEY
    c[(t >= 144) & (t < 192)] = C_VAL
    return c


class SeqState:
    """Incremental parse of the recovered token history."""

    def __init__(self, T):
        self.T = T
        self.tok = np.full(T, -1, dtype=np.int64)
        self.n = 0
        self.role = np.zeros(T, dtype=np.int64)
        self.pinned = np.zeros(T, dtype=np.int64)
        self.key_of = np.full(T, -1, dtype=np.int64)     # key id associated with a fact position
        self.quote_id = np.full(T, -1, dtype=np.int64)
        self.quote_off = np.full(T, -1, dtype=np.int64)
        self.quotes = []                                  # [start, length, n_requoted]
        self.fact_pos = {}                                # key -> [def_pos, key_pos, val_pos]
        self.asks = {}                                    # key -> list of ask positions
        self.topics = []                                  # positions of topic markers
        self._mode = None
        self._cur_quote = None
        self._re_words = None

    def push(self, toks):
        for t in toks:
            self._push1(int(t))

    def _push1(self, t):
        p = self.n
        self.tok[p] = t
        self.n += 1
        prev = self.tok[p - 1] if p >= 1 else -1
        prev2 = self.tok[p - 2] if p >= 2 else -1
        c = int(tok_class([t])[0])
        role = ROLE_OTHER
        if c == C_BOS:
            role = ROLE_BOS
        elif c == C_PIN:
            role = ROLE_PIN
        elif c == C_DEF:
            role = ROLE_FACT_DEF
            if prev == 2:
                self.pinned[p] = 1
        elif c == C_ASK:
            role = ROLE_ASK
        elif c == C_TOPIC:
            role = ROLE_TOPIC
            self.topics.append(p)
            self._mode = None
        elif c == C_QO:
            role = ROLE_QMARK
            self._mode = "quote"
            self._cur_quote = [p + 1, 0, 0]
            self.quotes.append(self._cur_quote)
        elif c == C_RE:
            role = ROLE_QMARK
            self._mode = "requote"
            self._re_words = []
        elif c == C_QC:
            role = ROLE_QMARK
            if self._mode == "requote" and self._re_words:
                for q in self.quotes:
                    L = q[1]
                    if L == len(self._re_words) and list(self.tok[q[0]:q[0] + L]) == self._re_words:
                        q[2] += 1
                        break
            self._mode = None
        elif c == C_KEY:
            if prev == 1:
                role = ROLE_FACT_KEY
                self.key_of[p] = t
                self.pinned[p] = self.pinned[p - 1]
                self.fact_pos[t] = [p - 1, p, -1]
            elif prev == 3:
                role = ROLE_ASK_KEY
                self.key_of[p] = t
                self.asks.setdefault(t, []).append(p)
                self.key_of[p - 1] = t
        elif c == C_VAL:
            if prev2 == 1 and tok_class([prev])[0] == C_KEY:
                role = ROLE_FACT_VAL
                self.key_of[p] = prev
                self.pinned[p] = self.pinned[p - 1]
                if prev in self.fact_pos:
                    self.fact_pos[prev][2] = p
            elif prev2 == 3:
                role = ROLE_ASK_VAL
                self.key_of[p] = prev
        elif c == C_WORD:
            if self._mode == "quote":
                role = ROLE_QUOTE
                q = self._cur_quote
                self.quote_id[p] = len(self.quotes) - 1
                self.quote_off[p] = q[1]
                q[1] += 1
            elif self._mode == "requote":
                role = ROLE_REQUOTE
                self._re_words.append(t)
            else:
                role = ROLE_FILLER
        self.role[p] = role

    def features(self, pos, e):
        """Feature matrix for cached positions `pos` at eviction time e (exclusive)."""
        pos = np.asarray(pos, dtype=np.int64)
        n = pos.shape[0]
        tok = self.tok[pos]
        role = self.role[pos]
        F = np.zeros((n, 20))
        F[:, 0] = e - 1 - pos
        F[:, 1] = pos / self.T
        F[:, 2] = (self.T - e) / self.T
        F[:, 3] = role
        F[:, 4] = tok_class(tok)
        prevt = np.where(pos >= 1, self.tok[np.maximum(pos - 1, 0)], -1)
        F[:, 5] = tok_class(prevt)
        F[:, 6] = self.pinned[pos]
        key = self.key_of[pos]
        F[:, 7] = np.where(key >= 0, key - 80, -1)
        n_ask = np.zeros(n)
        last_ask = np.full(n, -1.0)
        is_fact = (key >= 0) & np.isin(role, (ROLE_FACT_KEY, ROLE_FACT_VAL, ROLE_FACT_DEF))
        for i in np.flatnonzero(is_fact):
            lst = self.asks.get(int(key[i]))
            if lst:
                a = [x for x in lst if pos[i] < x < e]
                n_ask[i] = len(a)
                if a:
                    last_ask[i] = e - a[-1]
        F[:, 8] = n_ask
        F[:, 9] = last_ask
        qid = self.quote_id[pos]
        qinfo = np.array([[q[1], q[2]] for q in self.quotes] + [[-1, -1]], dtype=float)
        qlen = qinfo[qid, 0]
        qre = qinfo[qid, 1]
        F[:, 10] = qlen
        F[:, 11] = self.quote_off[pos]
        F[:, 12] = qre
        topics = np.asarray(self.topics, dtype=np.int64)
        tp = topics[topics < e]
        latest = tp.max() if tp.size else -1
        F[:, 13] = (role == ROLE_TOPIC) & (pos != latest)
        F[:, 14] = (pos == latest)
        F[:, 15] = len([1 for k, v in self.fact_pos.items() if v[2] >= 0 and v[2] < e])
        # tokens since the last topic marker (for filler)
        F[:, 16] = e - 1 - latest if latest >= 0 else e
        return F


N_ATT_FEATS = 4


def predict_trees(a, X):
    """Evaluate exported HistGradientBoosting trees (plain numpy)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    feat, thr, left, right = a["t_feat"], a["t_thr"], a["t_left"], a["t_right"]
    leaf, val, mgl, off = a["t_leaf"], a["t_val"], a["t_mgl"], a["t_off"]
    roots = off[:-1]
    # all (row, tree) pairs advance one level per iteration
    g = np.broadcast_to(roots[None, :], (n, roots.size)).copy()     # global node ids
    rows = np.broadcast_to(np.arange(n)[:, None], g.shape)
    base = np.broadcast_to(roots[None, :], g.shape)
    while True:
        act = leaf[g] == 0
        if not act.any():
            break
        ga = g[act]
        x = X[rows[act], feat[ga]]
        go_left = np.where(np.isnan(x), mgl[ga] == 1, x <= thr[ga])
        g[act] = base[act] + np.where(go_left, left[ga], right[ga])
    return float(a["t_base"]) + val[g].sum(1)


def fact_features(st, e, key_rate):
    """Fact-level features for every fact value position < e (planned mode)."""
    vpos = np.flatnonzero(st.role[:e] == ROLE_FACT_VAL)
    F = np.zeros((vpos.size, 7))
    for i, p in enumerate(vpos):
        k = int(st.key_of[p])
        asks = [x for x in st.asks.get(k, []) if p < x < e]
        F[i] = [st.pinned[p], key_rate[k - 80] if 80 <= k < 144 else 0.0, len(asks),
                (e - asks[-1]) if asks else -1.0, e - 1 - p, st.T - e, p]
    return vpos, F


def decode_table(w):
    g = w["g1_0"]
    emb = w["emb"]
    a = emb / np.sqrt(np.mean(emb * emb, axis=-1, keepdims=True) + EPS) * g
    return a @ w["Wk_0"]                     # (V, d) pre-RoPE layer-0 keys


class EvictionPolicy:
    def __init__(self, config, assets=None):
        self.cfg = config
        self.L, self.H = int(config["layers"]), int(config["heads"])
        self.budget = int(config["budget"])
        here = Path(__file__).resolve().parent
        app = Path(config.get("app_dir", "/app"))
        cands = [app / "model.npz", here / "model.npz", Path("/app/model.npz")]
        wpath = next((c for c in cands if c.exists()), cands[0])
        with np.load(wpath, allow_pickle=False) as z:
            w = {k: np.asarray(z[k], dtype=np.float64) for k in ("emb", "g1_0", "Wk_0")}
        self.table = decode_table(w)
        if assets is None:
            path = here / "ref_assets.npz"
            if not path.exists():
                path = app / "ref_assets.npz"
            with np.load(path, allow_pickle=False) as z:
                assets = {k: z[k] for k in z.files}
        self.assets = assets
        self.layer_frac = np.asarray(assets["layer_frac"], dtype=float)
        self.min_recent = int(assets["min_recent"])
        self.planned = "fact_heads" in assets and "fact_w" not in assets
        self.hybrid = "fact_w" in assets
        if self.hybrid:
            self.fact_w = float(assets["fact_w"])
            self.fact_b = float(assets.get("fact_b", 0.0))
        if "fact_heads" in assets:
            self.fact_heads = np.asarray(assets["fact_heads"], dtype=np.int64)     # (k, 2) layer, head
            self.topic_heads = np.asarray(assets["topic_heads"], dtype=np.int64)
            self.fact_cap = np.asarray(assets.get("fact_cap", np.zeros(len(self.fact_heads))), dtype=float)
            self.key_rate = np.asarray(assets["key_rate"], dtype=float)
            self.q_assets = {"t_" + k[2:]: v for k, v in assets.items() if k.startswith("q_")}
        hw = assets.get("head_w")
        self.head_w = np.ones((self.L, self.H)) if hw is None else np.asarray(hw, dtype=float)

    def begin_sequence(self, seq_len):
        self.st = SeqState(seq_len)
        self.acc = {}
        self._fact_cache = None

    # ---- planned mode -------------------------------------------------
    def _special(self):
        sp = {}
        for i, (l, h) in enumerate(self.fact_heads):
            sp[(int(l), int(h))] = ("fact", max(1, int(round(self.fact_cap[i] * self.budget))))
        for l, h in self.topic_heads:
            sp[(int(l), int(h))] = ("topic", 2)
        return sp

    def _fact_scores(self, e):
        if self._fact_cache is not None and self._fact_cache[0] == e:
            return self._fact_cache[1]
        vpos, F = fact_features(self.st, e, self.key_rate)
        pr = predict_trees(self.q_assets, F) if vpos.size else np.zeros(0)
        sc = np.zeros(self.st.T)
        sc[vpos] = pr
        self._fact_cache = (e, sc)
        return sc

    def _select_planned(self, view):
        e = view.chunk_end
        sp = self._special()
        special_total = sum(c for _, c in sp.values())
        rest = max(0, self.budget - special_total)
        keep = [None] * self.H
        others = []
        for h, pos in enumerate(view.positions):
            role = sp.get((view.layer, h))
            if role is None:
                others.append(h)
                continue
            kind, cap = role
            if kind == "fact":
                fs = self._fact_scores(e)
                isf = self.st.role[pos] == ROLE_FACT_VAL
                sc = np.where(isf, 1.0 + fs[pos], 0.0) + 1e-6 * pos
            else:
                tp = np.flatnonzero(self.st.role[:e] == ROLE_TOPIC)
                latest = tp.max() if tp.size else -1
                sc = 2.0 * (pos == latest) + 1e-6 * pos
            k = min(cap, len(pos))
            keep[h] = np.sort(np.argpartition(-sc, k - 1)[:k]) if 0 < k < len(pos) else np.arange(len(pos))[:k]
        if others:
            other_layers = sorted({l for l in range(self.L)
                                   if any((l, h) not in sp for h in range(self.H))})
            fr = self.layer_frac[other_layers]
            fr = fr / fr.sum()
            lb = int(rest * fr[other_layers.index(view.layer)])
            feats, owners, locals_ = [], [], []
            for h in others:
                pos, A = view.positions[h], view.attn[h]
                F = self.st.features(pos, e)
                acc = self.acc.setdefault((view.layer, h), np.zeros(self.st.T))
                acc[pos] += A.sum(0)
                att = np.stack([A.mean(0), A.max(0), A[-4:].mean(0), acc[pos] / np.maximum(1, e - pos)], 1)
                lh = np.full((len(pos), 2), [view.layer, view.layer * self.H + h], dtype=float)
                feats.append(np.concatenate([F, att, lh], 1))
                owners.append(np.full(len(pos), h))
                locals_.append(np.arange(len(pos)))
            owner = np.concatenate(owners)
            local = np.concatenate(locals_)
            score = predict_trees(self.assets, np.concatenate(feats)) * self.head_w[view.layer][owner]
            pos_all = np.concatenate([view.positions[h] for h in others])
            score = score + 10.0 * (pos_all >= e - self.min_recent)
            k = min(lb, score.shape[0])
            top = np.argpartition(-score, k - 1)[:k] if 0 < k < score.shape[0] else np.arange(score.shape[0])[:k]
            sel = {h: [] for h in others}
            for j in top:
                sel[owner[j]].append(local[j])
            for h in others:
                keep[h] = np.sort(np.asarray(sel[h], dtype=np.int64))
        return keep

    def _decode(self, view):
        C = view.chunk_end - view.chunk_start
        K = np.concatenate([k[-C:] for k in view.keys], axis=1)     # (C, d)
        d2 = ((K[:, None, :] - self.table[None, :, :]) ** 2).sum(-1)
        return d2.argmin(1)

    def layer_budget(self, view):
        held_other = 0
        for l in range(self.L):
            if l < view.layer:
                held_other += view.held[l]
            elif l > view.layer:
                # later layers still hold last chunk's entries; they will be cut to their share
                held_other += min(view.held[l], int(self.layer_frac[l] * self.budget))
        return int(min(self.layer_frac[view.layer] * self.budget, self.budget - held_other))

    def select(self, view):
        if view.layer == 0:
            self.st.push(self._decode(view))
        if self.planned:
            return self._select_planned(view)
        e = view.chunk_end
        feats, owners, locals_ = [], [], []
        for h, (pos, A) in enumerate(zip(view.positions, view.attn)):
            F = self.st.features(pos, e)
            acc = self.acc.setdefault((view.layer, h), np.zeros(self.st.T))
            acc[pos] += A.sum(0)
            att = np.stack([A.mean(0), A.max(0), A[-4:].mean(0),
                            acc[pos] / np.maximum(1, e - pos)], 1)
            lh = np.full((len(pos), 2), [view.layer, view.layer * self.H + h], dtype=float)
            feats.append(np.concatenate([F, att, lh], 1))
            owners.append(np.full(len(pos), h))
            locals_.append(np.arange(len(pos)))
        X = np.concatenate(feats)
        owner = np.concatenate(owners)
        score = predict_trees(self.assets, X) * self.head_w[view.layer][owner]
        local = np.concatenate(locals_)
        if self.hybrid:
            pos_cat = np.concatenate(view.positions)
            fs = self._fact_scores(e)
            for l, h in self.fact_heads:
                if l == view.layer:
                    m = (owner == h) & (self.st.role[pos_cat] == ROLE_FACT_VAL)
                    score[m] = self.fact_w * fs[pos_cat[m]] + self.fact_b
            tp = np.flatnonzero(self.st.role[:e] == ROLE_TOPIC)
            if tp.size:
                for l, h in self.topic_heads:
                    if l == view.layer:
                        score[(owner == h) & (pos_cat == tp.max())] += 10.0
        # always keep the newest entries of every head (cheap insurance for local heads)
        pos_all = np.concatenate(view.positions)
        score = score + 10.0 * (pos_all >= e - self.min_recent)
        lb = max(0, self.layer_budget(view))
        k = min(lb, score.shape[0])
        top = np.argpartition(-score, k - 1)[:k] if 0 < k < score.shape[0] else np.arange(score.shape[0])[:k]
        keep = [[] for _ in range(self.H)]
        for j in top:
            keep[owner[j]].append(local[j])
        return [np.sort(np.asarray(x, dtype=np.int64)) for x in keep]

# ===== END FILE: tmp/solution/policy.py =====

# ===== BEGIN FILE: tmp/solution/solve.sh =====
#!/bin/bash
# Install the reference policy into /app.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/policy.py" /app/policy.py
cp "$HERE/ref_assets.npz" /app/ref_assets.npz
# To rebuild the assets from agent-visible data only (about 15-25 min on 4 cores):
#   python "$HERE/build_reference.py" --app /app --out /app/ref_assets.npz

# ===== END FILE: tmp/solution/solve.sh =====

# ===== BEGIN FILE: tmp/tests/_paths.py =====
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    p = Path("/app")
    if (p / "policy.py").exists():
        return p
    return ROOT / "environment" / "app"


def app_policy_path() -> Path:
    return app_dir() / "policy.py"


def reference_path() -> Path:
    p = Path("/solution/policy.py")
    if p.exists():
        return p
    return ROOT / "solution" / "policy.py"


def verifier_log_dir() -> Path:
    p = Path("/logs/verifier")
    if p.exists() or Path("/logs").exists():
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = ROOT / "logs" / "verifier"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ===== END FILE: tmp/tests/_paths.py =====

# ===== BEGIN FILE: tmp/tests/categorize_failure.py =====
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def categorize(text: str) -> str:
    s = text.lower()
    for code in ("budget_violation", "contract_violation", "api_missing", "timeout", "bad_response"):
        if code in s:
            return code
    if "importerror" in s or "modulenotfounderror" in s:
        return "import_error"
    if "runtime_error" in s or "traceback" in s:
        return "runtime_error"
    return "unknown"


def main(argv):
    if len(argv) != 3:
        print("usage: categorize_failure.py STDERR_TXT OUT_TXT", file=sys.stderr)
        return 2
    text = Path(argv[1]).read_text(errors="replace") if Path(argv[1]).exists() else ""
    Path(argv[2]).write_text(categorize(text) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# ===== END FILE: tmp/tests/categorize_failure.py =====

# ===== BEGIN FILE: tmp/tests/generator.py =====
"""Private synthetic token language for the kv-cache eviction task.

Verifier-side only. The agent receives pre-generated training token sequences,
never this file.

Token map
---------
0            BOS
1            DEF   (start of a fact:  [PIN] DEF key val)
2            PIN   (optional marker before DEF)
3            ASK   (fact lookup:      ASK key val)
4            QO    (quote open:       QO w1..wn QC)
5            QC    (quote close)
6            RE    (re-quote:         RE w1..wn QC  -- exact copy of an earlier quote)
7..10        TOPIC markers; filler words follow the bigram table of the latest topic
11..15       unused / reserved
16..79       64 word tokens
80..143      64 key tokens
144..191     48 value tokens
192..255     64 quote-word tokens (quotes and re-quotes only)

Slice labels are attached to the *predicting* position t (the model's output at
t predicts token t+1):
  0 = other / filler
  1 = fact answer   (t is the key position in "ASK key", next token = val)
  2 = quote copy    (t inside a re-quote, next token is the copied word or QC)
  3 = filler word   (t is a filler word and t+1 is a filler word)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

BOS, DEF, PIN, ASK, QO, QC, RE = 0, 1, 2, 3, 4, 5, 6
TOPIC0, N_TOPIC = 7, 4
WORD0, N_WORD = 16, 64
KEY0, N_KEY = 80, 64
VAL0, N_VAL = 144, 48
QWORD0, N_QWORD = 192, 64
VOCAB = 256
SEQ_LEN = 512
N_HOT_KEYS = 16

_TABLE_SEED = 0x5EED_4B56


def _markov_table():
    rng = np.random.default_rng(_TABLE_SEED)
    succ = np.zeros((N_TOPIC, N_WORD, 3), dtype=np.int64)
    for z in range(N_TOPIC):
        for a in range(N_WORD):
            succ[z, a] = rng.choice(N_WORD, size=3, replace=False)
    return succ


_SUCC = _markov_table()
_SUCC_P = np.array([0.6, 0.25, 0.15])


@dataclass(frozen=True)
class Workload:
    p_fact: float
    p_quote: float
    p_pin: float
    p_query_plain: float
    short_gap_mean: float
    long_gap_min: int
    p_hot_extra: float
    p_requote_long: float
    p_requote_short: float
    topic_len_mean: float


IN_DIST = Workload(
    p_fact=0.060, p_quote=0.012, p_pin=0.35, p_query_plain=0.40,
    short_gap_mean=36.0, long_gap_min=60, p_hot_extra=0.5,
    p_requote_long=0.75, p_requote_short=0.20, topic_len_mean=110.0,
)

SHIFTED = Workload(
    p_fact=0.090, p_quote=0.016, p_pin=0.50, p_query_plain=0.40,
    short_gap_mean=70.0, long_gap_min=90, p_hot_extra=0.5,
    p_requote_long=0.80, p_requote_short=0.20, topic_len_mean=170.0,
)


def generate_sequence(rng: np.random.Generator, wl: Workload, T: int = SEQ_LEN):
    toks = [BOS]
    labels = [0]
    src = [-1]          # for labeled positions: position of the dependency source
    used_keys = set()
    facts = {}          # key -> (val, pos_of_val)
    quotes = []         # list of (start_pos_of_w1, words)
    sched = []          # (due, kind, payload)
    a = int(rng.integers(N_WORD))
    topic = int(rng.integers(N_TOPIC))
    topic_left = 0
    prev_filler = False

    def emit(tok, lab=0, s=-1):
        toks.append(int(tok))
        labels.append(0)
        src.append(-1)
        # label belongs to the predicting position (previous index)
        if lab:
            labels[-2] = lab
            src[-2] = s

    while len(toks) < T:
        t = len(toks)
        due = [i for i, it in enumerate(sched) if it[0] <= t]
        if due:
            i = min(due, key=lambda j: sched[j][0])
            _, kind, payload = sched.pop(i)
            if kind == "ask":
                key = payload
                val, vpos = facts[key]
                emit(ASK)
                emit(key)
                emit(val, lab=1, s=vpos)
            else:
                qstart, words = quotes[payload]
                emit(RE)
                emit(words[0])
                for j, w in enumerate(words[1:]):
                    emit(w, lab=2, s=qstart + j + 1)
                emit(QC, lab=2, s=qstart + len(words))
            continue
        u = rng.random()
        if u < wl.p_fact and len(used_keys) < N_KEY:
            free = [k for k in range(N_KEY) if k not in used_keys]
            kidx = int(rng.choice(free))
            used_keys.add(kidx)
            key = KEY0 + kidx
            val = VAL0 + int(rng.integers(N_VAL))
            pinned = rng.random() < wl.p_pin
            if pinned:
                emit(PIN)
            emit(DEF)
            emit(key)
            emit(val)
            vpos = len(toks) - 1
            facts[key] = (val, vpos)
            t = len(toks)
            if pinned:
                nq = 1 + int(rng.random() < 0.4)
                for _ in range(nq):
                    sched.append((t + int(rng.integers(wl.long_gap_min, T)), "ask", key))
            elif rng.random() < wl.p_query_plain:
                sched.append((t + 6 + int(rng.geometric(1.0 / wl.short_gap_mean)), "ask", key))
            if kidx < N_HOT_KEYS and rng.random() < wl.p_hot_extra:
                sched.append((t + int(rng.integers(20, T)), "ask", key))
        elif u < wl.p_fact + wl.p_quote:
            n = int(rng.integers(3, 9))
            words = [QWORD0 + int(w) for w in rng.integers(0, N_QWORD, size=n)]
            emit(QO)
            qstart = len(toks)
            for w in words:
                emit(w)
            emit(QC)
            quotes.append((qstart, words))
            p_re = wl.p_requote_long if n >= 6 else wl.p_requote_short
            if rng.random() < p_re:
                sched.append((len(toks) + int(rng.integers(20, T)), "re", len(quotes) - 1))
        else:
            if topic_left <= 0:
                topic = int((topic + rng.integers(1, N_TOPIC)) % N_TOPIC)
                topic_left = int(rng.geometric(1.0 / wl.topic_len_mean))
                emit(TOPIC0 + topic)
                prev_filler = False
                continue
            c = int(_SUCC[topic, a][rng.choice(3, p=_SUCC_P)])
            emit(WORD0 + c, lab=3 if prev_filler else 0)
            a = c
            topic_left -= 1
            prev_filler = True
            continue
        prev_filler = False
    return (np.asarray(toks[:T], dtype=np.int64),
            np.asarray(labels[:T], dtype=np.int64),
            np.asarray(src[:T], dtype=np.int64))


def generate_corpus(seed: int, n: int, wl: Workload = IN_DIST, T: int = SEQ_LEN):
    rng = np.random.default_rng(seed)
    toks = np.zeros((n, T), dtype=np.int64)
    labs = np.zeros((n, T), dtype=np.int64)
    srcs = np.zeros((n, T), dtype=np.int64)
    for i in range(n):
        toks[i], labs[i], srcs[i] = generate_sequence(rng, wl, T)
    # The final position predicts nothing.
    labs[:, -1] = 0
    return toks, labs, srcs


if __name__ == "__main__":
    for name, wl in (("in", IN_DIST), ("shift", SHIFTED)):
        t, l, s = generate_corpus(1, 200, wl)
        pos = np.arange(t.shape[1])[None, :]
        gap1 = (pos - s)[l == 1]
        gap2 = (pos - s)[l == 2]
        print(name, "filler", (l == 3).sum(1).mean(), "topics", ((t >= 7) & (t < 11)).sum(1).mean())
        print(name, "facts/seq", (t == DEF).sum(1).mean(), "pins", (t == PIN).sum(1).mean(),
              "asks", (l == 1).sum(1).mean(), "quotes", (t == QO).sum(1).mean(), "requote toks", (l == 2).sum(1).mean(),
              "ans gap med/p90", np.median(gap1), np.percentile(gap1, 90), "copy gap med", np.median(gap2))
    print(t[0][:120])

# ===== END FILE: tmp/tests/generator.py =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator import IN_DIST, SHIFTED, generate_corpus  # noqa: E402
from kvmodel import TinyLM, BudgetViolation, PolicyContractError, kl_full_vs  # noqa: E402
from sandbox_utils import PolicyClient, PolicyFailure, check_policy  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.npz"          # verifier-owned copy; /app/model.npz is never trusted

CHUNK = 32
SEQ_LEN = 512
BUDGET = 96
N_PER_SHARD = 60
SHARDS = {
    "in_dist": (IN_DIST, 424242),
    "shifted": (SHIFTED, 515151),
}
SHARD_WEIGHTS = {"in_dist": 0.4, "shifted": 0.6}
# Component weights inside a shard.
COMPONENT_WEIGHTS = {"answer_kl": 0.45, "answer_top1": 0.20, "filler_kl": 0.20, "other_kl": 0.15}
# Anchors (score 0 -> score 1), measured on these exact hidden shards at BUDGET=96.
#  * KL components are interpolated in log space, top-1 agreement linearly.
#  * Score 0: answer_kl 2.0 nats / top-1 0.5 (generic heavy-hitter, SnapKV-style
#    and sink+window policies all sit far below: answer_kl ~7.2-8.2, top-1 <0.17);
#    filler/other: the H2O baseline's value (sink+window starter is worse still).
#  * Score 1: reference KL x 1.08 and reference top-1 - 0.004, so the reference
#    reaches 1.0 with a small margin.
#  Reference (solution/policy.py) metrics:
#    in_dist: answer_kl 0.0853  top1 0.9889  filler_kl 0.0321  other_kl 0.3408
#    shifted: answer_kl 0.3242  top1 0.9568  filler_kl 0.0554  other_kl 0.4395
#  H2O metrics:
#    in_dist: answer_kl 7.1839  top1 0.1286  filler_kl 0.7723  other_kl 0.6592
#    shifted: answer_kl 7.8318  top1 0.0683  filler_kl 0.5875  other_kl 0.6650
ANCHORS = {
    "in_dist": {"answer_kl": (2.0, 0.0921), "answer_top1": (0.5, 0.985),
                "filler_kl": (0.7723, 0.0347), "other_kl": (0.6592, 0.3681)},
    "shifted": {"answer_kl": (2.0, 0.3501), "answer_top1": (0.5, 0.953),
                "filler_kl": (0.5875, 0.0598), "other_kl": (0.6650, 0.4747)},
}
POLICY_TIMEOUT_SEC = 1200.0


def policy_config():
    m = TinyLM(MODEL_PATH)
    return {"d": m.d, "heads": m.H, "layers": m.L, "head_dim": m.hd, "vocab": m.V,
            "chunk": CHUNK, "seq_len": SEQ_LEN, "budget": BUDGET}


def _score_kl(val, bad, good):
    if not np.isfinite(val):
        return 0.0
    val = max(val, 1e-12)
    t = (np.log(bad) - np.log(val)) / (np.log(bad) - np.log(good))
    return float(np.clip(t, 0.0, 1.0))


def _score_lin(val, bad, good):
    return float(np.clip((val - bad) / (good - bad), 0.0, 1.0))


def shard_metrics(full_lps, test_lps, labels):
    kls, ans_agree, labs = [], [], []
    for f, t, l in zip(full_lps, test_lps, labels):
        kl = kl_full_vs(f, t)
        kls.append(kl[CHUNK:-1])
        labs.append(l[CHUNK:-1])
        a = l[CHUNK:-1] == 1
        ans_agree.append((f[CHUNK:-1].argmax(-1) == t[CHUNK:-1].argmax(-1))[a])
    kl = np.concatenate(kls)
    lab = np.concatenate(labs)
    other = (lab == 0) | (lab == 2)
    return {
        "answer_kl": float(kl[lab == 1].mean()),
        "answer_top1": float(np.concatenate(ans_agree).mean()),
        "filler_kl": float(kl[lab == 3].mean()),
        "other_kl": float(kl[other].mean()),
        "n_answer": int((lab == 1).sum()),
        "n_filler": int((lab == 3).sum()),
        "n_other": int(other.sum()),
    }


def score_metrics(per_shard):
    reward = 0.0
    detail = {}
    for name, m in per_shard.items():
        comp = {}
        for c, wt in COMPONENT_WEIGHTS.items():
            bad, good = ANCHORS[name][c]
            comp[c] = _score_lin(m[c], bad, good) if c == "answer_top1" else _score_kl(m[c], bad, good)
        s = sum(COMPONENT_WEIGHTS[c] * comp[c] for c in COMPONENT_WEIGHTS)
        detail[name] = {"metrics": m, "component_scores": comp, "shard_score": s}
        reward += SHARD_WEIGHTS[name] * s
    return float(np.clip(reward, 0.0, 1.0)), detail


def hidden_shards():
    out = {}
    for name, (wl, seed) in SHARDS.items():
        toks, labs, _ = generate_corpus(seed, N_PER_SHARD, wl, T=SEQ_LEN)
        out[name] = (toks, labs)
    return out


def evaluate_policy_object(policy, model=None, shards=None, full_cache=None):
    """Author/local path: run a policy object (in-process). Returns per-shard metrics."""
    model = model or TinyLM(MODEL_PATH)
    shards = shards or hidden_shards()
    per = {}
    for name, (toks, labs) in shards.items():
        fulls = full_cache[name] if full_cache else [model.run(t) for t in toks]
        tests = [model.run(t, policy=policy, budget=BUDGET, chunk=CHUNK) for t in toks]
        per[name] = shard_metrics(fulls, tests, labs)
    return per


def evaluate(policy_path: str | Path, write_dir: str | Path | None = None):
    t0 = time.time()
    policy_path = Path(policy_path)
    if not check_policy(policy_path):
        return _finish(0.0, {"error": "api_missing"}, write_dir)
    model = TinyLM(MODEL_PATH)
    shards = hidden_shards()
    client = None
    try:
        client = PolicyClient(policy_path, policy_config(), total_timeout=POLICY_TIMEOUT_SEC)
        per = {}
        for name, (toks, labs) in shards.items():
            fulls, tests = [], []
            for t in toks:
                fulls.append(model.run(t))
                tests.append(model.run(t, policy=client, budget=BUDGET, chunk=CHUNK))
            per[name] = shard_metrics(fulls, tests, labs)
    except BudgetViolation as exc:
        return _finish(0.0, {"error": "budget_violation", "detail": str(exc)}, write_dir)
    except PolicyContractError as exc:
        return _finish(0.0, {"error": "contract_violation", "detail": str(exc)}, write_dir)
    except PolicyFailure as exc:
        return _finish(0.0, {"error": exc.code, "detail": exc.detail[-1500:]}, write_dir)
    except Exception as exc:  # pragma: no cover
        return _finish(0.0, {"error": "runtime_error", "detail": repr(exc)[-1500:]}, write_dir)
    finally:
        if client is not None:
            client.close()
    reward, detail = score_metrics(per)
    metrics = {"reward": reward, "shards": detail, "wall_sec": round(time.time() - t0, 1)}
    return _finish(reward, metrics, write_dir)


def _finish(reward, metrics, write_dir=None):
    if write_dir is not None:
        p = Path(write_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "reward.txt").write_text(f"{float(reward):.12f}\n")
        (p / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return float(reward), metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_path")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    r, m = evaluate(ns.policy_path, ns.out)
    print(json.dumps(m, indent=2, sort_keys=True))

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/kvmodel.py =====
"""Chunked numpy inference for the tiny decoder, with a KV-eviction hook.

This file is shipped to the agent (as /app/kvmodel.py) and the verifier uses an
identical copy. Inference is float64 numpy.

Processing model
----------------
The sequence is processed in chunks of `chunk` tokens. For every chunk and every
layer (in order 0..L-1):

  1. the chunk's queries attend to [kept cache entries of this layer/head] plus
     the chunk's own entries (causal inside the chunk);
  2. the chunk's keys/values are appended to the cache;
  3. the eviction policy is called with a `LayerView` and returns, per head,
     the indices (into that head's current cache arrays) to KEEP.

After the last layer of a chunk the total number of cache entries over all
layers and heads must be <= budget. Heads may hold different numbers of
entries and layers may hold different totals.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EPS = 1e-6
_C = np.sqrt(2.0 / np.pi)


class BudgetViolation(RuntimeError):
    pass


class PolicyContractError(RuntimeError):
    pass


def load_weights(path):
    with np.load(path, allow_pickle=False) as z:
        w = {k: np.asarray(z[k], dtype=np.float64) for k in z.files if not k.startswith("cfg_")}
        cfg = {k[4:]: int(z[k]) for k in z.files if k.startswith("cfg_")}
    return w, cfg


def _rms(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * g


def _gelu(u):
    return 0.5 * u * (1 + np.tanh(_C * (u + 0.044715 * u * u * u)))


def _rope_cs(pos, hd, base=10000.0):
    half = hd // 2
    inv = base ** (-np.arange(half) / half)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, pos):
    """Rotate pre-RoPE vectors x[..., n, hd] to absolute positions pos[n]."""
    cos, sin = _rope_cs(pos, x.shape[-1])
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


@dataclass
class LayerView:
    """Everything the policy may look at for one (chunk, layer) call.

    Lists are indexed by head. For head h, cache entries are aligned across
    positions[h], keys[h], keys_rot[h], values[h] and the columns of attn[h].
    The last `chunk_end - chunk_start` entries of every head are the current
    chunk's own entries (positions chunk_start..chunk_end-1, in order).
    """
    layer: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    seq_len: int
    positions: list      # [H] int64 (n_h,)
    keys: list           # [H] float64 (n_h, hd)   pre-RoPE keys
    keys_rot: list       # [H] float64 (n_h, hd)   keys after RoPE at their position
    values: list         # [H] float64 (n_h, hd)
    queries: list        # [H] float64 (C, hd)     pre-RoPE queries of the chunk tokens
    attn: list           # [H] float64 (C, n_h)    softmax attention of chunk queries
    budget: int          # global entry budget (all layers, all heads)
    held: np.ndarray     # (L,) entries currently held per layer *before* this call
                         #  (layers < layer already evicted for this chunk; this
                         #   layer's count includes the new chunk)


class TinyLM:
    def __init__(self, weights_path):
        self.w, self.cfg = load_weights(weights_path)
        self.d = self.cfg["d"]
        self.H = self.cfg["heads"]
        self.L = self.cfg["layers"]
        self.hd = self.d // self.H
        self.V = self.cfg["vocab"]

    # ------------------------------------------------------------------
    def run(self, tokens, policy=None, budget=None, chunk=32, collect=None):
        """Return log-probs [T, V]. policy=None means full cache (no eviction).

        collect: optional callback(layer, chunk_start, chunk_end, head, positions, attn)
        used by analysis code to observe attention without evicting.
        """
        w, H, L, hd = self.w, self.H, self.L, self.hd
        tokens = np.asarray(tokens, dtype=np.int64)
        T = tokens.shape[0]
        scale = 1.0 / np.sqrt(hd)
        cpos = [[np.zeros(0, np.int64) for _ in range(H)] for _ in range(L)]
        ck = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        ckr = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        cv = [[np.zeros((0, hd)) for _ in range(H)] for _ in range(L)]
        out = np.zeros((T, self.V))
        if policy is not None and hasattr(policy, "begin_sequence"):
            policy.begin_sequence(T)
        for ci, s in enumerate(range(0, T, chunk)):
            e = min(T, s + chunk)
            C = e - s
            pos = np.arange(s, e)
            h = w["emb"][tokens[s:e]]
            intra = np.triu(np.full((C, C), -np.inf), 1)
            for l in range(L):
                a = _rms(h, w[f"g1_{l}"])
                q = (a @ w[f"Wq_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                k = (a @ w[f"Wk_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                v = (a @ w[f"Wv_{l}"]).reshape(C, H, hd).transpose(1, 0, 2)
                qr = apply_rope(q, pos)
                kr = apply_rope(k, pos)
                o = np.zeros((C, H, hd))
                attn_all = []
                for hh in range(H):
                    K = np.concatenate([ckr[l][hh], kr[hh]], 0)
                    Vv = np.concatenate([cv[l][hh], v[hh]], 0)
                    n_old = ckr[l][hh].shape[0]
                    sc = (qr[hh] @ K.T) * scale
                    sc[:, n_old:] += intra
                    sc -= sc.max(-1, keepdims=True)
                    P = np.exp(sc)
                    P /= P.sum(-1, keepdims=True)
                    o[:, hh] = P @ Vv
                    attn_all.append(P)
                    cpos[l][hh] = np.concatenate([cpos[l][hh], pos])
                    ck[l][hh] = np.concatenate([ck[l][hh], k[hh]], 0)
                    ckr[l][hh] = K
                    cv[l][hh] = Vv
                    if collect is not None:
                        collect(l, s, e, hh, cpos[l][hh], P)
                h = h + o.reshape(C, self.d) @ w[f"Wo_{l}"]
                m = _rms(h, w[f"g2_{l}"])
                h = h + _gelu(m @ w[f"W1_{l}"]) @ w[f"W2_{l}"]
                if policy is not None:
                    held = np.array([sum(len(cpos[ll][hh]) for hh in range(H)) for ll in range(L)], dtype=np.int64)
                    view = LayerView(
                        layer=l, chunk_index=ci, chunk_start=s, chunk_end=e, seq_len=T,
                        positions=[cpos[l][hh].copy() for hh in range(H)],
                        keys=[ck[l][hh].copy() for hh in range(H)],
                        keys_rot=[ckr[l][hh].copy() for hh in range(H)],
                        values=[cv[l][hh].copy() for hh in range(H)],
                        queries=[q[hh].copy() for hh in range(H)],
                        attn=[attn_all[hh].copy() for hh in range(H)],
                        budget=int(budget), held=held,
                    )
                    keep = policy.select(view)
                    keep = validate_keep(keep, [len(cpos[l][hh]) for hh in range(H)])
                    for hh in range(H):
                        idx = keep[hh]
                        cpos[l][hh] = cpos[l][hh][idx]
                        ck[l][hh] = ck[l][hh][idx]
                        ckr[l][hh] = ckr[l][hh][idx]
                        cv[l][hh] = cv[l][hh][idx]
            if policy is not None:
                total = sum(len(cpos[ll][hh]) for ll in range(L) for hh in range(H))
                if total > budget:
                    raise BudgetViolation(f"chunk {ci}: {total} cache entries > budget {budget}")
            f = _rms(h, w["gf"])
            logits = f @ w["Wu"]
            logits -= logits.max(-1, keepdims=True)
            out[s:e] = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        return out


def validate_keep(keep, sizes):
    if not isinstance(keep, (list, tuple)) or len(keep) != len(sizes):
        raise PolicyContractError("select() must return one index array per head")
    res = []
    for hh, (idx, n) in enumerate(zip(keep, sizes)):
        idx = np.asarray(idx)
        if idx.dtype == bool:
            if idx.shape != (n,):
                raise PolicyContractError(f"head {hh}: boolean mask has wrong shape")
            idx = np.flatnonzero(idx)
        if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
            raise PolicyContractError(f"head {hh}: indices must be a 1-D integer array")
        idx = idx.astype(np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise PolicyContractError(f"head {hh}: index out of range")
        idx = np.unique(idx)
        res.append(idx)
    return res


def kl_full_vs(full_logp, test_logp):
    """Per-position KL(full || test) in nats."""
    pf = np.exp(full_logp)
    return np.sum(pf * (full_logp - test_logp), axis=-1)

# ===== END FILE: tmp/tests/kvmodel.py =====

# ===== BEGIN FILE: tmp/tests/policy_server.py =====
#!/usr/bin/env python3
"""Sandbox-side policy server for kv_cache_eviction.

Runs as an unprivileged subprocess. It loads the candidate policy module, then
answers framed requests on stdin/stdout. Each request carries only what the
candidate is allowed to see for one (chunk, layer) step; no token ids, no future
positions, no verifier objects. Only plain .npz bytes (allow_pickle=False) cross
the boundary in either direction.

Frame: 8-byte little-endian length + npz payload.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import io
import os
from pathlib import Path
import struct
import sys
import traceback

import numpy as np


@dataclass
class LayerView:
    layer: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    seq_len: int
    positions: list
    keys: list
    keys_rot: list
    values: list
    queries: list
    attn: list
    budget: int
    held: np.ndarray


def _read_frame(stream):
    hdr = stream.read(8)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    buf = stream.read(n)
    with np.load(io.BytesIO(buf), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _write_frame(stream, **arrays):
    bio = io.BytesIO()
    np.savez(bio, **arrays)
    data = bio.getvalue()
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _load(path):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_policy", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    policy_path = argv[1]
    mode = argv[2] if len(argv) > 2 else "serve"
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    # Keep the protocol channel clean: anything the candidate prints goes to stderr.
    sys.stdout = sys.stderr
    try:
        mod = _load(policy_path)
        cls = getattr(mod, "EvictionPolicy", None)
        if cls is None or not callable(getattr(cls, "select", None)):
            raise AttributeError("EvictionPolicy class with select() not found")
    except Exception:
        if mode == "check":
            traceback.print_exc()
            return 1
        _write_frame(out, error=np.array("api_missing"), detail=np.array(traceback.format_exc()[-1500:]))
        return 0
    if mode == "check":
        return 0
    policy = None
    while True:
        req = _read_frame(inp)
        if req is None:
            return 0
        cmd = str(req["cmd"])
        try:
            if cmd == "init":
                cfg = {k[4:]: int(req[k]) for k in req if k.startswith("cfg_")}
                cfg["app_dir"] = str(Path(policy_path).parent)
                policy = cls(cfg)
                _write_frame(out, ok=np.array(1))
            elif cmd == "begin":
                if hasattr(policy, "begin_sequence"):
                    policy.begin_sequence(int(req["seq_len"]))
                _write_frame(out, ok=np.array(1))
            elif cmd == "select":
                H = int(req["H"])
                view = LayerView(
                    layer=int(req["layer"]), chunk_index=int(req["chunk_index"]),
                    chunk_start=int(req["chunk_start"]), chunk_end=int(req["chunk_end"]),
                    seq_len=int(req["seq_len"]),
                    positions=[req[f"pos{h}"] for h in range(H)],
                    keys=[req[f"k{h}"] for h in range(H)],
                    keys_rot=[req[f"kr{h}"] for h in range(H)],
                    values=[req[f"v{h}"] for h in range(H)],
                    queries=[req[f"q{h}"] for h in range(H)],
                    attn=[req[f"a{h}"] for h in range(H)],
                    budget=int(req["budget"]), held=req["held"],
                )
                keep = policy.select(view)
                if not isinstance(keep, (list, tuple)) or len(keep) != H:
                    raise ValueError("select() must return one index array per head")
                resp = {}
                for h, k in enumerate(keep):
                    k = np.asarray(k)
                    if k.dtype == bool:
                        k = np.flatnonzero(k)
                    resp[f"keep{h}"] = np.asarray(k, dtype=np.int64)
                _write_frame(out, **resp)
            elif cmd == "quit":
                _write_frame(out, ok=np.array(1))
                return 0
            else:
                raise ValueError(f"unknown command {cmd}")
        except Exception:
            _write_frame(out, error=np.array("runtime_error"), detail=np.array(traceback.format_exc()[-1500:]))
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# ===== END FILE: tmp/tests/policy_server.py =====

# ===== BEGIN FILE: tmp/tests/sandbox_utils.py =====
"""Root-side helpers: run the candidate policy in an unprivileged subprocess.

On the platform (root + /sandbox/policy_server.py present) the policy server is
started as `nobody`. Locally it falls back to a plain subprocess speaking the
same framed-npz protocol.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np

RUNNER = "/sandbox/policy_server.py"


class PolicyFailure(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _sandbox_ready() -> bool:
    return os.path.exists(RUNNER) and getattr(os, "geteuid", lambda: 1000)() == 0


def _runner() -> str:
    return RUNNER if _sandbox_ready() else str(Path(__file__).resolve().parent / "policy_server.py")


def _cmd(args):
    if _sandbox_ready():
        if shutil.which("runuser"):
            return ["runuser", "-u", "nobody", "--", *args]
        if shutil.which("su"):
            quoted = " ".join(subprocess.list2cmdline([a]) for a in args)
            return ["su", "-s", "/bin/sh", "nobody", "-c", quoted]
    return args


def check_policy(policy_path, timeout=60) -> bool:
    try:
        r = subprocess.run(_cmd([sys.executable, _runner(), str(policy_path), "check"]),
                           capture_output=True, text=True, timeout=timeout,
                           cwd="/sandbox" if _sandbox_ready() else None)
        return r.returncode == 0
    except Exception:
        return False


class PolicyClient:
    """Duck-types the policy interface expected by TinyLM.run()."""

    def __init__(self, policy_path, cfg: dict, total_timeout: float = 900.0):
        self.proc = subprocess.Popen(
            _cmd([sys.executable, _runner(), str(policy_path), "serve"]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd="/sandbox" if _sandbox_ready() else None,
        )
        self._stderr_tail = []
        threading.Thread(target=self._drain, daemon=True).start()
        self.deadline = time.monotonic() + total_timeout
        self.timed_out = False
        self._timer = threading.Timer(total_timeout, self._kill)
        self._timer.daemon = True
        self._timer.start()
        self.calls = 0
        msg = {"cmd": np.array("init")}
        for k, v in cfg.items():
            msg["cfg_" + k] = np.array(int(v), dtype=np.int64)
        self._rpc(msg)

    def _drain(self):
        for line in self.proc.stderr:
            self._stderr_tail.append(line.decode(errors="replace"))
            if len(self._stderr_tail) > 50:
                self._stderr_tail.pop(0)

    def _kill(self):
        self.timed_out = True
        try:
            self.proc.kill()
        except Exception:
            pass

    def _rpc(self, arrays):
        bio = io.BytesIO()
        np.savez(bio, **arrays)
        data = bio.getvalue()
        try:
            self.proc.stdin.write(struct.pack("<Q", len(data)))
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
            hdr = self.proc.stdout.read(8)
            if len(hdr) < 8:
                raise EOFError
            (n,) = struct.unpack("<Q", hdr)
            buf = self.proc.stdout.read(n)
            with np.load(io.BytesIO(buf), allow_pickle=False) as z:
                resp = {k: z[k] for k in z.files}
        except Exception:
            if self.timed_out:
                raise PolicyFailure("timeout", "policy exceeded the total time limit")
            raise PolicyFailure("runtime_error", "policy process died: " + "".join(self._stderr_tail)[-1500:])
        if "error" in resp:
            raise PolicyFailure(str(resp["error"]), str(resp.get("detail", "")))
        return resp

    def begin_sequence(self, T):
        self._rpc({"cmd": np.array("begin"), "seq_len": np.array(int(T))})

    def select(self, view):
        H = len(view.positions)
        msg = {
            "cmd": np.array("select"), "H": np.array(H), "layer": np.array(view.layer),
            "chunk_index": np.array(view.chunk_index), "chunk_start": np.array(view.chunk_start),
            "chunk_end": np.array(view.chunk_end), "seq_len": np.array(view.seq_len),
            "budget": np.array(view.budget), "held": np.asarray(view.held, dtype=np.int64),
        }
        for h in range(H):
            msg[f"pos{h}"] = view.positions[h]
            msg[f"k{h}"] = view.keys[h]
            msg[f"kr{h}"] = view.keys_rot[h]
            msg[f"v{h}"] = view.values[h]
            msg[f"q{h}"] = view.queries[h]
            msg[f"a{h}"] = view.attn[h]
        resp = self._rpc(msg)
        self.calls += 1
        out = []
        for h in range(H):
            if f"keep{h}" not in resp:
                raise PolicyFailure("bad_response", f"missing keep{h}")
            out.append(resp[f"keep{h}"])
        return out

    def close(self):
        try:
            self._rpc({"cmd": np.array("quit")})
        except Exception:
            pass
        self._timer.cancel()
        try:
            self.proc.kill()
        except Exception:
            pass

# ===== END FILE: tmp/tests/sandbox_utils.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
mkdir -p /sandbox

# Make the policy server available outside /tests before /tests is locked down.
if [ -f /tests/policy_server.py ]; then
  cp /tests/policy_server.py /sandbox/policy_server.py
fi
chmod 755 /sandbox /sandbox/policy_server.py 2>/dev/null || true
# The candidate runs as `nobody`; it must be able to read its own /app files.
chmod -R a+rX /app 2>/dev/null || true

START_SEC=$(date +%s)

# - Smoke check -
set +e
pytest /tests/test_smoke.py -v --tb=short 2> /logs/verifier/smoke_stderr.txt
SMOKE_EXIT=$?
set -e
if [ $SMOKE_EXIT -ne 0 ]; then
  echo pipeline_broken > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# Lock verifier code, weights and generator away from the untrusted subprocess.
chmod 700 /tests 2>/dev/null || true

# - Sandbox/security regression -
set +e
pytest /tests/test_security.py -v --tb=short >> /logs/verifier/security_stdout.txt 2>&1
SECURITY_EXIT=$?
set -e
if [ $SECURITY_EXIT -ne 0 ]; then
  echo reward_hack_open > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# - Main continuous scoring -
set +e
pytest /tests/test_main.py \
  --json-report --json-report-file=/logs/verifier/ctrf.json \
  -v --tb=short \
  2> /logs/verifier/stderr.txt
EXIT=$?
set -e

if [ $EXIT -ne 0 ]; then
  python3 /tests/categorize_failure.py /logs/verifier/stderr.txt \
    /logs/verifier/failure_mode.txt 2>/dev/null \
    || echo unknown > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
else
  if [ ! -f /logs/verifier/reward.txt ]; then
    echo missing_reward > /logs/verifier/failure_mode.txt
    echo 0 > /logs/verifier/reward.txt
  fi
fi

# - Reference anchor + hidden-slice hygiene (informational only; never change reward) -
if [ "${RUN_REFERENCE_CHECK:-0}" = "1" ]; then
  set +e
  pytest /tests/test_reference.py -v --tb=short >> /logs/verifier/reference_stdout.txt 2>&1 || true
  set -e
fi
set +e
pytest /tests/test_holdout.py -v --tb=short >> /logs/verifier/holdout_stdout.txt 2>&1 || true
set -e

END_SEC=$(date +%s)
echo $(( END_SEC - START_SEC )) > /logs/verifier/wall_clock_sec.txt
exit 0

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_holdout.py =====
from __future__ import annotations

import numpy as np

from hidden_eval import hidden_shards


def test_hidden_slices_are_not_tiny():
    for name, (toks, labs) in hidden_shards().items():
        counts = {k: int((labs[:, 32:-1] == k).sum()) for k in (1, 3)}
        assert min(counts.values()) >= 150, (name, counts)

# ===== END FILE: tmp/tests/test_holdout.py =====

# ===== BEGIN FILE: tmp/tests/test_main.py =====
from __future__ import annotations

from _paths import app_policy_path, verifier_log_dir
from hidden_eval import evaluate


def test_main_scoring_writes_reward_and_metrics():
    out_dir = verifier_log_dir()
    reward, metrics = evaluate(app_policy_path(), out_dir)
    assert (out_dir / "reward.txt").exists(), "reward.txt was not written"
    assert (out_dir / "metrics.json").exists(), "metrics.json was not written"
    assert 0.0 <= reward <= 1.0
    assert "reward" in metrics or "error" in metrics

# ===== END FILE: tmp/tests/test_main.py =====

# ===== BEGIN FILE: tmp/tests/test_reference.py =====
from __future__ import annotations

from _paths import reference_path, verifier_log_dir
from hidden_eval import evaluate


def test_reference_anchor_reaches_full_reward():
    reward, metrics = evaluate(reference_path(), verifier_log_dir() / "reference")
    assert reward >= 0.97, f"reference reward too low: {reward:.6f}, metrics={metrics}"

# ===== END FILE: tmp/tests/test_reference.py =====

# ===== BEGIN FILE: tmp/tests/test_security.py =====
from __future__ import annotations

from pathlib import Path
import textwrap

from hidden_eval import evaluate

PROBE = r'''
import glob, inspect, os
import numpy as np

def _leaks():
    found = []
    for p in ("/tests/generator.py", "/tests/hidden_eval.py", "/tests/model.npz"):
        try:
            with open(p, "rb") as f:
                f.read(16)
            found.append(p)
        except Exception:
            pass
    for fi in inspect.stack():
        loc = fi.frame.f_locals
        for name in ("shards", "toks", "labs", "tokens", "full_cache"):
            if name in loc:
                found.append("frame:" + name)
    try:
        ppid = os.getppid()
        with open(f"/proc/{ppid}/mem", "rb") as f:
            f.read(1)
        found.append("parent_mem")
    except Exception:
        pass
    return found


class EvictionPolicy:
    def __init__(self, config):
        self.leaks = _leaks()

    def select(self, view):
        leaks = self.leaks + _leaks()
        if leaks:
            raise RuntimeError("LEAK:" + ",".join(sorted(set(leaks))))
        raise RuntimeError("NOLEAK")
'''


def test_policy_process_cannot_see_verifier_state(tmp_path):
    probe_dir = Path("/sandbox/probe") if Path("/sandbox").exists() else tmp_path
    probe_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe_dir.chmod(0o755)
    except Exception:
        pass
    probe = probe_dir / "policy.py"
    probe.write_text(textwrap.dedent(PROBE))
    probe.chmod(0o644)
    reward, metrics = evaluate(probe, tmp_path / "logs")
    detail = str(metrics.get("detail", ""))
    assert reward == 0.0
    assert "NOLEAK" in detail, f"policy process could observe verifier state: {detail[-600:]}"

# ===== END FILE: tmp/tests/test_security.py =====

# ===== BEGIN FILE: tmp/tests/test_smoke.py =====
from __future__ import annotations

from _paths import app_policy_path
from sandbox_utils import check_policy


def test_policy_file_exists():
    assert app_policy_path().exists(), "/app/policy.py not found"


def test_policy_class_present():
    assert check_policy(app_policy_path()), "EvictionPolicy with select() could not be loaded from /app/policy.py"

# ===== END FILE: tmp/tests/test_smoke.py =====

