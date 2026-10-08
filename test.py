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

Positions are grouped by the role their token plays in the sequence, and the groups are weighted separately. Each group's KL is scored on a log scale, from score 0 (roughly where generic attention-statistics policies such as heavy-hitter or observation-window eviction land) to score 1 (an idealized upper reference that a policy is not expected to reach). Part of the hidden set is drawn from a workload whose statistics differ from the training corpus.

The reward is continuous in [0, 1]. `python /app/evaluate_local.py` gives a quick unweighted local estimate.

## Rules

- The verifier runs your policy in a separate, unprivileged process with no network. `select` sees only what is in `view`.
- Your policy may read any files you place under `/app` (for example, tables or parameters you precompute), but must not write files at evaluation time.
- The total policy compute for the hidden evaluation (about 7,700 `select` calls) must stay under 20 minutes on 4 CPU cores.
- Exceeding the budget, returning invalid indices, crashing or timing out gives reward 0.



hidden
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
#  * Score 0 (unchanged from v0.1): answer_kl 2.0 nats / top-1 0.5; filler/other at
#    the H2O baseline's value. Generic attention-statistics policies score 0.
#  * Score 1 (v0.2): for every KL component, 0.8 x the better of
#      - the privileged ORACLE (solution/oracle_policy.py: knows the hidden
#        sequence, keeps what future queries actually attend to), and
#      - the best value observed in the v0.1 frontier-agent probe (10 runs),
#    so neither the oracle nor any observed agent saturates a component.
#    top-1 agreement: score 1 at 1.0.
#  ORACLE    in_dist: answer_kl 0.0054 top1 0.9972 filler_kl 0.0145 other_kl 0.1323
#            shifted: answer_kl 0.0251 top1 0.9930 filler_kl 0.0279 other_kl 0.2491
#  best agent in_dist: answer_kl 0.041  filler_kl 0.021  other_kl 0.088
#  (v0.1)    shifted: answer_kl 0.127  filler_kl 0.038  other_kl 0.122
#  H2O       in_dist: answer_kl 7.1839 top1 0.1286 filler_kl 0.7723 other_kl 0.6592
#            shifted: answer_kl 7.8318 top1 0.0683 filler_kl 0.5875 other_kl 0.6650
ANCHORS = {
    "in_dist": {"answer_kl": (2.0, 0.00432), "answer_top1": (0.5, 1.0),
                "filler_kl": (0.7723, 0.0116), "other_kl": (0.6592, 0.0704)},
    "shifted": {"answer_kl": (2.0, 0.0201), "answer_top1": (0.5, 1.0),
                "filler_kl": (0.5875, 0.0223), "other_kl": (0.6650, 0.0976)},
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

resource
#!/usr/bin/env python3
"""Re-score existing verifier metrics.json files with the current anchors.

    python rescore.py run1/metrics.json run2/metrics.json ...

Uses the raw per-shard metrics recorded by hidden_eval (unchanged between v0.1
and v0.2: same budget, same hidden shards), so no policy has to be re-run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hidden_eval import score_metrics  # noqa: E402

for path in sys.argv[1:]:
    m = json.load(open(path))
    if "shards" not in m:
        print(f"{path}: no shard metrics ({m.get('error')}) -> 0.0")
        continue
    per = {k: v["metrics"] for k, v in m["shards"].items()}
    r, d = score_metrics(per)
    comp = {s: {c: round(x, 3) for c, x in v["component_scores"].items()} for s, v in d.items()}
    print(f"{path}: old={m.get('reward', float('nan')):.4f} new={r:.4f} {json.dumps(comp)}")

solve.sh

#!/bin/bash
# Install the privileged ORACLE policy into /app (defines the top of the reward scale).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TESTS="${TESTS_DIR:-/tests}"
cp "$HERE/oracle_policy.py" /app/policy.py
cp "$HERE/oracle_params.npz" /app/oracle_params.npz
cp "$HERE/legit_reference/policy.py" /app/oracle_roles.py
if [ -f "$HERE/oracle_generator.py" ]; then
  cp "$HERE/oracle_generator.py" /app/oracle_generator.py
else
  cp "$TESTS/generator.py" /app/oracle_generator.py
fi

oracle_policy

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

oracle policy
"""Privileged ORACLE eviction policy for kv_cache_eviction (uses the hidden mechanism).

Not a legitimate solution: it regenerates the hidden evaluation sequences from
the task generator (fixed seeds), identifies the current sequence from its first
chunk (tokens decoded exactly from layer-0 keys), computes the full-cache
attention of every FUTURE query, and keeps the entries that future queries will
actually use. Selection = per-layer budget share, then a single in-layer top-k on
  head_w[l,h] * role_w[role(p)] * (max_future_attn^alpha + beta * mean_future_attn)
with the knobs tuned on training sequences (oracle_params.npz).

solve.sh installs it as /app/policy.py together with oracle_params.npz,
oracle_generator.py (= tests/generator.py) and oracle_roles.py (= legit reference
policy module, used only for its token-role parser).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HIDDEN = (("IN_DIST", 424242), ("SHIFTED", 515151))
N_PER_SHARD = 60


class EvictionPolicy:
    def __init__(self, config):
        import oracle_generator as G
        from oracle_roles import SeqState
        from kvmodel import TinyLM
        self.SeqState = SeqState
        app = Path(config.get("app_dir", HERE))
        mp = next(p for p in (app / "model.npz", HERE / "model.npz", Path("/app/model.npz")) if p.exists())
        self.m = TinyLM(mp)
        self.L, self.H = self.m.L, self.m.H
        self.budget = int(config["budget"])
        with np.load(HERE / "oracle_params.npz", allow_pickle=False) as z:
            p = {k: z[k] for k in z.files}
        self.p = {"layer_frac": p["layer_frac"].astype(float), "head_w": p["head_w"].astype(float),
                  "role_w": p["role_w"].astype(float), "alpha": float(p["alpha"]),
                  "beta": float(p["beta"]), "min_recent": int(p["min_recent"])}
        self.lookup = {}
        for wl_name, seed in HIDDEN:
            toks, _, _ = G.generate_corpus(seed, N_PER_SHARD, getattr(G, wl_name), T=int(config["seq_len"]))
            for t in toks:
                self.lookup[tuple(int(x) for x in t[:int(config["chunk"])])] = t
        w = self.m.w
        emb = w["emb"]
        a = emb / np.sqrt(np.mean(emb * emb, -1, keepdims=True) + 1e-6) * w["g1_0"]
        self.table = a @ w["Wk_0"]

    def begin_sequence(self, seq_len):
        self.ready = False

    def _prepare(self, view):
        C = view.chunk_end - view.chunk_start
        K = np.concatenate([k[-C:] for k in view.keys], 1)
        pref = tuple(int(x) for x in ((K[:, None] - self.table[None]) ** 2).sum(-1).argmin(1))
        tokens = np.asarray(self.lookup[pref])
        T = len(tokens)
        A = np.zeros((self.L, self.H, T, T))

        def cb(l, s, e, hh, pos, P):
            A[l, hh, s:e][:, pos] = P

        self.m.run(tokens, collect=cb, chunk=T)
        self.fut_max = np.maximum.accumulate(A[:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        self.fut_sum = np.cumsum(A[:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        st = self.SeqState(T)
        st.push(tokens)
        self.role = st.role.copy()
        self.T = T
        self.ready = True

    def select(self, view):
        if not self.ready:
            self._prepare(view)
        p, e, l = self.p, view.chunk_end, view.layer
        frac = p["layer_frac"]
        held_other = sum(view.held[ll] for ll in range(self.L) if ll < l) + \
            sum(min(view.held[ll], int(frac[ll] * self.budget)) for ll in range(self.L) if ll > l)
        lb = int(max(0, min(frac[l] * self.budget, self.budget - held_other)))
        scores, owners, locals_ = [], [], []
        for h, pos in enumerate(view.positions):
            if e < self.T:
                fm = self.fut_max[l, h, e, pos]
                fs = self.fut_sum[l, h, e, pos] / (self.T - e)
            else:
                fm = fs = np.zeros(len(pos))
            sc = p["head_w"][l, h] * p["role_w"][self.role[pos]] * (fm ** p["alpha"] + p["beta"] * fs)
            scores.append(sc + 10.0 * (pos >= e - p["min_recent"]))
            owners.append(np.full(len(pos), h))
            locals_.append(np.arange(len(pos)))
        sc = np.concatenate(scores)
        own = np.concatenate(owners)
        loc = np.concatenate(locals_)
        k = min(lb, sc.size)
        top = np.argpartition(-sc, k - 1)[:k] if 0 < k < sc.size else np.arange(sc.size)[:k]
        keep = [[] for _ in range(self.H)]
        for j in top:
            keep[own[j]].append(loc[j])
        return [np.sort(np.asarray(x, dtype=np.int64)) for x in keep]


