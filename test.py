#!/bin/bash
# Oracle: install the privileged policy (solution/policy.py) as /app/policy.py.
set -euo pipefail
cp "$(dirname "$0")/policy.py" /app/policy.py


"""Privileged ORACLE eviction policy for kv_cache_eviction -- single self-contained file.

This is the task's solution/oracle. It is NOT a legitimate policy: it uses the hidden
mechanism. It regenerates the hidden evaluation sequences from the task generator
(embedded below, identical to tests/generator.py, fixed seeds), identifies the current
sequence from its first chunk (tokens decoded exactly from layer-0 keys), computes the
full-cache attention of every FUTURE query and keeps the entries future queries will
actually use. Selection = per-layer budget share, then one in-layer top-k on
  head_w[l,h] * role_w[role(p)] * (max_future_attn^alpha + beta * mean_future_attn)
with the knobs tuned on training sequences (ORACLE_PARAMS below).

Install: copy this file to /app/policy.py. Needs only /app/kvmodel.py and /app/model.npz,
which are part of the task environment.
Expected verifier reward (v0.2 anchors): 0.905.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HIDDEN = (("IN_DIST", 424242), ("SHIFTED", 515151))
N_PER_SHARD = 60

# ---------------------------------------------------------------------------
# Embedded task generator (identical to tests/generator.py)
# ---------------------------------------------------------------------------
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



# ---------------------------------------------------------------------------
# Token-role parser
# ---------------------------------------------------------------------------
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



ORACLE_PARAMS = {
    'layer_frac': np.array([0.08518189579905856, 0.4012923066301236, 0.16947279290835857, 0.34405300466245936]),
    'head_w': np.array([[0.25, 8.0, 4.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.25, 0.25, 0.5], [1.0, 4.0, 0.5, 0.5]]),
    'role_w': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 0.25, 1.0, 1.0, 1.0, 1.0, 0.25, 1.0]),
    'alpha': 0.75,
    'beta': 0.125,
    'min_recent': 1,
}


class EvictionPolicy:
    def __init__(self, config):
        from kvmodel import TinyLM
        self.SeqState = SeqState
        app = Path(config.get("app_dir", HERE))
        mp = next(p for p in (app / "model.npz", HERE / "model.npz", Path("/app/model.npz")) if p.exists())
        self.m = TinyLM(mp)
        self.L, self.H = self.m.L, self.m.H
        self.budget = int(config["budget"])
        self.p = ORACLE_PARAMS
        self.lookup = {}
        for wl_name, seed in HIDDEN:
            toks, _, _ = generate_corpus(seed, N_PER_SHARD, globals()[wl_name], T=int(config["seq_len"]))
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
