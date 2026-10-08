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
