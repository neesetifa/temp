# ===== BEGIN FILE: tmp/environment/Dockerfile =====
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 pytest==8.3.5

WORKDIR /app
COPY app/ /app/
COPY tests/ /app/tests/

# ===== END FILE: tmp/environment/Dockerfile =====

# ===== BEGIN FILE: tmp/environment/app/evaluate.py =====
"""Proxy evaluation on the 64 visible unlabeled inputs per task.

    python3 /app/evaluate.py [path/to/merged.npz]

Prints mean KL(expert_k || merged head k) on the visible inputs.  If you trained on these inputs
this number is optimistic -- hold some out if you want an honest estimate.  The grader uses the
same formula on fresh hidden inputs drawn from the same distributions.
"""
import os
import sys

import numpy as np

from mlp import MERGED_SHAPES, N_TASKS, expert_logits, kl, load_expert, load_npz, load_unlabeled, merged_logits, softmax

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "merged.npz")
    m = load_npz(path)
    for k, s in MERGED_SHAPES.items():
        assert k in m and m[k].shape == s, f"bad or missing array {k}: expected {s}"
        assert np.isfinite(m[k]).all(), f"non-finite values in {k}"
    xs = load_unlabeled()
    vals = []
    for k in range(N_TASKS):
        p = softmax(expert_logits(load_expert(k), xs[k]))
        v = kl(p, merged_logits(m, xs[k], k))
        vals.append(v.mean())
        print(f"task {k}: mean KL {v.mean():.4f}   agreement {(p.argmax(1) == merged_logits(m, xs[k], k).argmax(1)).mean():.3f}")
    print(f"mean over tasks: {np.mean(vals):.4f}   geometric mean: {np.exp(np.mean(np.log(np.maximum(vals, 1e-12)))):.4f}")


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/environment/app/evaluate.py =====

# ===== BEGIN FILE: tmp/environment/app/make_merged.py =====
"""Starter: naive task arithmetic (average all expert trunks, keep each expert's head).

    python3 /app/make_merged.py        -> writes /app/merged.npz

This is a weak baseline (it ignores everything that makes the problem hard).  Replace it.
"""
import os

import numpy as np

from mlp import N_TASKS, TRUNK, load_base, load_expert, save_merged

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    base = load_base()
    experts = [load_expert(k) for k in range(N_TASKS)]
    m = {k: base[k] + sum(e[k] - base[k] for e in experts) / N_TASKS for k in TRUNK}
    m["HW"] = np.stack([e["head_W"] for e in experts])
    m["Hb"] = np.stack([e["head_b"] for e in experts])
    save_merged(os.path.join(HERE, "merged.npz"), m)
    print("wrote", os.path.join(HERE, "merged.npz"))


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/environment/app/make_merged.py =====

# ===== BEGIN FILE: tmp/environment/app/mlp.py =====
"""Model definition, loss/gradient and Adam in plain numpy (feel free to use torch instead).

Architecture (base, experts and the merged model all share the trunk shape):
    x (48) -> ReLU(x W1 + b1) (64) -> ReLU(. W2 + b2) (64) -> ReLU(. W3 + b3) (64) = h
    expert k : logits = h @ head_W + head_b            head_W (64,10), head_b (10,)
    merged   : logits_k = h @ HW[k] + Hb[k]            HW (6,64,10), Hb (6,10)  -- one shared trunk, 6 heads

Merged-model file format (what you must deliver as /app/merged.npz):
    W1 (48,64) b1 (64,) W2 (64,64) b2 (64,) W3 (64,64) b3 (64,) HW (6,64,10) Hb (6,10), all finite floats.
"""
import os

import numpy as np

D_IN, HID, N_TASKS, N_CLASSES = 48, 64, 6, 10
TRUNK = ["W1", "b1", "W2", "b2", "W3", "b3"]
MERGED_SHAPES = {"W1": (D_IN, HID), "b1": (HID,), "W2": (HID, HID), "b2": (HID,), "W3": (HID, HID), "b3": (HID,),
                 "HW": (N_TASKS, HID, N_CLASSES), "Hb": (N_TASKS, N_CLASSES)}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------- loading
def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def load_base():
    return load_npz(os.path.join(DATA, "base.npz"))


def load_expert(k):
    return load_npz(os.path.join(DATA, f"expert_{k}.npz"))


def load_unlabeled():
    d = load_npz(os.path.join(DATA, "unlabeled_inputs.npz"))
    return [d[f"task_{k}"] for k in range(N_TASKS)]


def save_merged(path, m):
    for k, s in MERGED_SHAPES.items():
        assert m[k].shape == s, (k, m[k].shape, s)
        assert np.isfinite(m[k]).all(), k
    np.savez(path, **{k: np.asarray(m[k], dtype=np.float64) for k in MERGED_SHAPES})


# ---------------------------------------------------------------- forward
def trunk(p, x):
    h1 = np.maximum(x @ p["W1"] + p["b1"], 0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
    return np.maximum(h2 @ p["W3"] + p["b3"], 0)


def expert_logits(e, x):
    return trunk(e, x) @ e["head_W"] + e["head_b"]


def merged_logits(m, x, k):
    return trunk(m, x) @ m["HW"][k] + m["Hb"][k]


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def softmax(z):
    return np.exp(log_softmax(z))


def kl(p_teacher, logits_student):
    """KL(p_teacher || softmax(logits_student)) per row -- the official per-input loss."""
    lt = np.log(np.maximum(p_teacher, 1e-30))
    return (p_teacher * (lt - log_softmax(logits_student))).sum(1)


# ---------------------------------------------------------------- training helpers
def grad(m, x, k, target):
    """Cross-entropy of merged head(s) k against probability targets.
    x (n,48); k int array (n,) head index per row; target (n,10).  Returns (loss, grads)."""
    h1 = np.maximum(x @ m["W1"] + m["b1"], 0)
    h2 = np.maximum(h1 @ m["W2"] + m["b2"], 0)
    h3 = np.maximum(h2 @ m["W3"] + m["b3"], 0)
    HWk = m["HW"][k]
    z = np.einsum("nh,nhc->nc", h3, HWk) + m["Hb"][k]
    ls = log_softmax(z)
    n = len(x)
    loss = -(target * ls).sum() / n
    dz = (np.exp(ls) - target) / n
    g = {"HW": np.zeros_like(m["HW"]), "Hb": np.zeros_like(m["Hb"])}
    for kk in np.unique(k):
        s = k == kk
        g["HW"][kk] = h3[s].T @ dz[s]
        g["Hb"][kk] = dz[s].sum(0)
    dh3 = np.einsum("nc,nhc->nh", dz, HWk) * (h3 > 0)
    g["W3"], g["b3"] = h2.T @ dh3, dh3.sum(0)
    dh2 = (dh3 @ m["W3"].T) * (h2 > 0)
    g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
    dh1 = (dh2 @ m["W2"].T) * (h1 > 0)
    g["W1"], g["b1"] = x.T @ dh1, dh1.sum(0)
    return loss, g


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t, self.lr, self.b1, self.b2 = 0, lr, b1, b2

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)

# ===== END FILE: tmp/environment/app/mlp.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Consolidate six experts into one model, with almost no data

You are given a pretrained **base** network and six **expert** networks, one per task
(`/app/data/expert_{0..5}.npz`). Each expert was fine-tuned from the base on its own private
labelled dataset. Deployment can only afford **one** network of the base's size, so your job is to
build a single **merged model**: one shared trunk with six task heads that reproduces every
expert's predictions as faithfully as possible.

## Architecture

Base, experts and the merged model share the same trunk shape (see `/app/mlp.py`):

```
x (48) -> ReLU(x W1 + b1) (64) -> ReLU(. W2 + b2) (64) -> ReLU(. W3 + b3) (64) = h
expert k : logits = h @ head_W + head_b                      (10 classes)
merged   : logits_k = h @ HW[k] + Hb[k]   for task k = 0..5    (one trunk, six heads)
```

The experts started from the base, but not all from the same parameterization. Some of them were
fine-tuned from a copy of the base whose hidden units had been permuted (a functionally identical
network with reordered neurons), so their weights are not directly aligned with the base or with
each other.

## Data

| file | content |
|---|---|
| `data/base.npz` | base trunk (`W1 b1 W2 b2 W3 b3`) |
| `data/expert_{k}.npz` | expert k: trunk + `head_W (64,10)`, `head_b (10,)` |
| `data/unlabeled_inputs.npz` | `task_{k}`: **64 unlabeled inputs** for each task, drawn i.i.d. from that task's input distribution |

There are no labels and no other data. Each task has its own unknown input distribution over R^48.
The 64 samples are the only data you have from it, but you can run the experts (and the base) on
any input you like.

`mlp.py` has loaders, the forward pass, the KL loss, a gradient for the merged model and Adam, all
in numpy. You may install and use anything else, e.g. torch. `make_merged.py` is a working but weak
starter (naive weight averaging). `evaluate.py` reports the score formula below on the 64 visible
inputs. That number is optimistic if you train on those inputs, so hold some out for an honest
estimate. `tests/test_public.py` checks the deliverable format.

## Deliverable

Write **`/app/merged.npz`** containing exactly these float arrays (extra arrays are ignored):
`W1 (48,64) b1 (64,) W2 (64,64) b2 (64,) W3 (64,64) b3 (64,) HW (6,64,10) Hb (6,10)`, all finite.
`mlp.save_merged()` writes the right format. Only this file is graded; none of your code is run
at grading time, so you can spend your own session's compute on producing it.

## Grading

For each task k, the grader draws 3,000 fresh hidden inputs from task k's true input distribution
(the same distribution your 64 samples came from). It computes the mean
**KL(expert_k(x) ‖ merged head k(x))** over the softmax outputs. Each task's mean KL is mapped to
a score in [0,1] on a log scale between two hidden references:

- **0** = plain distillation on the 64 visible inputs with Gaussian input noise;
- **1** = clearly better than a strong privileged reference.

The reward is the average of the six task scores. It is continuous; there is no pass/fail. A
missing or malformed file (wrong keys, shapes or dtypes, NaN/inf, pickled objects, symlinks,
files over 20 MB) gets reward 0.

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
"""Public format check for the deliverable (the hidden grader applies the same checks).

    cd /app && python3 -m pytest -q tests/test_public.py
"""
import os
import sys

import numpy as np

APP = os.environ.get("APP_DIR", "/app")
sys.path.insert(0, APP)
from mlp import MERGED_SHAPES, N_TASKS, expert_logits, kl, load_expert, load_unlabeled, merged_logits, softmax  # noqa: E402


def test_merged_file_format():
    path = os.path.join(APP, "merged.npz")
    assert os.path.isfile(path) and not os.path.islink(path), "deliver /app/merged.npz as a regular file"
    assert os.path.getsize(path) < 20 * 1024 * 1024
    with np.load(path, allow_pickle=False) as d:
        for k, s in MERGED_SHAPES.items():
            assert k in d.files, f"missing {k}"
            assert d[k].dtype.kind == "f" and d[k].shape == s, f"{k}: {d[k].dtype} {d[k].shape}, expected float {s}"
            assert np.isfinite(d[k]).all(), f"{k} has non-finite values"


def test_merged_runs_on_visible_inputs():
    with np.load(os.path.join(APP, "merged.npz"), allow_pickle=False) as d:
        m = {k: d[k].astype(np.float64) for k in d.files}
    xs = load_unlabeled()
    for k in range(N_TASKS):
        v = kl(softmax(expert_logits(load_expert(k), xs[k])), merged_logits(m, xs[k], k))
        assert np.isfinite(v).all()

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/solve.sh =====
#!/bin/bash
cp /solution/merged.npz /app/merged.npz

# ===== END FILE: tmp/solution/solve.sh =====

# ===== BEGIN FILE: tmp/tests/Dockerfile =====
# Verifier image.  Only reads /app/merged.npz (no candidate code is executed), so numpy is all it needs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN pip install numpy==2.2.6 pytest==8.3.5

# ===== END FILE: tmp/tests/Dockerfile =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
#!/usr/bin/env python3
"""Hidden evaluation for fewshot_expert_consolidation.

The candidate delivers only a weights file (/app/merged.npz); no candidate code is executed.
The file is treated as untrusted input: it must be a regular file (no symlink), small, a plain
npz without pickled objects, with the exact float arrays listed in scoring.SHAPES.
Any violation -> reward 0.
"""
import json
import os
import shutil
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import SHAPES, raw_metrics, reward_from_raw  # noqa: E402

APP = os.environ.get("FEC_APP_DIR", "/app")
LOGDIR = os.environ.get("FEC_LOG_DIR", "/logs/verifier")
HIDDEN = os.path.join(HERE, "hidden", "hidden_test.npz")
ANCHORS = os.path.join(HERE, "hidden", "anchors.json")
MAX_FILE = 20 * 1024 * 1024  # bytes on disk
MAX_MEMBER = 2 * 1024 * 1024  # uncompressed bytes per array (largest legit array is ~31 KB)


def load_candidate(path):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError("merged.npz missing or not a regular file")
    if os.path.getsize(path) > MAX_FILE:
        raise ValueError("merged.npz too large")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        raw = fh.read(MAX_FILE + 1)
    import io
    bio = io.BytesIO(raw)
    with zipfile.ZipFile(bio) as zf:
        for info in zf.infolist():
            if info.file_size > MAX_MEMBER:
                raise ValueError(f"array {info.filename} too large")
    bio.seek(0)
    m = {}
    with np.load(bio, allow_pickle=False) as d:
        for k, shape in SHAPES.items():
            if k not in d.files:
                raise ValueError(f"missing array {k}")
            a = d[k]
            if a.dtype.kind != "f" or a.shape != shape:
                raise ValueError(f"array {k}: dtype {a.dtype} shape {a.shape}, expected float {shape}")
            a = a.astype(np.float64)
            if not np.isfinite(a).all():
                raise ValueError(f"array {k} has non-finite values")
            m[k] = a
    return m


def write_out(reward, metrics):
    os.makedirs(LOGDIR, exist_ok=True)
    for fn in ("metrics.json", "reward.txt"):
        p = os.path.join(LOGDIR, fn)
        if os.path.lexists(p):
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.unlink(p)
    with open(os.path.join(LOGDIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(LOGDIR, "reward.txt"), "w") as f:
        f.write(f"{reward:.6f}\n")


def main():
    if os.geteuid() == 0:
        os.chmod(HERE, 0o700)
    try:
        m = load_candidate(os.path.join(APP, "merged.npz"))
    except Exception as e:  # malformed / missing deliverable -> 0
        write_out(0.0, {"error": f"{type(e).__name__}: {e}"[:2000], "reward": 0.0})
        print("invalid deliverable:", e)
        return
    with np.load(HIDDEN, allow_pickle=False) as d:
        H = {k: d[k] for k in d.files}
    anchors = json.load(open(ANCHORS))
    raw = raw_metrics(m, H)
    reward, scores = reward_from_raw(raw, anchors)
    write_out(reward, {"reward": reward, "group_scores": scores, "raw": raw})
    print(json.dumps({"reward": round(reward, 4), "group_scores": {k: round(v, 3) for k, v in scores.items()},
                      "mean_kl": {k: round(v["mean_kl"], 5) for k, v in raw.items()}}, indent=1))


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/rescore.py =====
#!/usr/bin/env python3
"""Offline re-scoring of finished runs with (possibly updated) anchors -- no re-running needed.

    python3 rescore.py metrics1.json [metrics2.json ...] [--anchors anchors.json]
                       [--set-best-agent]   # write per-task best observed agent mean_kl into anchors.json

metrics*.json are the files hidden_eval.py writes to /logs/verifier/metrics.json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import reward_from_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--anchors", default=os.path.join(HERE, "hidden", "anchors.json"))
    ap.add_argument("--set-best-agent", action="store_true")
    a = ap.parse_args()
    anchors = json.load(open(a.anchors))
    runs = []
    for p in a.metrics:
        m = json.load(open(p))
        if "raw" not in m:
            print(f"{p}: no raw metrics (invalid deliverable) -> 0")
            continue
        runs.append((p, m["raw"]))
    if a.set_best_agent and runs:
        for g in anchors["groups"]:
            anchors["groups"][g]["best_agent"] = min(r[g]["mean_kl"] for _, r in runs)
        json.dump(anchors, open(a.anchors, "w"), indent=1)
        print(f"updated best_agent in {a.anchors}")
    for p, raw in runs:
        r, sc = reward_from_raw(raw, anchors)
        print(f"{r:.4f}  {p}  " + " ".join(f"{k}={v:.2f}" for k, v in sc.items()))


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/tests/rescore.py =====

# ===== BEGIN FILE: tmp/tests/scoring.py =====
"""Scoring math shared by hidden_eval.py and rescore.py."""
import math

import numpy as np

K = 6
SHAPES = {"W1": (48, 64), "b1": (64,), "W2": (64, 64), "b2": (64,), "W3": (64, 64), "b3": (64,),
          "HW": (6, 64, 10), "Hb": (6, 10)}


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def merged_logits(m, x, k):
    h = np.maximum(x @ m["W1"] + m["b1"], 0)
    h = np.maximum(h @ m["W2"] + m["b2"], 0)
    h = np.maximum(h @ m["W3"] + m["b3"], 0)
    return h @ m["HW"][k] + m["Hb"][k]


def raw_metrics(m, H):
    """Per-task mean KL(expert || merged) and agreement on the hidden inputs (anchor independent)."""
    out = {}
    with np.errstate(all="ignore"):
        for k in range(K):
            x, p = H[f"x_{k}"], H[f"p_{k}"]
            z = merged_logits(m, x, k)
            if not np.isfinite(z).all():
                out[f"task_{k}"] = {"mean_kl": float("inf"), "agreement": 0.0}
                continue
            lt = np.log(np.maximum(p, 1e-30))
            klv = (p * (lt - log_softmax(z))).sum(1)
            out[f"task_{k}"] = {"mean_kl": float(klv.mean()) if np.isfinite(klv).all() else float("inf"),
                                "p90_kl": float(np.percentile(klv, 90)) if np.isfinite(klv).all() else float("inf"),
                                "agreement": float((p.argmax(1) == z.argmax(1)).mean())}
    return out


def group_score(L, L0, L1):
    if not np.isfinite(L) or L >= L0:
        return 0.0
    if L <= L1:
        return 1.0
    return float(min(1.0, max(0.0, math.log(L0 / L) / math.log(L0 / L1))))


def reward_from_raw(raw, anchors):
    scores = {}
    for k in range(K):
        g = anchors["groups"][f"task_{k}"]
        ref = g["oracle"] if g.get("best_agent") is None else min(g["oracle"], g["best_agent"])
        scores[f"task_{k}"] = group_score(raw[f"task_{k}"]["mean_kl"], g["L0"], 0.8 * ref)
    return float(np.mean(list(scores.values()))), scores

# ===== END FILE: tmp/tests/scoring.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
# Verifier entry point: scores /app/merged.npz on the hidden inputs and writes
# /logs/verifier/reward.txt (+ metrics.json).
set -u
mkdir -p /logs/verifier
chmod 700 /tests 2>/dev/null || true
python3 /tests/hidden_eval.py
if [ ! -s /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
exit 0

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_security.py =====
"""Regression test for the verifier's handling of untrusted deliverables.

    python3 /tests/test_security.py      (or: pytest -q /tests/test_security.py)

No candidate code is ever executed; the only attack surface is /app/merged.npz and the log dir.
Every malformed / hostile deliverable must give reward 0 without crashing the verifier, and a
pre-planted symlink at the reward path must not redirect the write.  A valid file must still score.
"""
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import SHAPES  # noqa: E402


def _valid(rng):
    return {k: rng.normal(0, 0.1, s) for k, s in SHAPES.items()}


def _save(path, arrays):
    np.savez(path, **arrays)


def _cases(rng):
    v = _valid(rng)
    cases = {}
    cases["missing_key"] = lambda p: _save(p, {k: a for k, a in v.items() if k != "W2"})
    cases["wrong_shape"] = lambda p: _save(p, dict(v, W1=np.zeros((48, 65))))
    cases["int_dtype"] = lambda p: _save(p, dict(v, b1=np.zeros(64, dtype=np.int64)))
    cases["nan"] = lambda p: _save(p, dict(v, W3=np.full((64, 64), np.nan)))
    cases["huge_values"] = lambda p: _save(p, dict(v, W1=np.full((48, 64), 1e200)))

    def obj(p):  # pickled object array inside the npz
        buf = io.BytesIO()
        np.save(buf, np.array([{"x": 1}], dtype=object), allow_pickle=True)
        with zipfile.ZipFile(p, "w") as zf:
            for k, a in v.items():
                b = io.BytesIO()
                np.save(b, a)
                zf.writestr(k + ".npy", b.getvalue())
            zf.writestr("W1.npy", buf.getvalue())
    cases["pickled_object"] = obj

    def raw_pickle(p):
        with open(p, "wb") as f:
            pickle.dump(v, f)
    cases["raw_pickle"] = raw_pickle

    def bomb(p):  # 400 MB of zeros, compresses to < 1 MB
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            hdr = io.BytesIO()
            np.lib.format.write_array_header_1_0(hdr, {"descr": "<f8", "fortran_order": False, "shape": (50_000_000,)})
            zf.writestr("W1.npy", hdr.getvalue() + b"\0" * 400_000_000)
    cases["zip_bomb"] = bomb
    cases["symlink_to_hidden"] = lambda p: os.symlink(os.path.join(HERE, "hidden", "hidden_test.npz"), p)
    cases["directory"] = lambda p: os.mkdir(p)
    cases["missing_file"] = lambda p: None
    return cases, v


def _run(app, logdir):
    env = dict(os.environ, FEC_APP_DIR=app, FEC_LOG_DIR=logdir)
    r = subprocess.run([sys.executable, os.path.join(HERE, "hidden_eval.py")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"verifier crashed: {r.stderr[-2000:]}"
    return float(open(os.path.join(logdir, "reward.txt")).read())


def test_malformed_deliverables_score_zero():
    rng = np.random.default_rng(0)
    cases, valid = _cases(rng)
    tmp = tempfile.mkdtemp(prefix="fec_sec_")
    try:
        for name, make in cases.items():
            app, logdir = os.path.join(tmp, name, "app"), os.path.join(tmp, name, "logs")
            os.makedirs(app)
            make(os.path.join(app, "merged.npz"))
            r = _run(app, logdir)
            assert r == 0.0, f"{name}: reward {r}"
            assert "error" in json.load(open(os.path.join(logdir, "metrics.json"))) or name == "huge_values", name
        # valid file + planted symlink at reward path -> verifier writes a real file, target untouched
        app, logdir = os.path.join(tmp, "ok", "app"), os.path.join(tmp, "ok", "logs")
        os.makedirs(app)
        os.makedirs(logdir)
        target = os.path.join(tmp, "victim.txt")
        open(target, "w").write("untouched")
        os.symlink(target, os.path.join(logdir, "reward.txt"))
        _save(os.path.join(app, "merged.npz"), valid)
        _run(app, logdir)
        assert not os.path.islink(os.path.join(logdir, "reward.txt"))
        assert open(target).read() == "untouched"
        m = json.load(open(os.path.join(logdir, "metrics.json")))
        assert "raw" in m and all(np.isfinite(v["mean_kl"]) for v in m["raw"].values())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_malformed_deliverables_score_zero()
    print("security test passed")

# ===== END FILE: tmp/tests/test_security.py =====

