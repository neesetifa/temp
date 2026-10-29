# Consolidate six task models into one before on-device deployment

A product ships six small classifiers, one per task, all fine-tuned by different teams from the
same pretrained **base** network. The next device release has room for only **one** network of
the base's size. Your job is to build that single **merged model**: one shared trunk with six
task heads. Once it ships, it replaces all six task models on the device, so each head has to
reproduce its original model's predictions as faithfully as possible.

## What you have

All files are in `/app/data`:

| file | content |
|---|---|
| `base.npz` | the pretrained base trunk (`W1 b1 W2 b2 W3 b3`) |
| `expert_{k}.npz` | task model k (k = 0..5): trunk + `head_W (64,10)`, `head_b (10,)` |
| `unlabeled_inputs.npz` | `task_{k}`: **64 inputs logged from task k's production traffic** |

There are no labels and no other data. The input distribution of each task's traffic over R^48 is
unknown, and some tasks may share the same input distribution. The 64 logged inputs per task
are all you have from it, but you can run the six task models (and the base) on any input you
like.

The six task models started from the base, but not all from the same export of it. Some teams
fine-tuned from a re-exported copy whose neurons are stored in a different order. That copy
computes exactly the same function as the base, but its weights are not directly aligned with
the base or with each other.

## Architecture

The base, the task models and the merged model share the trunk shape (see `/app/mlp.py`):

```
x (48) -> ReLU(x W1 + b1) (64) -> ReLU(. W2 + b2) (64) -> ReLU(. W3 + b3) (64) = h
task model k : logits = h @ head_W + head_b                       (10 classes)
merged model : logits_k = h @ HW[k] + Hb[k]   for task k = 0..5    (one trunk, six heads)
```

## What matters

The merged model will serve the same production traffic the 64 logged inputs came from, i.e. many
more inputs from each task's real input distribution. For task k, what counts is how closely head
k matches task model k on that traffic, measured as the mean **KL(task_model_k(x) ‖ merged
head k(x))** over the softmax outputs, averaged over task k's real inputs. Lower is better.

All six tasks ship together, so the release is only as good as its weakest head. A head that is
far off its task model on its traffic (for example an untrained head, or one task model's weights
reused for every task) breaks that product feature, however good the other five heads are. A
release whose six heads are all close to their task models beats one with five excellent heads
and one poor one. Improving a head that is already better than the others helps much less than
improving the weakest one. Getting the divergence down on every task, relative to how hard that
task is, is the goal.

Tools in `/app`:

- `mlp.py`: loaders, the forward pass, the KL divergence above, a gradient for the merged model and
  Adam, all in numpy. You may install and use anything else, e.g. torch.
- `make_merged.py`: a working but weak starting point (naive weight averaging).
- `evaluate.py`: reports each task's mean KL on the 64 logged inputs. If you train on those inputs,
  this number is optimistic, so hold some out for an honest estimate of how the model will do on
  real traffic.
- `tests/test_public.py`: checks that your file loads in the deployment format.

## Deliverable

The device build pipeline loads **`/app/merged.npz`**, which must contain these float arrays:
`W1 (48,64) b1 (64,) W2 (64,64) b2 (64,) W3 (64,64) b3 (64,) HW (6,64,10) Hb (6,10)`, all finite.
The pipeline ignores any other arrays in the file and never reads them. The file itself must be
a regular file of at most 20 MB. `mlp.save_merged()` writes the right format. The pipeline cannot
ship a file with missing arrays, wrong shapes or dtypes, NaN/inf values or pickled objects, or a
file that is a symlink or larger than 20 MB.

Only the weights file ships; none of your code runs on the device. You can spend your own
session's compute on producing it.


hidden eval

#!/usr/bin/env python3
"""Hidden evaluation for fewshot_expert_consolidation.

The candidate delivers only a weights file (/app/merged.npz); no candidate code is executed.
The file is treated as untrusted input: it must be a regular file (no symlink, <= 20 MB), a zip/npz
whose REQUIRED arrays (scoring.SHAPES) are plain float arrays of the exact shapes, finite, and not
pickled.  Any other members of the archive are ignored and never read or decompressed.
Any violation -> reward 0.
"""
import json
import os
import resource
import shutil
import sys
import time
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
MAX_MEMBER = 2 * 1024 * 1024  # uncompressed bytes per REQUIRED array (largest legit one is ~31 KB)


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
        names = set(zf.namelist())
        for k in SHAPES:  # only the required members are inspected; extra members are ignored
            if k + ".npy" not in names:
                raise ValueError(f"missing array {k}")
            if zf.getinfo(k + ".npy").file_size > MAX_MEMBER:
                raise ValueError(f"array {k} too large")
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


def resources(t0):
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    return {"wall_s": round(time.time() - t0, 3), "peak_rss_mb": round(peak_kb / 1024, 1)}


def main():
    t0 = time.time()
    if os.geteuid() == 0:
        os.chmod(HERE, 0o700)
    try:
        m = load_candidate(os.path.join(APP, "merged.npz"))
    except Exception as e:  # malformed / missing deliverable -> 0
        res = resources(t0)
        write_out(0.0, {"error": f"{type(e).__name__}: {e}"[:2000], "reward": 0.0, "resources": res})
        print("invalid deliverable:", e)
        print(f"grader resources: wall {res['wall_s']} s, peak RSS {res['peak_rss_mb']} MB")
        return
    with np.load(HIDDEN, allow_pickle=False) as d:
        H = {k: d[k] for k in d.files}
    anchors = json.load(open(ANCHORS))
    raw = raw_metrics(m, H)
    reward, det = reward_from_raw(raw, anchors)
    res = resources(t0)
    write_out(reward, {"reward": reward, "R": det["R"], "R1": det["R1"], "worst_ratio": det["worst_ratio"], "ratios": det["ratios"],
                       "per_task_diagnostic": det["per_task_diagnostic"], "raw": raw, "resources": res})
    print(json.dumps({"reward": round(reward, 4), "R": round(det["R"], 4), "R1": round(det["R1"], 4),
                      "worst_ratio": round(det["worst_ratio"], 4),
                      "kl_ratio_to_zero_point": {k: round(v, 4) for k, v in det["ratios"].items()},
                      "mean_kl": {k: round(v["mean_kl"], 5) for k, v in raw.items()}, "resources": res}, indent=1))


if __name__ == "__main__":
    main()

scoring
"""Scoring math shared by hidden_eval.py, rescore.py and reference/reproduce.py.

Reward (v0.3): driven by the weakest head.
  r_k = L_k / L0_k          L_k  = mean KL(expert_k || merged head k) on task k's hidden inputs
                            L0_k = zero-point reference for task k (anchors.json)
  floor: if any r_k >= 1 (some head no better than its zero point)  ->  reward = 0
  R   = ( mean_k r_k^P )^(1/P),  P = 8      power mean = soft maximum: for 6 tasks
                                             0.80 * max_k r_k  <=  R  <=  max_k r_k
  R1  = same power mean of (0.8 * ref_k / L0_k),  ref_k = min(oracle_k, best_agent_k)
  reward = clip( log(1/R) / log(1/R1), 0, 1 )

So the reward is set almost entirely by the worst head: one head at its zero point scores 0 however
good the others are, a release of six balanced heads beats five excellent heads plus one poor
one with the same average ratio, and improving a non-worst head still helps a little (no
plateaus).  For balanced solutions (all r_k equal) R = r_k, i.e. the usual log-scale position
between the zero point and 0.8 x the reference.  Per-task scores are reported for diagnostics only.
v0.2 used the arithmetic mean of r_k, which let good heads cover a failed one.
"""
import math

import numpy as np

K = 6
P = 8  # power-mean exponent (soft maximum over the task ratios)
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


def _ref(g):
    return g["oracle"] if g.get("best_agent") is None else min(g["oracle"], g["best_agent"])


def per_task_score(L, L0, L1):
    """Diagnostic only (not used in the reward): single-task log-scale position, clipped to [0,1]."""
    if not np.isfinite(L) or L >= L0:
        return 0.0
    if L <= L1:
        return 1.0
    return float(math.log(L0 / L) / math.log(L0 / L1))


def reward_from_raw(raw, anchors):
    """Returns (reward, details).  details has R, R1, worst ratio, per-task ratios and diagnostic scores."""
    g = anchors["groups"]
    ratios, diag = {}, {}
    for k in range(K):
        t = f"task_{k}"
        L = raw[t]["mean_kl"]
        ratios[t] = L / g[t]["L0"] if np.isfinite(L) else float("inf")
        diag[t] = per_task_score(L, g[t]["L0"], 0.8 * _ref(g[t]))
    r = np.array(list(ratios.values()), dtype=np.float64)
    r1 = np.array([0.8 * _ref(g[f"task_{k}"]) / g[f"task_{k}"]["L0"] for k in range(K)])
    worst = float(r.max())
    with np.errstate(over="ignore", invalid="ignore"):
        R = float(np.mean(r ** P) ** (1.0 / P)) if np.isfinite(r).all() else float("inf")
    R1 = float(np.mean(r1 ** P) ** (1.0 / P))
    if not np.isfinite(R) or worst >= 1.0 or R >= 1.0:
        reward = 0.0
    elif R <= R1:
        reward = 1.0
    else:
        reward = float(math.log(1.0 / R) / math.log(1.0 / R1))
    return reward, {"R": R, "R1": R1, "worst_ratio": worst, "ratios": ratios, "per_task_diagnostic": diag}

test scoring
"""Checks that the reward combination keeps the promise "the release is only as good as its weakest head".

    python3 /tests/test_scoring.py      (or: pytest -q /tests/test_scoring.py)

Works on per-task KL ratios r_k = L_k / L0_k only (no models needed), using the shipped anchors.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import K, reward_from_raw  # noqa: E402

ANCH = json.load(open(os.path.join(HERE, "hidden", "anchors.json")))
L0 = np.array([ANCH["groups"][f"task_{k}"]["L0"] for k in range(K)])
REF = np.array([ANCH["groups"][f"task_{k}"]["oracle"] for k in range(K)])
R_REF = REF / L0  # reference-quality ratio of every head


def reward(r):
    return reward_from_raw({f"task_{k}": {"mean_kl": float(r[k] * L0[k])} for k in range(K)}, ANCH)[0]


def test_head_at_or_above_zero_point_gives_zero():
    for k in range(K):
        for worst in (1.0, 1.8, 50.0):
            r = R_REF.copy()
            r[k] = worst
            assert reward(r) == 0.0, (k, worst, reward(r))


def test_balanced_beats_one_poor_head_with_same_average():
    for k in range(K):
        for worst in (0.3, 0.5, 0.8, 0.99):
            r = R_REF.copy()
            r[k] = worst
            assert reward(np.full(K, r.mean())) > reward(r), (k, worst)


def test_weakest_head_dominates_but_others_still_count():
    r = np.full(K, 0.4)
    better_other = r.copy()
    better_other[0] = 0.2  # improve a non-worst head
    worse_one = r.copy()
    worse_one[0] = 0.6  # degrade one head by the same factor
    assert reward(better_other) > reward(r)  # no plateau
    assert reward(r) - reward(worse_one) > 2 * (reward(better_other) - reward(r))  # worst head weighs much more


def test_balanced_solutions_keep_order_and_reference_scores_high():
    vals = [reward(np.full(K, x)) for x in (0.9, 0.6, 0.4, 0.2, 0.1)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert reward(R_REF) > 0.9
    assert reward(np.ones(K)) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("scoring promise tests passed")

anchors
{
 "_doc": "Per-task anchors on mean KL(expert||merged) over 3000 hidden inputs. L0 (zero point) = distillation on the 64 visible inputs + N(0,0.3^2) input noise, 8k Adam steps; oracle = same distillation on unlimited fresh samples of the true input distributions, 300k steps (code: tests/reference/). reward = clip(log(1/R)/log(1/R1),0,1) with R = power mean (p=8, a soft maximum) of r_k = L_k/L0_k and R1 the same power mean of 0.8*min(oracle_k,best_agent_k)/L0_k; reward = 0 if any r_k >= 1 (see scoring.py).",
 "groups": {
  "task_0": {
   "L0": 0.032004,
   "oracle": 0.002539,
   "best_agent": null
  },
  "task_1": {
   "L0": 0.037234,
   "oracle": 0.003952,
   "best_agent": null
  },
  "task_2": {
   "L0": 0.1174,
   "oracle": 0.009772,
   "best_agent": null
  },
  "task_3": {
   "L0": 0.065318,
   "oracle": 0.005621,
   "best_agent": null
  },
  "task_4": {
   "L0": 0.033183,
   "oracle": 0.002945,
   "best_agent": null
  },
  "task_5": {
   "L0": 0.049341,
   "oracle": 0.003428,
   "best_agent": null
  }
 },
 "_expected_oracle_reward": 0.914932
}
