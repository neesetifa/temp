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

There are no labels and no other data. Each task's input distribution over R^48 is unknown (some
tasks may share the same input distribution). The 64 samples per task are the only data you have
from it, but you can run the experts (and the base) on any input you like.

`mlp.py` has loaders, the forward pass, the KL loss, a gradient for the merged model and Adam, all
in numpy. You may install and use anything else, e.g. torch. `make_merged.py` is a working but weak
starter (naive weight averaging). `evaluate.py` reports, on the 64 visible inputs, the
quantity the grader measures: each task's mean KL(expert ‖ merged head), not the reward itself
(the reward also needs the hidden reference values). That number is optimistic if you train on
those inputs, so hold some out for an honest estimate. `tests/test_public.py` checks the deliverable format.

## Deliverable

Write **`/app/merged.npz`** containing these float arrays:
`W1 (48,64) b1 (64,) W2 (64,64) b2 (64,) W3 (64,64) b3 (64,) HW (6,64,10) Hb (6,10)`, all finite.
Any other arrays in the file are ignored (never read). The file as a whole must be a regular file
of at most 20 MB. `mlp.save_merged()` writes the right format. Only this file is graded; none of
your code is run at grading time, so you can spend your own session's compute on producing it.

## Grading

For each task k, the grader uses a fixed hidden set of 3,000 inputs drawn i.i.d. from task k's
true input distribution (the same distribution your 64 samples came from, disjoint from them) and
computes the mean
**L_k = KL(expert_k(x) ‖ merged head k(x))** over the softmax outputs.

Two hidden references fix the scale. Both run the same procedure: start from the base with the
experts' hidden units aligned to it and their task vectors added (scaled), then distill all six
experts into the merged model with Adam. They differ only in the inputs used for distillation:

- **zero point** (L0_k): your 64 visible inputs per task plus Gaussian input noise (σ = 0.3),
  8,000 steps;
- **top reference** (ref_k): unlimited fresh samples from each task's *true* input distribution,
  300,000 steps. That distribution is the only thing it knows that you don't. It has no labels,
  no expert training data and no other information, and it uses the same base, experts and
  64 inputs you have.

The tasks are combined so that every head matters. Let r_k = L_k / L0_k and R = mean_k r_k (an
arithmetic mean, so one bad head dominates). Then

    reward = clip( log(1/R) / log(1/R1), 0, 1 ),   R1 = mean_k (0.8 · ref_k / L0_k)

So the reward is 0 at the zero point or worse, and 1 when you are clearly better than the top
reference. One badly broken head (for example an untrained or zeroed head, or a single expert
copied for all tasks) pushes R far above 1 and the reward to 0. The reward is continuous; there is
no pass/fail. A missing or malformed file (missing required arrays, wrong shapes or dtypes,
NaN/inf, pickled objects, a symlink, more than 20 MB) gets reward 0.

  
  rescore.py
  #!/usr/bin/env python3
"""Offline re-scoring of finished runs with (possibly updated) anchors -- no re-running needed.

    python3 rescore.py metrics1.json [metrics2.json ...] [--anchors anchors.json]
                       [--set-best-agent --write-anchors NEW.json]

--set-best-agent puts the per-task best observed agent mean_kl into the anchors used for this rescore.
The anchors file given by --anchors is never modified; the updated anchors are written only to the
path given by --write-anchors (which must differ from --anchors).

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
    ap.add_argument("--write-anchors", default=None, help="where to save anchors updated by --set-best-agent")
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
        if a.write_anchors:
            if os.path.abspath(a.write_anchors) == os.path.abspath(a.anchors):
                sys.exit("--write-anchors must be a different file from --anchors (refusing to overwrite)")
            json.dump(anchors, open(a.write_anchors, "w"), indent=1)
            print(f"wrote updated anchors to {a.write_anchors} ({a.anchors} unchanged)")
        else:
            print("best_agent applied for this rescore only (pass --write-anchors NEW.json to save)")
    for p, raw in runs:
        r, det = reward_from_raw(raw, anchors)
        print(f"{r:.4f}  {p}  R={det['R']:.4f}  ratios: " + " ".join(f"{v:.3f}" for v in det["ratios"].values()))


if __name__ == "__main__":
    main()

test.sh
#!/bin/bash
# Verifier entry point: scores /app/merged.npz on the hidden inputs and writes
# /logs/verifier/reward.txt (+ metrics.json, which also records grader wall time and peak memory).
set -u
mkdir -p /logs/verifier
chmod 700 /tests 2>/dev/null || true
START=$(date +%s.%N)
python3 /tests/hidden_eval.py
STATUS=$?
END=$(date +%s.%N)
echo "test.sh: hidden_eval exit status ${STATUS}, wall-clock $(python3 -c "print(f'{$END-$START:.2f}')") s" \
  | tee /logs/verifier/test_sh_timing.txt
if [ ! -s /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
exit 0

hidden
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
    write_out(reward, {"reward": reward, "R": det["R"], "R1": det["R1"], "ratios": det["ratios"],
                       "per_task_diagnostic": det["per_task_diagnostic"], "raw": raw, "resources": res})
    print(json.dumps({"reward": round(reward, 4), "R": round(det["R"], 4), "R1": round(det["R1"], 4),
                      "kl_ratio_to_zero_point": {k: round(v, 4) for k, v in det["ratios"].items()},
                      "mean_kl": {k: round(v["mean_kl"], 5) for k, v in raw.items()}, "resources": res}, indent=1))


if __name__ == "__main__":
    main()

