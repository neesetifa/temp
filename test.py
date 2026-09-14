# Compact novel next-basket recommendation

`/app/train.csv` contains real purchase histories derived from UCI Online Retail II. Each row is one training customer: an ordered basket history plus the items in that customer's next basket that had **not appeared anywhere in the visible history**.

Build `/output/predict.py`.

It will be called as:

```bash
python /output/predict.py <in.csv> <out.csv>
```

`<in.csv>` has these columns:

```text
query_id,history,gaps_days
```

`history` is a sequence of baskets. Baskets are separated by `|`; item IDs inside a basket are separated by spaces. The last basket in `history` is the most recent observed basket. `gaps_days` contains one non-negative day gap per history basket, in the same order.

Write `<out.csv>` with exactly one column named `prediction`. Each row must contain exactly **10 distinct item IDs**, separated by single spaces, best recommendation first. Output rows must correspond to input rows in the same order.

The held-out label contains only **novel** next-basket items: items already present in that customer's visible history receive no credit. Held-out customers with no novel item in their next basket are not part of evaluation.

The complete `/output` artifact is limited to **64 KiB (65,536 bytes)**. This includes `predict.py` and every model, table, index, or other file under `/output`. Symlinks and other special files are not allowed.

The held-out customers are disjoint from the customers in `/app/train.csv`. Their visible histories are supplied only through `<in.csv>` at evaluation time. The predictor cannot access `/app/train.csv` during final evaluation, so any cross-customer statistics or learned model state needed at inference time must fit inside `/output`.

Training columns are:

```text
query_id,history,gaps_days,target_items
```

`target_items` contains only the next-basket items that are novel relative to that training customer's visible history. IDs are opaque remapped identifiers; do not infer meaning from their numeric values.

The score is based on how highly hidden novel next-basket items appear in each held-out customer's top-10 ranking. A train-derived global novel-item popularity recommender, filtered to remove items already seen by each customer, is the score floor. The useful part of the task is deciding which cross-customer item relationships or compressed representations are worth encoding under the artifact-size limit.

Keep the submission deterministic. Reordering input rows must only reorder the corresponding outputs.


predict

#!/usr/bin/env python3
"""Author-only unseen-popularity smoke baseline; not an oracle/reference/optimum."""
from __future__ import annotations

import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
POPULAR = HERE / 'popular_items.txt'


def parse_history(s):
    return [[x for x in b.split() if x] for b in str(s).split('|') if str(b).strip()]


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: predict.py <in.csv> <out.csv>')
    items = POPULAR.read_text(encoding='utf-8').strip().split()
    if len(items) < 10:
        raise SystemExit('popular_items.txt missing/invalid')
    with open(sys.argv[1], newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames != ['query_id', 'history', 'gaps_days']:
            raise SystemExit('bad input schema')
        rows = list(r)
    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['prediction'])
        for row in rows:
            seen = {x for b in parse_history(row['history']) for x in b}
            pred = []
            for x in items:
                if x not in seen:
                    pred.append(x)
                    if len(pred) == 10:
                        break
            if len(pred) != 10:
                raise SystemExit('not enough unseen popularity items')
            w.writerow([' '.join(pred)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


verifier
from __future__ import annotations
import contextlib,csv,math,os,pathlib,shutil,subprocess,tempfile
ARTIFACT_LIMIT=65_536
OUTPUT_DIR=pathlib.Path('/output'); PREDICT=OUTPUT_DIR/'predict.py'; TEST_DIR=pathlib.Path(__file__).resolve().parent; CATALOG_PATH=TEST_DIR/'catalog.csv'

def artifact_size_and_validate(root=OUTPUT_DIR):
    if not root.exists() or not root.is_dir(): raise AssertionError('/output directory missing')
    total=0
    for p in root.rglob('*'):
        if p.is_symlink(): raise AssertionError(f'symlink not allowed in /output: {p}')
        if p.is_dir(): continue
        if not p.is_file(): raise AssertionError(f'special file not allowed in /output: {p}')
        total += p.stat().st_size
    if total>ARTIFACT_LIMIT: raise AssertionError(f'artifact too large: {total} > {ARTIFACT_LIMIT} bytes')
    return total

def load_catalog():
    with CATALOG_PATH.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames != ['item_id']: raise AssertionError('invalid verifier catalog')
        out={str(row['item_id']).strip() for row in r}
    if len(out)<10: raise AssertionError('catalog too small')
    return out

@contextlib.contextmanager
def _protected_verifier_paths():
    paths=[pathlib.Path('/tests'),pathlib.Path('/app'),pathlib.Path('/output'),pathlib.Path('/logs/verifier')]; modes=[]
    for p in paths:
        try:
            if p.exists(): modes.append((p,p.stat().st_mode & 0o7777)); p.chmod(0o700)
        except OSError: pass
    try: yield
    finally:
        for p,m in reversed(modes):
            try:p.chmod(m)
            except OSError:pass

def _chown_tree(root,uid,gid):
    for p in [root,*root.rglob('*')]:
        try:os.chown(p,uid,gid)
        except OSError:pass

def run_predict(input_csv,output_csv,timeout=60):
    artifact_size_and_validate()
    if not PREDICT.exists() or not PREDICT.is_file() or PREDICT.stat().st_size==0: raise AssertionError('/output/predict.py missing or empty')
    sandbox=pathlib.Path(tempfile.mkdtemp(prefix='oe_exec_',dir='/var/tmp' if pathlib.Path('/var/tmp').exists() else None)); artifact=sandbox/'artifact'
    shutil.copytree(OUTPUT_DIR,artifact,symlinks=True); sandbox_in=sandbox/'input.csv'; sandbox_out=sandbox/'result.csv'; shutil.copy2(input_csv,sandbox_in)
    home=sandbox/'home'; tmp=sandbox/'tmp'; home.mkdir(); tmp.mkdir(); preexec=None
    if os.geteuid()==0:
        try:
            import pwd; nobody=pwd.getpwnam('nobody'); uid,gid=nobody.pw_uid,nobody.pw_gid
        except Exception: uid=gid=65534
        _chown_tree(sandbox,uid,gid)
        def drop(): os.setgroups([]); os.setgid(gid); os.setuid(uid)
        preexec=drop
    env={'PATH':os.environ.get('PATH','/usr/local/bin:/usr/bin:/bin'),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','HOME':str(home),'TMPDIR':str(tmp),'XDG_CACHE_HOME':str(home/'.cache'),'PYTHONHASHSEED':'0'}
    try:
        with _protected_verifier_paths():
            cp=subprocess.run(['python',str(artifact/'predict.py'),str(sandbox_in),str(sandbox_out)],cwd=str(artifact),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout,preexec_fn=preexec)
        if cp.returncode!=0: raise AssertionError(f'predict.py failed with code {cp.returncode}: {(cp.stderr or cp.stdout)[-4000:]}')
        if not sandbox_out.exists() or sandbox_out.is_symlink() or not sandbox_out.is_file(): raise AssertionError('submission did not create regular output CSV')
        shutil.copy2(sandbox_out,output_csv)
    finally: shutil.rmtree(sandbox,ignore_errors=True)

def parse_predictions(output_csv,expected_rows,catalog):
    with output_csv.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames != ['prediction']: raise AssertionError('output must have exactly one column named prediction')
        rows=list(r)
    if len(rows)!=expected_rows: raise AssertionError(f'wrong row count: {len(rows)} != {expected_rows}')
    out=[]
    for i,row in enumerate(rows):
        toks=str(row['prediction']).strip().split()
        if len(toks)!=10: raise AssertionError(f'row {i}: exactly 10 item IDs required')
        if len(set(toks))!=10: raise AssertionError(f'row {i}: item IDs must be distinct')
        bad=[x for x in toks if x not in catalog]
        if bad: raise AssertionError(f'row {i}: unknown item IDs {bad[:3]}')
        out.append(tuple(toks))
    return out

def write_input(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query_id','history','gaps_days'])
        for r in rows:w.writerow([r[0],r[1],r[2]])

def ndcg_at_10(pred,target):
    if not target:return 0.0
    dcg=sum(1/math.log2(rank+1) for rank,item in enumerate(pred,1) if item in target); k=min(10,len(target)); idcg=sum(1/math.log2(rank+1) for rank in range(1,k+1)); return dcg/idcg if idcg else 0.0

test main
from __future__ import annotations

import csv
import json
import math
import pathlib
import resource
import tempfile

from verifier_utils import artifact_size_and_validate, load_catalog, ndcg_at_10, parse_predictions, run_predict, write_input

TEST_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR = pathlib.Path('/logs/verifier')


def _load_cases():
    out = []
    with (TEST_DIR / 'test.csv').open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        assert r.fieldnames == ['query_id', 'history', 'gaps_days', 'target_items'], f'bad test schema: {r.fieldnames}'
        for row in r:
            target = {x for x in row['target_items'].split() if x}
            if not target:
                continue
            seen = {x for basket in row['history'].split('|') for x in basket.split() if x}
            assert target.isdisjoint(seen), 'v0.4 verifier target contains a repeated history item'
            out.append((row['query_id'], row['history'], row['gaps_days'], target))
    assert out, 'empty test.csv'
    return out


def _score(preds, cases):
    vals = [ndcg_at_10(p, c[3]) for p, c in zip(preds, cases)]
    hits = [1.0 if any(x in c[3] for x in p) else 0.0 for p, c in zip(preds, cases)]
    return sum(vals) / len(vals), sum(hits) / len(hits)


def test_reward():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    artifact_bytes = artifact_size_and_validate()
    catalog = load_catalog()
    cases = _load_cases()
    config = json.loads((TEST_DIR / 'scoring_config.json').read_text())
    floor = float(config['unseen_popularity_ndcg_at_10'])
    anchor = float(config['calibration_anchor_ndcg_at_10'])
    anchor_reward = float(config['calibration_anchor_reward'])
    assert config['novel_target_only'] is True
    assert 0.0 <= floor < anchor <= 1.0, 'invalid reward anchors'
    assert 0.0 < anchor_reward < 1.0, 'invalid calibration anchor reward'

    rows = [(c[0], c[1], c[2]) for c in cases]
    with tempfile.TemporaryDirectory(prefix='oe_main_') as td:
        td = pathlib.Path(td)
        p_in = td / 'in.csv'; p_out = td / 'out.csv'
        p_rev = td / 'rev.csv'; p_rev_out = td / 'rev_out.csv'
        write_input(p_in, rows)
        run_predict(p_in, p_out, timeout=60)
        preds = parse_predictions(p_out, len(cases), catalog)
        write_input(p_rev, list(reversed(rows)))
        run_predict(p_rev, p_rev_out, timeout=60)
        rp = parse_predictions(p_rev_out, len(cases), catalog)
        assert rp == list(reversed(preds)), 'predictions depend on input row position'

    raw, hit = _score(preds, cases)
    z = max(0.0, (raw - floor) / max(1e-12, anchor - floor))
    reward = 1.0 - (1.0 - anchor_reward) ** z
    reward = max(0.0, min(1.0, reward))
    assert math.isfinite(reward)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    metrics = {
        'artifact_bytes': artifact_bytes,
        'artifact_limit_bytes': int(config['artifact_limit_bytes']),
        'calibration_anchor_ndcg_at_10': anchor,
        'calibration_anchor_reward': anchor_reward,
        'catalog_size': len(catalog),
        'customer_disjoint_eval': True,
        'evaluated_customers': len(cases),
        'hit_rate_at_10': hit,
        'ndcg_at_10': raw,
        'novel_target_only': True,
        'peak_child_memory_mb': peak,
        'reward': reward,
        'reward_gap_units': z,
        'unseen_popularity_ndcg_at_10': floor,
    }
    (LOG_DIR / 'reward.txt').write_text(f'{reward:.12f}\n')
    (LOG_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')


finalize
from __future__ import annotations
import json
import math
import pathlib
import time

LOG = pathlib.Path("/logs/verifier")
LOG.mkdir(parents=True, exist_ok=True)
reward_path = LOG / "reward.txt"
metrics_path = LOG / "metrics.json"

failure = "none"
reward = 0.0
metrics = {}
try:
    reward = float(reward_path.read_text(encoding="utf-8").strip())
    if not math.isfinite(reward) or not (0.0 <= reward <= 1.0):
        raise ValueError("reward must be finite and in [0,1]")
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception as e:
    failure = f"verifier_failure: {type(e).__name__}: {e}"
    reward = 0.0
    reward_path.write_text("0.0\n", encoding="utf-8")
    metrics = {"reward": 0.0}
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

(LOG / "failure_mode.txt").write_text(failure + "\n", encoding="utf-8")
(LOG / "ctrf.json").write_text(json.dumps({"reward": reward}, indent=2) + "\n", encoding="utf-8")
start_path = LOG / ".start_time"
try:
    start = float(start_path.read_text().strip())
    wall = max(0.0, time.time() - start)
except Exception:
    wall = 0.0
(LOG / "wall_clock_sec.txt").write_text(f"{wall:.6f}\n", encoding="utf-8")
peak = float(metrics.get("peak_child_memory_mb", 0.0) or 0.0)
(LOG / "peak_memory_mb.txt").write_text(f"{peak:.3f}\n", encoding="utf-8")

test smoke
from pathlib import Path


def test_predict_exists():
    p = Path("/output/predict.py")
    assert p.exists() and p.is_file() and not p.is_symlink() and p.stat().st_size > 0

schema
from __future__ import annotations
import csv,pathlib,tempfile
from verifier_utils import artifact_size_and_validate,load_catalog,parse_predictions,run_predict,write_input
TEST_DIR=pathlib.Path(__file__).resolve().parent

def _rows(n=4):
    vals=[]
    with (TEST_DIR/'test.csv').open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        for _,row in zip(range(n),r): vals.append((row['query_id'],row['history'],row['gaps_days']))
    assert len(vals)>=3; return vals

def test_schema_determinism_and_order():
    artifact_size_and_validate(); catalog=load_catalog(); rows=_rows()
    with tempfile.TemporaryDirectory(prefix='oe_schema_') as td:
        td=pathlib.Path(td); p=td/'in.csv'; o1=td/'o1.csv'; o2=td/'o2.csv'
        write_input(p,rows); run_predict(p,o1,60); run_predict(p,o2,60); a=parse_predictions(o1,len(rows),catalog); b=parse_predictions(o2,len(rows),catalog); assert a==b,'submission is not deterministic'
