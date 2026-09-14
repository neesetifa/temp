#!/usr/bin/env python3
"""Author reference submission builder for compact-next-basket-oe v0.4.

This is a valid, budget-compliant reference implementation used to exercise the
submission path.  It is not an oracle, not the calibration anchor, and not a
claimed optimum.  It learns a compact top-5 item-association table from the
visible training customers and writes the inference artifact to /output.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import struct
from collections import Counter, defaultdict

ARTIFACT_LIMIT = 65_536
K = 5
SENTINEL = 65_535


def parse_history(text: str) -> list[list[int]]:
    return [
        [int(x) for x in basket.split()]
        for basket in str(text).split("|")
        if basket.strip()
    ]


def build(train_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    source_count: Counter[int] = Counter()
    target_count: Counter[int] = Counter()
    assoc: dict[int, Counter[int]] = defaultdict(Counter)
    all_items: set[int] = set()

    with train_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["query_id", "history", "gaps_days", "target_items"]
        if reader.fieldnames != expected:
            raise SystemExit(f"unexpected train schema: {reader.fieldnames}")
        for row in reader:
            seen = {x for basket in parse_history(row["history"]) for x in basket}
            target = {int(x) for x in row["target_items"].split() if x}
            if not seen or not target:
                continue
            all_items.update(seen)
            all_items.update(target)
            source_count.update(seen)
            target_count.update(target)
            for src in seen:
                counts = assoc[src]
                for dst in target:
                    counts[dst] += 1

    if not all_items:
        raise SystemExit("training data produced no items")
    nslots = max(all_items) + 1
    if nslots >= SENTINEL:
        raise SystemExit("reference solution assumes uint16 remapped item ids")

    # Stable global fallback: target frequency, then source frequency, then id.
    popularity = sorted(
        all_items,
        key=lambda item: (-target_count[item], -source_count[item], item),
    )

    table = [[SENTINEL] * K for _ in range(nslots)]
    for src, counts in assoc.items():
        ranked: list[tuple[float, int]] = []
        src_n = max(1, source_count[src])
        for dst, joint_n in counts.items():
            dst_n = max(1, target_count[dst])
            score = joint_n / math.sqrt(src_n * dst_n)
            ranked.append((score, dst))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        for j, (_, dst) in enumerate(ranked[:K]):
            table[src][j] = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_dir():
            import shutil
            shutil.rmtree(old)
        else:
            old.unlink()

    model_path = output_dir / "model.bin"
    with model_path.open("wb") as f:
        f.write(struct.pack("<HH", nslots, len(popularity)))
        f.write(struct.pack("<" + "H" * len(popularity), *popularity))
        flat = [item for row in table for item in row]
        f.write(struct.pack("<" + "H" * len(flat), *flat))

    predictor = f'''#!/usr/bin/env python3
import csv, pathlib, struct, sys
from collections import Counter
K={K}; SENT={SENTINEL}
HERE=pathlib.Path(__file__).resolve().parent
blob=(HERE/'model.bin').read_bytes()
n,m=struct.unpack_from('<HH',blob,0); off=4
pop=struct.unpack_from('<'+'H'*m,blob,off); off+=2*m
assoc=struct.unpack_from('<'+'H'*(n*K),blob,off)
def parse_hist(s):
    return [[int(x) for x in b.split()] for b in str(s).split('|') if b.strip()]
def main():
    if len(sys.argv)!=3: raise SystemExit('usage: predict.py <in.csv> <out.csv>')
    with open(sys.argv[1],newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['query_id','history','gaps_days']: raise SystemExit('bad input schema')
        rows=list(r)
    answers=[]
    for row in rows:
        seen={{x for basket in parse_hist(row['history']) for x in basket}}
        votes=Counter()
        for src in sorted(seen):
            if 0<=src<n:
                base=src*K
                for rank in range(K):
                    dst=assoc[base+rank]
                    if dst!=SENT and dst not in seen:
                        votes[dst]+=K-rank
        out=[item for item,_ in sorted(votes.items(),key=lambda kv:(-kv[1],kv[0]))[:10]]
        have=set(out)
        if len(out)<10:
            for item in pop:
                if item not in seen and item not in have:
                    out.append(item); have.add(item)
                    if len(out)==10: break
        if len(out)!=10: raise SystemExit('could not form ten recommendations')
        answers.append(' '.join(map(str,out)))
    with open(sys.argv[2],'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['prediction']); w.writerows([[x] for x in answers])
if __name__=='__main__': main()
'''
    pred_path = output_dir / "predict.py"
    pred_path.write_text(predictor, encoding="utf-8")
    pred_path.chmod(0o755)

    artifact_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    if artifact_bytes > ARTIFACT_LIMIT:
        raise SystemExit(
            f"reference artifact exceeds budget: {artifact_bytes} > {ARTIFACT_LIMIT}"
        )
    print(f"reference artifact: {artifact_bytes} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/output"))
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve()
    task_root = here.parents[1]
    train = args.train
    if train is None:
        if pathlib.Path("/app/train.csv").exists():
            train = pathlib.Path("/app/train.csv")
        else:
            train = task_root / "environment/data/train.csv"
    if not train.exists():
        raise SystemExit(f"training data not found: {train}")
    build(train, args.output)


if __name__ == "__main__":
    main()


sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /output
python "$SCRIPT_DIR/reference_solution.py" --output /output
