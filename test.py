os
import pathlib
import random
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"
ARTIFACT_LIMIT = 65_536


def parse_args():
    p = argparse.ArgumentParser(description="Materialize compact-next-basket-oe from UCI Online Retail II")
    p.add_argument("--raw", type=pathlib.Path, help="Path to online_retail_II.xlsx (or the UCI zip). If omitted, download from UCI.")
    p.add_argument("--task-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    p.add_argument("--secret", default=None, help="Private remapping secret. If omitted, a random secret is generated and NOT written to the task.")
    p.add_argument("--min-user-orders", type=int, default=3)
    p.add_argument("--min-item-train-orders", type=int, default=5)
    p.add_argument("--max-users", type=int, default=0, help="Optional development cap after eligibility filtering; 0 keeps all users.")
    p.add_argument("--make-zip", action="store_true", help="Also write a final task zip excluding author_tools.")
    return p.parse_args()


def _download_source(dst_dir: pathlib.Path) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zpath = dst_dir / "online_retail_ii.zip"
    print(f"Downloading {UCI_ZIP} ...")
    urllib.request.urlretrieve(UCI_ZIP, zpath)
    return zpath


def _resolve_xlsx(raw: pathlib.Path | None, work: pathlib.Path) -> pathlib.Path:
    if raw is None:
        raw = _download_source(work)
    raw = raw.expanduser().resolve()
    if not raw.exists():
        raise FileNotFoundError(raw)
    if raw.suffix.lower() == ".xlsx":
        return raw
    if raw.suffix.lower() == ".zip":
        with zipfile.ZipFile(raw) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            if not names:
                raise RuntimeError("zip does not contain an xlsx file")
            name = names[0]
            out = work / pathlib.Path(name).name
            with zf.open(name) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return out
    raise ValueError("--raw must be .xlsx or .zip")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canon = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    aliases = {
        "invoice": ["invoice", "invoice no", "invoiceno"],
        "stock": ["stockcode", "stock code"],
        "quantity": ["quantity"],
        "date": ["invoicedate", "invoice date"],
        "price": ["price", "unitprice", "unit price"],
        "customer": ["customer id", "customerid"],
    }
    chosen = {}
    for out, names in aliases.items():
        for n in names:
            key = n.strip().lower().replace("_", " ")
            if key in canon:
                chosen[out] = canon[key]
                break
        if out not in chosen:
            raise KeyError(f"could not find required column for {out}; columns={list(df.columns)}")
    return df[[chosen[k] for k in ["invoice", "stock", "quantity", "date", "price", "customer"]]].rename(
        columns={v: k for k, v in chosen.items()}
    )


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(raw).copy()
    df = df.dropna(subset=["invoice", "stock", "quantity", "date", "price", "customer"])
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock"] = df["stock"].astype(str).str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["quantity", "price", "date"])
    df = df[~df["invoice"].str.upper().str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    # Keep merchandise-like stock codes and remove administrative lines such as POST/M/DOT.
    df = df[df["stock"].str.fullmatch(r"\d{4,6}[A-Z]?")]
    # Normalize CustomerID so Excel's 13085.0 and textual 13085 agree.
    cnum = pd.to_numeric(df["customer"], errors="coerce")
    df = df[cnum.notna()].copy()
    df["customer"] = cnum[cnum.notna()].astype("int64").astype(str).to_numpy()
    # Aggregate duplicate line items inside an invoice.
    df = (
        df.groupby(["customer", "invoice", "date", "stock"], as_index=False, sort=False)["quantity"]
        .sum()
        .sort_values(["customer", "date", "invoice", "stock"], kind="stable")
    )
    return df


def _make_split(df: pd.DataFrame, min_user_orders: int, min_item_train_orders: int, max_users: int, rng: random.Random):
    inv_meta = df[["customer", "invoice", "date"]].drop_duplicates()
    inv_meta = inv_meta.sort_values(["customer", "date", "invoice"], kind="stable")
    counts = inv_meta.groupby("customer")["invoice"].nunique()
    eligible = set(counts[counts >= min_user_orders].index.astype(str))
    df = df[df["customer"].isin(eligible)].copy()
    inv_meta = inv_meta[inv_meta["customer"].isin(eligible)].copy()

    if max_users and len(eligible) > max_users:
        keep = sorted(eligible)
        rng.shuffle(keep)
        keep = set(keep[:max_users])
        df = df[df["customer"].isin(keep)].copy()
        inv_meta = inv_meta[inv_meta["customer"].isin(keep)].copy()

    last_inv = inv_meta.groupby("customer", sort=False).tail(1)[["customer", "invoice"]]
    last_keys = set(map(tuple, last_inv.astype(str).to_numpy()))
    key_pairs = list(zip(df["customer"].astype(str), df["invoice"].astype(str)))
    is_target = np.fromiter((k in last_keys for k in key_pairs), dtype=bool, count=len(key_pairs))
    train = df.loc[~is_target].copy()
    target = df.loc[is_target].copy()

    # Define a catalog from merchandise that occurs in enough distinct visible training orders.
    item_orders = train[["stock", "invoice"]].drop_duplicates().groupby("stock")["invoice"].nunique()
    catalog = set(item_orders[item_orders >= min_item_train_orders].index.astype(str))
    train = train[train["stock"].isin(catalog)].copy()
    target = target[target["stock"].isin(catalog)].copy()

    # Drop customers with no evaluable hidden target or fewer than two non-empty visible orders after catalog filtering.
    train_order_counts = train[["customer", "invoice"]].drop_duplicates().groupby("customer")["invoice"].nunique()
    target_counts = target.groupby("customer")["stock"].nunique()
    keep_users = set(train_order_counts[train_order_counts >= 2].index.astype(str)) & set(target_counts[target_counts >= 1].index.astype(str))
    train = train[train["customer"].isin(keep_users)].copy()
    target = target[target["customer"].isin(keep_users)].copy()

    # Recompute catalog after dropping users; keep >=10 items invariant.
    live_items = sorted(set(train["stock"].astype(str)))
    if len(live_items) < 10:
        raise RuntimeError("catalog collapsed below 10 items")
    target = target[target["stock"].isin(live_items)].copy()
    target_users = set(target["customer"].astype(str))
    train = train[train["customer"].isin(target_users)].copy()
    return train, target, live_items


def _dense_private_maps(customers: list[str], items: list[str], secret: str):
    seed = int.from_bytes(hashlib.sha256(secret.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    cs = sorted(customers)
    it = sorted(items)
    rng.shuffle(cs)
    rng.shuffle(it)
    return {c: i for i, c in enumerate(cs)}, {x: i for i, x in enumerate(it)}


def _materialize_train(train: pd.DataFrame, c_map: dict[str, int], i_map: dict[str, int]) -> pd.DataFrame:
    out_rows = []
    for customer, g in train.groupby("customer", sort=False):
        orders = (
            g[["invoice", "date"]].drop_duplicates().sort_values(["date", "invoice"], kind="stable").reset_index(drop=True)
        )
        if len(orders) < 2:
            continue
        first_date = orders.loc[0, "date"]
        prev_date = first_date
        order_info = {}
        for idx, row in orders.iterrows():
            date = row["date"]
            order_info[str(row["invoice"])] = (
                int(idx),
                max(0, int((date - first_date).total_seconds() // 86400)),
                0 if idx == 0 else max(0, int((date - prev_date).total_seconds() // 86400)),
            )
            prev_date = date
        agg = g.groupby(["invoice", "stock"], as_index=False)["quantity"].sum()
        for row in agg.itertuples(index=False):
            oi, dsf, dsp = order_info[str(row.invoice)]
            out_rows.append(
                (
                    c_map[str(customer)],
                    oi,
                    i_map[str(row.stock)],
                    min(99, max(1, int(round(float(row.quantity))))),
                    dsf,
                    dsp,
                )
            )
    return pd.DataFrame(
        out_rows,
        columns=["customer_id", "order_index", "item_id", "quantity", "days_since_first", "days_since_prev"],
    ).sort_values(["customer_id", "order_index", "item_id"], kind="stable")


def _materialize_test(target: pd.DataFrame, c_map: dict[str, int], i_map: dict[str, int]) -> pd.DataFrame:
    rows = []
    for customer, g in target.groupby("customer", sort=False):
        if str(customer) not in c_map:
            continue
        items = sorted({i_map[str(x)] for x in g["stock"] if str(x) in i_map})
        if items:
            rows.append((c_map[str(customer)], " ".join(map(str, items))))
    return pd.DataFrame(rows, columns=["customer_id", "target_items"]).sort_values("customer_id")


def _ndcg(pred: list[str], target: set[str]) -> float:
    if not target:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r, x in enumerate(pred, start=1) if x in target)
    k = min(10, len(target))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, k + 1))
    return dcg / idcg


def _popularity(train_out: pd.DataFrame, test_out: pd.DataFrame):
    # Basket frequency, not purchased quantity, to avoid wholesale quantities dominating.
    bf = train_out[["customer_id", "order_index", "item_id"]].drop_duplicates()["item_id"].value_counts()
    top = [str(int(x)) for x in bf.index[:10]]
    vals = []
    for t in test_out["target_items"]:
        vals.append(_ndcg(top, set(str(t).split())))
    return top, float(np.mean(vals))


def _zip_final(root: pathlib.Path):
    out = root.parent / f"{root.name.replace('_source', '')}.zip"
    allowed = ["baseline", "environment", "tests", "README.md", "instruction.md", "task.toml"]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in allowed:
            p = root / name
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(root))
            elif p.is_file():
                zf.write(p, p.relative_to(root))
    print(f"Wrote final zip: {out}")
    return out


def main():
    a = parse_args()
    root = a.task_root.resolve()
    secret = a.secret or os.urandom(24).hex()
    # Deliberately do not persist the secret. Materialized IDs are enough for the task.
    rng = random.Random(int.from_bytes(hashlib.sha256((secret + "subset").encode()).digest()[:8], "big"))

    with tempfile.TemporaryDirectory(prefix="retail_oe_build_") as td:
        xlsx = _resolve_xlsx(a.raw, pathlib.Path(td))
        sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
        raw = pd.concat(list(sheets.values()), ignore_index=True)
    source_rows = len(raw)
    clean = _clean(raw)
    train, target, items = _make_split(clean, a.min_user_orders, a.min_item_train_orders, a.max_users, rng)
    customers = sorted(set(train["customer"].astype(str)) & set(target["customer"].astype(str)))
    c_map, i_map = _dense_private_maps(customers, items, secret)
    train = train[train["customer"].isin(customers) & train["stock"].isin(i_map)].copy()
    target = target[target["customer"].isin(customers) & target["stock"].isin(i_map)].copy()

    train_out = _materialize_train(train, c_map, i_map)
    test_out = _materialize_test(target, c_map, i_map)
    # Keep only customers that survived materialization in both files.
    test_users = set(test_out["customer_id"].astype(int))
    train_out = train_out[train_out["customer_id"].isin(test_users)].copy()
    if len(test_out) < 200:
        raise RuntimeError(f"too few evaluation customers after filtering: {len(test_out)}")

    catalog_ids = sorted(train_out["item_id"].unique().astype(int).tolist())
    if len(catalog_ids) < 50:
        raise RuntimeError(f"too few catalog items after filtering: {len(catalog_ids)}")
    # Remove target items not present in final train catalog and any empty cases.
    catset = set(catalog_ids)
    cleaned_test = []
    for row in test_out.itertuples(index=False):
        vals = [int(x) for x in str(row.target_items).split() if int(x) in catset]
        if vals:
            cleaned_test.append((int(row.customer_id), " ".join(map(str, sorted(set(vals))))))
    test_out = pd.DataFrame(cleaned_test, columns=["customer_id", "target_items"]).sort_values("customer_id")
    keep = set(test_out["customer_id"].astype(int))
    train_out = train_out[train_out["customer_id"].isin(keep)].copy()

    top10, floor = _popularity(train_out, test_out)

    (root / "environment/data").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "baseline").mkdir(parents=True, exist_ok=True)
    train_out.to_csv(root / "environment/data/train.csv", index=False)
    test_out.to_csv(root / "tests/test.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame({"item_id": catalog_ids}).to_csv(root / "tests/catalog.csv", index=False)
    (root / "baseline/popular_items.txt").write_text(" ".join(top10) + "\n", encoding="utf-8")
    scoring = {
        "popularity_ndcg_at_10": floor,
        "artifact_limit_bytes": ARTIFACT_LIMIT,
        "metric": "macro_binary_ndcg_at_10",
        "normalization": "clip((raw-popularity)/(1-popularity),0,1)",
    }
    (root / "tests/scoring_config.json").write_text(json.dumps(scoring, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "source": "UCI Online Retail II (Daqing Chen), DOI 10.24432/C5CG6D",
        "source_rows": source_rows,
        "clean_rows_before_split": len(clean),
        "visible_train_rows": len(train_out),
        "evaluation_customers": len(test_out),
        "catalog_size": len(catalog_ids),
        "popularity_ndcg_at_10": floor,
        "median_target_items": float(test_out["target_items"].map(lambda x: len(str(x).split())).median()),
        "artifact_limit_bytes": ARTIFACT_LIMIT,
        "private_id_remap": True,
        "min_user_orders_source": a.min_user_orders,
        "min_item_train_orders_source": a.min_item_train_orders,
    }
    (root / "author_tools/build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Private remapping secret was not written to disk.")
    if a.make_zip:
        _zip_final(root)


if __name__ == "__main__":
    main()


# ============================================================================
# FILE: author_tools/package_task.py
# ============================================================================
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pathlib
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=pathlib.Path, default=None)
    a = p.parse_args()
    root = a.task_root.resolve()
    train = root / "environment/data/train.csv"
    test = root / "tests/test.csv"
    if train.stat().st_size < 100 or test.stat().st_size < 100:
        raise SystemExit("dataset is not materialized; run build_dataset.py first")
    output = a.output or root.parent / "compact_next_basket_oe_v0_1.zip"
    include = ["baseline", "environment", "tests", "README.md", "instruction.md", "task.toml"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in include:
            pth = root / name
            if pth.is_dir():
                for f in pth.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(root))
            else:
                zf.write(pth, pth.relative_to(root))
    print(output)

if __name__ == "__main__":
    main()


# ============================================================================
# FILE: author_tools/probe.py
# ============================================================================
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def ndcg(pred, target):
    target = set(target)
    dcg = sum(1 / math.log2(r + 1) for r, x in enumerate(pred, 1) if x in target)
    k = min(10, len(target))
    if not k:
        return 0.0
    idcg = sum(1 / math.log2(r + 1) for r in range(1, k + 1))
    return dcg / idcg


def fill10(items, pop):
    seen = set()
    out = []
    for x in list(items) + list(pop):
        x = int(x)
        if x not in seen:
            out.append(x); seen.add(x)
            if len(out) == 10:
                return out
    return out


def score_map(preds, targets):
    return float(np.mean([ndcg(preds[c], t) for c, t in targets.items()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    root = a.task_root
    tr = pd.read_csv(root / "environment/data/train.csv")
    te = pd.read_csv(root / "tests/test.csv")
    if tr.empty or te.empty:
        raise SystemExit("dataset not built; run build_dataset.py first")
    targets = {int(r.customer_id): [int(x) for x in str(r.target_items).split()] for r in te.itertuples(index=False)}
    customers = sorted(targets)

    basket = tr[["customer_id", "order_index", "item_id"]].drop_duplicates()
    pop = basket["item_id"].value_counts().index.astype(int).tolist()
    pop_preds = {c: pop[:10] for c in customers}

    recent = {}
    freq = {}
    hybrid = {}
    unique_counts = {}
    for c in customers:
        g = basket[basket.customer_id == c].sort_values(["order_index", "item_id"])
        unique_counts[c] = int(g.item_id.nunique())
        seq = []
        for oi in sorted(g.order_index.unique(), reverse=True):
            # Within a basket there is no true item order; numeric order makes this deterministic.
            seq.extend(sorted(g[g.order_index == oi].item_id.astype(int).tolist()))
        recent[c] = fill10(seq, pop)
        counts = Counter(g.item_id.astype(int).tolist())
        last_seen = g.groupby("item_id")["order_index"].max().to_dict()
        f = sorted(counts, key=lambda x: (-counts[x], -last_seen[x], x))
        freq[c] = fill10(f, pop)
        max_oi = int(g.order_index.max())
        hs = {}
        for item, cnt in counts.items():
            age = max_oi - int(last_seen[item])
            hs[item] = 1.0 * cnt + 2.0 / (1.0 + age)
        h = sorted(hs, key=lambda x: (-hs[x], x))
        hybrid[c] = fill10(h, pop)

    scores = {
        "global_popularity": score_map(pop_preds, targets),
        "recent_unique_then_popularity": score_map(recent, targets),
        "frequency_then_popularity": score_map(freq, targets),
        "frequency_plus_recency_then_popularity": score_map(hybrid, targets),
    }
    n_users = len(customers)
    catalog_size = int(tr.item_id.nunique())
    item_bytes = 2 if catalog_size <= 65535 else 4
    # Dense remapped customer IDs allow direct row indexing, so this is a useful lower-bound footprint estimate.
    sizes = {
        "dense_top1_per_user_bytes": n_users * item_bytes,
        "dense_top3_per_user_bytes": n_users * 3 * item_bytes,
        "dense_top5_per_user_bytes": n_users * 5 * item_bytes,
        "dense_top10_per_user_bytes": n_users * 10 * item_bytes,
        "dense_all_seen_item_ids_lower_bound_bytes": int(sum(unique_counts.values()) * item_bytes),
        "rank8_int8_user_item_factors_lower_bound_bytes": int((n_users + catalog_size) * 8),
        "rank16_int8_user_item_factors_lower_bound_bytes": int((n_users + catalog_size) * 16),
        "rank8_float32_user_item_factors_lower_bound_bytes": int((n_users + catalog_size) * 8 * 4),
    }
    result = {
        "scores": scores,
        "state_size_lower_bounds": sizes,
        "evaluation_customers": n_users,
        "catalog_size": catalog_size,
        "artifact_limit_bytes": 65536,
        "checks": {
            "personalized_signal": max(scores[k] for k in scores if k != "global_popularity") > scores["global_popularity"] + 1e-4,
            "top10_table_exceeds_budget_before_code": sizes["dense_top10_per_user_bytes"] > 65536,
            "rank16_int8_exceeds_budget_before_code": sizes["rank16_int8_user_item_factors_lower_bound_bytes"] > 65536,
        },
    }
    out = root / "author_tools/probe_metrics.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


# ============================================================================
# FILE: baseline/predict.py
# ============================================================================
#!/usr/bin/env python3
"""Author-only smoke baseline.

This is NOT an oracle, reference solution, claimed optimum, or ceiling, and it
is not shipped to the agent.  `author_tools/build_dataset.py` writes
`popular_items.txt` next to this file.
"""
from __future__ import annotations
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
POPULAR = HERE / "popular_items.txt"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: predict.py <in.csv> <out.csv>")
    if not POPULAR.exists():
        raise SystemExit("popular_items.txt missing; run author_tools/build_dataset.py first")
    items = POPULAR.read_text(encoding="utf-8").strip().split()
    if len(items) < 10:
        raise SystemExit("baseline requires at least 10 popularity items")
    pred = " ".join(items[:10])
    with open(sys.argv[1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and list(rows[0].keys()) != ["customer_id"]:
        raise SystemExit("input must have exactly customer_id")
    with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prediction"])
        for _ in rows:
            w.writerow([pred])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ============================================================================
# FILE: environment/Dockerfile
# ============================================================================
FROM python:3.11-slim

RUN pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2
WORKDIR /app
COPY data/train.csv /app/train.csv


# ============================================================================
# FILE: tests/finalize_artifacts.py
# ============================================================================
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


# ============================================================================
# FILE: tests/verifier_utils.py
# ============================================================================
from __future__ import annotations

import contextlib
import csv
import math
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Iterable

ARTIFACT_LIMIT = 65_536
OUTPUT_DIR = pathlib.Path("/output")
PREDICT = OUTPUT_DIR / "predict.py"
TEST_DIR = pathlib.Path(__file__).resolve().parent
CATALOG_PATH = TEST_DIR / "catalog.csv"


def artifact_size_and_validate(root: pathlib.Path = OUTPUT_DIR) -> int:
    if not root.exists() or not root.is_dir():
        raise AssertionError("/output directory missing")
    total = 0
    for p in root.rglob("*"):
        if p.is_symlink():
            raise AssertionError(f"symlink not allowed in /output: {p}")
        if p.is_dir():
            continue
        if not p.is_file():
            raise AssertionError(f"special file not allowed in /output: {p}")
        total += p.stat().st_size
    if total > ARTIFACT_LIMIT:
        raise AssertionError(f"artifact too large: {total} > {ARTIFACT_LIMIT} bytes")
    return total


def load_catalog() -> set[str]:
    with CATALOG_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["item_id"]:
            raise AssertionError("invalid verifier catalog")
        out = {str(row["item_id"]).strip() for row in r}
    if len(out) < 10:
        raise AssertionError("verifier catalog has fewer than 10 items")
    return out


@contextlib.contextmanager
def _protected_verifier_paths():
    """Best-effort local isolation for the submission process.

    On the standard root-run verifier we execute a copied /output artifact as
    `nobody`, while /tests, /app, the original /output and verifier logs are
    temporarily owner-only.  Production infrastructure should still treat
    /output as the only carried-forward submission artifact.
    """
    paths = [pathlib.Path("/tests"), pathlib.Path("/app"), pathlib.Path("/output"), pathlib.Path("/logs/verifier")]
    modes = []
    for p in paths:
        try:
            if p.exists():
                modes.append((p, p.stat().st_mode & 0o7777))
                p.chmod(0o700)
        except OSError:
            pass
    try:
        yield
    finally:
        for p, mode in reversed(modes):
            try:
                p.chmod(mode)
            except OSError:
                pass


def _chown_tree(root: pathlib.Path, uid: int, gid: int) -> None:
    for p in [root, *root.rglob("*")]:
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass


def run_predict(input_csv: pathlib.Path, output_csv: pathlib.Path, timeout: int = 300) -> None:
    artifact_size_and_validate()
    if not PREDICT.exists() or not PREDICT.is_file() or PREDICT.stat().st_size == 0:
        raise AssertionError("/output/predict.py missing or empty")

    # Execute a copy so relative artifact files work, while absolute /output
    # references do not bypass the self-contained artifact contract.
    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="oe_exec_", dir="/var/tmp" if pathlib.Path("/var/tmp").exists() else None))
    artifact = sandbox / "artifact"
    shutil.copytree(OUTPUT_DIR, artifact, symlinks=True)
    sandbox_in = sandbox / "input.csv"
    sandbox_out = sandbox / "result.csv"
    shutil.copy2(input_csv, sandbox_in)
    home = sandbox / "home"
    tmp = sandbox / "tmp"
    home.mkdir(); tmp.mkdir()

    preexec = None
    run_uid = None
    run_gid = None
    if os.geteuid() == 0:
        try:
            import pwd
            nobody = pwd.getpwnam("nobody")
            run_uid, run_gid = nobody.pw_uid, nobody.pw_gid
        except Exception:
            run_uid = run_gid = 65534
        _chown_tree(sandbox, run_uid, run_gid)
        def _drop_privileges():
            os.setgroups([])
            os.setgid(run_gid)
            os.setuid(run_uid)
        preexec = _drop_privileges

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PYTHONHASHSEED": "0",
    }
    try:
        # Non-root fallback at least removes the conventional train path.
        train = pathlib.Path("/app/train.csv")
        moved = None
        if os.geteuid() != 0 and train.exists():
            moved = sandbox / ".train_hidden"
            os.replace(train, moved)
        try:
            with _protected_verifier_paths():
                cp = subprocess.run(
                    ["python", str(artifact / "predict.py"), str(sandbox_in), str(sandbox_out)],
                    cwd=str(artifact),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    preexec_fn=preexec,
                )
        finally:
            if moved is not None and moved.exists():
                train.parent.mkdir(parents=True, exist_ok=True)
                os.replace(moved, train)
        if cp.returncode != 0:
            msg = cp.stderr[-4000:] if cp.stderr else cp.stdout[-4000:]
            raise AssertionError(f"predict.py failed with code {cp.returncode}: {msg}")
        if not sandbox_out.exists() or sandbox_out.is_symlink() or not sandbox_out.is_file():
            raise AssertionError("submission did not create a regular output CSV")
        shutil.copy2(sandbox_out, output_csv)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

def parse_predictions(output_csv: pathlib.Path, expected_rows: int, catalog: set[str]) -> list[tuple[str, ...]]:
    if not output_csv.exists():
        raise AssertionError("output CSV missing")
    if output_csv.is_symlink() or not output_csv.is_file():
        raise AssertionError("output CSV must be a regular file, not a symlink")
    with output_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["prediction"]:
            raise AssertionError("output must have exactly one column named prediction")
        rows = list(r)
    if len(rows) != expected_rows:
        raise AssertionError(f"wrong row count: got {len(rows)}, expected {expected_rows}")
    preds: list[tuple[str, ...]] = []
    for i, row in enumerate(rows):
        raw = str(row["prediction"]).strip()
        toks = raw.split()
        if len(toks) != 10:
            raise AssertionError(f"row {i}: prediction must contain exactly 10 item IDs")
        if len(set(toks)) != 10:
            raise AssertionError(f"row {i}: item IDs must be distinct")
        bad = [x for x in toks if x not in catalog]
        if bad:
            raise AssertionError(f"row {i}: unknown item IDs: {bad[:3]}")
        preds.append(tuple(toks))
    return preds


def write_input(path: pathlib.Path, customer_ids: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_id"])
        for c in customer_ids:
            w.writerow([c])


def ndcg_at_10(pred: tuple[str, ...], target: set[str]) -> float:
    if not target:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(pred, start=1):
        if item in target:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(10, len(target))
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# FILE: tests/test_main.py
# ============================================================================
from __future__ import annotations
import csv
import json
import math
import pathlib
import tempfile
import resource

from verifier_utils import (
    artifact_size_and_validate,
    load_catalog,
    ndcg_at_10,
    parse_predictions,
    run_predict,
    write_input,
)

TEST_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR = pathlib.Path("/logs/verifier")


def _load_cases():
    cases = []
    with (TEST_DIR / "test.csv").open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        expected = ["customer_id", "target_items"]
        assert r.fieldnames == expected, f"bad verifier test schema: {r.fieldnames}"
        for row in r:
            target = {x for x in row["target_items"].split() if x}
            if target:
                cases.append((row["customer_id"], target))
    assert cases, "test.csv empty; run author_tools/build_dataset.py"
    return cases


def _score(preds, cases):
    vals = [ndcg_at_10(p, target) for p, (_, target) in zip(preds, cases)]
    hits = [1.0 if any(x in target for x in p) else 0.0 for p, (_, target) in zip(preds, cases)]
    return sum(vals) / len(vals), sum(hits) / len(hits)


def test_reward():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    artifact_bytes = artifact_size_and_validate()
    catalog = load_catalog()
    cases = _load_cases()
    customer_ids = [c for c, _ in cases]
    config = json.loads((TEST_DIR / "scoring_config.json").read_text(encoding="utf-8"))
    pop_floor = float(config["popularity_ndcg_at_10"])
    assert 0.0 <= pop_floor < 1.0

    with tempfile.TemporaryDirectory(prefix="oe_main_") as td:
        td = pathlib.Path(td)
        p_in = td / "in.csv"
        p_out = td / "out.csv"
        p_rev = td / "rev.csv"
        p_rev_out = td / "rev_out.csv"
        write_input(p_in, customer_ids)
        run_predict(p_in, p_out, timeout=600)
        preds = parse_predictions(p_out, len(cases), catalog)
        write_input(p_rev, reversed(customer_ids))
        run_predict(p_rev, p_rev_out, timeout=600)
        rev_preds = parse_predictions(p_rev_out, len(cases), catalog)
        assert rev_preds == list(reversed(preds)), "full-test predictions depend on input row position"

    raw_ndcg, hit_rate = _score(preds, cases)
    reward = (raw_ndcg - pop_floor) / max(1e-12, 1.0 - pop_floor)
    reward = max(0.0, min(1.0, reward))
    assert math.isfinite(reward)

    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    metrics = {
        "reward": reward,
        "ndcg_at_10": raw_ndcg,
        "popularity_ndcg_at_10": pop_floor,
        "hit_rate_at_10": hit_rate,
        "evaluated_customers": len(cases),
        "catalog_size": len(catalog),
        "artifact_bytes": artifact_bytes,
        "artifact_limit_bytes": 65_536,
        "peak_child_memory_mb": peak_kb / 1024.0,
    }
    (LOG_DIR / "reward.txt").write_text(f"{reward:.12f}\n", encoding="utf-8")
    (LOG_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ============================================================================
# FILE: tests/test_schema.py
# ============================================================================
from __future__ import annotations
import csv
import pathlib
import tempfile

from verifier_utils import artifact_size_and_validate, load_catalog, parse_predictions, run_predict, write_input

TEST_DIR = pathlib.Path(__file__).resolve().parent


def _customers(n=4):
    with (TEST_DIR / "test.csv").open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        vals = [row["customer_id"] for _, row in zip(range(n), r)]
    assert len(vals) >= 3, "test.csv not materialized; run author_tools/build_dataset.py"
    return vals


def test_schema_determinism_and_order():
    artifact_size_and_validate()
    catalog = load_catalog()
    ids = _customers()
    with tempfile.TemporaryDirectory(prefix="oe_schema_") as td:
        td = pathlib.Path(td)
        p_in = td / "in.csv"
        p_out1 = td / "out1.csv"
        p_out2 = td / "out2.csv"
        p_rev = td / "rev.csv"
        p_rev_out = td / "rev_out.csv"
        write_input(p_in, ids)
        run_predict(p_in, p_out1, timeout=120)
        run_predict(p_in, p_out2, timeout=120)
        a = parse_predictions(p_out1, len(ids), catalog)
        b = parse_predictions(p_out2, len(ids), catalog)
        assert a == b, "submission is not deterministic"
        write_input(p_rev, list(reversed(ids)))
        run_predict(p_rev, p_rev_out, timeout=120)
        c = parse_predictions(p_rev_out, len(ids), catalog)
        assert c == list(reversed(a)), "predictions must follow input row order"


# ============================================================================
# FILE: tests/test_smoke.py
# ============================================================================
from pathlib import Path


def test_predict_exists():
    p = Path("/output/predict.py")
    assert p.exists() and p.is_file() and not p.is_symlink() and p.stat().st_size > 0


# ============================================================================
# FILE: tests/test.sh
# ============================================================================
#!/bin/bash
set +e
mkdir -p /logs/verifier
python - <<'PY'
import pathlib,time
pathlib.Path('/logs/verifier/.start_time').write_text(str(time.time()))
PY
cd /tests
python -m pytest -q test_smoke.py test_schema.py test_main.py
status=$?
python /tests/finalize_artifacts.py
exit $status
