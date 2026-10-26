# ===== BEGIN FILE: tmp/environment/Dockerfile =====
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 scikit-learn==1.6.1
WORKDIR /app
COPY app/ /app/

# ===== END FILE: tmp/environment/Dockerfile =====

# ===== BEGIN FILE: tmp/environment/app/tools/backtest.py =====
"""Backtest a release (model.json + policy.json) on the warehouse extract.

    python3 tools/backtest.py RELEASE_DIR [--from 2026-08-01] [--to 2026-09-30]

Replays the decision engine on extracted auths in the date range and reports the dashboard metrics
with the labels available in the extract (chargebacks and issuer reports received so far).
Caveats the team keeps tripping over:
  * recent weeks look artificially clean: most chargebacks for them have not arrived yet;
  * transactions v7 or an analyst declined (outside the decline holdout) have no outcome; they are counted in
    `unknown_outcome_tpv_share` and contribute no cost either way;
  * the extract is a stratified sample, so everything is weighted by `sample_weight`, and the review
    queue is approximated by filling each PT hour's capacity with weighted rows.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import riskengine as RE  # noqa: E402

DATA = os.environ.get('RISK_DATA_DIR', os.path.join(HERE, '..', 'data'))


def epoch_seconds(col):
    t = pd.to_datetime(col, utc=True)
    return ((t - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def load_extract(path=None):
    df = pd.read_csv(path or os.path.join(DATA, 'warehouse', 'auth_extract.csv.gz'))
    return df.drop_duplicates('txn_id').reset_index(drop=True)


def visible_labels(df):
    cb = pd.read_csv(os.path.join(DATA, 'disputes', 'chargebacks.csv'), dtype={'reason_code': str})
    fin = json.load(open(os.path.join(DATA, 'config', 'finance_cost_model.json')))
    codes = {c for v in fin['fraud_loss']['fraud_reason_codes'].values() for c in v}
    fraud_cb = cb[cb.reason_code.isin(codes) & (cb.representment_status != 'won')]
    return df.txn_id.isin(set(fraud_cb.txn_id)).to_numpy()


def weighted_decide(s, X, ts, policy, capacity, w):
    R, K = policy
    v = X[:, RE.FEATURES.index('vertical')].astype(int)
    g = X[:, RE.FEATURES.index('region')].astype(int)
    amt = np.nan_to_num(X[:, RE.FEATURES.index('amount_usd')], nan=0.0)
    b = RE.amount_band(amt)
    act = np.zeros(len(s), np.int8)
    act[s >= K[v, g, b]] = 2
    cand = np.where((act == 0) & (s >= R[v, g, b]))[0]
    if len(cand):
        hb = RE.pt_hour_index(ts)[cand]
        order = np.lexsort((-(s[cand] * amt[cand]), hb))
        c, hb_o, w_o = cand[order], hb[order], w[cand][order]
        cum = pd.Series(w_o).groupby(hb_o).cumsum().to_numpy()
        cap = np.asarray(capacity)[(hb_o % 24).astype(int)]
        act[c[cum <= cap]] = 1
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('release')
    ap.add_argument('--from', dest='d0', default='2026-07-06')
    ap.add_argument('--to', dest='d1', default='2026-10-25')
    a = ap.parse_args()
    model, policy = RE.load_artifacts(os.path.join(a.release, 'model.json'), os.path.join(a.release, 'policy.json'))
    fin = json.load(open(os.path.join(DATA, 'config', 'finance_cost_model.json')))
    ops = json.load(open(os.path.join(DATA, 'config', 'review_ops.json')))
    margin = np.array([fin['false_decline_cost']['lost_margin_rate_by_vertical'][v] for v in RE.VERTICALS])
    df = load_extract()
    day = df.auth_ts_utc.str[:10]
    df = df[(day >= a.d0) & (day <= a.d1)].reset_index(drop=True)
    X = df[RE.FEATURES].to_numpy(np.float64)
    ts = epoch_seconds(df.auth_ts_utc)
    w = df.sample_weight.to_numpy(float)
    s = RE.score(model, X)
    act = weighted_decide(s, X, ts, policy, ops['capacity_per_pt_hour'], w)
    fraud = visible_labels(df)
    unknown = ((df.decision == 'DECLINE') | (df.review_outcome == 'declined')).to_numpy()   # never observed
    amt = df.amount_local.to_numpy(float)
    usd = df.amount_usd.to_numpy(float)
    rate = (df.amount_usd / df.amount_local.where(df.amount_local > 0)).groupby(df.currency).transform('median')
    usd = np.where(np.isnan(usd), amt * rate.to_numpy(float), usd)
    v = df.vertical.to_numpy(int)
    qa_catch, qa_fd = 0.8, 0.03
    p_decl = np.where(act == 2, 1.0, np.where(act == 1, np.where(fraud, qa_catch, qa_fd), 0.0))
    fraud_loss = np.where(fraud & ~unknown, (1 - p_decl) * (usd + fin['fraud_loss']['chargeback_fee_usd']), 0.0)
    fd = np.where(~fraud & ~unknown, p_decl * margin[v] * usd, 0.0)
    week = pd.to_datetime(df.auth_ts_utc.str[:10]).dt.to_period('W-SUN').dt.start_time.dt.date.astype(str).to_numpy()
    port = np.where(df.region.to_numpy(int) == 0, 'NA', 'INTL')
    out = pd.DataFrame(dict(week=week, portfolio=port, tpv=w * usd, fraud=w * fraud_loss, fd=w * fd,
                            unk=w * usd * unknown, rev=w * (act == 1), dec=w * (act == 2), n=w))
    g = out.groupby(['week', 'portfolio']).sum()
    rep = pd.DataFrame({'fraud_bps': g.fraud / g.tpv * 1e4, 'false_decline_bps': g.fd / g.tpv * 1e4,
                        'net_bps': (g.fraud + g.fd) / g.tpv * 1e4, 'review_rate': g.rev / g.n,
                        'decline_rate': g.dec / g.n, 'unknown_outcome_tpv_share': g.unk / g.tpv})
    pd.set_option('display.width', 160)
    print(rep.round(4).to_string())


if __name__ == '__main__':
    main()

# ===== END FILE: tmp/environment/app/tools/backtest.py =====

# ===== BEGIN FILE: tmp/environment/app/tools/check_release.py =====
"""Pre-release check: run the release pipeline's artifact validation and look at the decision mix
the release would produce on recent traffic.

    python3 tools/check_release.py RELEASE_DIR [--days 14]

Exits non-zero if the release pipeline would reject the artifacts.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import riskengine as RE  # noqa: E402
from backtest import load_extract, weighted_decide, epoch_seconds, DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('release')
    ap.add_argument('--days', type=int, default=14)
    a = ap.parse_args()
    try:
        model, policy = RE.load_artifacts(os.path.join(a.release, 'model.json'), os.path.join(a.release, 'policy.json'))
    except (RE.ArtifactError, OSError, ValueError) as e:
        print('REJECTED:', e)
        sys.exit(1)
    nodes = sum(len(t[5]) for t in model['trees'])
    depth = max(t[6] for t in model['trees'])
    print(f"model ok: {len(model['trees'])} trees, {nodes} nodes, max depth {depth} "
          f"(limits {RE.MAX_TREES} / {RE.MAX_TOTAL_NODES} / {RE.MAX_DEPTH})")
    print('policy ok')
    ops = json.load(open(os.path.join(DATA, 'config', 'review_ops.json')))
    df = load_extract()
    last = pd.to_datetime(df.auth_ts_utc.str[:10])
    df = df[last > last.max() - pd.Timedelta(days=a.days)].reset_index(drop=True)
    X = df[RE.FEATURES].to_numpy(np.float64)
    ts = epoch_seconds(df.auth_ts_utc)
    w = df.sample_weight.to_numpy(float)
    s = RE.score(model, X)
    R, K = policy
    v = X[:, 3].astype(int); g = X[:, 4].astype(int)
    b = RE.amount_band(np.nan_to_num(X[:, 0]))
    flagged = (s >= R[v, g, b]) & (s < K[v, g, b])
    act = weighted_decide(s, X, ts, policy, ops['capacity_per_pt_hour'], w)
    hb = RE.pt_hour_index(ts)
    per_hour = pd.Series(w * flagged).groupby(hb).sum()
    cap = np.asarray(ops['capacity_per_pt_hour'])[(per_hour.index.to_numpy() % 24)]
    print(f"last {a.days} days (weighted): decline {np.sum(w * (act == 2)) / w.sum():.3%}, "
          f"review {np.sum(w * (act == 1)) / w.sum():.3%}, flagged for review {np.sum(w * flagged) / w.sum():.3%}")
    print(f"review queue: {np.mean(per_hour.to_numpy() > cap):.1%} of hours overflow; "
          f"flagged/capacity median {np.median(per_hour.to_numpy() / cap):.2f}")


if __name__ == '__main__':
    main()

# ===== END FILE: tmp/environment/app/tools/check_release.py =====

# ===== BEGIN FILE: tmp/environment/app/tools/hgb_export.py =====
"""Convert a fitted sklearn HistGradientBoostingClassifier (binary, numeric features) to the decision
service's "tree_ensemble_v1" JSON. Feature order must be riskengine.FEATURES.
Train with max_depth <= 12 and max_leaf_nodes <= 256 to stay inside the serving limits."""
import numpy as np

try:
    from riskengine import FEATURES
except ImportError:  # pragma: no cover
    from tools.riskengine import FEATURES


def export_hgb(clf, feature_names=None):
    preds = clf._predictors
    if len(preds) and len(preds[0]) != 1:
        raise ValueError('only binary classifiers are supported')
    base = float(np.ravel(clf._baseline_prediction)[0])
    trees = []
    for it in preds:
        nodes = it[0].nodes
        if np.any(nodes['is_categorical']):
            raise ValueError('categorical splits are not supported')
        leaf = nodes['is_leaf'].astype(bool)
        trees.append(dict(
            split_feature=[int(f) if not l else 0 for f, l in zip(nodes['feature_idx'], leaf)],
            threshold=[float(np.clip(t, -1e300, 1e300)) if not l else 0.0 for t, l in zip(nodes['num_threshold'], leaf)],
            left=[int(c) if not l else -1 for c, l in zip(nodes['left'], leaf)],
            right=[int(c) if not l else -1 for c, l in zip(nodes['right'], leaf)],
            default_left=[int(bool(m)) if not l else 0 for m, l in zip(nodes['missing_go_to_left'], leaf)],
            leaf_value=[float(v) if l else 0.0 for v, l in zip(nodes['value'], leaf)],
        ))
    return dict(format='tree_ensemble_v1', feature_names=list(feature_names or FEATURES), base_margin=base, trees=trees)

# ===== END FILE: tmp/environment/app/tools/hgb_export.py =====

# ===== BEGIN FILE: tmp/environment/app/tools/riskengine.py =====
"""Offline copy of the risk decision service: model artifact loader/scorer and policy engine.

The production decision service loads two artifacts per release:
  model.json   - tree ensemble (format "tree_ensemble_v1"), scored on the logged feature vector
  policy.json  - per-segment review/decline thresholds on the model score (format "risk_policy_v1")
Both are validated at load time; an artifact that fails validation is rejected by the release
pipeline. This module is the same code the service uses (numpy only).
"""
import json, math
import numpy as np

FEATURES = ['amount_usd', 'hour_local', 'is_weekend', 'vertical', 'region', 'channel_app', 'is_recurring',
            'card_type', 'card_age_days', 'card_txn_count_24h', 'card_distinct_merchants_24h', 'card_decline_count_24h',
            'bin_risk_90d', 'ip_bin_country_mismatch', 'avs_result', 'cvv_match',
            'device_present', 'device_age_days', 'device_distinct_cards_24h', 'emulator_flag', 'ip_proxy_score',
            'billing_distance_km', 'email_age_days', 'email_domain_risk', 'account_age_days', 'prev_orders',
            'new_shipping_address', 'shipping_billing_match', 'amount_vs_card_avg',
            'merchant_age_days', 'merchant_cb_rate_90d']
VERTICALS = ['digital_goods', 'electronics', 'fashion', 'travel', 'food_delivery', 'subscriptions', 'marketplace', 'home_goods']
REGIONS = ['NA', 'EU', 'LATAM', 'APAC']
AMOUNT_BANDS_USD = [25.0, 100.0, 500.0]          # band 0: <25, 1: 25-100, 2: 100-500, 3: >=500
NBANDS = len(AMOUNT_BANDS_USD) + 1

# serving limits (p99 scoring latency budget of the decision service)
MAX_TREES = 400
MAX_NODES_PER_TREE = 511
MAX_TOTAL_NODES = 120_000
MAX_DEPTH = 12
MAX_FILE_BYTES = 40_000_000


class ArtifactError(ValueError):
    pass


def _arr(t, key, n, dtype):
    v = t.get(key)
    if not isinstance(v, list) or len(v) != n:
        raise ArtifactError(f'tree field {key!r} must be a list of length {n}')
    ok = (int,) if dtype is np.int64 else (int, float)
    if not all(type(x) in ok for x in v):
        raise ArtifactError(f'tree field {key!r} must contain only {"integers" if dtype is np.int64 else "numbers"}')
    try:
        a = np.asarray(v, dtype=dtype)
    except (TypeError, ValueError):
        raise ArtifactError(f'tree field {key!r} has non-numeric entries')
    if a.shape != (n,):
        raise ArtifactError(f'tree field {key!r} must be flat')
    return a


def load_model(obj):
    """obj: parsed JSON (dict). Returns a compact dict of numpy arrays. Raises ArtifactError."""
    if not isinstance(obj, dict) or obj.get('format') != 'tree_ensemble_v1':
        raise ArtifactError('model format must be "tree_ensemble_v1"')
    if obj.get('feature_names') != FEATURES:
        raise ArtifactError('feature_names must equal the serving feature list, in order')
    base = obj.get('base_margin', 0.0)
    if not isinstance(base, (int, float)) or isinstance(base, bool) or not math.isfinite(base):
        raise ArtifactError('base_margin must be a finite number')
    trees = obj.get('trees')
    if not isinstance(trees, list) or not 1 <= len(trees) <= MAX_TREES:
        raise ArtifactError(f'trees must be a list of 1..{MAX_TREES} trees')
    out, total = [], 0
    nf = len(FEATURES)
    for ti, t in enumerate(trees):
        if not isinstance(t, dict):
            raise ArtifactError(f'tree {ti} must be an object')
        lv = t.get('leaf_value')
        n = len(lv) if isinstance(lv, list) else -1
        if not 1 <= n <= MAX_NODES_PER_TREE:
            raise ArtifactError(f'tree {ti}: node count must be 1..{MAX_NODES_PER_TREE}')
        total += n
        if total > MAX_TOTAL_NODES:
            raise ArtifactError(f'model exceeds {MAX_TOTAL_NODES} nodes in total')
        feat = _arr(t, 'split_feature', n, np.int64)
        thr = _arr(t, 'threshold', n, np.float64)
        left = _arr(t, 'left', n, np.int64)
        right = _arr(t, 'right', n, np.int64)
        dl = _arr(t, 'default_left', n, np.int64)
        val = _arr(t, 'leaf_value', n, np.float64)
        leaf = left < 0
        if np.any(leaf != (right < 0)):
            raise ArtifactError(f'tree {ti}: left/right must both be -1 at leaves')
        inner = ~leaf
        if np.any((feat[inner] < 0) | (feat[inner] >= nf)):
            raise ArtifactError(f'tree {ti}: split_feature out of range')
        if np.any(~np.isfinite(thr[inner])):
            raise ArtifactError(f'tree {ti}: thresholds must be finite')
        if np.any(~np.isfinite(val[leaf])):
            raise ArtifactError(f'tree {ti}: leaf values must be finite')
        if np.any(~np.isin(dl[inner], [0, 1])):
            raise ArtifactError(f'tree {ti}: default_left must be 0/1')
        # children must point forward (no cycles) and every node except root has exactly one parent
        idx = np.arange(n)
        if np.any((left[inner] <= idx[inner]) | (left[inner] >= n) | (right[inner] <= idx[inner]) | (right[inner] >= n)):
            raise ArtifactError(f'tree {ti}: children must have larger indices than their parent')
        par = np.zeros(n, np.int64)
        np.add.at(par, left[inner], 1)
        np.add.at(par, right[inner], 1)
        if par[0] != 0 or np.any(par[1:] != 1):
            raise ArtifactError(f'tree {ti}: nodes must form a single binary tree rooted at node 0')
        depth = np.zeros(n, np.int64)
        for i in np.where(inner)[0]:
            depth[left[i]] = depth[right[i]] = depth[i] + 1
        if depth.max() > MAX_DEPTH:
            raise ArtifactError(f'tree {ti}: depth exceeds {MAX_DEPTH}')
        out.append((feat.astype(np.int32), thr, left.astype(np.int32), right.astype(np.int32), dl.astype(bool),
                    val, int(depth.max())))
    return dict(base=float(base), trees=out)


def margin(model, X):
    """X: (n, F) float array with NaN for missing. Returns raw margin (n,)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    m = np.full(n, model['base'])
    rows = np.arange(n)
    for feat, thr, left, right, dl, val, depth in model['trees']:
        node = np.zeros(n, np.int32)
        for _ in range(depth):
            f = feat[node]
            l = left[node]
            internal = l >= 0
            if not internal.any():
                break
            x = X[rows, np.where(internal, f, 0)]
            go_left = np.where(np.isnan(x), dl[node], x <= thr[node])
            node = np.where(internal, np.where(go_left, l, right[node]), node)
        m += val[node]
    return m


def score(model, X):
    z = np.clip(margin(model, X), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def load_policy(obj):
    """Returns (review_thr, decline_thr) arrays of shape (8 verticals, 4 regions, 4 amount bands)."""
    if not isinstance(obj, dict) or obj.get('format') != 'risk_policy_v1':
        raise ArtifactError('policy format must be "risk_policy_v1"')

    def thr_pair(d, where):
        if not isinstance(d, dict):
            raise ArtifactError(f'{where}: must be an object')
        r, k = d.get('review'), d.get('decline')
        for name, v in (('review', r), ('decline', k)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ArtifactError(f'{where}: {name} must be a finite number')
        if not (0.0 <= r <= k <= 1.0 + 1e-9):
            raise ArtifactError(f'{where}: need 0 <= review <= decline <= 1')
        return float(r), float(k)

    r0, k0 = thr_pair(obj.get('default'), 'default')
    R = np.full((len(VERTICALS), len(REGIONS), NBANDS), r0)
    K = np.full((len(VERTICALS), len(REGIONS), NBANDS), k0)
    rows = obj.get('segments', [])
    if not isinstance(rows, list) or len(rows) > len(VERTICALS) * len(REGIONS) * NBANDS:
        raise ArtifactError('segments must be a list of at most 128 rows')
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ArtifactError(f'segments[{i}] must be an object')
        v, g, b = row.get('vertical'), row.get('region'), row.get('amount_band')
        if v not in VERTICALS or g not in REGIONS or not isinstance(b, int) or isinstance(b, bool) or not 0 <= b < NBANDS:
            raise ArtifactError(f'segments[{i}]: unknown vertical/region/amount_band')
        key = (v, g, b)
        if key in seen:
            raise ArtifactError(f'segments[{i}]: duplicate segment {key}')
        seen.add(key)
        r, k = thr_pair(row, f'segments[{i}]')
        R[VERTICALS.index(v), REGIONS.index(g), b] = r
        K[VERTICALS.index(v), REGIONS.index(g), b] = k
    return R, K


def load_artifacts(model_path, policy_path):
    import os
    for p in (model_path, policy_path):
        if os.path.getsize(p) > MAX_FILE_BYTES:
            raise ArtifactError(f'{os.path.basename(p)} exceeds {MAX_FILE_BYTES} bytes')
    with open(model_path) as f:
        model = load_model(json.load(f, parse_constant=_reject_const))
    with open(policy_path) as f:
        policy = load_policy(json.load(f, parse_constant=_reject_const))
    return model, policy


def _reject_const(c):
    raise ArtifactError(f'invalid JSON constant {c}')


def amount_band(amount_usd):
    a = np.nan_to_num(np.asarray(amount_usd, dtype=np.float64), nan=0.0)
    return np.searchsorted(np.asarray(AMOUNT_BANDS_USD), a, side='right')


# ---- manual review queue (fraud ops, Pacific time staffing) ----
# 2026-11-01 09:00 UTC = end of US daylight time
_DST_END_UTC = 1793523600


def pt_hour_index(ts_utc):
    """Absolute PT-hour bucket for each auth timestamp (epoch seconds, UTC)."""
    ts = np.asarray(ts_utc, dtype=np.int64)
    off = np.where(ts < _DST_END_UTC, -7 * 3600, -8 * 3600)
    return (ts + off) // 3600


def decide(s, X, ts_utc, policy, capacity_by_pt_hour):
    """Decision engine. Returns actions: 0 approve, 1 manual review, 2 decline.

    decline if score >= decline threshold of the segment; otherwise review if score >= review threshold,
    subject to analyst capacity: within each PT hour, flagged transactions are worked in order of
    expected loss (score x amount_usd); flagged transactions beyond that hour's capacity are released
    (approved) because orders cannot be held longer.
    """
    R, K = policy
    X = np.asarray(X, dtype=np.float64)
    v = X[:, FEATURES.index('vertical')].astype(np.int64)
    g = X[:, FEATURES.index('region')].astype(np.int64)
    amt = np.nan_to_num(X[:, FEATURES.index('amount_usd')], nan=0.0)
    b = amount_band(amt)
    rt, kt = R[v, g, b], K[v, g, b]
    act = np.zeros(len(s), np.int8)
    act[s >= kt] = 2
    cand = np.where((act == 0) & (s >= rt))[0]
    if len(cand):
        hb = pt_hour_index(ts_utc)[cand]
        pri = s[cand] * amt[cand]
        order = np.lexsort((-pri, hb))
        hb_o = hb[order]
        starts = np.r_[0, np.flatnonzero(np.diff(hb_o)) + 1]
        lens = np.diff(np.r_[starts, len(order)])
        rank = np.arange(len(order)) - np.repeat(starts, lens)
        cap = np.asarray(capacity_by_pt_hour)[(hb_o % 24).astype(np.int64)]
        act[cand[order[rank < cap]]] = 1
    return act

# ===== END FILE: tmp/environment/app/tools/riskengine.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Refresh the card-not-present risk model for the Oct 26 release

You are on the Risk ML team of a payment processor serving about 1,500 card-not-present merchants
in North America, Europe, LATAM and APAC. Every authorization is scored by the **risk decision
service**: a tree-ensemble model produces a risk score, and a per-segment policy turns the score
into *approve*, *manual review* or *decline*.

`txn-risk-v7` has been in production since July 1. Model refreshes were frozen while the feature
store was migrated; the migration finished last week, and fraud losses have been climbing all
quarter. The release train restarts tonight: whatever is in `/app/release/` at cut-over is loaded at
**00:00 PT, Monday Oct 26** and stays frozen until the next release on Nov 9. That window covers
Halloween, the start of the holiday season, and wave 2 of the device-SDK v3 migration.

Your job is to ship a new `model.json` and `policy.json` into `/app/release/` (it currently holds a
copy of v7's artifacts). Only those two files are deployed; anything else you write under `/app`
stays on your workstation.

## What the team watches

Fraud and finance track **weekly net fraud cost** in bps of TPV, separately for the **NA** and the
**international** (EU + LATAM + APAC) portfolios:

- **fraud loss** — fraud-coded chargebacks (Visa 10.4, Mastercard 4837, Amex F29) on transactions we
  approved: dispute amount + $15 fee, recovered when the merchant wins representment;
- **false-decline cost** — lost margin (rate per vertical in `config/finance_cost_model.json`) ×
  amount for legitimate orders that were declined, by the model or by an analyst after review.

Chargebacks keep arriving for months and finance books them back to the auth week, so the release's
numbers are only final long after Nov 8 — the final numbers are what the team cares about, not the
early dashboard. Non-fraud disputes (Visa 13.x, Mastercard 4853/4841, Amex C08/C31) are worked by the
disputes team and are not part of this metric.

## How the decision service works

- **`model.json`** — format `"tree_ensemble_v1"`, scored on the 31 logged features in the order of
  `docs/feature_catalog.md`; risk score = sigmoid(`base_margin` + sum of the leaf values reached).
  Missing feature values arrive as NaN and follow each node's `default_left`.
- **`policy.json`** — format `"risk_policy_v1"`: default `review` / `decline` thresholds on the
  score, plus optional overrides per segment (vertical × region × amount band; bands are
  < $25, $25–100, $100–500, ≥ $500 by `amount_usd`).
- A transaction is declined if its score is at or above its segment's decline threshold; otherwise it
  goes to manual review if the score is at or above the review threshold. Analysts work the review
  queue per Pacific-time hour with the capacity in `config/review_ops.json`, highest
  score × `amount_usd` first; whatever is still in the queue at the end of the hour is approved.
  The same file has the Q3 QA audit of analyst accuracy.
- The release pipeline validates both artifacts and rejects the release if either fails: format,
  exact feature list, serving limits (≤ 400 trees, ≤ 511 nodes per tree, ≤ 120,000 nodes in total,
  depth ≤ 12, each file ≤ 40 MB), thresholds `0 ≤ review ≤ decline ≤ 1`.

`tools/riskengine.py` is the service's own loader, scorer and decision engine.
`tools/check_release.py RELEASE_DIR` runs the pipeline's validation and shows the decision mix and
review-queue load on recent traffic. `tools/hgb_export.py` converts a fitted scikit-learn
`HistGradientBoostingClassifier` into `tree_ensemble_v1`. `tools/backtest.py RELEASE_DIR` replays a
release on the extract with the labels available so far (read its caveats).

## Data (`/app/data`)

- `warehouse/auth_extract.csv.gz` — authorizations Jul 6 – Oct 25 (UTC). Stratified extract: every
  auth that has a chargeback or issuer fraud report so far, every review and decline, every
  decline-holdout approval, plus a 5% random sample of all other auths (`sample_weight` = 20). Each
  row has ids, timestamp, merchant, card network, local amount and currency, the 31 logged features,
  v7's score and decision, and the review outcome.
- `disputes/chargebacks.csv` — chargebacks received through Oct 25, with reason code, received date
  and representment status.
- `disputes/issuer_fraud_reports.csv` — issuer fraud reports (Visa TC40, Mastercard SAFE, Amex)
  received through Oct 25. They arrive faster than chargebacks, but not every fraud gets one and they
  do not move money.
- `reference/merchants.csv` — merchant registry, including scheduled go-lives and SDK v3 migration
  dates.
- `docs/` — feature catalog and the feature store changelog (incidents and definition changes).
- `config/` — finance cost model and review-ops staffing.
- `release_v7/` — v7's artifacts as deployed.

The box is CPU-only (4 cores, 12 GB RAM) with numpy, scipy, pandas and scikit-learn installed.

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
"""Pre-flight check: the release artifacts exist and pass the release pipeline's validation."""
import subprocess
import sys
from pathlib import Path


def test_release_files_exist():
    assert Path("/app/release/model.json").is_file()
    assert Path("/app/release/policy.json").is_file()


def test_release_passes_pipeline_validation():
    r = subprocess.run([sys.executable, "/app/tools/check_release.py", "/app/release"],
                       capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/solve.sh =====
#!/bin/bash
cp /solution/model.json /app/release/model.json
cp /solution/policy.json /app/release/policy.json

# ===== END FILE: tmp/solution/solve.sh =====

# ===== BEGIN FILE: tmp/tests/Dockerfile =====
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.2.6

# ===== END FILE: tmp/tests/Dockerfile =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
"""Runs the release in /app/release (model.json + policy.json) through the decision service on the
actual Oct 26 - Nov 8 authorizations, computes final net fraud cost per portfolio-week and writes
/logs/verifier/reward.txt and metrics.json. The artifacts are parsed as data; nothing from /app is
executed or imported."""
import json
import math
import os
import stat
import sys
import time

TESTS = os.environ.get("PRR_TESTS_DIR", os.path.dirname(os.path.abspath(__file__)))
APP = os.environ.get("PRR_APP_DIR", "/app")
LOGS = os.environ.get("PRR_LOG_DIR", "/logs/verifier")
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402
import riskengine as RE  # noqa: E402

GROUPS = ["NA_w1", "NA_w2", "INTL_w1", "INTL_w2"]
CUT_DAY = 112
CB_FEE = 15.0


def write_out(reward, metrics):
    os.makedirs(LOGS, exist_ok=True)
    for name in ("reward.txt", "metrics.json"):
        p = os.path.join(LOGS, name)
        try:
            if os.path.islink(p) or os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass
    metrics["reward"] = reward
    for name, text in (("metrics.json", json.dumps(metrics, indent=1)), ("reward.txt", f"{reward:.6f}\n")):
        p = os.path.join(LOGS, name)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(text)
    print(f"reward {reward:.6f}")


def read_bytes(path):
    """Open a candidate file without following symlinks or blocking on special files."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise RE.ArtifactError(f"{os.path.basename(path)} is not a regular file")
        if st.st_size > RE.MAX_FILE_BYTES:
            raise RE.ArtifactError(f"{os.path.basename(path)} exceeds {RE.MAX_FILE_BYTES} bytes")
        chunks, total = [], 0
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            total += len(b)
            if total > RE.MAX_FILE_BYTES:
                raise RE.ArtifactError(f"{os.path.basename(path)} exceeds {RE.MAX_FILE_BYTES} bytes")
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def parse_json(raw, name):
    def bad_const(c):
        raise RE.ArtifactError(f"{name}: invalid JSON constant {c}")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=bad_const)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        if isinstance(e, RE.ArtifactError):
            raise
        raise RE.ArtifactError(f"{name} is not valid JSON: {type(e).__name__}")


def load_release():
    rel = os.path.join(APP, "release")
    model = RE.load_model(parse_json(read_bytes(os.path.join(rel, "model.json")), "model.json"))
    policy = RE.load_policy(parse_json(read_bytes(os.path.join(rel, "policy.json")), "policy.json"))
    return model, policy


def load_hidden():
    hd = os.path.join(TESTS, "hidden")
    with open(os.path.join(hd, "anchors.json")) as f:
        anchors = json.load(f)
    with open(os.path.join(hd, "window_meta.json")) as f:
        wmeta = json.load(f)
    return anchors, wmeta


def group_costs(act, W, margin):
    amt = W["amount_usd_true"].astype(np.float64)
    v = W["X"][:, RE.FEATURES.index("vertical")].astype(np.int64)
    na = W["X"][:, RE.FEATURES.index("region")] == 0
    rev_decl = (act == 1) & (W["rev_u"] < W["p_catch"])
    declined = (act == 2) | rev_decl
    fraud = np.where(~declined & W["loss_flag"], amt + CB_FEE, 0.0)
    fdc = np.where(declined & (W["y"] == 0), margin[v] * amt, 0.0)
    wk = (W["day"].astype(np.int64) - CUT_DAY) // 7
    out = {}
    for g in GROUPS:
        port, w = g.split("_w")
        m = (wk == int(w) - 1) & (na if port == "NA" else ~na)
        tpv = amt[m].sum()
        out[g] = dict(net_bps=float((fraud[m].sum() + fdc[m].sum()) / tpv * 1e4),
                      fraud_bps=float(fraud[m].sum() / tpv * 1e4), false_decline_bps=float(fdc[m].sum() / tpv * 1e4),
                      decline_rate=float((act[m] == 2).mean()), review_rate=float((act[m] == 1).mean()))
    return out


def reward_from(groups, anchors):
    per = {}
    for g in GROUPS:
        c = max(groups[g]["net_bps"], 1e-6)
        c0, c1 = anchors["cost0"][g], anchors["cost1"][g]
        s = math.log(c0 / c) / math.log(c0 / c1)
        per[g] = float(min(1.0, max(0.0, s)))
    return float(sum(anchors["weights"][g] * per[g] for g in GROUPS)), per


def main():
    t0 = time.time()
    metrics = {"version": "0.1"}
    anchors, wmeta = load_hidden()
    try:
        model, policy = load_release()
    except FileNotFoundError as e:
        metrics["status"] = f"missing artifact: {os.path.basename(str(e.filename))}"
        return write_out(0.0, metrics)
    except (RE.ArtifactError, OSError) as e:
        metrics["status"] = f"rejected: {str(e)[:300]}"
        return write_out(0.0, metrics)
    z = np.load(os.path.join(TESTS, "hidden", "window_futB.npz"), allow_pickle=False)
    W = {k: z[k] for k in z.files}
    s = RE.score(model, W["X"])
    act = RE.decide(s, W["X"], W["ts"], policy, wmeta["capacity_per_pt_hour"])
    margin = np.array([wmeta["lost_margin_rate"][v] for v in RE.VERTICALS])
    groups = group_costs(act, W, margin)
    reward, per = reward_from(groups, anchors)
    metrics.update(status="ok", groups=groups, group_reward=per, trees=len(model["trees"]),
                   seconds=round(time.time() - t0, 1))
    write_out(reward, metrics)


if __name__ == "__main__":
    main()

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/rescore.py =====
"""Recompute reward from metrics.json files with the current anchors (no replay needed).
    python3 rescore.py metrics.json [more.json ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_eval import reward_from, TESTS

anchors = json.load(open(os.path.join(TESTS, "hidden", "anchors.json")))
for p in sys.argv[1:]:
    m = json.load(open(p))
    if m.get("status") != "ok":
        print(p, 0.0, m.get("status")); continue
    r, per = reward_from(m["groups"], anchors)
    print(p, round(r, 6), {k: round(v, 3) for k, v in per.items()})

# ===== END FILE: tmp/tests/rescore.py =====

# ===== BEGIN FILE: tmp/tests/riskengine.py =====
"""Offline copy of the risk decision service: model artifact loader/scorer and policy engine.

The production decision service loads two artifacts per release:
  model.json   - tree ensemble (format "tree_ensemble_v1"), scored on the logged feature vector
  policy.json  - per-segment review/decline thresholds on the model score (format "risk_policy_v1")
Both are validated at load time; an artifact that fails validation is rejected by the release
pipeline. This module is the same code the service uses (numpy only).
"""
import json, math
import numpy as np

FEATURES = ['amount_usd', 'hour_local', 'is_weekend', 'vertical', 'region', 'channel_app', 'is_recurring',
            'card_type', 'card_age_days', 'card_txn_count_24h', 'card_distinct_merchants_24h', 'card_decline_count_24h',
            'bin_risk_90d', 'ip_bin_country_mismatch', 'avs_result', 'cvv_match',
            'device_present', 'device_age_days', 'device_distinct_cards_24h', 'emulator_flag', 'ip_proxy_score',
            'billing_distance_km', 'email_age_days', 'email_domain_risk', 'account_age_days', 'prev_orders',
            'new_shipping_address', 'shipping_billing_match', 'amount_vs_card_avg',
            'merchant_age_days', 'merchant_cb_rate_90d']
VERTICALS = ['digital_goods', 'electronics', 'fashion', 'travel', 'food_delivery', 'subscriptions', 'marketplace', 'home_goods']
REGIONS = ['NA', 'EU', 'LATAM', 'APAC']
AMOUNT_BANDS_USD = [25.0, 100.0, 500.0]          # band 0: <25, 1: 25-100, 2: 100-500, 3: >=500
NBANDS = len(AMOUNT_BANDS_USD) + 1

# serving limits (p99 scoring latency budget of the decision service)
MAX_TREES = 400
MAX_NODES_PER_TREE = 511
MAX_TOTAL_NODES = 120_000
MAX_DEPTH = 12
MAX_FILE_BYTES = 40_000_000


class ArtifactError(ValueError):
    pass


def _arr(t, key, n, dtype):
    v = t.get(key)
    if not isinstance(v, list) or len(v) != n:
        raise ArtifactError(f'tree field {key!r} must be a list of length {n}')
    ok = (int,) if dtype is np.int64 else (int, float)
    if not all(type(x) in ok for x in v):
        raise ArtifactError(f'tree field {key!r} must contain only {"integers" if dtype is np.int64 else "numbers"}')
    try:
        a = np.asarray(v, dtype=dtype)
    except (TypeError, ValueError):
        raise ArtifactError(f'tree field {key!r} has non-numeric entries')
    if a.shape != (n,):
        raise ArtifactError(f'tree field {key!r} must be flat')
    return a


def load_model(obj):
    """obj: parsed JSON (dict). Returns a compact dict of numpy arrays. Raises ArtifactError."""
    if not isinstance(obj, dict) or obj.get('format') != 'tree_ensemble_v1':
        raise ArtifactError('model format must be "tree_ensemble_v1"')
    if obj.get('feature_names') != FEATURES:
        raise ArtifactError('feature_names must equal the serving feature list, in order')
    base = obj.get('base_margin', 0.0)
    if not isinstance(base, (int, float)) or isinstance(base, bool) or not math.isfinite(base):
        raise ArtifactError('base_margin must be a finite number')
    trees = obj.get('trees')
    if not isinstance(trees, list) or not 1 <= len(trees) <= MAX_TREES:
        raise ArtifactError(f'trees must be a list of 1..{MAX_TREES} trees')
    out, total = [], 0
    nf = len(FEATURES)
    for ti, t in enumerate(trees):
        if not isinstance(t, dict):
            raise ArtifactError(f'tree {ti} must be an object')
        lv = t.get('leaf_value')
        n = len(lv) if isinstance(lv, list) else -1
        if not 1 <= n <= MAX_NODES_PER_TREE:
            raise ArtifactError(f'tree {ti}: node count must be 1..{MAX_NODES_PER_TREE}')
        total += n
        if total > MAX_TOTAL_NODES:
            raise ArtifactError(f'model exceeds {MAX_TOTAL_NODES} nodes in total')
        feat = _arr(t, 'split_feature', n, np.int64)
        thr = _arr(t, 'threshold', n, np.float64)
        left = _arr(t, 'left', n, np.int64)
        right = _arr(t, 'right', n, np.int64)
        dl = _arr(t, 'default_left', n, np.int64)
        val = _arr(t, 'leaf_value', n, np.float64)
        leaf = left < 0
        if np.any(leaf != (right < 0)):
            raise ArtifactError(f'tree {ti}: left/right must both be -1 at leaves')
        inner = ~leaf
        if np.any((feat[inner] < 0) | (feat[inner] >= nf)):
            raise ArtifactError(f'tree {ti}: split_feature out of range')
        if np.any(~np.isfinite(thr[inner])):
            raise ArtifactError(f'tree {ti}: thresholds must be finite')
        if np.any(~np.isfinite(val[leaf])):
            raise ArtifactError(f'tree {ti}: leaf values must be finite')
        if np.any(~np.isin(dl[inner], [0, 1])):
            raise ArtifactError(f'tree {ti}: default_left must be 0/1')
        # children must point forward (no cycles) and every node except root has exactly one parent
        idx = np.arange(n)
        if np.any((left[inner] <= idx[inner]) | (left[inner] >= n) | (right[inner] <= idx[inner]) | (right[inner] >= n)):
            raise ArtifactError(f'tree {ti}: children must have larger indices than their parent')
        par = np.zeros(n, np.int64)
        np.add.at(par, left[inner], 1)
        np.add.at(par, right[inner], 1)
        if par[0] != 0 or np.any(par[1:] != 1):
            raise ArtifactError(f'tree {ti}: nodes must form a single binary tree rooted at node 0')
        depth = np.zeros(n, np.int64)
        for i in np.where(inner)[0]:
            depth[left[i]] = depth[right[i]] = depth[i] + 1
        if depth.max() > MAX_DEPTH:
            raise ArtifactError(f'tree {ti}: depth exceeds {MAX_DEPTH}')
        out.append((feat.astype(np.int32), thr, left.astype(np.int32), right.astype(np.int32), dl.astype(bool),
                    val, int(depth.max())))
    return dict(base=float(base), trees=out)


def margin(model, X):
    """X: (n, F) float array with NaN for missing. Returns raw margin (n,)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    m = np.full(n, model['base'])
    rows = np.arange(n)
    for feat, thr, left, right, dl, val, depth in model['trees']:
        node = np.zeros(n, np.int32)
        for _ in range(depth):
            f = feat[node]
            l = left[node]
            internal = l >= 0
            if not internal.any():
                break
            x = X[rows, np.where(internal, f, 0)]
            go_left = np.where(np.isnan(x), dl[node], x <= thr[node])
            node = np.where(internal, np.where(go_left, l, right[node]), node)
        m += val[node]
    return m


def score(model, X):
    z = np.clip(margin(model, X), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def load_policy(obj):
    """Returns (review_thr, decline_thr) arrays of shape (8 verticals, 4 regions, 4 amount bands)."""
    if not isinstance(obj, dict) or obj.get('format') != 'risk_policy_v1':
        raise ArtifactError('policy format must be "risk_policy_v1"')

    def thr_pair(d, where):
        if not isinstance(d, dict):
            raise ArtifactError(f'{where}: must be an object')
        r, k = d.get('review'), d.get('decline')
        for name, v in (('review', r), ('decline', k)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ArtifactError(f'{where}: {name} must be a finite number')
        if not (0.0 <= r <= k <= 1.0 + 1e-9):
            raise ArtifactError(f'{where}: need 0 <= review <= decline <= 1')
        return float(r), float(k)

    r0, k0 = thr_pair(obj.get('default'), 'default')
    R = np.full((len(VERTICALS), len(REGIONS), NBANDS), r0)
    K = np.full((len(VERTICALS), len(REGIONS), NBANDS), k0)
    rows = obj.get('segments', [])
    if not isinstance(rows, list) or len(rows) > len(VERTICALS) * len(REGIONS) * NBANDS:
        raise ArtifactError('segments must be a list of at most 128 rows')
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ArtifactError(f'segments[{i}] must be an object')
        v, g, b = row.get('vertical'), row.get('region'), row.get('amount_band')
        if v not in VERTICALS or g not in REGIONS or not isinstance(b, int) or isinstance(b, bool) or not 0 <= b < NBANDS:
            raise ArtifactError(f'segments[{i}]: unknown vertical/region/amount_band')
        key = (v, g, b)
        if key in seen:
            raise ArtifactError(f'segments[{i}]: duplicate segment {key}')
        seen.add(key)
        r, k = thr_pair(row, f'segments[{i}]')
        R[VERTICALS.index(v), REGIONS.index(g), b] = r
        K[VERTICALS.index(v), REGIONS.index(g), b] = k
    return R, K


def load_artifacts(model_path, policy_path):
    import os
    for p in (model_path, policy_path):
        if os.path.getsize(p) > MAX_FILE_BYTES:
            raise ArtifactError(f'{os.path.basename(p)} exceeds {MAX_FILE_BYTES} bytes')
    with open(model_path) as f:
        model = load_model(json.load(f, parse_constant=_reject_const))
    with open(policy_path) as f:
        policy = load_policy(json.load(f, parse_constant=_reject_const))
    return model, policy


def _reject_const(c):
    raise ArtifactError(f'invalid JSON constant {c}')


def amount_band(amount_usd):
    a = np.nan_to_num(np.asarray(amount_usd, dtype=np.float64), nan=0.0)
    return np.searchsorted(np.asarray(AMOUNT_BANDS_USD), a, side='right')


# ---- manual review queue (fraud ops, Pacific time staffing) ----
# 2026-11-01 09:00 UTC = end of US daylight time
_DST_END_UTC = 1793523600


def pt_hour_index(ts_utc):
    """Absolute PT-hour bucket for each auth timestamp (epoch seconds, UTC)."""
    ts = np.asarray(ts_utc, dtype=np.int64)
    off = np.where(ts < _DST_END_UTC, -7 * 3600, -8 * 3600)
    return (ts + off) // 3600


def decide(s, X, ts_utc, policy, capacity_by_pt_hour):
    """Decision engine. Returns actions: 0 approve, 1 manual review, 2 decline.

    decline if score >= decline threshold of the segment; otherwise review if score >= review threshold,
    subject to analyst capacity: within each PT hour, flagged transactions are worked in order of
    expected loss (score x amount_usd); flagged transactions beyond that hour's capacity are released
    (approved) because orders cannot be held longer.
    """
    R, K = policy
    X = np.asarray(X, dtype=np.float64)
    v = X[:, FEATURES.index('vertical')].astype(np.int64)
    g = X[:, FEATURES.index('region')].astype(np.int64)
    amt = np.nan_to_num(X[:, FEATURES.index('amount_usd')], nan=0.0)
    b = amount_band(amt)
    rt, kt = R[v, g, b], K[v, g, b]
    act = np.zeros(len(s), np.int8)
    act[s >= kt] = 2
    cand = np.where((act == 0) & (s >= rt))[0]
    if len(cand):
        hb = pt_hour_index(ts_utc)[cand]
        pri = s[cand] * amt[cand]
        order = np.lexsort((-pri, hb))
        hb_o = hb[order]
        starts = np.r_[0, np.flatnonzero(np.diff(hb_o)) + 1]
        lens = np.diff(np.r_[starts, len(order)])
        rank = np.arange(len(order)) - np.repeat(starts, lens)
        cap = np.asarray(capacity_by_pt_hour)[(hb_o % 24).astype(np.int64)]
        act[cand[order[rank < cap]]] = 1
    return act

# ===== END FILE: tmp/tests/riskengine.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
set -u
chmod -R go-rwx /tests 2>/dev/null || true
mkdir -p /logs/verifier
chown root:root /logs/verifier 2>/dev/null || true
chmod 755 /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/metrics.json
cd /tests
python3 -I /tests/hidden_eval.py || {
  rm -f /logs/verifier/reward.txt
  echo "0.000000" > /logs/verifier/reward.txt
}

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_security.py =====
"""Regression checks for the verifier's handling of untrusted /app contents.

Runs hidden_eval.py against crafted /app directories (via PRR_APP_DIR / PRR_LOG_DIR) and checks
that every malformed or hostile release is rejected cleanly with reward 0, that nothing outside the
log directory is written, and that /tests is not readable by an unprivileged user once test.sh has
run. Positive controls: the v7 release and a worst-case legal model are accepted.

    python3 /tests/test_security.py            (as root, after test.sh)
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
import riskengine as RE  # noqa: E402

EVAL = os.path.join(TESTS, "hidden_eval.py")
results = []


def leaf_tree(v=0.0):
    return dict(split_feature=[0], threshold=[0.0], left=[-1], right=[-1], default_left=[0], leaf_value=[v])


def stump(f=0, t=100.0):
    return dict(split_feature=[f, 0, 0], threshold=[t, 0.0, 0.0], left=[1, -1, -1], right=[2, -1, -1],
                default_left=[1, 0, 0], leaf_value=[0.0, -0.5, 0.5])


def full_tree(depth):
    """complete binary tree in pre-order with children after parents"""
    sf, th, le, ri, dl, lv = [], [], [], [], [], []

    def build(d):
        i = len(sf)
        sf.append(d % len(RE.FEATURES)); th.append(float(d)); le.append(-1); ri.append(-1); dl.append(d % 2); lv.append(0.0)
        if d < depth:
            le[i] = build(d + 1)
            ri[i] = build(d + 1)
        else:
            lv[i] = 0.001 * (i % 7)
        return i
    build(0)
    return dict(split_feature=sf, threshold=th, left=le, right=ri, default_left=dl, leaf_value=lv)


MODEL_OK = dict(format="tree_ensemble_v1", feature_names=list(RE.FEATURES), base_margin=-5.0, trees=[stump()])
POLICY_OK = dict(format="risk_policy_v1", default=dict(review=0.02, decline=0.6), segments=[])


def run(setup, expect_ok=False, name="", timeout=900):
    root = tempfile.mkdtemp()
    os.chmod(root, 0o755)
    app, logs = os.path.join(root, "app"), os.path.join(root, "logs")
    os.makedirs(os.path.join(app, "release")); os.makedirs(logs)
    extra = setup(os.path.join(app, "release"), logs)
    extra = extra if isinstance(extra, dict) else {}
    env = dict(os.environ, PRR_APP_DIR=app, PRR_LOG_DIR=logs, PRR_TESTS_DIR=TESTS)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-I", EVAL], env=env, capture_output=True, text=True, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    dt = time.time() - t0
    rpath = os.path.join(logs, "reward.txt"); mpath = os.path.join(logs, "metrics.json")
    ok = rc == 0 and os.path.isfile(rpath) and not os.path.islink(rpath)
    reward = float(open(rpath).read()) if ok else None
    status = json.load(open(mpath)).get("status") if ok and os.path.isfile(mpath) else None
    passed = (ok and status == "ok") if expect_ok else (ok and reward == 0.0 and status != "ok")
    for check in extra.get("post", []):
        passed = passed and check()
    results.append((name, passed, rc, reward, (status or "")[:70], round(dt, 1)))
    shutil.rmtree(root, ignore_errors=True)


def write(rel, model=None, policy=None, raw_model=None):
    with open(os.path.join(rel, "model.json"), "w") as f:
        f.write(raw_model if raw_model is not None else json.dumps(model if model is not None else MODEL_OK))
    with open(os.path.join(rel, "policy.json"), "w") as f:
        f.write(json.dumps(policy if policy is not None else POLICY_OK))


def bad_model(mut):
    m = copy.deepcopy(MODEL_OK)
    mut(m)
    return lambda r, l: write(r, model=m)


def bad_policy(mut):
    p = copy.deepcopy(POLICY_OK)
    mut(p)
    return lambda r, l: write(r, policy=p)


# ------------------------------------------------------------------ files
run(lambda r, l: None, name="missing release files")
run(lambda r, l: write(r) or os.unlink(os.path.join(r, "policy.json")), name="missing policy")
run(lambda r, l: (write(r), os.unlink(os.path.join(r, "model.json")),
                  os.symlink(os.path.join(TESTS, "hidden", "anchors.json"), os.path.join(r, "model.json"))),
    name="symlink to hidden file")
run(lambda r, l: (write(r), os.unlink(os.path.join(r, "model.json")), os.mkfifo(os.path.join(r, "model.json"))),
    name="fifo (must not block)", timeout=120)
run(lambda r, l: (write(r), os.unlink(os.path.join(r, "policy.json")), os.makedirs(os.path.join(r, "policy.json"))),
    name="directory")
run(lambda r, l: write(r, raw_model=" " * (RE.MAX_FILE_BYTES + 10)), name="oversized file")
run(lambda r, l: write(r, raw_model="[" * 200000 + "]" * 200000), name="deep nesting")
run(lambda r, l: write(r, raw_model=json.dumps(MODEL_OK).replace("-5.0", "NaN")), name="NaN constant")
run(lambda r, l: write(r, raw_model=json.dumps(MODEL_OK).replace("-5.0", "1e400")), name="overflowing number")
run(lambda r, l: write(r, raw_model="\udcff".encode("utf-8", "surrogatepass").decode("latin-1")), name="not utf-8")
# ------------------------------------------------------------------ model schema
run(bad_model(lambda m: m.update(format="xgboost")), name="wrong format")
run(bad_model(lambda m: m.update(feature_names=list(reversed(RE.FEATURES)))), name="wrong feature order")
run(bad_model(lambda m: m.update(trees=[])), name="no trees")
run(bad_model(lambda m: m.update(trees=[leaf_tree()] * (RE.MAX_TREES + 1))), name="too many trees")
run(bad_model(lambda m: m.update(trees=[full_tree(13)])), name="tree too deep / too many nodes")
run(bad_model(lambda m: m.update(trees=[full_tree(8)] * 300)), name="total node budget exceeded")
run(bad_model(lambda m: m["trees"][0].update(left=[0, -1, -1])), name="self loop")
run(bad_model(lambda m: m["trees"][0].update(left=[2, -1, -1], right=[2, -1, -1])), name="node with two parents")
run(bad_model(lambda m: m["trees"][0].update(split_feature=[31, 0, 0])), name="split feature out of range")
run(bad_model(lambda m: m["trees"][0].update(threshold=["7", 0, 0])), name="string threshold")
run(bad_model(lambda m: m["trees"][0].update(leaf_value=[0.0, 0.5, None])), name="null leaf")
run(bad_model(lambda m: m["trees"][0].update(default_left=[2, 0, 0])), name="bad default_left")
run(bad_model(lambda m: m["trees"][0].update(left=[1, -1])), name="ragged arrays")
run(bad_model(lambda m: m.update(base_margin=True)), name="bool base margin")
# ------------------------------------------------------------------ policy schema
run(bad_policy(lambda p: p.update(format="risk_policy_v0")), name="policy wrong format")
run(bad_policy(lambda p: p.update(default=dict(review=0.5, decline=0.4))), name="review above decline")
run(bad_policy(lambda p: p.update(default=dict(review=-0.1, decline=0.4))), name="negative threshold")
run(bad_policy(lambda p: p.update(default=dict(review=0.1, decline=1.5))), name="decline above 1")
run(bad_policy(lambda p: p.update(default=dict(review=False, decline=0.4))), name="bool threshold")
run(bad_policy(lambda p: p.update(segments=[dict(vertical="crypto", region="NA", amount_band=0, review=0.1, decline=0.5)])),
    name="unknown segment")
run(bad_policy(lambda p: p.update(segments=[dict(vertical="travel", region="NA", amount_band=4, review=0.1, decline=0.5)])),
    name="amount band out of range")
run(bad_policy(lambda p: p.update(segments=[dict(vertical="travel", region="NA", amount_band=1, review=0.1, decline=0.5)] * 2)),
    name="duplicate segment")
run(bad_policy(lambda p: p.update(segments=[dict(vertical="travel", region="NA", amount_band=1, review=0.1, decline=0.5)] * 200)),
    name="too many segments")


def pre_symlink(r, l):
    victim = os.path.join(os.path.dirname(l), "victim.txt")
    open(victim, "w").write("untouched")
    os.symlink(victim, os.path.join(l, "reward.txt"))
    os.symlink(victim, os.path.join(l, "metrics.json"))
    write(r)
    return {"post": [lambda: open(victim).read() == "untouched"]}


run(pre_symlink, expect_ok=True, name="pre-existing symlinks in log dir")
# ------------------------------------------------------------------ positive controls
run(lambda r, l: write(r), expect_ok=True, name="valid minimal release accepted")
WORST = dict(format="tree_ensemble_v1", feature_names=list(RE.FEATURES), base_margin=-5.0,
             trees=[full_tree(8)] * 234)
run(lambda r, l: write(r, model=WORST), expect_ok=True, name="worst-case legal model (~120k nodes)")


def chain(depth):
    # internal node at 2i (i < depth): left child 2i+1 is a leaf, right child 2i+2 continues the chain
    n = 2 * depth + 1
    inner = [k % 2 == 0 and k < 2 * depth for k in range(n)]
    return dict(split_feature=[(k // 2) % 31 for k in range(n)], threshold=[-1e9 if inner[k] else 0.0 for k in range(n)],
                left=[k + 1 if inner[k] else -1 for k in range(n)], right=[k + 2 if inner[k] else -1 for k in range(n)],
                default_left=[0] * n, leaf_value=[0.0 if inner[k] else 0.001 for k in range(n)])


DEEP = dict(format="tree_ensemble_v1", feature_names=list(RE.FEATURES), base_margin=-5.0, trees=[chain(12)] * 400)
run(lambda r, l: write(r, model=DEEP), expect_ok=True, name="400 trees of depth 12 (chains)")

# ------------------------------------------------------------------ permissions (Harbor layout only)
if os.geteuid() == 0 and TESTS == "/tests":
    r = subprocess.run(["su", "nobody", "-s", "/bin/sh", "-c", "cat /tests/hidden/anchors.json"], capture_output=True)
    results.append(("/tests unreadable by nobody", r.returncode != 0, r.returncode, None, "", 0))

bad = [x for x in results if not x[1]]
for name, passed, rc, reward, status, dt in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name:40s} rc={rc} reward={reward} t={dt}s status={status}")
print(f"{len(results) - len(bad)}/{len(results)} passed")
sys.exit(1 if bad else 0)

# ===== END FILE: tmp/tests/test_security.py =====


