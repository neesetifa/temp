# ===== BEGIN FILE: tmp/environment/app/solve.py =====
"""Starter solution for transaction_dispute_outcome.

The fallback model predicts the training mean refund fraction. Replace these
functions with a model that uses case, party, and event evidence.
"""
from __future__ import annotations
import numpy as np


def fit_dispute_model(
    train_case_X,
    train_party_X,
    train_party_code,
    train_event_X,
    train_event_code,
    train_event_party,
    train_party_offsets,
    train_event_offsets,
    train_y,
):
    y = np.asarray(train_y, dtype=float)
    mean = float(np.mean(y)) if y.size else 0.5
    return {"mean": mean}


def predict_dispute_outcome(
    case_X,
    party_X,
    party_code,
    event_X,
    event_code,
    event_party,
    party_offsets,
    event_offsets,
    params,
):
    n = np.asarray(case_X).shape[0]
    mean = float(params.get("mean", 0.5)) if isinstance(params, dict) else 0.5
    return np.full(n, mean, dtype=float)

# ===== END FILE: tmp/environment/app/solve.py =====

# ===== BEGIN FILE: tmp/environment/instruction.md =====
# Transaction dispute outcome

The file `app/solve.py` contains a simple fallback for a dispute outcome model. Replace it with a deterministic model that predicts the final refund fraction for each dispute case.

Each case has case-level features, a packed set of parties, and a packed set of evidence events. Parties have role/category codes. Events have event/source/document codes, numeric evidence fields, and local party routing indices. The target is a continuous refund fraction between 0 and 1.

Implement these functions in `app/solve.py`:

```python
def fit_dispute_model(
    train_case_X,
    train_party_X,
    train_party_code,
    train_event_X,
    train_event_code,
    train_event_party,
    train_party_offsets,
    train_event_offsets,
    train_y,
):
    ...


def predict_dispute_outcome(
    case_X,
    party_X,
    party_code,
    event_X,
    event_code,
    event_party,
    party_offsets,
    event_offsets,
    params,
):
    ...
```

`party_offsets` and `event_offsets` use the usual packed-array convention: entries for case `i` are in `[offsets[i], offsets[i+1])`. `event_party[:, 0]` and `event_party[:, 1]` are local party indices within the case; `-1` means that side is not attached to a known party.

Predictions must be finite, deterministic, one-dimensional, and have shape `(n_cases,)`. Do not mutate input arrays. No external data or network access is needed.

# ===== END FILE: tmp/environment/instruction.md =====

# ===== BEGIN FILE: tmp/environment/tests/test_public.py =====
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "solve.py"
DATA = np.load(ROOT / "app" / "train_data.npz", allow_pickle=False)
PUBLIC = np.load(ROOT / "app" / "public_eval.npz", allow_pickle=False)


def load_mod():
    spec = importlib.util.spec_from_file_location("solve", str(APP))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def fit_args():
    return [
        DATA["train_case_X"], DATA["train_party_X"], DATA["train_party_code"],
        DATA["train_event_X"], DATA["train_event_code"], DATA["train_event_party"],
        DATA["train_party_offsets"], DATA["train_event_offsets"], DATA["train_y"],
    ]


def pred_args():
    return [
        PUBLIC["case_X"], PUBLIC["party_X"], PUBLIC["party_code"],
        PUBLIC["event_X"], PUBLIC["event_code"], PUBLIC["event_party"],
        PUBLIC["party_offsets"], PUBLIC["event_offsets"],
    ]


def test_api_shape_and_finite():
    mod = load_mod()
    params = mod.fit_dispute_model(*fit_args())
    pred = np.asarray(mod.predict_dispute_outcome(*pred_args(), params), dtype=float)
    assert pred.shape == PUBLIC["y"].shape
    assert np.all(np.isfinite(pred))


def test_predict_is_deterministic():
    mod = load_mod()
    params = mod.fit_dispute_model(*fit_args())
    p1 = np.asarray(mod.predict_dispute_outcome(*pred_args(), params), dtype=float)
    p2 = np.asarray(mod.predict_dispute_outcome(*pred_args(), params), dtype=float)
    assert np.allclose(p1, p2, rtol=0, atol=1e-12)


def test_does_not_mutate_inputs():
    mod = load_mod()
    fa = [x.copy() for x in fit_args()]
    before = [x.copy() for x in fa]
    params = mod.fit_dispute_model(*fa)
    for a, b in zip(fa, before):
        assert np.array_equal(a, b)
    pa = [x.copy() for x in pred_args()]
    before = [x.copy() for x in pa]
    _ = mod.predict_dispute_outcome(*pa, params)
    for a, b in zip(pa, before):
        assert np.array_equal(a, b)


def test_offsets_are_consistent():
    assert DATA["train_party_offsets"][0] == 0
    assert DATA["train_party_offsets"][-1] == DATA["train_party_X"].shape[0]
    assert DATA["train_event_offsets"][0] == 0
    assert DATA["train_event_offsets"][-1] == DATA["train_event_X"].shape[0]
    assert PUBLIC["party_offsets"][0] == 0
    assert PUBLIC["party_offsets"][-1] == PUBLIC["party_X"].shape[0]
    assert PUBLIC["event_offsets"][0] == 0
    assert PUBLIC["event_offsets"][-1] == PUBLIC["event_X"].shape[0]
    assert np.all(np.diff(DATA["train_party_offsets"]) > 0)
    assert np.all(np.diff(DATA["train_event_offsets"]) > 0)
    assert np.all(np.diff(PUBLIC["party_offsets"]) > 0)
    assert np.all(np.diff(PUBLIC["event_offsets"]) > 0)

# ===== END FILE: tmp/environment/tests/test_public.py =====

# ===== BEGIN FILE: tmp/solution/reference_solution.py =====
"""Reference solution for transaction_dispute_outcome v0.1 core probe.

The reference is a structured party-routed evidence model. It does not use the
synthetic generator latents; it builds visible evidence features from party roles,
event routing, source codes, missingness, and contradiction patterns, then fits a
regularized linear model with nonlinear summary features.
"""
from __future__ import annotations

import numpy as np

N_ROLES = 5
N_FAMILIES = 12
N_SOURCES = 6
N_DOC = 6
ROLE_CLAIMANT = 0
ROLE_RESPONDENT = 1
ROLE_INTERMEDIARY = 2
ROLE_SHIPPER = 3
ROLE_WITNESS = 4

F_CLAIM = 0
F_REBUTTAL = 1
F_DOCUMENT = 2
F_COUNTER_DOCUMENT = 3
F_DELIVERY_OK = 4
F_QUALITY_ISSUE = 5
F_POLICY_EXCEPTION = 6
F_THIRD_PARTY_VERIFY = 7
F_MISSING_DOC = 8
F_DUPLICATE_REPORT = 9
F_LATE_EVIDENCE = 10
F_AUTOMATED_SIGNAL = 11

S_SELF = 0
S_COUNTERPARTY = 1
S_PLATFORM = 2
S_THIRD_PARTY = 3
S_AUTOMATED = 4
S_MISSING = 5

D_RECEIPT = 1
D_DELIVERY = 2
D_QUALITY = 3
D_POLICY = 4
D_IDENTITY = 5


def _role_desire(role: int) -> int:
    if role == ROLE_CLAIMANT:
        return 1
    if role == ROLE_RESPONDENT:
        return -1
    return 0


def _side_from_visible(fam: int, src: int, doc: int, role: int, regime_proxy: float) -> int:
    """Visible sign heuristic: +1 supports claimant/refund; -1 supports respondent."""
    if fam in (F_CLAIM, F_REBUTTAL, F_DOCUMENT, F_COUNTER_DOCUMENT):
        return _role_desire(role)
    if fam == F_DELIVERY_OK:
        return -1 if role != ROLE_CLAIMANT else 1
    if fam == F_QUALITY_ISSUE:
        return 1 if role != ROLE_RESPONDENT else -1
    if fam == F_POLICY_EXCEPTION:
        strict_like = regime_proxy > 0.52
        if strict_like:
            return -1 if doc in (D_POLICY, D_DELIVERY) else 1
        return 1 if doc in (D_QUALITY, D_IDENTITY) else -1
    if fam == F_THIRD_PARTY_VERIFY:
        d = _role_desire(role)
        if d != 0:
            return d
        return 1 if doc in (D_QUALITY, D_IDENTITY) else -1
    if fam == F_MISSING_DOC:
        d = _role_desire(role)
        return -d if d != 0 else (1 if doc in (D_DELIVERY, D_POLICY) else -1)
    if fam == F_DUPLICATE_REPORT:
        d = _role_desire(role)
        return -d if d != 0 else 0
    if fam == F_LATE_EVIDENCE:
        return _role_desire(role)
    if fam == F_AUTOMATED_SIGNAL:
        return 1 if (regime_proxy > 0.55 and doc in (D_QUALITY, D_IDENTITY)) else -1
    return 0


def _source_weight(src: int) -> float:
    return [0.42, 0.40, 0.78, 1.00, 0.65, 0.72][int(src)]


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 6.0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = (X - mean) / std
    Z = np.column_stack([np.ones(Z.shape[0]), Z])
    reg = np.eye(Z.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return {"mean": mean, "std": std, "coef": coef, "alpha": float(alpha)}


def _ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = (X - model["mean"]) / model["std"]
    Z = np.column_stack([np.ones(Z.shape[0]), Z])
    pred = Z @ model["coef"]
    return np.clip(pred, 0.0, 1.0)


def _safe_idx(local_idx: int, start: int, stop: int) -> int | None:
    if local_idx < 0 or start + local_idx >= stop:
        return None
    return start + local_idx


def _extract_features(case_X, party_X, party_code, event_X, event_code, event_party,
                      party_offsets, event_offsets, feature_set: str = "reference") -> np.ndarray:
    case_X = np.asarray(case_X, dtype=np.float64)
    party_X = np.asarray(party_X, dtype=np.float64)
    party_code = np.asarray(party_code, dtype=np.int64)
    event_X = np.asarray(event_X, dtype=np.float64)
    event_code = np.asarray(event_code, dtype=np.int64)
    event_party = np.asarray(event_party, dtype=np.int64)
    party_offsets = np.asarray(party_offsets, dtype=np.int64)
    event_offsets = np.asarray(event_offsets, dtype=np.int64)

    n = case_X.shape[0]
    rows = []
    for i in range(n):
        ps, pe = int(party_offsets[i]), int(party_offsets[i + 1])
        es, ee = int(event_offsets[i]), int(event_offsets[i + 1])
        cX = case_X[i]
        pX = party_X[ps:pe]
        pC = party_code[ps:pe]
        eX = event_X[es:ee]
        eC = event_code[es:ee]
        eP = event_party[es:ee]
        roles = pC[:, 0] if pC.size else np.zeros(0, dtype=np.int64)
        regime_proxy = float(0.55 * cX[4] + 0.45 * cX[7]) if cX.shape[0] >= 8 else 0.5

        feats = []
        # Basic case features and simple nonlinearities.
        if feature_set not in ("event_count",):
            feats.extend(cX.tolist())
            feats.extend([cX[0] ** 2, cX[1] ** 2, cX[0] * cX[1], cX[4] * cX[7]])
        else:
            feats.extend([1.0])

        # Party role counts and visible party proxies.
        role_counts = np.zeros(N_ROLES)
        role_proxy_mean = np.zeros((N_ROLES, min(3, pX.shape[1] if pX.ndim == 2 else 0)))
        role_proxy_count = np.zeros(N_ROLES)
        for j, r in enumerate(roles):
            if 0 <= r < N_ROLES:
                role_counts[r] += 1
                if role_proxy_mean.shape[1] > 0:
                    role_proxy_mean[r] += pX[j, :role_proxy_mean.shape[1]]
                    role_proxy_count[r] += 1
        if role_proxy_mean.shape[1] > 0:
            role_proxy_mean = role_proxy_mean / np.maximum(role_proxy_count[:, None], 1.0)
        if feature_set not in ("amount_proxy", "event_count"):
            feats.extend(role_counts.tolist())
            feats.extend(role_proxy_mean.ravel().tolist())

        # Event aggregate containers.
        bag_family = np.zeros(N_FAMILIES)
        bag_source = np.zeros(N_SOURCES)
        bag_doc = np.zeros(N_DOC)
        side_family = np.zeros((2, N_FAMILIES))  # claimant, respondent
        side_source = np.zeros((2, N_SOURCES))
        side_family_source = np.zeros((2, N_FAMILIES, N_SOURCES))
        role_family_source = np.zeros((N_ROLES, N_FAMILIES, N_SOURCES))
        role_family_doc = np.zeros((N_ROLES, N_FAMILIES, N_DOC))
        signed_family_source = np.zeros((N_FAMILIES, N_SOURCES))
        signed_doc = np.zeros(N_DOC)
        signed_time = np.zeros(4)  # early/late x claimant/respondent sign bins
        counter_side_family_source = np.zeros((2, N_FAMILIES, N_SOURCES))
        counter_role_family = np.zeros((N_ROLES, N_FAMILIES))
        counts = np.zeros(6)
        claimant_support = respondent_support = 0.0
        claimant_counter_support = respondent_counter_support = 0.0
        claimant_self = respondent_self = 0.0
        claimant_verified = respondent_verified = 0.0
        missing_claimant = missing_respondent = 0.0
        late_self_claimant = late_self_respondent = 0.0
        late_verified_claimant = late_verified_respondent = 0.0
        amount_support = np.zeros(2)
        max_claim = max_resp = 0.0

        for k in range(eC.shape[0]):
            fam, src, doc = map(int, eC[k, :3])
            fam = int(np.clip(fam, 0, N_FAMILIES - 1))
            src = int(np.clip(src, 0, N_SOURCES - 1))
            doc = int(np.clip(doc, 0, N_DOC - 1))
            p_local = int(eP[k, 0]) if eP.shape[1] > 0 else -1
            c_local = int(eP[k, 1]) if eP.shape[1] > 1 else -1
            p_global = _safe_idx(p_local, ps, pe)
            c_global = _safe_idx(c_local, ps, pe)
            role = int(party_code[p_global, 0]) if p_global is not None else ROLE_INTERMEDIARY
            counter_role = int(party_code[c_global, 0]) if c_global is not None else ROLE_INTERMEDIARY
            strength = float(eX[k, 2]) if eX.shape[1] > 2 else 0.5
            time = float(eX[k, 0]) if eX.shape[1] > 0 else 0.5
            amount = float(eX[k, 1]) if eX.shape[1] > 1 else 1.0
            conf = float(eX[k, 4]) if eX.shape[1] > 4 else 0.5
            comp = float(eX[k, 5]) if eX.shape[1] > 5 else 0.5
            rel_proxy = float(party_X[p_global, 0]) if p_global is not None and party_X.shape[1] > 0 else 0.5
            side = _side_from_visible(fam, src, doc, role, regime_proxy)
            # Some neutral platform/witness/automated records are primarily evidence
            # *against* the counterparty they are attached to. This is deliberately
            # absent from the simple role-event-source formula probe.
            counter_side = 0
            if role in (ROLE_INTERMEDIARY, ROLE_SHIPPER, ROLE_WITNESS) and counter_role in (ROLE_CLAIMANT, ROLE_RESPONDENT):
                if fam in (F_THIRD_PARTY_VERIFY, F_POLICY_EXCEPTION, F_AUTOMATED_SIGNAL):
                    counter_side = -_role_desire(counter_role)
                    if fam == F_THIRD_PARTY_VERIFY:
                        if doc in (D_QUALITY, D_IDENTITY):
                            counter_side = +1 if counter_role == ROLE_RESPONDENT else -1
                        elif doc in (D_DELIVERY, D_POLICY):
                            counter_side = -1 if counter_role == ROLE_CLAIMANT else +1
            if feature_set in ("reference", "reference_no_missingness", "reference_no_contradiction", "reference_no_source_reliability", "counter_routed", "nonlinear_formula_probe") and counter_side != 0:
                side = counter_side
            if feature_set == "reference_no_missingness" and fam == F_MISSING_DOC:
                # Strict ablation: missing/censored markers are not interpreted as evidence.
                side = 0
            if feature_set in ("source_agnostic_party", "reference_no_source_reliability"):
                src_factor = 0.66
            else:
                src_factor = _source_weight(src)
            w = strength * (0.35 + 0.65 * conf) * (0.45 + 0.55 * comp) * src_factor * (0.75 + 0.5 * rel_proxy)
            if time > 0.72:
                if src in (S_PLATFORM, S_THIRD_PARTY):
                    w *= 0.85
                else:
                    w *= 0.45
            w_amt = w * np.clip(0.6 + 0.25 * np.log1p(max(amount, 0.0)), 0.45, 1.25)

            bag_family[fam] += 1.0
            bag_source[src] += 1.0
            bag_doc[doc] += 1.0
            if 0 <= role < N_ROLES:
                role_family_source[role, fam, src] += w_amt
                role_family_doc[role, fam, doc] += w_amt
            if 0 <= counter_role < N_ROLES:
                counter_role_family[counter_role, fam] += w_amt
            if counter_side > 0:
                claimant_counter_support += w_amt
                counter_side_family_source[0, fam, src] += w_amt
            elif counter_side < 0:
                respondent_counter_support += w_amt
                counter_side_family_source[1, fam, src] += w_amt
            if side > 0:
                claimant_support += w_amt
                side_family[0, fam] += w_amt
                side_source[0, src] += w_amt
                side_family_source[0, fam, src] += w_amt
                amount_support[0] += w_amt * amount
                max_claim = max(max_claim, w_amt)
                if src in (S_SELF, S_COUNTERPARTY):
                    claimant_self += w_amt
                if src in (S_PLATFORM, S_THIRD_PARTY, S_AUTOMATED):
                    claimant_verified += w_amt
                if fam == F_LATE_EVIDENCE:
                    if src in (S_PLATFORM, S_THIRD_PARTY): late_verified_claimant += w_amt
                    else: late_self_claimant += w_amt
            elif side < 0:
                respondent_support += w_amt
                side_family[1, fam] += w_amt
                side_source[1, src] += w_amt
                side_family_source[1, fam, src] += w_amt
                amount_support[1] += w_amt * amount
                max_resp = max(max_resp, w_amt)
                if src in (S_SELF, S_COUNTERPARTY):
                    respondent_self += w_amt
                if src in (S_PLATFORM, S_THIRD_PARTY, S_AUTOMATED):
                    respondent_verified += w_amt
                if fam == F_LATE_EVIDENCE:
                    if src in (S_PLATFORM, S_THIRD_PARTY): late_verified_respondent += w_amt
                    else: late_self_respondent += w_amt
            if fam == F_MISSING_DOC:
                if role == ROLE_CLAIMANT:
                    missing_claimant += w_amt
                elif role == ROLE_RESPONDENT:
                    missing_respondent += w_amt
            signed_family_source[fam, src] += side * w_amt
            signed_doc[doc] += side * w_amt
            signed_time[(0 if time < 0.55 else 2) + (0 if side >= 0 else 1)] += abs(w_amt)
            counts[0] += 1.0
            counts[1] += strength
            counts[2] += conf
            counts[3] += time
            counts[4] += amount
            counts[5] += float(time > 0.72)

        if counts[0] > 0:
            counts[1:5] /= counts[0]
        event_total = max(counts[0], 1.0)
        # Normalize some bag counts to reduce pure case-size proxy dominance.
        bag_norm = np.concatenate([bag_family, bag_source, bag_doc]) / event_total

        contrast = claimant_support - respondent_support
        counter_contrast = claimant_counter_support - respondent_counter_support
        verified_contrast = claimant_verified - respondent_verified
        self_contrast = claimant_self - respondent_self
        contradiction = min(claimant_self + 0.35 * claimant_verified, respondent_self + 0.35 * respondent_verified)
        override = np.tanh(verified_contrast) * np.tanh(contradiction)
        missing_contrast = missing_respondent - missing_claimant
        late_valid_contrast = late_verified_claimant - late_verified_respondent
        late_weak_contrast = late_self_claimant - late_self_respondent
        support_sum = claimant_support + respondent_support

        if feature_set == "amount_proxy":
            row = [1.0, cX[0], cX[0] ** 2, counts[4], amount_support[0] - amount_support[1]]
            rows.append(np.asarray(row, dtype=np.float64)); continue
        if feature_set == "event_count":
            row = [counts[0], counts[5], counts[1], counts[2]]
            rows.append(np.asarray(row, dtype=np.float64)); continue

        if feature_set in ("case_bag", "reference_no_party_routing"):
            feats.extend(counts.tolist())
            feats.extend(bag_norm.tolist())
            feats.extend((bag_family / event_total * counts[1]).tolist())
            rows.append(np.asarray(feats, dtype=np.float64)); continue

        # Party-routed base features.
        if feature_set != "reference_no_party_routing":
            feats.extend(counts.tolist())
            feats.extend([claimant_support, respondent_support, contrast, support_sum,
                          max_claim, max_resp, max_claim - max_resp,
                          np.tanh(contrast), np.tanh(support_sum)])
            feats.extend(side_family.ravel().tolist())
            feats.extend(signed_doc.tolist())
            feats.extend(signed_time.tolist())

        if feature_set == "source_agnostic_party":
            rows.append(np.asarray(feats, dtype=np.float64)); continue

        if feature_set in ("counter_routed", "reference", "nonlinear_formula_probe"):
            feats.extend([claimant_counter_support, respondent_counter_support, counter_contrast,
                          np.tanh(counter_contrast), counter_contrast * cX[4]])
            feats.extend(counter_side_family_source.ravel().tolist())
            feats.extend(counter_role_family.ravel().tolist())

        # Source-aware features.
        if feature_set != "reference_no_source_reliability":
            feats.extend(side_source.ravel().tolist())
            feats.extend(signed_family_source.ravel().tolist())
            if feature_set in ("formula_probe",):
                feats.extend(side_family_source.ravel().tolist())
                feats.extend(role_family_source.ravel().tolist())
                feats.extend(role_family_doc.ravel().tolist())
            elif feature_set in ("nonlinear_formula_probe",):
                feats.extend(side_family_source.ravel().tolist())
                feats.extend(role_family_source.ravel().tolist())
                feats.extend(role_family_doc.ravel().tolist())
                # Add routed linear features but not the full hand-built nonlinear
                # settlement summaries used by the reference.
                feats.extend((counter_side_family_source.reshape(2, -1).sum(axis=0)).tolist())
            elif feature_set == "reference":
                # Reference keeps structured source-by-family side features, but not the
                # huge role-cross basis used by the formula-recovery probe.
                feats.extend(side_family_source.ravel().tolist())
        else:
            # Collapse sources away deliberately.
            feats.extend(side_family.ravel().tolist())
            feats.extend((side_family.sum(axis=1)).tolist())

        # Missingness features.
        if feature_set != "reference_no_missingness":
            strict_proxy = float(regime_proxy > 0.50)
            lenient_proxy = 1.0 - strict_proxy
            feats.extend([missing_claimant, missing_respondent, missing_contrast,
                          missing_contrast * strict_proxy, missing_contrast * lenient_proxy,
                          missing_claimant * cX[4], missing_respondent * cX[4]])

        # Counter-routed baseline stops here: it knows the counterparty routing but not
        # the nonlinear override / settlement heuristics.
        if feature_set == "counter_routed":
            rows.append(np.asarray(feats, dtype=np.float64)); continue

        # A stronger recovered-formula probe gets generic nonlinear contrasts, but not
        # the reference's hand-built settlement score / calibrated override summaries.
        if feature_set == "nonlinear_formula_probe":
            feats.extend([claimant_self, respondent_self, claimant_verified, respondent_verified,
                          self_contrast, verified_contrast, contradiction, override,
                          min(claimant_support, respondent_support), abs(contrast),
                          late_valid_contrast, late_weak_contrast, counter_contrast,
                          np.tanh(contrast), np.tanh(verified_contrast), np.tanh(counter_contrast),
                          np.tanh(missing_contrast), np.tanh(contradiction) * np.sign(verified_contrast + counter_contrast + 1e-9)])
            rows.append(np.asarray(feats, dtype=np.float64)); continue

        # Contradiction/override nonlinearities.
        if feature_set != "reference_no_contradiction":
            feats.extend([claimant_self, respondent_self, claimant_verified, respondent_verified,
                          self_contrast, verified_contrast, contradiction, override,
                          contradiction * verified_contrast,
                          min(claimant_support, respondent_support),
                          abs(contrast), np.tanh(2.0 * verified_contrast),
                          late_valid_contrast, late_weak_contrast,
                          late_valid_contrast - 0.4 * late_weak_contrast,
                          counter_contrast, np.tanh(counter_contrast),
                          counter_contrast * np.tanh(contradiction)])
            # Low-dimensional nonlinear interaction summaries are what formula probes often miss.
            strict_level = 0.5 + 0.5 * np.clip(cX[4], 0.0, 1.0)
            hand_evidence = (
                0.55 * np.tanh(0.85 * self_contrast)
                + 1.05 * np.tanh(1.05 * verified_contrast)
                + 1.45 * np.tanh(verified_contrast) * np.tanh(1.25 * contradiction)
                - 0.55 * np.tanh(self_contrast) * np.tanh(1.10 * contradiction) * (1.0 - abs(np.tanh(verified_contrast)))
                + 0.90 * np.tanh(1.25 * missing_contrast) * strict_level
                + 0.42 * np.tanh(late_valid_contrast) - 0.25 * np.tanh(late_weak_contrast)
                + 0.55 * np.tanh(counter_contrast) * (0.75 + 0.25 * strict_level)
            )
            prior_like = -0.08 + 0.08 * cX[5] - 0.06 * cX[6] + 0.04 * cX[4]
            hand_pred = 1.0 / (1.0 + np.exp(-1.35 * (prior_like + 1.05 * np.tanh(hand_evidence))))
            feats.extend([
                np.tanh(contrast + 0.6 * override + 0.35 * missing_contrast + 0.35 * counter_contrast),
                np.tanh(verified_contrast) * (1.0 + 0.25 * cX[1]),
                np.tanh(missing_contrast) * strict_level,
                np.tanh(contradiction) * np.sign(verified_contrast + counter_contrast + 1e-9),
                hand_evidence, np.tanh(hand_evidence), hand_pred,
                hand_pred * (1.0 + 0.2 * cX[1]), hand_pred * strict_level,
            ])

        rows.append(np.asarray(feats, dtype=np.float64))

    # Pad not needed because feature_set fixed; ensure rectangular.
    return np.vstack(rows).astype(np.float64)




def fit_dispute_model(train_case_X, train_party_X, train_party_code, train_event_X,
                      train_event_code, train_event_party, train_party_offsets,
                      train_event_offsets, train_y):
    X = _extract_features(train_case_X, train_party_X, train_party_code, train_event_X,
                          train_event_code, train_event_party, train_party_offsets,
                          train_event_offsets, feature_set="reference")
    model = _ridge_fit(X, train_y, alpha=8.0)
    return {"model": model, "feature_set": "reference"}


def predict_dispute_outcome(case_X, party_X, party_code, event_X, event_code,
                            event_party, party_offsets, event_offsets, params):
    feature_set = params.get("feature_set", "reference") if isinstance(params, dict) else "reference"
    model = params["model"] if isinstance(params, dict) and "model" in params else params
    X = _extract_features(case_X, party_X, party_code, event_X, event_code,
                          event_party, party_offsets, event_offsets, feature_set=feature_set)
    return _ridge_predict(model, X)


# Utilities used by calibration/baselines.
def fit_feature_model(train_arrays: dict, feature_set: str, alpha: float = 10.0) -> dict:
    X = _extract_features(train_arrays["case_X"], train_arrays["party_X"], train_arrays["party_code"],
                          train_arrays["event_X"], train_arrays["event_code"], train_arrays["event_party"],
                          train_arrays["party_offsets"], train_arrays["event_offsets"], feature_set=feature_set)
    model = _ridge_fit(X, train_arrays["y"], alpha=alpha)
    return {"model": model, "feature_set": feature_set}


def predict_feature_model(model: dict, arrays: dict) -> np.ndarray:
    X = _extract_features(arrays["case_X"], arrays["party_X"], arrays["party_code"],
                          arrays["event_X"], arrays["event_code"], arrays["event_party"],
                          arrays["party_offsets"], arrays["event_offsets"], feature_set=model["feature_set"])
    return _ridge_predict(model["model"], X)

# ===== END FILE: tmp/solution/reference_solution.py =====

# ===== BEGIN FILE: tmp/tests/agent_call.py =====
#!/usr/bin/env python3
"""Sandbox-side agent runner for transaction_dispute_outcome."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REQUIRED = ("fit_dispute_model", "predict_dispute_outcome")


def _load(path: str | os.PathLike):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_solution", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(path: str | os.PathLike) -> bool:
    try:
        mod = _load(path)
        return all(callable(getattr(mod, name, None)) for name in REQUIRED)
    except Exception:
        return False


def _arrays_equal(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _save_error(out_path: str | os.PathLike, code: str, detail: str = "") -> None:
    np.savez(out_path, n=np.array(0, dtype=np.int64), error=np.array(str(code)), detail=np.array(str(detail)[:1200]))


def run_eval(solve_path: str | os.PathLike, in_path: str | os.PathLike, out_path: str | os.PathLike) -> int:
    try:
        with np.load(in_path, allow_pickle=False) as z:
            train_case_X = np.asarray(z["train_case_X"], dtype=np.float64)
            train_party_X = np.asarray(z["train_party_X"], dtype=np.float64)
            train_party_code = np.asarray(z["train_party_code"], dtype=np.int64)
            train_event_X = np.asarray(z["train_event_X"], dtype=np.float64)
            train_event_code = np.asarray(z["train_event_code"], dtype=np.int64)
            train_event_party = np.asarray(z["train_event_party"], dtype=np.int64)
            train_party_offsets = np.asarray(z["train_party_offsets"], dtype=np.int64)
            train_event_offsets = np.asarray(z["train_event_offsets"], dtype=np.int64)
            train_y = np.asarray(z["train_y"], dtype=np.float64)
            case_X = np.asarray(z["case_X"], dtype=np.float64)
            party_X = np.asarray(z["party_X"], dtype=np.float64)
            party_code = np.asarray(z["party_code"], dtype=np.int64)
            event_X = np.asarray(z["event_X"], dtype=np.float64)
            event_code = np.asarray(z["event_code"], dtype=np.int64)
            event_party = np.asarray(z["event_party"], dtype=np.int64)
            party_offsets = np.asarray(z["party_offsets"], dtype=np.int64)
            event_offsets = np.asarray(z["event_offsets"], dtype=np.int64)
        mod = _load(solve_path)
        if not all(callable(getattr(mod, name, None)) for name in REQUIRED):
            _save_error(out_path, "api_missing", "required fit_dispute_model/predict_dispute_outcome not found")
            return 0

        fit_args_a = [
            train_case_X.copy(), train_party_X.copy(), train_party_code.copy(), train_event_X.copy(),
            train_event_code.copy(), train_event_party.copy(), train_party_offsets.copy(),
            train_event_offsets.copy(), train_y.copy(),
        ]
        fit_before = [x.copy() for x in fit_args_a]
        params_a = mod.fit_dispute_model(*fit_args_a)
        if not all(_arrays_equal(a, b) for a, b in zip(fit_args_a, fit_before)):
            _save_error(out_path, "input_mutation_in_fit", "fit_dispute_model modified one or more input arrays")
            return 0

        fit_args_b = [
            train_case_X.copy(), train_party_X.copy(), train_party_code.copy(), train_event_X.copy(),
            train_event_code.copy(), train_event_party.copy(), train_party_offsets.copy(),
            train_event_offsets.copy(), train_y.copy(),
        ]
        params_b = mod.fit_dispute_model(*fit_args_b)

        pred_args_1 = [case_X.copy(), party_X.copy(), party_code.copy(), event_X.copy(), event_code.copy(), event_party.copy(), party_offsets.copy(), event_offsets.copy()]
        pred_before = [x.copy() for x in pred_args_1]
        pred1 = np.asarray(mod.predict_dispute_outcome(*pred_args_1, params_a), dtype=np.float64)
        if not all(_arrays_equal(a, b) for a, b in zip(pred_args_1, pred_before)):
            _save_error(out_path, "input_mutation_in_predict", "predict_dispute_outcome modified one or more input arrays")
            return 0

        pred_args_2 = [case_X.copy(), party_X.copy(), party_code.copy(), event_X.copy(), event_code.copy(), event_party.copy(), party_offsets.copy(), event_offsets.copy()]
        pred2 = np.asarray(mod.predict_dispute_outcome(*pred_args_2, params_a), dtype=np.float64)
        if pred1.shape != pred2.shape or not np.allclose(pred1, pred2, rtol=0, atol=1e-10, equal_nan=True):
            _save_error(out_path, "nondeterministic_prediction", "two predict calls with the same params differed")
            return 0

        pred_args_3 = [case_X.copy(), party_X.copy(), party_code.copy(), event_X.copy(), event_code.copy(), event_party.copy(), party_offsets.copy(), event_offsets.copy()]
        pred3 = np.asarray(mod.predict_dispute_outcome(*pred_args_3, params_b), dtype=np.float64)
        if pred1.shape != pred3.shape or not np.allclose(pred1, pred3, rtol=0, atol=1e-10, equal_nan=True):
            _save_error(out_path, "nondeterministic_fit", "two independent fit calls led to different predictions")
            return 0

        np.savez(out_path, n=np.array(1, dtype=np.int64), p0=pred1.astype(np.float64))
        return 0
    except Exception:
        _save_error(out_path, "runtime_error", traceback.format_exc())
        return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: agent_call.py SOLVE_PATH {check|eval} [IN_NPZ OUT_NPZ]", file=sys.stderr)
        return 2
    solve_path = argv[1]
    cmd = argv[2]
    if cmd == "check":
        ok = check(solve_path)
        if ok:
            print(json.dumps({"ok": True}))
            return 0
        print(json.dumps({"ok": False}), file=sys.stderr)
        return 1
    if cmd == "eval":
        if len(argv) != 5:
            print("eval requires IN_NPZ and OUT_NPZ", file=sys.stderr)
            return 2
        return run_eval(solve_path, argv[3], argv[4])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# ===== END FILE: tmp/tests/agent_call.py =====

# ===== BEGIN FILE: tmp/tests/generator.py =====
"""Synthetic generator for transaction_dispute_outcome v0.1 core probe.

The task is a reusable fit/predict benchmark over packed multi-party dispute records.
This generator is intentionally synthetic but tries to avoid a single scalar proxy:
case outcome depends on party-routed evidence, source reliability, missing/censored
markers, contradiction resolution, and policy-regime interactions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ROLE_CLAIMANT = 0
ROLE_RESPONDENT = 1
ROLE_INTERMEDIARY = 2
ROLE_SHIPPER = 3
ROLE_WITNESS = 4
N_ROLES = 5

F_CLAIM = 0
F_REBUTTAL = 1
F_DOCUMENT = 2
F_COUNTER_DOCUMENT = 3
F_DELIVERY_OK = 4
F_QUALITY_ISSUE = 5
F_POLICY_EXCEPTION = 6
F_THIRD_PARTY_VERIFY = 7
F_MISSING_DOC = 8
F_DUPLICATE_REPORT = 9
F_LATE_EVIDENCE = 10
F_AUTOMATED_SIGNAL = 11
N_FAMILIES = 12

S_SELF = 0
S_COUNTERPARTY = 1
S_PLATFORM = 2
S_THIRD_PARTY = 3
S_AUTOMATED = 4
S_MISSING = 5
N_SOURCES = 6

D_NONE = 0
D_RECEIPT = 1
D_DELIVERY = 2
D_QUALITY = 3
D_POLICY = 4
D_IDENTITY = 5
N_DOC = 6

SCENARIOS = [
    "simple_uncontested",
    "strong_claimant_evidence",
    "strong_respondent_evidence",
    "contradictory_evidence",
    "third_party_overrides_self_report",
    "missing_document_penalty",
    "missing_not_penalized_regime",
    "late_valid_evidence",
    "late_weak_evidence",
    "multi_party_routing",
    "amount_proxy_trap",
    "role_base_rate_trap",
    "rare_policy_regime",
]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _role_desire(role: int) -> int:
    """+1 means event from this party generally supports claimant/refund."""
    if role == ROLE_CLAIMANT:
        return +1
    if role == ROLE_RESPONDENT:
        return -1
    return 0


def _append_event(events, rng, family, source, primary, counter, *, strength=None,
                  time=None, amount=None, confidence=None, completeness=None,
                  doc_type=None):
    if strength is None:
        strength = rng.beta(2.2, 2.0)
    if time is None:
        time = rng.beta(1.4, 1.8)
    if amount is None:
        amount = np.clip(rng.lognormal(mean=-0.25, sigma=0.5), 0.05, 2.0)
    if confidence is None:
        base = {S_SELF: 0.40, S_COUNTERPARTY: 0.42, S_PLATFORM: 0.72,
                S_THIRD_PARTY: 0.86, S_AUTOMATED: 0.62, S_MISSING: 0.70}[source]
        confidence = np.clip(base + rng.normal(0, 0.11), 0.05, 0.98)
    if completeness is None:
        completeness = np.clip(rng.beta(2.5, 1.7), 0.02, 1.0)
    if doc_type is None:
        if family in (F_DELIVERY_OK,):
            doc_type = D_DELIVERY
        elif family in (F_QUALITY_ISSUE,):
            doc_type = D_QUALITY
        elif family in (F_POLICY_EXCEPTION,):
            doc_type = D_POLICY
        elif family == F_MISSING_DOC:
            doc_type = rng.choice([D_RECEIPT, D_DELIVERY, D_QUALITY, D_IDENTITY])
        else:
            doc_type = int(rng.integers(0, N_DOC))
    delay = np.clip(time + rng.normal(0, 0.12), 0, 1)
    events.append({
        "family": int(family), "source": int(source), "doc": int(doc_type),
        "primary": int(primary), "counter": int(counter),
        "x": [float(time), float(amount), float(strength), float(delay),
              float(confidence), float(completeness)],
    })


def _event_support(ev, roles, party_latent, regime, case_complexity):
    """Return claimant-support and respondent-support latent evidence units."""
    fam = ev["family"]
    src = ev["source"]
    doc = ev["doc"]
    p = ev["primary"]
    c = ev.get("counter", -1)
    role = roles[p] if p >= 0 else ROLE_INTERMEDIARY
    counter_role = roles[c] if c >= 0 and c < len(roles) else ROLE_INTERMEDIARY
    strength = ev["x"][2]
    time = ev["x"][0]
    amount = ev["x"][1]
    conf = ev["x"][4]
    comp = ev["x"][5]

    src_w = np.array([0.38, 0.40, 0.75, 0.98, 0.63, 0.70])[src]
    # Reliability latent is only partially visible through party_X proxies.
    rel = party_latent[p, 0] if p >= 0 else 0.0
    party_w = np.clip(0.75 + 0.35 * rel, 0.35, 1.25)
    base = strength * (0.45 + 0.55 * conf) * (0.55 + 0.45 * comp) * src_w * party_w
    base *= np.clip(0.85 + 0.10 * np.log1p(amount), 0.75, 1.12)

    # Family-specific late evidence handling: third-party/platform late evidence is
    # still useful; self late evidence is heavily discounted.
    if time > 0.72:
        if src in (S_THIRD_PARTY, S_PLATFORM):
            late_w = 0.88 if fam in (F_LATE_EVIDENCE, F_THIRD_PARTY_VERIFY) else 0.76
        else:
            late_w = 0.38 if fam in (F_LATE_EVIDENCE, F_DOCUMENT, F_COUNTER_DOCUMENT) else 0.55
    else:
        late_w = 1.0
    base *= late_w

    sign = 0
    if fam in (F_CLAIM, F_DOCUMENT):
        sign = _role_desire(role)
    elif fam in (F_REBUTTAL, F_COUNTER_DOCUMENT):
        sign = _role_desire(role)
    elif fam == F_DELIVERY_OK:
        sign = -1  # verified delivery tends to favor respondent/no refund
        if role == ROLE_CLAIMANT and src == S_SELF:
            sign = +1  # claimant may assert delivery problem despite code ambiguity
            base *= 0.45
    elif fam == F_QUALITY_ISSUE:
        sign = +1
        if role == ROLE_RESPONDENT and src == S_SELF:
            sign = -1
            base *= 0.42
    elif fam == F_POLICY_EXCEPTION:
        if regime in (0, 3):
            sign = +1 if doc in (D_QUALITY, D_IDENTITY) else -1
        else:
            sign = -1 if doc in (D_POLICY, D_DELIVERY) else +1
        # In review-heavy regimes, a policy exception tied to a neutral primary is often
        # interpreted as evidence about the counterparty. This makes event routing matter
        # beyond a case-level bag of family/source codes.
        if role in (ROLE_INTERMEDIARY, ROLE_WITNESS, ROLE_SHIPPER) and counter_role in (ROLE_CLAIMANT, ROLE_RESPONDENT):
            routed = -_role_desire(counter_role)
            if routed != 0 and doc in (D_POLICY, D_DELIVERY, D_QUALITY):
                sign = routed if regime in (0, 3) else -routed
                base *= 1.12
        base *= 0.8 + 0.25 * (regime == 3)
    elif fam == F_THIRD_PARTY_VERIFY:
        sign = _role_desire(role)
        if role in (ROLE_INTERMEDIARY, ROLE_WITNESS, ROLE_SHIPPER):
            # Neutral verifier: the counterparty routing is load-bearing. The same
            # family/source/doc bag can imply opposite outcomes depending on who the
            # neutral record is attached against.
            routed = -_role_desire(counter_role)
            if routed != 0:
                sign = routed
                if doc in (D_QUALITY, D_IDENTITY):
                    sign = +1 if counter_role == ROLE_RESPONDENT else -1
                elif doc in (D_DELIVERY, D_POLICY):
                    sign = -1 if counter_role == ROLE_CLAIMANT else +1
            else:
                sign = +1 if doc in (D_QUALITY, D_IDENTITY) else -1
        base *= 1.45
    elif fam == F_MISSING_DOC:
        # Missing document hurts the primary party in strict regimes; in lenient
        # regimes it is censored/ordinary and only weakly informative.
        if regime == 1:
            base *= 0.18
        elif regime == 3 and doc == D_POLICY:
            base *= 1.25
        sign = -_role_desire(role)
        if sign == 0:
            sign = +1 if rng_hash(p, doc, regime) % 2 == 0 else -1
    elif fam == F_DUPLICATE_REPORT:
        # Duplicate/self reports are weak and can hurt credibility if repeated.
        sign = -_role_desire(role)
        base *= 0.30 + 0.15 * case_complexity
    elif fam == F_LATE_EVIDENCE:
        sign = _role_desire(role)
        if src in (S_PLATFORM, S_THIRD_PARTY):
            base *= 0.95
        else:
            base *= 0.45
    elif fam == F_AUTOMATED_SIGNAL:
        # Automated signals are regime-dependent and intentionally ambiguous. Neutral
        # automated records attached against a counterparty carry a routed sign, but
        # only in stricter regimes.
        sign = +1 if (regime in (0, 3) and doc in (D_QUALITY, D_IDENTITY)) else -1
        if role in (ROLE_INTERMEDIARY, ROLE_WITNESS, ROLE_SHIPPER) and counter_role in (ROLE_CLAIMANT, ROLE_RESPONDENT) and regime in (0, 3):
            sign = -_role_desire(counter_role)
        if src != S_AUTOMATED:
            base *= 0.6

    if sign > 0:
        return base, 0.0
    if sign < 0:
        return 0.0, base
    return 0.0, 0.0


def rng_hash(*vals):
    h = 2166136261
    for v in vals:
        h = (h ^ (int(v) + 0x9e3779b9)) * 16777619 & 0xFFFFFFFF
    return h


def _make_case(rng: np.random.Generator, scenario_id: int):
    scenario = SCENARIOS[scenario_id]
    n_parties = int(rng.choice([2, 3, 4, 5], p=[0.43, 0.31, 0.18, 0.08]))
    roles = [ROLE_CLAIMANT, ROLE_RESPONDENT]
    if n_parties > 2:
        roles += list(rng.choice([ROLE_INTERMEDIARY, ROLE_SHIPPER, ROLE_WITNESS], size=n_parties - 2, replace=True))
    roles = np.array(roles, dtype=np.int64)

    regime_probs = np.array([0.35, 0.28, 0.27, 0.10])
    if scenario == "rare_policy_regime":
        regime = 3
    elif scenario == "missing_not_penalized_regime":
        regime = 1
    elif scenario == "missing_document_penalty":
        regime = int(rng.choice([0, 2, 3], p=[0.45, 0.35, 0.20]))
    else:
        regime = int(rng.choice(np.arange(4), p=regime_probs))

    log_amount = float(np.clip(rng.normal(0.0, 0.85), -2.0, 2.6))
    complexity = float(np.clip(rng.beta(2.0, 2.7) + 0.18 * (n_parties - 2), 0, 1.3))
    urgency = float(np.clip(rng.beta(1.5, 3.0), 0, 1))
    item_risk = float(np.clip(rng.beta(2.2, 2.2), 0, 1))
    relationship_age = float(np.clip(rng.beta(2.0, 2.8), 0, 1))
    channel_proxy = float(rng.integers(0, 4) / 3.0 + rng.normal(0, 0.03))
    # No direct regime parameter: noisy proxies only.
    regime_proxy_a = float(np.clip((regime / 3.0) + rng.normal(0, 0.22), -0.3, 1.3))
    regime_proxy_b = float(np.clip((regime in (0, 3)) + rng.normal(0, 0.30), -0.4, 1.4))
    case_X = np.array([log_amount, complexity, urgency, channel_proxy,
                       regime_proxy_a, item_risk, relationship_age, regime_proxy_b], dtype=np.float64)

    # Latents: reliability, reporting bias, leverage. No direct exposure.
    party_latent = rng.normal(0, 0.75, size=(n_parties, 3))
    party_X = np.zeros((n_parties, 5), dtype=np.float64)
    party_code = np.zeros((n_parties, 4), dtype=np.int64)
    for i, r in enumerate(roles):
        rel, bias, lev = party_latent[i]
        party_X[i, 0] = np.clip(0.50 + 0.22 * rel + rng.normal(0, 0.18), 0, 1)
        party_X[i, 1] = np.clip(0.50 + 0.20 * bias + rng.normal(0, 0.18), 0, 1)
        party_X[i, 2] = np.clip(0.45 + 0.18 * lev + rng.normal(0, 0.20), 0, 1)
        party_X[i, 3] = np.clip(rng.beta(2, 3) + 0.1 * (r == ROLE_RESPONDENT), 0, 1)
        party_X[i, 4] = np.clip(rng.beta(2.2, 2.2), 0, 1)
        party_code[i, 0] = r
        party_code[i, 1] = int(rng.integers(0, 5))
        party_code[i, 2] = int(rng.integers(0, 4))
        party_code[i, 3] = int(np.clip(np.floor(4 * party_X[i, 0]), 0, 3))

    events = []
    # Background event noise.
    n_bg = int(rng.integers(3, 9 + n_parties))
    for _ in range(n_bg):
        fam = int(rng.choice(np.arange(N_FAMILIES), p=np.array([0.10, 0.08, 0.13, 0.10, 0.10, 0.11, 0.07, 0.07, 0.07, 0.04, 0.06, 0.07])))
        src = int(rng.choice(np.arange(N_SOURCES), p=[0.31, 0.12, 0.20, 0.13, 0.17, 0.07]))
        primary = int(rng.integers(0, n_parties))
        counter = int(1 if primary == 0 else 0)
        _append_event(events, rng, fam, src, primary, counter)

    # Scenario-specific informative patterns.
    if scenario in ("strong_claimant_evidence", "simple_uncontested"):
        _append_event(events, rng, F_DOCUMENT, S_SELF, 0, 1, strength=rng.uniform(0.65, 0.95), confidence=rng.uniform(0.45, 0.70), doc_type=D_QUALITY)
        _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_THIRD_PARTY, 0 if rng.random() < 0.65 else min(2, n_parties-1), 1, strength=rng.uniform(0.55, 0.95), confidence=rng.uniform(0.75, 0.98), doc_type=D_QUALITY)
    if scenario == "strong_respondent_evidence" or (scenario == "simple_uncontested" and rng.random() < 0.45):
        _append_event(events, rng, F_COUNTER_DOCUMENT, S_SELF, 1, 0, strength=rng.uniform(0.65, 0.95), confidence=rng.uniform(0.45, 0.70), doc_type=D_DELIVERY)
        _append_event(events, rng, F_DELIVERY_OK, S_PLATFORM, 1 if n_parties <= 2 else int(rng.choice([1, min(3, n_parties-1)])), 0, strength=rng.uniform(0.60, 0.95), confidence=rng.uniform(0.70, 0.98), doc_type=D_DELIVERY)
    if scenario == "contradictory_evidence":
        _append_event(events, rng, F_DOCUMENT, S_SELF, 0, 1, strength=rng.uniform(0.7, 1.0), confidence=rng.uniform(0.35, 0.65), doc_type=D_QUALITY)
        _append_event(events, rng, F_COUNTER_DOCUMENT, S_SELF, 1, 0, strength=rng.uniform(0.7, 1.0), confidence=rng.uniform(0.35, 0.65), doc_type=D_DELIVERY)
        if rng.random() < 0.55:
            side = int(rng.choice([0, 1]))
            _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_THIRD_PARTY, side, 1-side, strength=rng.uniform(0.5, 0.85), confidence=rng.uniform(0.75, 0.98), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
    if scenario == "third_party_overrides_self_report":
        _append_event(events, rng, F_DOCUMENT, S_SELF, 0, 1, strength=rng.uniform(0.8, 1.0), confidence=rng.uniform(0.45, 0.65), doc_type=D_QUALITY)
        _append_event(events, rng, F_COUNTER_DOCUMENT, S_SELF, 1, 0, strength=rng.uniform(0.75, 1.0), confidence=rng.uniform(0.40, 0.60), doc_type=D_DELIVERY)
        # Third-party decisive, randomly chooses side.
        side = int(rng.choice([0, 1], p=[0.55, 0.45]))
        _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_THIRD_PARTY, side, 1-side, strength=rng.uniform(0.78, 1.0), confidence=rng.uniform(0.85, 0.99), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
    if scenario in ("missing_document_penalty", "missing_not_penalized_regime"):
        primary = int(rng.choice([0, 1], p=[0.48, 0.52]))
        _append_event(events, rng, F_MISSING_DOC, S_MISSING, primary, 1-primary, strength=rng.uniform(0.65, 1.0), confidence=rng.uniform(0.70, 0.95), doc_type=int(rng.choice([D_RECEIPT, D_DELIVERY, D_QUALITY, D_POLICY])))
        # Add weak evidence in opposite direction to make missingness important.
        _append_event(events, rng, F_DOCUMENT if primary == 1 else F_COUNTER_DOCUMENT, S_SELF, 1-primary, primary, strength=rng.uniform(0.45, 0.75), confidence=rng.uniform(0.35, 0.60))
    if scenario == "late_valid_evidence":
        side = int(rng.choice([0, 1]))
        _append_event(events, rng, F_LATE_EVIDENCE, S_THIRD_PARTY, side, 1-side, strength=rng.uniform(0.7, 1.0), confidence=rng.uniform(0.78, 0.98), time=rng.uniform(0.78, 1.0), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
    if scenario == "late_weak_evidence":
        side = int(rng.choice([0, 1]))
        _append_event(events, rng, F_LATE_EVIDENCE, S_SELF, side, 1-side, strength=rng.uniform(0.7, 1.0), confidence=rng.uniform(0.35, 0.60), time=rng.uniform(0.78, 1.0), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
    if scenario == "multi_party_routing":
        # Similar-looking event bags are made routing-dependent by assigning neutral
        # platform/witness records against different counterparties. A case-level bag
        # and even primary-role-only features are deliberately misleading here.
        third = min(2, n_parties - 1)
        routed_against = int(rng.choice([0, 1]))
        other = 1 - routed_against
        _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_PLATFORM, third, routed_against, strength=rng.uniform(0.72, 1.0), confidence=rng.uniform(0.78, 0.98), doc_type=D_QUALITY if routed_against == 1 else D_DELIVERY)
        _append_event(events, rng, F_POLICY_EXCEPTION, S_PLATFORM, third, routed_against, strength=rng.uniform(0.58, 0.95), confidence=rng.uniform(0.70, 0.96), doc_type=D_POLICY if routed_against == 0 else D_QUALITY)
        _append_event(events, rng, F_QUALITY_ISSUE, S_THIRD_PARTY, third, other, strength=rng.uniform(0.50, 0.85), confidence=rng.uniform(0.70, 0.98), doc_type=D_QUALITY)
        _append_event(events, rng, F_REBUTTAL if routed_against == 0 else F_CLAIM, S_SELF, other, routed_against, strength=rng.uniform(0.55, 0.85), confidence=rng.uniform(0.35, 0.65))
    if scenario == "amount_proxy_trap":
        # Large amounts but mixed evidence; amount alone is anti-predictive here.
        log_amount = float(np.clip(rng.normal(1.5, 0.45), 0.5, 2.8))
        case_X[0] = log_amount
        side = int(rng.choice([0, 1]))
        _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_THIRD_PARTY, side, 1-side, strength=rng.uniform(0.45, 0.75), confidence=rng.uniform(0.75, 0.98), amount=rng.uniform(0.1, 0.35), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
        _append_event(events, rng, F_DOCUMENT if 1-side == 0 else F_COUNTER_DOCUMENT, S_SELF, 1-side, side, strength=rng.uniform(0.75, 1.0), confidence=rng.uniform(0.35, 0.55), amount=rng.uniform(1.2, 2.0))
    if scenario == "role_base_rate_trap":
        # Role priors suggest the common direction but event evidence reverses it.
        side = int(rng.choice([0, 1]))
        _append_event(events, rng, F_THIRD_PARTY_VERIFY, S_THIRD_PARTY, side, 1-side, strength=rng.uniform(0.7, 1.0), confidence=rng.uniform(0.80, 0.99), doc_type=D_QUALITY if side == 0 else D_DELIVERY)
        for _ in range(2):
            _append_event(events, rng, F_CLAIM if side == 1 else F_REBUTTAL, S_SELF, 1-side, side, strength=rng.uniform(0.55, 0.85), confidence=rng.uniform(0.35, 0.55))
    if scenario == "rare_policy_regime":
        _append_event(events, rng, F_POLICY_EXCEPTION, S_PLATFORM, int(rng.choice([0, 1])), int(rng.choice([0, 1])), strength=rng.uniform(0.65, 1.0), confidence=rng.uniform(0.70, 0.98), doc_type=D_POLICY)
        _append_event(events, rng, F_AUTOMATED_SIGNAL, S_AUTOMATED, int(rng.choice([0, 1])), int(rng.choice([0, 1])), strength=rng.uniform(0.45, 0.85), confidence=rng.uniform(0.55, 0.85), doc_type=int(rng.choice([D_POLICY, D_QUALITY, D_DELIVERY])))

    # Sort events by time so ordered information is present but not enough by itself.
    events.sort(key=lambda e: (e["x"][0], e["family"], e["source"]))

    c_sup = 0.0
    r_sup = 0.0
    c_self = r_self = c_third = r_third = 0.0
    missing_claimant = missing_respondent = 0.0
    late_self = late_verified = 0.0
    duplicate_penalty_claimant = duplicate_penalty_respondent = 0.0
    for ev in events:
        cs, rs = _event_support(ev, roles, party_latent, regime, complexity)
        c_sup += cs
        r_sup += rs
        role = roles[ev["primary"]] if ev["primary"] >= 0 else ROLE_INTERMEDIARY
        if ev["source"] in (S_SELF, S_COUNTERPARTY):
            c_self += cs; r_self += rs
        if ev["source"] in (S_PLATFORM, S_THIRD_PARTY, S_AUTOMATED):
            c_third += cs; r_third += rs
        if ev["family"] == F_MISSING_DOC:
            if role == ROLE_CLAIMANT:
                missing_claimant += ev["x"][2]
            elif role == ROLE_RESPONDENT:
                missing_respondent += ev["x"][2]
        if ev["family"] == F_LATE_EVIDENCE:
            if ev["source"] in (S_PLATFORM, S_THIRD_PARTY):
                late_verified += ev["x"][2]
            else:
                late_self += ev["x"][2]
        if ev["family"] == F_DUPLICATE_REPORT:
            if role == ROLE_CLAIMANT:
                duplicate_penalty_claimant += ev["x"][2]
            elif role == ROLE_RESPONDENT:
                duplicate_penalty_respondent += ev["x"][2]

    # Prior and nonlinear dispute resolution. Do not make amount a strong direct proxy.
    prior = -0.10 + 0.08 * item_risk - 0.06 * relationship_age + 0.06 * (regime == 0) - 0.05 * (regime == 2)
    self_gap = c_self - r_self
    verified_gap = c_third - r_third
    contradiction = min(c_self + 0.25 * c_third, r_self + 0.25 * r_third)
    # Source reliability is load-bearing: self reports help, but verified evidence dominates
    # when the record is contradictory. This interaction is intentionally not a simple
    # additive bag-of-events formula.
    evidence = 0.48 * np.tanh(0.80 * self_gap) + 1.18 * np.tanh(1.05 * verified_gap)
    evidence += 2.15 * np.tanh(verified_gap) * np.tanh(1.55 * contradiction)
    # If both sides have self reports but no verified separation, settlement pulls toward
    # the middle; if verified evidence is decisive it overrides self-report volume.
    evidence -= 1.05 * np.tanh(self_gap) * np.tanh(1.35 * contradiction) * (1.0 - abs(np.tanh(1.15 * verified_gap)))
    evidence += 0.35 * np.sign(verified_gap + 1e-9) * np.tanh(max(c_third, r_third)) * np.tanh(contradiction)
    # Missingness is regime-dependent and can dominate weak claims; the strict regimes
    # use a threshold-like rule rather than a fixed one-hot coefficient.
    if regime != 1:
        miss_gap = np.tanh(1.65 * missing_respondent) - np.tanh(1.65 * missing_claimant)
        miss_gate = 1.0 if (missing_claimant + missing_respondent) > (0.55 + 0.25 * complexity) else 0.45
        evidence += 1.35 * miss_gap * miss_gate
    else:
        evidence += 0.03 * (missing_respondent - missing_claimant)
    # Late verified evidence remains meaningful; late self-report is weak.
    evidence += 0.48 * np.tanh(late_verified) - 0.30 * np.tanh(late_self)
    evidence += 0.18 * np.tanh(duplicate_penalty_respondent - duplicate_penalty_claimant)
    if regime == 3:
        evidence += 0.45 * np.tanh(1.25 * verified_gap) + 0.24 * np.sin(1.55 * (c_sup + r_sup))

    # Party leverage and reliability affect negotiation, but only softly.
    leverage_shift = 0.06 * (party_latent[0, 2] - party_latent[1, 2])
    reliability_shift = 0.06 * (party_latent[0, 0] - party_latent[1, 0])
    raw_score = prior + 1.05 * np.tanh(evidence) + leverage_shift + reliability_shift
    raw_score += rng.normal(0, 0.065 + 0.020 * complexity)
    y = float(np.clip(_sigmoid(1.45 * raw_score), 0.0, 1.0))

    event_X = np.array([e["x"] for e in events], dtype=np.float64)
    event_code = np.array([[e["family"], e["source"], e["doc"]] for e in events], dtype=np.int64)
    event_party = np.array([[e["primary"], e["counter"]] for e in events], dtype=np.int64)

    return case_X, party_X, party_code, event_X, event_code, event_party, y, scenario


def generate_split(n_cases: int, seed: int, scenario_probs=None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if scenario_probs is None:
        scenario_probs = np.array([0.09, 0.09, 0.09, 0.10, 0.08, 0.08, 0.07,
                                   0.07, 0.07, 0.08, 0.07, 0.06, 0.05], dtype=float)
        scenario_probs = scenario_probs / scenario_probs.sum()
    case_X_list = []
    party_X_list = []
    party_code_list = []
    event_X_list = []
    event_code_list = []
    event_party_list = []
    party_offsets = [0]
    event_offsets = [0]
    y = []
    scenarios = []
    for _ in range(n_cases):
        sid = int(rng.choice(np.arange(len(SCENARIOS)), p=scenario_probs))
        case_X, party_X, party_code, event_X, event_code, event_party, yy, scenario = _make_case(rng, sid)
        case_X_list.append(case_X)
        party_X_list.append(party_X)
        party_code_list.append(party_code)
        event_X_list.append(event_X)
        event_code_list.append(event_code)
        event_party_list.append(event_party)
        party_offsets.append(party_offsets[-1] + party_X.shape[0])
        event_offsets.append(event_offsets[-1] + event_X.shape[0])
        y.append(yy)
        scenarios.append(SCENARIOS.index(scenario))
    data = {
        "case_X": np.vstack(case_X_list).astype(np.float64),
        "party_X": np.vstack(party_X_list).astype(np.float64),
        "party_code": np.vstack(party_code_list).astype(np.int64),
        "event_X": np.vstack(event_X_list).astype(np.float64),
        "event_code": np.vstack(event_code_list).astype(np.int64),
        "event_party": np.vstack(event_party_list).astype(np.int64),
        "party_offsets": np.asarray(party_offsets, dtype=np.int64),
        "event_offsets": np.asarray(event_offsets, dtype=np.int64),
        "y": np.asarray(y, dtype=np.float64),
        "scenario_id": np.asarray(scenarios, dtype=np.int64),
    }
    for i, name in enumerate(SCENARIOS):
        data[f"slice_{name}"] = (data["scenario_id"] == i)
    return data


def save_trial_npz(out_dir: str | Path, train_n=5200, public_n=900, hidden_n=1400):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = generate_split(train_n, 12031)
    public = generate_split(public_n, 12032)
    hidden = generate_split(hidden_n, 12033)

    np.savez_compressed(out_dir / "train_data.npz",
        train_case_X=train["case_X"],
        train_party_X=train["party_X"],
        train_party_code=train["party_code"],
        train_event_X=train["event_X"],
        train_event_code=train["event_code"],
        train_event_party=train["event_party"],
        train_party_offsets=train["party_offsets"],
        train_event_offsets=train["event_offsets"],
        train_y=train["y"],
    )
    np.savez_compressed(out_dir / "public_eval.npz",
        case_X=public["case_X"],
        party_X=public["party_X"],
        party_code=public["party_code"],
        event_X=public["event_X"],
        event_code=public["event_code"],
        event_party=public["event_party"],
        party_offsets=public["party_offsets"],
        event_offsets=public["event_offsets"],
        y=public["y"],
    )
    hidden_payload = {
        "case_X": hidden["case_X"],
        "party_X": hidden["party_X"],
        "party_code": hidden["party_code"],
        "event_X": hidden["event_X"],
        "event_code": hidden["event_code"],
        "event_party": hidden["event_party"],
        "party_offsets": hidden["party_offsets"],
        "event_offsets": hidden["event_offsets"],
        "y": hidden["y"],
        "scenario_id": hidden["scenario_id"],
    }
    for name in SCENARIOS:
        hidden_payload[f"slice_{name}"] = hidden[f"slice_{name}"]
    np.savez_compressed(out_dir / "hidden_eval_payload.npz", **hidden_payload)
    meta = {
        "task": "transaction_dispute_outcome_v0_2",
        "scenario_names": SCENARIOS,
        "train_n": train_n,
        "public_n": public_n,
        "hidden_n": hidden_n,
        "seeds": {"train": 12031, "public": 12032, "hidden": 12033},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    save_trial_npz(Path(__file__).parent / "trial_npz")

# ===== END FILE: tmp/tests/generator.py =====

# ===== BEGIN FILE: tmp/tests/hidden_eval.py =====
"""Verifier scoring for transaction_dispute_outcome v0.2."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

try:
    from generator import generate_split, SCENARIOS
    from sandbox_utils import run_eval, check_solution
except Exception:  # pragma: no cover
    from .generator import generate_split, SCENARIOS
    from .sandbox_utils import run_eval, check_solution

# Anchors from v0.2 hardening calibration. GOOD is slightly stricter than
# the measured reference so the reference does not clip to 1.0. BAD is above
# weak-but-valid simple/routed baselines to preserve reward gradient.
OVERALL_GOOD = 0.104965623401
OVERALL_BAD = 0.245057681719
SCORE_POWER = 2.2

SLICE_GOOD = {
    "simple_uncontested": 0.11497387219785692,
    "strong_claimant_evidence": 0.09707485357095831,
    "strong_respondent_evidence": 0.09107542823498246,
    "contradictory_evidence": 0.09861798083416425,
    "third_party_overrides_self_report": 0.10476549878044529,
    "missing_document_penalty": 0.12135953150018779,
    "missing_not_penalized_regime": 0.12449676140683406,
    "late_valid_evidence": 0.09950577412898556,
    "late_weak_evidence": 0.08276272824840235,
    "multi_party_routing": 0.12487235670452923,
    "amount_proxy_trap": 0.09882294161178991,
    "role_base_rate_trap": 0.104379937923512,
    "rare_policy_regime": 0.08417514493490233,
}

SLICE_BAD = {
    "simple_uncontested": 0.24969331548656476,
    "strong_claimant_evidence": 0.257809344334476,
    "strong_respondent_evidence": 0.22180158149749574,
    "contradictory_evidence": 0.23431929417670538,
    "third_party_overrides_self_report": 0.28187450165577294,
    "missing_document_penalty": 0.2747811754469242,
    "missing_not_penalized_regime": 0.1816223090678328,
    "late_valid_evidence": 0.23300347872108326,
    "late_weak_evidence": 0.18544028357791537,
    "multi_party_routing": 0.24930063279006417,
    "amount_proxy_trap": 0.2563101349016127,
    "role_base_rate_trap": 0.2781801265201534,
    "rare_policy_regime": 0.2591644642740709,
}

SLICE_WEIGHTS = {
    "simple_uncontested": 0.55,
    "strong_claimant_evidence": 0.85,
    "strong_respondent_evidence": 0.85,
    "contradictory_evidence": 1.35,
    "third_party_overrides_self_report": 1.35,
    "missing_document_penalty": 1.35,
    "missing_not_penalized_regime": 1.2,
    "late_valid_evidence": 1.05,
    "late_weak_evidence": 1.05,
    "multi_party_routing": 1.5,
    "amount_proxy_trap": 1.25,
    "role_base_rate_trap": 1.25,
    "rare_policy_regime": 1.25,
}


def _rmse(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _score_rmse(value, good, bad):
    value = float(value)
    good = max(float(good), 1e-9)
    bad = max(float(bad), good * 1.0001)
    if not np.isfinite(value):
        return 0.0
    raw = (math.log(bad) - math.log(max(value, 1e-9))) / (math.log(bad) - math.log(good))
    return float(np.clip(raw, 0.0, 1.0) ** SCORE_POWER)


def _make_train_hidden():
    train = generate_split(5200, 12031)
    hidden = generate_split(1400, 12033)
    return train, hidden


def _payload(train, hidden):
    return {
        "train_case_X": train["case_X"],
        "train_party_X": train["party_X"],
        "train_party_code": train["party_code"],
        "train_event_X": train["event_X"],
        "train_event_code": train["event_code"],
        "train_event_party": train["event_party"],
        "train_party_offsets": train["party_offsets"],
        "train_event_offsets": train["event_offsets"],
        "train_y": train["y"],
        "case_X": hidden["case_X"],
        "party_X": hidden["party_X"],
        "party_code": hidden["party_code"],
        "event_X": hidden["event_X"],
        "event_code": hidden["event_code"],
        "event_party": hidden["event_party"],
        "party_offsets": hidden["party_offsets"],
        "event_offsets": hidden["event_offsets"],
    }


def evaluate(solve_path: str | Path, write_dir: str | Path | None = None):
    solve_path = Path(solve_path)
    if not check_solution(solve_path):
        return _finish(0.0, {"error": "api_missing"}, write_dir)
    train, hidden = _make_train_hidden()
    try:
        out = run_eval(solve_path, _payload(train, hidden))
    except Exception as exc:
        return _finish(0.0, {"error": "runtime_error", "detail": str(exc)[:1200]}, write_dir)
    if "error" in out:
        return _finish(0.0, {"error": out.get("error"), "detail": out.get("detail", "")}, write_dir)
    pred = np.asarray(out["preds"][0], dtype=float)
    y = np.asarray(hidden["y"], dtype=float)
    if pred.shape != y.shape:
        return _finish(0.0, {"error": "wrong_shape", "shape": list(pred.shape), "expected": list(y.shape)}, write_dir)
    if not np.all(np.isfinite(pred)):
        return _finish(0.0, {"error": "nonfinite_prediction"}, write_dir)
    if np.nanmax(np.abs(pred)) > 1e9:
        return _finish(0.0, {"error": "catastrophic_scale"}, write_dir)

    pred = np.clip(pred, 0.0, 1.0)
    overall_rmse = _rmse(y, pred)
    overall_score = _score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    slice_metrics = {}
    weighted = []
    for name in SCENARIOS:
        key = "slice_" + str(name)
        mask = np.asarray(hidden.get(key, np.zeros_like(y, dtype=bool)), dtype=bool)
        if not np.any(mask):
            continue
        r = _rmse(y, pred, mask)
        s = _score_rmse(r, SLICE_GOOD[str(name)], SLICE_BAD[str(name)])
        w = min(1.0, max(0.25, float(np.sum(mask)) / 120.0)) * SLICE_WEIGHTS.get(str(name), 1.0)
        slice_metrics[str(name)] = {"n": int(np.sum(mask)), "rmse": r, "score": s, "weight": w}
        weighted.append((w, s))
    if weighted:
        ws = np.array([x[0] for x in weighted], dtype=float)
        ss = np.array([x[1] for x in weighted], dtype=float)
        slice_score = float(np.sum(ws * ss) / np.sum(ws))
    else:
        slice_score = overall_score
    reward = 0.50 * overall_score + 0.50 * slice_score
    metrics = {
        "reward": float(np.clip(reward, 0.0, 1.0)),
        "overall_rmse": overall_rmse,
        "overall_score": overall_score,
        "slice_score": slice_score,
        "slice_metrics": slice_metrics,
    }
    return _finish(metrics["reward"], metrics, write_dir)


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
    ap.add_argument("solve_path")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    reward, metrics = evaluate(ns.solve_path, ns.out)
    print(json.dumps(metrics, indent=2, sort_keys=True))

# ===== END FILE: tmp/tests/hidden_eval.py =====

# ===== BEGIN FILE: tmp/tests/sandbox_utils.py =====
"""Root-side sandbox helper for transaction_dispute_outcome."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

RUNNER = "/sandbox/agent_call.py"
IO_DIR = "/sandbox/io"
_counter = [0]
REQUIRED = ("fit_dispute_model", "predict_dispute_outcome")


def _sandbox_ready() -> bool:
    return os.path.exists(RUNNER) and getattr(os, "geteuid", lambda: 1000)() == 0


def _local_runner() -> str:
    return str(Path(__file__).resolve().parent / "agent_call.py")


def _demote():  # pragma: no cover
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(38, 1, 0, 0, 0)
    except Exception:
        pass
    try:
        import pwd
        p = pwd.getpwnam("nobody")
        try:
            os.setgroups([])
        except Exception:
            pass
        os.setgid(p.pw_gid)
        os.setuid(p.pw_uid)
    except Exception:
        pass


def _nobody_command(args: list[str]) -> tuple[list[str], object | None]:
    if shutil.which("runuser"):
        return ["runuser", "-u", "nobody", "--", *args], None
    if shutil.which("su"):
        quoted = " ".join(subprocess.list2cmdline([a]) for a in args)
        return ["su", "-s", "/bin/sh", "nobody", "-c", quoted], None
    return args, _demote


def _payload_to_npz(path: str | os.PathLike, payload: dict) -> None:
    np.savez(
        path,
        train_case_X=np.asarray(payload["train_case_X"], dtype=np.float64),
        train_party_X=np.asarray(payload["train_party_X"], dtype=np.float64),
        train_party_code=np.asarray(payload["train_party_code"], dtype=np.int64),
        train_event_X=np.asarray(payload["train_event_X"], dtype=np.float64),
        train_event_code=np.asarray(payload["train_event_code"], dtype=np.int64),
        train_event_party=np.asarray(payload["train_event_party"], dtype=np.int64),
        train_party_offsets=np.asarray(payload["train_party_offsets"], dtype=np.int64),
        train_event_offsets=np.asarray(payload["train_event_offsets"], dtype=np.int64),
        train_y=np.asarray(payload["train_y"], dtype=np.float64),
        case_X=np.asarray(payload["case_X"], dtype=np.float64),
        party_X=np.asarray(payload["party_X"], dtype=np.float64),
        party_code=np.asarray(payload["party_code"], dtype=np.int64),
        event_X=np.asarray(payload["event_X"], dtype=np.float64),
        event_code=np.asarray(payload["event_code"], dtype=np.int64),
        event_party=np.asarray(payload["event_party"], dtype=np.int64),
        party_offsets=np.asarray(payload["party_offsets"], dtype=np.int64),
        event_offsets=np.asarray(payload["event_offsets"], dtype=np.int64),
    )


def _read_output(out_path: str | os.PathLike) -> dict:
    with np.load(out_path, allow_pickle=False) as z:
        if "error" in z.files:
            detail = ""
            if "detail" in z.files:
                detail = str(z["detail"].item())
            return {"error": str(z["error"].item()), "detail": detail}
        n = int(z["n"])
        return {"preds": [np.asarray(z[f"p{i}"], dtype=np.float64) for i in range(n)]}


def run_eval(solve_path: str | os.PathLike, payload: dict, timeout: int = 180) -> dict:
    if not _sandbox_ready():
        import tempfile
        runner = _local_runner()
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in_eval.npz")
            out_path = os.path.join(td, "out_eval.npz")
            _payload_to_npz(in_path, payload)
            r = subprocess.run([sys.executable, runner, str(solve_path), "eval", in_path, out_path], capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0 or not os.path.exists(out_path):
                raise RuntimeError(f"local eval failed (rc={r.returncode}): {r.stderr[-800:]}")
            return _read_output(out_path)

    Path(IO_DIR).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(IO_DIR, 0o777)
    except Exception:
        pass
    _counter[0] += 1
    tag = f"eval_{os.getpid()}_{_counter[0]}"
    in_path = os.path.join(IO_DIR, f"{tag}_in.npz")
    out_path = os.path.join(IO_DIR, f"{tag}_out.npz")
    _payload_to_npz(in_path, payload)
    try:
        os.chmod(in_path, 0o644)
    except Exception:
        pass
    args = [sys.executable, RUNNER, str(solve_path), "eval", in_path, out_path]
    cmd, preexec = _nobody_command(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=preexec)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"sandbox eval failed (rc={r.returncode}): {r.stderr[-800:]}")
    return _read_output(out_path)


def check_solution(solve_path: str | os.PathLike, timeout: int = 60) -> bool:
    runner = RUNNER if _sandbox_ready() else _local_runner()
    try:
        r = subprocess.run([sys.executable, runner, str(solve_path), "check"], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False

# ===== END FILE: tmp/tests/sandbox_utils.py =====

# ===== BEGIN FILE: tmp/tests/test.sh =====
#!/bin/bash
set -uo pipefail

TEST_DIR=/tests
if [ ! -d "$TEST_DIR" ]; then
  TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

mkdir -p /logs/verifier 2>/dev/null || true
mkdir -p /sandbox/io 2>/dev/null || true
rm -f /logs/verifier/jr_*.json /logs/verifier/ctrf.json 2>/dev/null || true

if [ -f "$TEST_DIR/agent_call.py" ]; then
  cp "$TEST_DIR/agent_call.py" /sandbox/agent_call.py 2>/dev/null || true
fi
chmod 755 /sandbox/agent_call.py 2>/dev/null || true
chmod 777 /sandbox/io 2>/dev/null || true

START_SEC=$(date +%s)

finalize() {
  python - <<'PY'
import json, glob, os
from pathlib import Path
log = Path('/logs/verifier')
log.mkdir(parents=True, exist_ok=True)
reports = []
for p in sorted(glob.glob('/logs/verifier/jr_*.json')):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        reports.append((os.path.basename(p), d))
    except Exception as e:
        reports.append((os.path.basename(p), {'error': str(e), 'tests': []}))

tests = []
summary = {'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0, 'xfailed': 0, 'xpassed': 0, 'total': 0}
for name, d in reports:
    for t in d.get('tests', []) or []:
        item = dict(t)
        item['report_file'] = name
        tests.append(item)
        outcome = item.get('outcome')
        if outcome == 'passed':
            summary['passed'] += 1
        elif outcome == 'failed':
            summary['failed'] += 1
        elif outcome == 'error':
            summary['error'] += 1
        elif outcome == 'skipped':
            summary['skipped'] += 1
        elif outcome == 'xfailed':
            summary['xfailed'] += 1
        elif outcome == 'xpassed':
            summary['xpassed'] += 1
summary['total'] = len(tests)
(log / 'ctrf.json').write_text(json.dumps({'summary': summary, 'tests': tests}, indent=2), encoding='utf-8')

failure = None
if summary['total'] == 0 or summary['passed'] == 0:
    failure = 'pipeline_broken'
elif summary['failed'] or summary['error'] or summary['xpassed']:
    # Determine stage from report filename.
    bad_reports = {t.get('report_file','') for t in tests if t.get('outcome') in ('failed','error','xpassed')}
    if any('smoke' in r for r in bad_reports):
        failure = 'pipeline_broken'
    elif any('schema' in r for r in bad_reports):
        failure = 'schema_violation'
    else:
        failure = 'runtime_error'

reward_path = log / 'reward.txt'
if failure is None:
    if not reward_path.exists():
        failure = 'missing_reward'
    else:
        try:
            v = float(reward_path.read_text().strip())
            if not (0.0 <= v <= 1.0):
                failure = 'schema_violation'
        except Exception:
            failure = 'schema_violation'

if failure is not None:
    (log / 'failure_mode.txt').write_text(failure + '\n', encoding='utf-8')
    reward_path.write_text('0\n', encoding='utf-8')
PY
  END_SEC=$(date +%s)
  echo $(( END_SEC - START_SEC )) > /logs/verifier/wall_clock_sec.txt 2>/dev/null || true
  if [ -f /proc/self/status ]; then
    awk '/VmHWM/ {printf "%.0f\n", $2/1024}' /proc/self/status \
      > /logs/verifier/peak_memory_mb.txt 2>/dev/null || true
  fi
}
trap finalize EXIT

set +e
pytest "$TEST_DIR/test_smoke.py" -v --tb=short -p no:cacheprovider \
  --json-report --json-report-file=/logs/verifier/jr_smoke.json \
  2> /logs/verifier/smoke_stderr.txt

pytest "$TEST_DIR/test_schema.py" -v --tb=short -p no:cacheprovider \
  --json-report --json-report-file=/logs/verifier/jr_schema.json \
  2>> /logs/verifier/smoke_stderr.txt

chmod 700 /tests 2>/dev/null || true

pytest "$TEST_DIR/test_main.py" -v --tb=short -p no:cacheprovider \
  --json-report --json-report-file=/logs/verifier/jr_main.json \
  2> /logs/verifier/stderr.txt
set -e

exit 0

# ===== END FILE: tmp/tests/test.sh =====

# ===== BEGIN FILE: tmp/tests/test_main.py =====
from __future__ import annotations

from pathlib import Path

from hidden_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]


def app_solve_path() -> Path:
    p = Path("/app/solve.py")
    return p if p.exists() else ROOT / "environment" / "app" / "solve.py"


def verifier_log_dir() -> Path:
    p = Path("/logs/verifier")
    if p.exists() or Path("/logs").exists():
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = ROOT / "logs" / "verifier"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_main_scoring_writes_reward_and_metrics():
    out_dir = verifier_log_dir()
    reward, metrics = evaluate(app_solve_path(), out_dir)
    assert (out_dir / "reward.txt").exists(), "reward.txt was not written"
    assert (out_dir / "metrics.json").exists(), "metrics.json was not written"
    assert 0.0 <= reward <= 1.0
    assert "reward" in metrics

# ===== END FILE: tmp/tests/test_main.py =====

# ===== BEGIN FILE: tmp/tests/test_schema.py =====
from __future__ import annotations

from pathlib import Path
import numpy as np

from generator import generate_split, SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
N_CASE = 8
N_PARTY_X = 5
N_PARTY_CODE = 4
N_EVENT_X = 6
N_EVENT_CODE = 3
N_EVENT_PARTY = 2


def app_data_path(name: str) -> Path:
    p = Path("/app") / name
    return p if p.exists() else ROOT / "environment" / "app" / name


def _check_dataset(d, *, train: bool, has_y: bool, has_slices: bool = False):
    ckey = "train_case_X" if train else "case_X"
    pxkey = "train_party_X" if train else "party_X"
    pckey = "train_party_code" if train else "party_code"
    exkey = "train_event_X" if train else "event_X"
    eckey = "train_event_code" if train else "event_code"
    epkey = "train_event_party" if train else "event_party"
    poff = "train_party_offsets" if train else "party_offsets"
    eoff = "train_event_offsets" if train else "event_offsets"
    ykey = "train_y" if train else "y"
    assert d[ckey].ndim == 2 and d[ckey].shape[1] == N_CASE
    assert d[pxkey].ndim == 2 and d[pxkey].shape[1] == N_PARTY_X
    assert d[pckey].ndim == 2 and d[pckey].shape[1] == N_PARTY_CODE
    assert d[exkey].ndim == 2 and d[exkey].shape[1] == N_EVENT_X
    assert d[eckey].ndim == 2 and d[eckey].shape[1] == N_EVENT_CODE
    assert d[epkey].ndim == 2 and d[epkey].shape[1] == N_EVENT_PARTY
    assert d[poff].ndim == 1 and d[eoff].ndim == 1
    assert d[poff][0] == 0 and d[eoff][0] == 0
    assert d[poff][-1] == d[pxkey].shape[0]
    assert d[eoff][-1] == d[exkey].shape[0]
    assert d[pxkey].shape[0] == d[pckey].shape[0]
    assert d[exkey].shape[0] == d[eckey].shape[0] == d[epkey].shape[0]
    assert d[poff].shape[0] == d[ckey].shape[0] + 1
    assert d[eoff].shape[0] == d[ckey].shape[0] + 1
    assert np.all(np.diff(d[poff]) > 0)
    assert np.all(np.diff(d[eoff]) > 0)
    assert np.all(np.isfinite(d[ckey]))
    assert np.all(np.isfinite(d[pxkey]))
    assert np.all(np.isfinite(d[exkey]))
    if has_y:
        assert ykey in d
        assert d[ykey].shape == (d[ckey].shape[0],)
        assert np.all(np.isfinite(d[ykey]))
        assert np.all((d[ykey] >= -1e-9) & (d[ykey] <= 1.0 + 1e-9))
    if has_slices:
        slice_keys = [k for k in d.keys() if str(k).startswith("slice_")]
        assert len(slice_keys) >= len(SCENARIOS)
        for k in slice_keys:
            assert d[k].shape == (d[ckey].shape[0],)


def test_agent_visible_schema():
    with np.load(app_data_path("train_data.npz"), allow_pickle=False) as z:
        _check_dataset(z, train=True, has_y=True)
    with np.load(app_data_path("public_eval.npz"), allow_pickle=False) as z:
        _check_dataset(z, train=False, has_y=True)


def test_verifier_generated_schema():
    train = generate_split(5200, 12031)
    public = generate_split(900, 12032)
    hidden = generate_split(1400, 12033)
    td = {
        "train_case_X": train["case_X"],
        "train_party_X": train["party_X"],
        "train_party_code": train["party_code"],
        "train_event_X": train["event_X"],
        "train_event_code": train["event_code"],
        "train_event_party": train["event_party"],
        "train_party_offsets": train["party_offsets"],
        "train_event_offsets": train["event_offsets"],
        "train_y": train["y"],
    }
    _check_dataset(td, train=True, has_y=True)
    for split in (public, hidden):
        _check_dataset(split, train=False, has_y=True, has_slices=True)


def test_train_public_hidden_are_distinct_sizes():
    train = generate_split(5200, 12031)
    public = generate_split(900, 12032)
    hidden = generate_split(1400, 12033)
    assert train["case_X"].shape[0] > public["case_X"].shape[0] > 0
    assert hidden["case_X"].shape[0] > public["case_X"].shape[0] > 0

# ===== END FILE: tmp/tests/test_schema.py =====

# ===== BEGIN FILE: tmp/tests/test_smoke.py =====
from __future__ import annotations

from pathlib import Path
import numpy as np

from sandbox_utils import check_solution

ROOT = Path(__file__).resolve().parents[1]


def app_solve_path() -> Path:
    p = Path("/app/solve.py")
    return p if p.exists() else ROOT / "environment" / "app" / "solve.py"


def app_data_path(name: str) -> Path:
    p = Path("/app") / name
    return p if p.exists() else ROOT / "environment" / "app" / name


def test_candidate_file_exists():
    assert app_solve_path().exists(), "solve.py not found"


def test_required_functions_present():
    assert check_solution(app_solve_path()), "required fit_dispute_model / predict_dispute_outcome functions not found"


def test_visible_data_loadable():
    for name in ("train_data.npz", "public_eval.npz"):
        path = app_data_path(name)
        assert path.exists(), f"{name} not found"
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            assert ({"train_case_X", "train_party_X", "train_event_X"} & files) or ({"case_X", "party_X", "event_X"} & files), f"{name} missing core arrays"

# ===== END FILE: tmp/tests/test_smoke.py =====

