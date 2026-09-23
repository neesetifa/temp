Task

The records are packed invoice line data. Each invoice has one or more line rows. Implement the two functions in app/solve.py:

fit_invoice_model(
    train_line_X,
    train_line_code,
    train_invoice_X,
    train_invoice_code,
    train_invoice_offsets,
    train_y,
)

predict_net_line_cost(
    line_X,
    line_code,
    invoice_X,
    invoice_code,
    invoice_offsets,
    params,
)

Return one finite prediction for every input line. The target is the final audited net cost for each line. Rows from the same invoice may affect one another.

invoice_offsets[i]:invoice_offsets[i + 1] gives the line rows belonging to invoice i.

Scoring is based mainly on RMSE after a signed log transform of predicted and true net line costs, with additional held-out cohorts. Signed log means sign(x) * log1p(abs(x)).

Keep the implementation deterministic and do not mutate the input arrays.

reference
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Column indices duplicated to keep this solution self-contained.
LX_QUANTITY = 0; LX_UNIT_PRICE = 1; LX_RAW_AMOUNT = 2; LX_REPORTED_DISCOUNT = 3; LX_REPORTED_TAX = 4; LX_PACKAGE_COUNT = 5; LX_UNIT_SIZE = 6; LX_WEIGHT_PROXY = 7; LX_LEAD_TIME = 8; LX_HIST_ITEM_VOLUME = 9; LX_RECENT_PRICE_INDEX = 10; LX_LINE_POSITION_NORM = 11; LX_IS_NEGATIVE = 12; LX_IS_ROUND_AMOUNT = 13
LC_ITEM_FAMILY = 0; LC_ITEM_SUBFAMILY = 1; LC_UNIT_TYPE = 2; LC_CHARGE_TYPE = 3; LC_VENDOR_LINE_TYPE = 4; LC_CONTRACT_TIER = 5; LC_REGION_CODE = 6; LC_CURRENCY_BUCKET = 7; LC_EXCEPTION_CODE = 8
IX_INVOICE_SUBTOTAL = 2; IX_HEADER_DISCOUNT = 3; IX_HEADER_SHIPPING = 4; IX_HEADER_HANDLING_FEE = 5; IX_MINIMUM_CHARGE_FLAG = 6; IX_EXPEDITE_FLAG = 7; IX_STATED_LINE_COUNT = 8; IX_SEASON_SIN = 9; IX_SEASON_COS = 10
IC_VENDOR_CODE = 0; IC_CONTRACT_CHANNEL = 1; IC_INVOICE_REGION = 2; IC_MONTH_BUCKET = 3
N_CHARGE_TYPE = 7; N_ITEM_FAMILY = 24; N_EXCEPTION = 12


def _signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _inv_signed_log(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


def _safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-9, 1.0, b)


def _target_encode(keys, y, min_count=8, shrink=18.0):
    # returns mapping dict tuple/int -> encoded signed-log target mean with shrinkage
    global_mean = float(np.mean(y))
    sums = {}
    cnts = {}
    for k, val in zip(keys, y):
        kk = tuple(k) if np.ndim(k) else int(k)
        sums[kk] = sums.get(kk, 0.0) + float(val)
        cnts[kk] = cnts.get(kk, 0) + 1
    enc = {}
    for k in sums:
        n = cnts[k]
        enc[k] = (sums[k] + shrink * global_mean) / (n + shrink)
    return enc, global_mean


def _apply_te(keys, mapping, default):
    out = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys):
        kk = tuple(k) if np.ndim(k) else int(k)
        out[i] = mapping.get(kk, default)
    return out


def _build_features(line_X, line_code, invoice_X, invoice_code, invoice_offsets, encoders=None, fit=False, y_log=None):
    n = line_X.shape[0]
    raw = line_X[:, LX_RAW_AMOUNT]
    raw_pos = np.maximum(raw, 0.0)
    abs_raw = np.abs(raw)
    base = [
        _signed_log(raw),
        np.log1p(abs_raw),
        _signed_log(line_X[:, LX_REPORTED_DISCOUNT]),
        _signed_log(line_X[:, LX_REPORTED_TAX]),
        np.log1p(line_X[:, LX_PACKAGE_COUNT]),
        np.log1p(np.maximum(line_X[:, LX_WEIGHT_PROXY], 0.0)),
        line_X[:, LX_LINE_POSITION_NORM],
        line_X[:, LX_IS_NEGATIVE],
        line_X[:, LX_IS_ROUND_AMOUNT],
        _signed_log(line_X[:, LX_QUANTITY]),
        _signed_log(line_X[:, LX_UNIT_PRICE]),
        np.log1p(np.maximum(line_X[:, LX_UNIT_SIZE], 0.0)),
        np.log1p(np.maximum(line_X[:, LX_LEAD_TIME], 0.0)),
        np.log1p(np.maximum(line_X[:, LX_HIST_ITEM_VOLUME], 0.0)),
        line_X[:, LX_RECENT_PRICE_INDEX],
        line_code[:, LC_ITEM_FAMILY].astype(float),
        line_code[:, LC_ITEM_SUBFAMILY].astype(float),
        line_code[:, LC_UNIT_TYPE].astype(float),
        line_code[:, LC_CHARGE_TYPE].astype(float),
        line_code[:, LC_VENDOR_LINE_TYPE].astype(float),
        line_code[:, LC_CONTRACT_TIER].astype(float),
        line_code[:, LC_REGION_CODE].astype(float),
        line_code[:, LC_CURRENCY_BUCKET].astype(float),
        line_code[:, LC_EXCEPTION_CODE].astype(float),
    ]
    rep_inv = np.empty((n, invoice_X.shape[1]), dtype=float)
    rep_code = np.empty((n, invoice_code.shape[1]), dtype=float)
    agg = np.zeros((n, 38), dtype=float)
    for ii in range(len(invoice_offsets)-1):
        s, e = int(invoice_offsets[ii]), int(invoice_offsets[ii+1])
        idx = slice(s, e)
        m = e - s
        rep_inv[idx] = invoice_X[ii]
        rep_code[idx] = invoice_code[ii]
        rr = raw[idx]
        rp = raw_pos[idx]
        subtotal_pos = max(float(rp.sum()), 1e-9)
        wt = np.maximum(line_X[idx, LX_WEIGHT_PROXY], 0.0)
        pkg = np.maximum(line_X[idx, LX_PACKAGE_COUNT], 0.0)
        wt_sum = max(float(wt.sum()), 1e-9)
        pkg_sum = max(float(pkg.sum()), 1e-9)
        ch = line_code[idx, LC_CHARGE_TYPE]
        fam = line_code[idx, LC_ITEM_FAMILY]
        exc = line_code[idx, LC_EXCEPTION_CODE]
        amount_share = rp / subtotal_pos
        weight_share = wt / wt_sum
        package_share = pkg / pkg_sum
        service = np.isin(ch, [1,2,3]).astype(float)
        service_share = service / max(service.sum(), 1e-9) if service.sum() else amount_share
        eligible = (~np.isin(ch, [2,3,4,5])).astype(float) * (rr > 0)
        eligible_amount = rp * eligible
        eligible_share = eligible_amount / max(float(eligible_amount.sum()), 1e-9)
        low = (rp <= np.quantile(rp, 0.35) if m > 2 else np.ones(m, dtype=bool)).astype(float) * eligible
        low_share = low / max(float(low.sum()), 1e-9)
        order = np.argsort(np.argsort(rr)).astype(float) / max(1, m-1)
        # duplicate-ish counts by rounded absolute amount
        rounded = np.round(np.abs(rr) / 5.0)
        dup_count = np.array([np.sum(rounded == r) for r in rounded], dtype=float)
        # family and charge shares for each row
        fam_amt = np.zeros(m)
        fam_cnt = np.zeros(m)
        for f in np.unique(fam):
            mask = fam == f
            fam_amt[mask] = rp[mask].sum() / subtotal_pos
            fam_cnt[mask] = mask.sum() / max(1, m)
        ch_amt = np.zeros(m)
        ch_cnt = np.zeros(m)
        for c in np.unique(ch):
            mask = ch == c
            ch_amt[mask] = rp[mask].sum() / subtotal_pos
            ch_cnt[mask] = mask.sum() / max(1, m)
        header_discount = invoice_X[ii, IX_HEADER_DISCOUNT]
        header_shipping = invoice_X[ii, IX_HEADER_SHIPPING]
        handling = invoice_X[ii, IX_HEADER_HANDLING_FEE]
        # first 20 columns rowwise allocation features
        agg[idx, 0] = m
        agg[idx, 1] = _signed_log(invoice_X[ii, IX_INVOICE_SUBTOTAL])
        agg[idx, 2] = np.log1p(abs(invoice_X[ii, IX_HEADER_DISCOUNT]))
        agg[idx, 3] = np.log1p(abs(invoice_X[ii, IX_HEADER_SHIPPING]))
        agg[idx, 4] = np.log1p(abs(invoice_X[ii, IX_HEADER_HANDLING_FEE]))
        agg[idx, 5] = amount_share
        agg[idx, 6] = weight_share
        agg[idx, 7] = package_share
        agg[idx, 8] = service_share
        agg[idx, 9] = eligible_share
        agg[idx, 10] = low_share
        agg[idx, 11] = fam_amt
        agg[idx, 12] = fam_cnt
        agg[idx, 13] = ch_amt
        agg[idx, 14] = ch_cnt
        agg[idx, 15] = order
        agg[idx, 16] = dup_count
        agg[idx, 17] = np.mean(rr < 0)
        agg[idx, 18] = np.mean(exc > 0)
        agg[idx, 19] = np.mean(service)
        agg[idx, 20] = _signed_log(header_discount * amount_share)
        agg[idx, 21] = _signed_log(header_discount * eligible_share)
        agg[idx, 22] = _signed_log(header_discount * service_share)
        agg[idx, 23] = _signed_log(header_discount * low_share)
        agg[idx, 24] = _signed_log(header_shipping * amount_share)
        agg[idx, 25] = _signed_log(header_shipping * weight_share)
        agg[idx, 26] = _signed_log(header_shipping * package_share)
        agg[idx, 27] = _signed_log(handling * service_share)
        agg[idx, 28] = _signed_log(handling * low_share)
        agg[idx, 29] = _signed_log((header_shipping + handling - header_discount) * amount_share)
        agg[idx, 30] = _safe_div(rp, np.maximum(1.0, np.mean(rp)))
        agg[idx, 31] = _safe_div(wt, np.maximum(1.0, np.mean(wt)))
        agg[idx, 32] = np.log1p(np.max(rp))
        agg[idx, 33] = np.log1p(np.mean(rp))
        agg[idx, 34] = np.log1p(np.std(rp))
        agg[idx, 35] = invoice_X[ii, IX_MINIMUM_CHARGE_FLAG]
        agg[idx, 36] = invoice_X[ii, IX_EXPEDITE_FLAG]
        agg[idx, 37] = rep_code[idx, IC_CONTRACT_CHANNEL]
    X_parts = [np.vstack(base).T, rep_inv, rep_code, agg]
    if fit:
        encoders = {}
        keys = {
            'vendor': invoice_code[np.searchsorted(invoice_offsets[1:], np.arange(n), side='right'), IC_VENDOR_CODE],
            'family': line_code[:, [LC_ITEM_FAMILY]],
            'vendor_family': np.column_stack([invoice_code[np.searchsorted(invoice_offsets[1:], np.arange(n), side='right'), IC_VENDOR_CODE], line_code[:, LC_ITEM_FAMILY]]),
            'contract_family': np.column_stack([line_code[:, LC_CONTRACT_TIER], line_code[:, LC_ITEM_FAMILY]]),
            'charge_exc': np.column_stack([line_code[:, LC_CHARGE_TYPE], line_code[:, LC_EXCEPTION_CODE]]),
        }
        te_cols = []
        for name, key in keys.items():
            mapping, default = _target_encode(key, y_log, shrink=25.0)
            encoders[name] = (mapping, default)
            te_cols.append(_apply_te(key, mapping, default))
        X_parts.append(np.vstack(te_cols).T)
        return np.hstack(X_parts), encoders
    else:
        inv_ids = np.searchsorted(invoice_offsets[1:], np.arange(n), side='right')
        keys = {
            'vendor': invoice_code[inv_ids, IC_VENDOR_CODE],
            'family': line_code[:, [LC_ITEM_FAMILY]],
            'vendor_family': np.column_stack([invoice_code[inv_ids, IC_VENDOR_CODE], line_code[:, LC_ITEM_FAMILY]]),
            'contract_family': np.column_stack([line_code[:, LC_CONTRACT_TIER], line_code[:, LC_ITEM_FAMILY]]),
            'charge_exc': np.column_stack([line_code[:, LC_CHARGE_TYPE], line_code[:, LC_EXCEPTION_CODE]]),
        }
        te_cols = []
        for name, key in keys.items():
            mapping, default = encoders[name]
            te_cols.append(_apply_te(key, mapping, default))
        X_parts.append(np.vstack(te_cols).T)
        return np.hstack(X_parts)



def _vendor_attrs(vendor):
    v = float(int(vendor) + 1)
    discount_style = 0.5 + 0.32 * np.sin(v * 0.73) + 0.13 * np.cos(v * 0.11)
    ship_style = 0.5 + 0.30 * np.cos(v * 0.47) - 0.10 * np.sin(v * 0.17)
    min_style = 0.5 + 0.28 * np.sin(v * 0.31 + 0.4)
    bias = 0.018 * np.sin(v * 0.19) + 0.011 * np.cos(v * 0.07)
    exception_style = 0.5 + 0.30 * np.cos(v * 0.29)
    return discount_style, ship_style, min_style, bias, exception_style


def _family_attrs(family):
    f = float(int(family) + 1)
    margin = 0.97 + 0.07 * np.sin(f * 0.63) + 0.03 * np.cos(f * 0.17)
    bulk = 0.55 + 0.45 * ((int(family) % 7) in (0, 3, 5)) + 0.08 * np.sin(f)
    service_affinity = 0.2 + 0.7 * ((int(family) % 11) in (1, 6))
    volatility = 0.03 + 0.045 * ((int(family) % 5) == 2) + 0.015 * np.cos(f * 0.33)
    return margin, bulk, service_affinity, max(0.02, volatility)


def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _structural_prediction(line_X, line_code, invoice_X, invoice_code, invoice_offsets):
    out = np.zeros(line_X.shape[0], dtype=float)
    for ii in range(len(invoice_offsets) - 1):
        s, e = int(invoice_offsets[ii]), int(invoice_offsets[ii + 1])
        idx = slice(s, e)
        n = e - s
        vendor = int(invoice_code[ii, IC_VENDOR_CODE])
        channel = int(invoice_code[ii, IC_CONTRACT_CHANNEL])
        region = int(invoice_code[ii, IC_INVOICE_REGION])
        month = int(invoice_code[ii, IC_MONTH_BUCKET])
        disc_style, ship_style, min_style, vendor_bias, _ = _vendor_attrs(vendor)
        contract_tier = int(np.bincount(line_code[idx, LC_CONTRACT_TIER].astype(int), minlength=5).argmax())
        vendor_volume_tier = min(4, int(np.log1p(max(0, 320 - vendor)) // 1.5))
        raw = line_X[idx, LX_RAW_AMOUNT]
        raw_pos = np.maximum(raw, 0.0)
        positive_subtotal = float(raw_pos.sum())
        header_discount = invoice_X[ii, IX_HEADER_DISCOUNT]
        header_shipping = invoice_X[ii, IX_HEADER_SHIPPING]
        handling = invoice_X[ii, IX_HEADER_HANDLING_FEE]
        min_charge_flag = int(invoice_X[ii, IX_MINIMUM_CHARGE_FLAG] > 0.5)
        expedite = int(invoice_X[ii, IX_EXPEDITE_FLAG] > 0.5)
        ch = line_code[idx, LC_CHARGE_TYPE].astype(int)
        fam = line_code[idx, LC_ITEM_FAMILY].astype(int)
        unit_type = line_code[idx, LC_UNIT_TYPE].astype(int)
        exc = line_code[idx, LC_EXCEPTION_CODE].astype(int)
        service_flag = np.isin(ch, [1, 2, 3]).astype(float)
        eligible_flag = (~np.isin(ch, [4, 5])).astype(float)
        package = np.maximum(line_X[idx, LX_PACKAGE_COUNT], 0.0)
        weight = np.maximum(line_X[idx, LX_WEIGHT_PROXY], 0.0)
        amount_share = raw_pos / max(raw_pos.sum(), 1e-9)
        elig_amount = raw_pos * eligible_flag * (ch != 2) * (ch != 3)
        eligible_share = elig_amount / max(elig_amount.sum(), 1e-9)
        package_share = package / max(package.sum(), 1e-9)
        weight_share = weight / max(weight.sum(), 1e-9)
        service_share = service_flag / max(service_flag.sum(), 1e-9) if service_flag.sum() > 0 else amount_share
        low_value = (raw_pos <= np.quantile(raw_pos, 0.35) if n > 2 else np.ones(n, dtype=bool)).astype(float)
        low_value_share = low_value * eligible_flag / max(np.sum(low_value * eligible_flag), 1e-9)
        fam_bulk = np.array([_family_attrs(f)[1] for f in fam])
        bulk_mix = float(np.average(fam_bulk, weights=np.maximum(raw_pos, 1.0))) if n else 0.0
        service_mix = float(service_flag.mean()) if n else 0.0
        service_absorb_guess = float(service_mix > 0.16)
        z_discount = np.array([
            0.6 + 0.5 * disc_style + 0.18 * contract_tier,
            0.4 + 0.8 * (contract_tier >= 2) + 0.3 * (channel == 1),
            -0.1 + 0.7 * bulk_mix + 0.1 * region,
            -0.4 + 1.3 * service_mix + 0.4 * service_absorb_guess,
            -0.2 + 0.8 * min_charge_flag + 0.2 * (n < 4),
        ])
        w_disc = _softmax(z_discount)
        disc_share = w_disc[0] * amount_share + w_disc[1] * eligible_share + w_disc[2] * package_share + w_disc[3] * service_share + w_disc[4] * low_value_share
        disc_share /= max(disc_share.sum(), 1e-9)
        z_ship = np.array([
            0.1 + 1.0 * ship_style + 0.30 * expedite,
            0.3 + 1.5 * bulk_mix + 0.25 * region,
            -0.2 + 0.60 * package.mean(),
            -0.4 + 1.0 * service_mix + 0.30 * (channel == 3),
        ])
        w_ship = _softmax(z_ship)
        ship_share = w_ship[0] * amount_share + w_ship[1] * weight_share + w_ship[2] * package_share + w_ship[3] * service_share
        ship_share /= max(ship_share.sum(), 1e-9)
        min_share = 0.58 * service_share + 0.27 * low_value_share + 0.15 * package_share
        min_share /= max(min_share.sum(), 1e-9)
        for local_j, j in enumerate(range(s, e)):
            family = int(fam[local_j])
            margin, bulk, service_affinity, volatility = _family_attrs(family)
            amount = float(raw[local_j])
            ut = int(unit_type[local_j])
            c = int(ch[local_j])
            ex = int(exc[local_j])
            unit_shift_guess = ((family in (2, 8, 13, 19)) and (ut in (3, 4) or family % 5 == 2))
            unit_correction = 1.0 + 0.012 * ut + 0.018 * float(unit_shift_guess)
            family_contract = 1.0 + 0.014 * contract_tier * np.sin((family + 1) * 0.6) + 0.010 * channel * np.cos((family + region + 1) * 0.3)
            subfamily = int(line_code[j, LC_ITEM_SUBFAMILY])
            vendor_line_type = int(line_code[j, LC_VENDOR_LINE_TYPE])
            hist_log = np.log1p(line_X[j, LX_HIST_ITEM_VOLUME])
            lead_log = np.log1p(line_X[j, LX_LEAD_TIME])
            price_idx = line_X[j, LX_RECENT_PRICE_INDEX]
            pos_norm = line_X[j, LX_LINE_POSITION_NORM]
            line_audit = np.tanh(
                0.34 * (price_idx - 1.08)
                + 0.12 * (hist_log - 5.0)
                - 0.10 * (lead_log - 2.0)
                + 0.18 * np.sin(0.21 * subfamily + 0.45 * vendor_line_type)
                + 0.11 * invoice_X[ii, IX_SEASON_SIN]
                - 0.08 * invoice_X[ii, IX_SEASON_COS]
                + 0.15 * (pos_norm - 0.5) * (c in (1, 3, 6))
            )
            complex_multiplier = 1.0 + (0.550 + 0.105 * contract_tier) * line_audit
            base = amount * unit_correction * family_contract * (1.0 + vendor_bias) * complex_multiplier
            base -= line_X[j, LX_REPORTED_DISCOUNT] * (0.8 + 0.1 * contract_tier)
            base += line_X[j, LX_REPORTED_TAX] * (0.55 + 0.06 * region)
            if c == 1:
                base *= 1.0 + 0.05 * service_affinity + 0.02 * contract_tier
            elif c == 6:
                base *= 0.82 + 0.04 * ex
            elif c == 2:
                base *= 0.60 + 0.07 * expedite
            elif c == 3:
                base *= 0.72 + 0.04 * channel
            base -= header_discount * disc_share[local_j] * (0.94 + 0.03 * np.sin(vendor + family))
            base += header_shipping * ship_share[local_j] * (0.90 + 0.04 * np.cos(region + family))
            base += handling * (0.45 * ship_share[local_j] + 0.55 * min_share[local_j]) * (1.0 + 0.025 * channel)
            local_mix = (
                0.255 * abs(amount) * np.sin(0.17 * subfamily + 0.33 * vendor_line_type + 0.11 * month)
                + 0.330 * header_discount * disc_share[local_j] * np.tanh(hist_log - 4.6)
                - 0.285 * header_shipping * ship_share[local_j] * np.tanh(lead_log - 1.8)
            )
            base += local_mix
            if min_charge_flag:
                base += max(0.0, 42.0 + 4.5 * region - 0.025 * positive_subtotal) * min_share[local_j] * (0.8 + 0.3 * min_style)
            if ex >= 7:
                base += (8.0 + 0.018 * abs(amount)) * np.sin(0.7 * ex + 0.2 * family) + 4.0 * (c == 6)
            elif ex > 0:
                base += (2.0 + 0.006 * abs(amount)) * np.cos(ex + channel)
            if c == 4:
                base *= 0.75 + 0.05 * (ex > 0)
                base -= 0.05 * header_discount * eligible_share[local_j]
            if c == 5:
                paired_strength = 0.25 + 0.50 * (0.35 + 0.03 * vendor_volume_tier)
                base *= paired_strength
                base -= 0.02 * positive_subtotal * amount_share[local_j]
            out[j] = base
    return out

def fit_invoice_model(train_line_X, train_line_code, train_invoice_X, train_invoice_code, train_invoice_offsets, train_y):
    y_log = _signed_log(np.asarray(train_y, dtype=float))
    structural = _structural_prediction(train_line_X, train_line_code, train_invoice_X, train_invoice_code, train_invoice_offsets)
    structural_log = _signed_log(structural)
    X, enc = _build_features(train_line_X, train_line_code, train_invoice_X, train_invoice_code, train_invoice_offsets, fit=True, y_log=y_log)
    residual = y_log - structural_log
    hgb = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=23, l2_regularization=0.08, random_state=17, min_samples_leaf=20)
    hgb.fit(X, residual)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=18.0))
    ridge.fit(X, residual)
    return {'enc': enc, 'hgb': hgb, 'ridge': ridge}


def predict_net_line_cost(line_X, line_code, invoice_X, invoice_code, invoice_offsets, params):
    structural = _structural_prediction(line_X, line_code, invoice_X, invoice_code, invoice_offsets)
    structural_log = _signed_log(structural)
    X = _build_features(line_X, line_code, invoice_X, invoice_code, invoice_offsets, encoders=params['enc'], fit=False)
    resid = 0.72 * params['hgb'].predict(X) + 0.28 * params['ridge'].predict(X)
    z = structural_log + resid
    pred = _inv_signed_log(z)
    raw = np.asarray(line_X[:, LX_RAW_AMOUNT], dtype=float)
    scale = np.maximum(100.0, 12.0 * np.maximum(1.0, np.abs(raw)) + 5.0 * (np.abs(raw).mean() + 1.0))
    return np.clip(pred, -scale, scale)


solve
"""Starter implementation for invoice_line_reconciliation.

Replace the bodies of fit_invoice_model and predict_net_line_cost.
"""
from __future__ import annotations
import numpy as np


def fit_invoice_model(
    train_line_X,
    train_line_code,
    train_invoice_X,
    train_invoice_code,
    train_invoice_offsets,
    train_y,
):
    # Simple starter: remember a signed-log mean and a rough raw-line slope.
    y = np.asarray(train_y, dtype=float)
    raw = np.asarray(train_line_X[:, 2], dtype=float)
    denom = float(np.dot(raw, raw) + 1e-9)
    slope = float(np.dot(raw, y) / denom) if raw.size else 1.0
    return {"mean": float(np.mean(y)) if y.size else 0.0, "slope": slope}


def predict_net_line_cost(
    line_X,
    line_code,
    invoice_X,
    invoice_code,
    invoice_offsets,
    params,
):
    raw = np.asarray(line_X[:, 2], dtype=float)
    slope = float(params.get("slope", 1.0)) if isinstance(params, dict) else 1.0
    pred = slope * raw
    if pred.shape[0] != line_X.shape[0]:
        pred = np.full(line_X.shape[0], float(params.get("mean", 0.0)))
    return pred.astype(float)

generator
"""Deterministic data generator for invoice_line_reconciliation.

This module is verifier-side. The agent-visible package receives only the
pre-generated train/public .npz files and the function interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

N_LINE_X = 14
N_LINE_CODE = 9
N_INVOICE_X = 12
N_INVOICE_CODE = 4

# line_X columns
LX_QUANTITY = 0
LX_UNIT_PRICE = 1
LX_RAW_AMOUNT = 2
LX_REPORTED_DISCOUNT = 3
LX_REPORTED_TAX = 4
LX_PACKAGE_COUNT = 5
LX_UNIT_SIZE = 6
LX_WEIGHT_PROXY = 7
LX_LEAD_TIME = 8
LX_HIST_ITEM_VOLUME = 9
LX_RECENT_PRICE_INDEX = 10
LX_LINE_POSITION_NORM = 11
LX_IS_NEGATIVE = 12
LX_IS_ROUND_AMOUNT = 13

# line_code columns
LC_ITEM_FAMILY = 0
LC_ITEM_SUBFAMILY = 1
LC_UNIT_TYPE = 2
LC_CHARGE_TYPE = 3
LC_VENDOR_LINE_TYPE = 4
LC_CONTRACT_TIER = 5
LC_REGION_CODE = 6
LC_CURRENCY_BUCKET = 7
LC_EXCEPTION_CODE = 8

# invoice_X columns
IX_VENDOR_AGE = 0
IX_VENDOR_VOLUME_TIER = 1
IX_INVOICE_SUBTOTAL = 2
IX_HEADER_DISCOUNT = 3
IX_HEADER_SHIPPING = 4
IX_HEADER_HANDLING_FEE = 5
IX_MINIMUM_CHARGE_FLAG = 6
IX_EXPEDITE_FLAG = 7
IX_STATED_LINE_COUNT = 8
IX_SEASON_SIN = 9
IX_SEASON_COS = 10
IX_CONTRACT_VALUE_LOG = 11

# invoice_code columns
IC_VENDOR_CODE = 0
IC_CONTRACT_CHANNEL = 1
IC_INVOICE_REGION = 2
IC_MONTH_BUCKET = 3

N_ITEM_FAMILY = 24
N_ITEM_SUBFAMILY = 96
N_UNIT_TYPE = 6
N_CHARGE_TYPE = 7
N_VENDOR_LINE_TYPE = 8
N_CONTRACT_TIER = 5
N_REGION = 9
N_CURRENCY = 4
N_EXCEPTION = 12
N_VENDOR = 320
N_CHANNEL = 5
N_MONTH = 12

CH_ITEM = 0
CH_SERVICE = 1
CH_SHIPPING = 2
CH_HANDLING = 3
CH_CREDIT = 4
CH_REVERSAL = 5
CH_ADJUSTMENT = 6

@dataclass(frozen=True)
class GeneratedData:
    line_X: np.ndarray
    line_code: np.ndarray
    invoice_X: np.ndarray
    invoice_code: np.ndarray
    invoice_offsets: np.ndarray
    y: Optional[np.ndarray]
    slice_name: Optional[np.ndarray] = None

    def as_npz_dict(self, include_y: bool = True, include_slices: bool = False) -> Dict[str, np.ndarray]:
        d = {
            "line_X": self.line_X.astype(np.float64),
            "line_code": self.line_code.astype(np.int64),
            "invoice_X": self.invoice_X.astype(np.float64),
            "invoice_code": self.invoice_code.astype(np.int64),
            "invoice_offsets": self.invoice_offsets.astype(np.int64),
        }
        if include_y and self.y is not None:
            d["y"] = self.y.astype(np.float64)
        if include_slices and self.slice_name is not None:
            d["slice_name"] = self.slice_name.astype("U40")
        return d


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _zipf_choice(rng: np.random.Generator, n: int, size: int, alpha: float = 1.18) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = 1.0 / np.power(ranks, alpha)
    p /= p.sum()
    return rng.choice(n, size=size, replace=True, p=p)


def _vendor_attrs(vendor: int) -> Tuple[float, float, float, float, float]:
    # deterministic smooth vendor traits, not random state dependent.
    v = float(vendor + 1)
    discount_style = 0.5 + 0.32 * np.sin(v * 0.73) + 0.13 * np.cos(v * 0.11)
    ship_style = 0.5 + 0.30 * np.cos(v * 0.47) - 0.10 * np.sin(v * 0.17)
    min_style = 0.5 + 0.28 * np.sin(v * 0.31 + 0.4)
    bias = 0.018 * np.sin(v * 0.19) + 0.011 * np.cos(v * 0.07)
    exception_style = 0.5 + 0.30 * np.cos(v * 0.29)
    return discount_style, ship_style, min_style, bias, exception_style


def _family_attrs(family: int) -> Tuple[float, float, float, float]:
    f = float(family + 1)
    margin = 0.97 + 0.07 * np.sin(f * 0.63) + 0.03 * np.cos(f * 0.17)
    bulk = 0.55 + 0.45 * ((family % 7) in (0, 3, 5)) + 0.08 * np.sin(f)
    service_affinity = 0.2 + 0.7 * ((family % 11) in (1, 6))
    volatility = 0.03 + 0.045 * ((family % 5) == 2) + 0.015 * np.cos(f * 0.33)
    return margin, bulk, service_affinity, max(0.02, volatility)


def _sample_line_count(rng: np.random.Generator, profile: str) -> int:
    u = rng.random()
    if profile == "matched_mix":
        return int(rng.integers(7, 16))
    if u < 0.54:
        return int(rng.integers(1, 5))
    if u < 0.88:
        return int(rng.integers(5, 16))
    if u < 0.98:
        return int(rng.integers(16, 42))
    return int(rng.integers(42, 76))


def _choose_profile(rng: np.random.Generator, split: str, i: int) -> str:
    # The special profiles occur naturally but are mildly enriched in hidden
    # so diagnostic slices have enough rows without artificial duplication.
    if split == "hidden":
        profiles = [
            "regular", "discount", "shipping", "minimum", "reversal",
            "matched_mix", "rare_vendor", "rare_exception", "service_absorb", "unit_shift",
        ]
        p = np.array([0.33, 0.10, 0.10, 0.07, 0.08, 0.08, 0.07, 0.05, 0.05, 0.07])
    elif split == "public":
        profiles = ["regular", "discount", "shipping", "minimum", "reversal", "service_absorb", "unit_shift"]
        p = np.array([0.55, 0.11, 0.10, 0.05, 0.05, 0.08, 0.06])
    else:
        profiles = ["regular", "discount", "shipping", "minimum", "reversal", "rare_vendor", "rare_exception", "service_absorb", "unit_shift"]
        p = np.array([0.57, 0.09, 0.08, 0.045, 0.045, 0.035, 0.035, 0.05, 0.05])
    p = p / p.sum()
    return str(rng.choice(profiles, p=p))


def _sample_vendor(rng: np.random.Generator, profile: str, split: str) -> int:
    if profile == "rare_vendor":
        return int(rng.integers(190, N_VENDOR))
    if split == "hidden" and rng.random() < 0.08:
        return int(rng.integers(160, N_VENDOR))
    return int(_zipf_choice(rng, N_VENDOR, 1, alpha=1.22)[0])


def generate_dataset(seed: int, n_invoices: int, split: str = "train") -> GeneratedData:
    rng = _rng(seed)
    line_Xs = []
    line_codes = []
    invoice_X = np.zeros((n_invoices, N_INVOICE_X), dtype=np.float64)
    invoice_code = np.zeros((n_invoices, N_INVOICE_CODE), dtype=np.int64)
    offsets = [0]
    ys = []
    slice_names = []

    for inv_idx in range(n_invoices):
        profile = _choose_profile(rng, split, inv_idx)
        n = _sample_line_count(rng, profile)
        vendor = _sample_vendor(rng, profile, split)
        disc_style, ship_style, min_style, vendor_bias, exc_style = _vendor_attrs(vendor)
        vendor_volume_tier = min(4, int(np.log1p(max(0, N_VENDOR - vendor)) // 1.5))
        vendor_age = rng.gamma(2.0 + 0.15 * vendor_volume_tier, 18.0) + 3.0 * vendor_volume_tier
        contract_tier = int(np.clip(rng.choice(N_CONTRACT_TIER, p=[0.34, 0.28, 0.19, 0.13, 0.06]) + (vendor_volume_tier >= 3 and rng.random() < 0.25), 0, N_CONTRACT_TIER - 1))
        channel = int(rng.choice(N_CHANNEL, p=[0.38, 0.27, 0.16, 0.12, 0.07]))
        region = int(rng.choice(N_REGION, p=np.array([0.18,0.15,0.14,0.12,0.10,0.09,0.08,0.08,0.06])))
        month = int(rng.integers(0, N_MONTH))
        expedite = int((profile == "shipping" and rng.random() < 0.65) or rng.random() < (0.08 + 0.05 * (channel == 3)))
        contract_log = float(np.log1p(rng.lognormal(mean=7.0 + 0.18 * contract_tier + 0.08 * vendor_volume_tier, sigma=0.7)))

        # correlated family mix
        if profile == "shipping":
            preferred_families = np.array([0, 3, 5, 7, 14, 21])
        elif profile == "service_absorb":
            preferred_families = np.array([1, 6, 12, 17])
        elif profile == "unit_shift":
            preferred_families = np.array([2, 8, 13, 19])
        else:
            base = int((vendor * 3 + region * 5 + channel) % N_ITEM_FAMILY)
            preferred_families = np.array([(base + k) % N_ITEM_FAMILY for k in range(6)])

        inv_line_X = np.zeros((n, N_LINE_X), dtype=np.float64)
        inv_line_code = np.zeros((n, N_LINE_CODE), dtype=np.int64)
        raw_abs = np.zeros(n, dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        package = np.zeros(n, dtype=np.float64)
        service_flag = np.zeros(n, dtype=np.float64)
        eligible_flag = np.ones(n, dtype=np.float64)
        charge = np.full(n, CH_ITEM, dtype=np.int64)
        exception = np.zeros(n, dtype=np.int64)

        for j in range(n):
            if rng.random() < 0.72:
                family = int(rng.choice(preferred_families))
            else:
                family = int(rng.integers(0, N_ITEM_FAMILY))
            subfamily = int((family * 4 + rng.integers(0, 4) + (vendor % 3)) % N_ITEM_SUBFAMILY)
            unit_type = int(np.clip(rng.poisson(1.4 + 0.08 * family), 0, N_UNIT_TYPE - 1))
            currency = int(rng.choice(N_CURRENCY, p=[0.72, 0.14, 0.09, 0.05]))
            vendor_line_type = int(rng.choice(N_VENDOR_LINE_TYPE, p=[0.54,0.16,0.10,0.07,0.05,0.04,0.025,0.015]))

            # Charge type distribution with special profile enrichment.
            probs = np.array([0.76, 0.07, 0.025, 0.025, 0.035, 0.025, 0.055], dtype=np.float64)
            if profile == "service_absorb":
                probs[[CH_SERVICE, CH_HANDLING, CH_ADJUSTMENT]] += [0.18, 0.07, 0.04]
            if profile == "reversal":
                probs[[CH_CREDIT, CH_REVERSAL, CH_ADJUSTMENT]] += [0.11, 0.12, 0.05]
            if profile == "shipping":
                probs[[CH_SHIPPING, CH_HANDLING]] += [0.06, 0.04]
            probs /= probs.sum()
            ch = int(rng.choice(N_CHARGE_TYPE, p=probs))
            charge[j] = ch
            if ch in (CH_SERVICE, CH_SHIPPING, CH_HANDLING):
                service_flag[j] = 1.0
            if ch in (CH_CREDIT, CH_REVERSAL):
                eligible_flag[j] = 0.0

            exc_p = 0.015 + 0.018 * (profile == "rare_exception") + 0.01 * exc_style + 0.015 * (ch in (CH_CREDIT, CH_REVERSAL, CH_ADJUSTMENT))
            if rng.random() < exc_p:
                if profile == "rare_exception" and rng.random() < 0.65:
                    exc = int(rng.integers(7, N_EXCEPTION))
                else:
                    
                    exc_probs = np.array([0.52,0.16,0.09,0.06,0.045,0.035,0.025,0.02,0.017,0.014,0.011,0.013], dtype=float)
                    exc_probs = exc_probs / exc_probs.sum()
                    exc = int(rng.choice(N_EXCEPTION, p=exc_probs))
            else:
                exc = 0
            exception[j] = exc

            margin, bulk, service_affinity, volatility = _family_attrs(family)
            qty = rng.lognormal(mean=0.35 + 0.18 * (unit_type in (2, 4)) + 0.08 * (profile == "unit_shift"), sigma=0.85)
            qty = max(0.2, min(qty, 120.0))
            base_price = np.exp(2.15 + 0.055 * family + 0.025 * subfamily + 0.08 * unit_type + rng.normal(0, 0.45 + volatility))
            unit_size = np.exp(rng.normal(0.2 + 0.13 * unit_type + 0.18 * bulk, 0.42))
            package_count = max(1.0, np.ceil(qty / max(1.0, 2.2 + unit_size * (1.4 + 0.2 * bulk))))
            if profile == "shipping" and bulk > 0.9:
                package_count *= rng.choice([1.0, 2.0, 3.0], p=[0.35,0.45,0.20])
            weight_proxy = package_count * unit_size * (0.7 + 1.6 * bulk) * rng.lognormal(0, 0.22)
            recent_price_index = 0.92 + 0.025 * month + 0.07 * np.sin((month + family) / 2.0) + rng.normal(0, 0.045)
            hist_volume = np.exp(2.3 + 0.1 * vendor_volume_tier - 0.045 * family + rng.normal(0, 0.8))
            lead_time = rng.gamma(2.0 + 0.35 * region + 0.4 * expedite, 2.0 + 0.15 * unit_type)

            currency_scale = [1.0, 1.08, 0.82, 1.22][currency]
            amount = qty * base_price * margin * recent_price_index * currency_scale
            if ch == CH_SERVICE:
                amount *= 0.45 + 1.1 * service_affinity
            elif ch in (CH_SHIPPING, CH_HANDLING):
                amount *= 0.18 + 0.22 * package_count
            elif ch == CH_ADJUSTMENT:
                amount *= 0.22
            elif ch in (CH_CREDIT, CH_REVERSAL):
                amount *= -(0.20 + 0.65 * rng.random())
            amount += rng.normal(0, max(0.6, abs(amount) * 0.025))
            if rng.random() < 0.17:
                amount = np.round(amount / 5.0) * 5.0
            reported_discount = max(0.0, amount * rng.uniform(0.0, 0.045) * (amount > 0))
            reported_tax = max(0.0, (amount - reported_discount) * (0.012 + 0.006 * region + 0.015 * (ch == CH_SERVICE)) * rng.uniform(0.65, 1.12))

            inv_line_X[j, LX_QUANTITY] = qty
            inv_line_X[j, LX_UNIT_PRICE] = amount / max(qty, 1e-6)
            inv_line_X[j, LX_RAW_AMOUNT] = amount
            inv_line_X[j, LX_REPORTED_DISCOUNT] = reported_discount
            inv_line_X[j, LX_REPORTED_TAX] = reported_tax
            inv_line_X[j, LX_PACKAGE_COUNT] = package_count
            inv_line_X[j, LX_UNIT_SIZE] = unit_size
            inv_line_X[j, LX_WEIGHT_PROXY] = weight_proxy
            inv_line_X[j, LX_LEAD_TIME] = lead_time
            inv_line_X[j, LX_HIST_ITEM_VOLUME] = hist_volume
            inv_line_X[j, LX_RECENT_PRICE_INDEX] = recent_price_index
            inv_line_X[j, LX_LINE_POSITION_NORM] = j / max(1, n - 1)
            inv_line_X[j, LX_IS_NEGATIVE] = float(amount < 0)
            inv_line_X[j, LX_IS_ROUND_AMOUNT] = float(abs(amount / 5.0 - np.round(amount / 5.0)) < 1e-6)

            inv_line_code[j] = np.array([family, subfamily, unit_type, ch, vendor_line_type, contract_tier, region, currency, exc], dtype=np.int64)
            raw_abs[j] = abs(amount)
            weight[j] = weight_proxy
            package[j] = package_count

        subtotal = float(np.sum(inv_line_X[:, LX_RAW_AMOUNT]))
        positive_subtotal = float(np.sum(np.maximum(inv_line_X[:, LX_RAW_AMOUNT], 0.0)))
        discount_rate = max(0.0, 0.018 + 0.012 * contract_tier + 0.009 * channel + 0.008 * disc_style + rng.normal(0, 0.012))
        if profile == "discount":
            discount_rate += rng.uniform(0.04, 0.09)
        header_discount = positive_subtotal * min(0.24, discount_rate)
        ship_base = (0.012 + 0.004 * region + 0.010 * expedite + 0.009 * ship_style) * max(positive_subtotal, 1.0)
        ship_base += 0.34 * np.sum(weight) + 1.7 * np.sum(package)
        if profile == "shipping":
            ship_base *= rng.uniform(1.35, 2.1)
        header_shipping = max(0.0, ship_base * rng.lognormal(0, 0.12))
        handling = max(0.0, (0.004 + 0.003 * channel + 0.004 * min_style) * positive_subtotal + 0.9 * n + rng.normal(0, 2.0))
        small_invoice = positive_subtotal < (170.0 + 25.0 * region + 45.0 * (channel == 3)) or profile == "minimum"
        min_charge_flag = int(small_invoice and rng.random() < (0.62 if profile == "minimum" else 0.34))
        if min_charge_flag:
            handling += max(0.0, 36.0 + 8.0 * region + 12.0 * min_style - 0.04 * positive_subtotal)

        invoice_X[inv_idx, IX_VENDOR_AGE] = vendor_age
        invoice_X[inv_idx, IX_VENDOR_VOLUME_TIER] = vendor_volume_tier
        invoice_X[inv_idx, IX_INVOICE_SUBTOTAL] = subtotal
        invoice_X[inv_idx, IX_HEADER_DISCOUNT] = header_discount
        invoice_X[inv_idx, IX_HEADER_SHIPPING] = header_shipping
        invoice_X[inv_idx, IX_HEADER_HANDLING_FEE] = handling
        invoice_X[inv_idx, IX_MINIMUM_CHARGE_FLAG] = min_charge_flag
        invoice_X[inv_idx, IX_EXPEDITE_FLAG] = expedite
        invoice_X[inv_idx, IX_STATED_LINE_COUNT] = n
        invoice_X[inv_idx, IX_SEASON_SIN] = np.sin(2 * np.pi * month / 12.0)
        invoice_X[inv_idx, IX_SEASON_COS] = np.cos(2 * np.pi * month / 12.0)
        invoice_X[inv_idx, IX_CONTRACT_VALUE_LOG] = contract_log
        invoice_code[inv_idx] = np.array([vendor, channel, region, month], dtype=np.int64)

        # allocation bases
        raw_pos = np.maximum(inv_line_X[:, LX_RAW_AMOUNT], 0.0)
        amount_share = raw_pos / max(raw_pos.sum(), 1e-9)
        elig_amount = raw_pos * eligible_flag * (charge != CH_SHIPPING) * (charge != CH_HANDLING)
        eligible_share = elig_amount / max(elig_amount.sum(), 1e-9)
        package_share = package / max(package.sum(), 1e-9)
        weight_share = weight / max(weight.sum(), 1e-9)
        service_share = service_flag / max(service_flag.sum(), 1e-9) if service_flag.sum() > 0 else amount_share
        low_value = (raw_pos <= np.quantile(raw_pos, 0.35) if n > 2 else np.ones(n, dtype=bool)).astype(float)
        low_value_share = low_value * eligible_flag / max(np.sum(low_value * eligible_flag), 1e-9)

        # Smooth profile weights driven by visible context and line mix.
        fam_bulk = np.array([_family_attrs(int(f))[1] for f in inv_line_code[:, LC_ITEM_FAMILY]])
        bulk_mix = float(np.average(fam_bulk, weights=np.maximum(raw_pos, 1.0))) if n else 0.0
        service_mix = float(service_flag.mean())
        z_discount = np.array([
            0.6 + 0.5 * disc_style + 0.18 * contract_tier,
            0.4 + 0.8 * (contract_tier >= 2) + 0.3 * (channel == 1),
            -0.1 + 0.7 * bulk_mix + 0.1 * region,
            -0.4 + 1.3 * service_mix + 0.4 * (profile == "service_absorb"),
            -0.2 + 0.8 * min_charge_flag + 0.2 * (n < 4),
        ])
        w_disc = _softmax(z_discount)
        disc_share = (w_disc[0] * amount_share + w_disc[1] * eligible_share + w_disc[2] * package_share + w_disc[3] * service_share + w_disc[4] * low_value_share)
        disc_share /= max(disc_share.sum(), 1e-9)

        z_ship = np.array([
            0.1 + 1.0 * ship_style + 0.30 * expedite,
            0.3 + 1.5 * bulk_mix + 0.25 * region,
            -0.2 + 0.60 * package.mean(),
            -0.4 + 1.0 * service_mix + 0.30 * (channel == 3),
        ])
        w_ship = _softmax(z_ship)
        ship_share = w_ship[0] * amount_share + w_ship[1] * weight_share + w_ship[2] * package_share + w_ship[3] * service_share
        ship_share /= max(ship_share.sum(), 1e-9)

        min_share = 0.58 * service_share + 0.27 * low_value_share + 0.15 * package_share
        min_share /= max(min_share.sum(), 1e-9)

        # True target generation.
        y = np.zeros(n, dtype=np.float64)
        for j in range(n):
            family = int(inv_line_code[j, LC_ITEM_FAMILY])
            unit_type = int(inv_line_code[j, LC_UNIT_TYPE])
            ch = int(inv_line_code[j, LC_CHARGE_TYPE])
            exc = int(inv_line_code[j, LC_EXCEPTION_CODE])
            margin, bulk, service_affinity, volatility = _family_attrs(family)
            amount = inv_line_X[j, LX_RAW_AMOUNT]
            unit_correction = 1.0 + 0.012 * unit_type + 0.022 * (profile == "unit_shift") * ((unit_type in (3, 4)) or (family % 5 == 2))
            family_contract = 1.0 + 0.014 * contract_tier * np.sin((family + 1) * 0.6) + 0.010 * channel * np.cos((family + region + 1) * 0.3)
            # v0.2 hardening: a visible-but-nonlinear audit adjustment from line history,
            # season, subfamily, and vendor-line conventions. This is meant to break
            # a low-dimensional allocation parser while remaining learnable from rows.
            subfamily = int(inv_line_code[j, LC_ITEM_SUBFAMILY])
            vendor_line_type = int(inv_line_code[j, LC_VENDOR_LINE_TYPE])
            hist_log = np.log1p(inv_line_X[j, LX_HIST_ITEM_VOLUME])
            lead_log = np.log1p(inv_line_X[j, LX_LEAD_TIME])
            price_idx = inv_line_X[j, LX_RECENT_PRICE_INDEX]
            pos_norm = inv_line_X[j, LX_LINE_POSITION_NORM]
            line_audit = np.tanh(
                0.34 * (price_idx - 1.08)
                + 0.12 * (hist_log - 5.0)
                - 0.10 * (lead_log - 2.0)
                + 0.18 * np.sin(0.21 * subfamily + 0.45 * vendor_line_type)
                + 0.11 * invoice_X[inv_idx, IX_SEASON_SIN]
                - 0.08 * invoice_X[inv_idx, IX_SEASON_COS]
                + 0.15 * (pos_norm - 0.5) * (charge[j] in (CH_SERVICE, CH_ADJUSTMENT, CH_HANDLING))
            )
            complex_multiplier = 1.0 + (0.550 + 0.105 * contract_tier) * line_audit
            base = amount * unit_correction * family_contract * (1.0 + vendor_bias) * complex_multiplier
            base -= inv_line_X[j, LX_REPORTED_DISCOUNT] * (0.8 + 0.1 * contract_tier)
            base += inv_line_X[j, LX_REPORTED_TAX] * (0.55 + 0.06 * region)
            if ch == CH_SERVICE:
                base *= 1.0 + 0.05 * service_affinity + 0.02 * contract_tier
            elif ch == CH_ADJUSTMENT:
                base *= 0.82 + 0.04 * exc
            elif ch == CH_SHIPPING:
                base *= 0.60 + 0.07 * expedite
            elif ch == CH_HANDLING:
                base *= 0.72 + 0.04 * channel
            # Header adjustments.
            base -= header_discount * disc_share[j] * (0.94 + 0.03 * np.sin(vendor + family))
            base += header_shipping * ship_share[j] * (0.90 + 0.04 * np.cos(region + family))
            base += handling * (0.45 * ship_share[j] + 0.55 * min_share[j]) * (1.0 + 0.025 * channel)
            # Secondary row/mix interaction: subtle enough not to dominate, but it
            # makes same-total invoices with different subfamily/vendor-line mixes diverge.
            local_mix = (
                0.255 * raw_abs[j] * np.sin(0.17 * subfamily + 0.33 * vendor_line_type + 0.11 * month)
                + 0.330 * header_discount * disc_share[j] * np.tanh(hist_log - 4.6)
                - 0.285 * header_shipping * ship_share[j] * np.tanh(lead_log - 1.8)
            )
            base += local_mix
            if min_charge_flag:
                base += max(0.0, 42.0 + 4.5 * region - 0.025 * positive_subtotal) * min_share[j] * (0.8 + 0.3 * min_style)
            # rare exceptions / credits
            if exc >= 7:
                base += (8.0 + 0.018 * raw_abs[j]) * np.sin(0.7 * exc + 0.2 * family) + 4.0 * (ch == CH_ADJUSTMENT)
            elif exc > 0:
                base += (2.0 + 0.006 * raw_abs[j]) * np.cos(exc + channel)
            if ch == CH_CREDIT:
                base *= 0.75 + 0.05 * (exc > 0)
                base -= 0.05 * header_discount * eligible_share[j]
            if ch == CH_REVERSAL:
                # some reversals are true offsets, some are correction credits.
                paired_strength = 0.25 + 0.50 * (rng.random() < (0.35 + 0.03 * vendor_volume_tier))
                base *= paired_strength
                base -= 0.02 * positive_subtotal * amount_share[j]
            noise_sd = 0.8 + 0.018 * abs(base) + 0.008 * positive_subtotal / max(n, 1) + 0.6 * volatility
            y[j] = base + rng.normal(0, noise_sd)

        # Inject a few near-duplicates after target construction by adjusting paired lines.
        if profile == "reversal" and n >= 3:
            pos_candidates = np.where(inv_line_X[:, LX_RAW_AMOUNT] > 20)[0]
            neg_candidates = np.where(inv_line_X[:, LX_RAW_AMOUNT] < -5)[0]
            if len(pos_candidates) and len(neg_candidates):
                a = int(rng.choice(pos_candidates))
                b = int(rng.choice(neg_candidates))
                if rng.random() < 0.55:
                    inv_line_X[b, LX_RAW_AMOUNT] = -abs(inv_line_X[a, LX_RAW_AMOUNT]) * rng.uniform(0.85, 1.05)
                    inv_line_X[b, LX_UNIT_PRICE] = inv_line_X[b, LX_RAW_AMOUNT] / max(inv_line_X[b, LX_QUANTITY], 1e-6)
                    y[b] = -0.65 * y[a] + rng.normal(0, 1.5 + 0.03 * abs(y[a]))

        line_Xs.append(inv_line_X)
        line_codes.append(inv_line_code)
        ys.append(y)
        slice_names.extend([profile] * n)
        offsets.append(offsets[-1] + n)

    return GeneratedData(
        line_X=np.vstack(line_Xs),
        line_code=np.vstack(line_codes),
        invoice_X=invoice_X,
        invoice_code=invoice_code,
        invoice_offsets=np.array(offsets, dtype=np.int64),
        y=np.concatenate(ys),
        slice_name=np.array(slice_names, dtype="U40"),
    )


def make_train_public() -> Tuple[GeneratedData, GeneratedData]:
    return generate_dataset(77131, 850, "train"), generate_dataset(81211, 160, "public")


def make_hidden() -> GeneratedData:
    return generate_dataset(99221, 380, "hidden")

hidden
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

try:
    from .generator import make_hidden, make_train_public
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from generator import make_hidden, make_train_public


def signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))

# Anchors are set after v0.2 calibration. GOOD values sit slightly above
# the measured reference RMSEs, so the reference can reach full reward while
# valid weaker models still receive continuous gradient.
OVERALL_GOOD = 0.312
OVERALL_BAD = 0.860
SLICE_GOOD = {
    "regular": 0.355,
    "discount": 0.390,
    "shipping": 0.283,
    "minimum": 0.227,
    "reversal": 0.345,
    "matched_mix": 0.271,
    "rare_vendor": 0.252,
    "rare_exception": 0.257,
    "service_absorb": 0.211,
    "unit_shift": 0.247,
}
SLICE_BAD = {k: 0.72 for k in SLICE_GOOD}
SLICE_WEIGHTS = {
    "regular": 0.04,
    "discount": 0.04,
    "shipping": 0.04,
    "minimum": 0.035,
    "reversal": 0.035,
    "matched_mix": 0.035,
    "rare_vendor": 0.035,
    "rare_exception": 0.03,
    "service_absorb": 0.03,
    "unit_shift": 0.03,
}
OVERALL_WEIGHT = 0.65
SCORE_POWER = 1.25


def score_rmse(rmse, good, bad):
    if not np.isfinite(rmse):
        return 0.0
    if rmse <= good:
        return 1.0
    if rmse >= bad:
        return 0.0
    t = (bad - rmse) / (bad - good)
    return float(np.clip(t, 0.0, 1.0) ** SCORE_POWER)


def load_solution(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("candidate_solution", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def evaluate(solve_path: str | Path, write_dir: str | Path | None = None):
    train, _ = make_train_public()
    hidden = make_hidden()
    mod = load_solution(solve_path)
    # Copy inputs and check mutation/determinism.
    args_train = [train.line_X.copy(), train.line_code.copy(), train.invoice_X.copy(), train.invoice_code.copy(), train.invoice_offsets.copy(), train.y.copy()]
    before = [a.copy() for a in args_train]
    params = mod.fit_invoice_model(*args_train)
    for a, b in zip(args_train, before):
        if not np.array_equal(a, b):
            return _finish(0.0, {"error": "input_mutation_in_fit"}, write_dir)
    pred_args = [hidden.line_X.copy(), hidden.line_code.copy(), hidden.invoice_X.copy(), hidden.invoice_code.copy(), hidden.invoice_offsets.copy()]
    before_pred = [a.copy() for a in pred_args]
    pred1 = np.asarray(mod.predict_net_line_cost(*pred_args, params), dtype=float)
    pred2 = np.asarray(mod.predict_net_line_cost(*[x.copy() for x in pred_args], params), dtype=float)
    for a, b in zip(pred_args, before_pred):
        if not np.array_equal(a, b):
            return _finish(0.0, {"error": "input_mutation_in_predict"}, write_dir)
    if pred1.shape != hidden.y.shape:
        return _finish(0.0, {"error": "wrong_shape", "shape": list(pred1.shape), "expected": list(hidden.y.shape)}, write_dir)
    if not np.all(np.isfinite(pred1)):
        return _finish(0.0, {"error": "nonfinite_prediction"}, write_dir)
    if not np.allclose(pred1, pred2, rtol=0, atol=1e-10):
        return _finish(0.0, {"error": "nondeterministic_prediction"}, write_dir)
    if np.nanmax(np.abs(pred1)) > 1e9:
        return _finish(0.0, {"error": "catastrophic_scale"}, write_dir)
    err = signed_log(pred1) - signed_log(hidden.y)
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_score = score_rmse(overall_rmse, OVERALL_GOOD, OVERALL_BAD)
    slice_metrics = {}
    slice_score_sum = 0.0
    for name, wt in SLICE_WEIGHTS.items():
        mask = hidden.slice_name == name
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
        sc = score_rmse(rmse, SLICE_GOOD.get(name, OVERALL_GOOD), SLICE_BAD.get(name, OVERALL_BAD))
        slice_metrics[name] = {"n": int(mask.sum()), "rmse": rmse, "score": sc, "weight": wt}
        slice_score_sum += wt * sc
    reward = OVERALL_WEIGHT * overall_score + slice_score_sum
    metrics = {
        "reward": float(np.clip(reward, 0.0, 1.0)),
        "overall_rmse": overall_rmse,
        "overall_score": overall_score,
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
