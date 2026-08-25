# Route profile quality

The old route summary model treats each route like a bag of readings. That is where it breaks. Routes with similar averages can land in different final condition scores, especially when the profile has handoffs, short excursions, or a long recovery section.

Implement the model in `app/solve.py`.

Expose these two functions:

```python
def fit_quality_model(train_step_X, train_step_code, train_route_X, train_route_offsets, train_y):
    ...

def predict_quality_score(step_X, step_code, route_X, route_offsets, params):
    ...
```

`fit_quality_model` receives the labeled training routes and returns any object that `predict_quality_score` can use later.

`predict_quality_score` receives unlabeled routes in the same packed format and returns one numeric score per route.

## Data layout

`train_data.npz` contains:

- `train_step_X`: packed step-level numeric records.
- `train_step_code`: packed step-level integer codes.
- `train_route_X`: one route-level numeric row per route.
- `train_route_offsets`: route boundaries into the packed step arrays.
- `train_y`: final condition score for each route.

`public_eval.npz` has the same layout with `public_` prefixes and includes labels for local checks only.

Step rows are ordered inside each route. For route `i`, its steps are:

```python
start = route_offsets[i]
end = route_offsets[i + 1]
steps = step_X[start:end]
codes = step_code[start:end]
```

The numeric step columns are:

```text
duration, reported_temperature, humidity, load_fraction,
delay_flag, vibration, scan_gap, relative_position
```

The step code columns are:

```text
phase_code, handling_code, sensor_code
```

The route-level columns are:

```text
package_class, product_family, declared_sensitivity,
carrier_type, planned_duration, route_risk
```

`package_class`, `product_family`, and `carrier_type` are integer-coded values stored in the numeric route array.

## Constraints

- Return a one-dimensional array with length `len(route_offsets) - 1`.
- Predictions must be finite numbers.
- Do not mutate the input arrays.
- Do not read files outside the provided app/data area.
- Do not use network access.

Public tests only check the interface and basic behavior. The hidden evaluation uses separate routes.


  solve
  import numpy as np


def fit_quality_model(train_step_X, train_step_code, train_route_X, train_route_offsets, train_y):
    y = np.asarray(train_y, dtype=float).reshape(-1)
    mean = float(np.mean(y)) if y.size else 0.0
    return {'mean': mean}


def predict_quality_score(step_X, step_code, route_X, route_offsets, params):
    n_routes = len(route_offsets) - 1
    mean = float(params.get('mean', 0.0))
    return np.full(n_routes, mean, dtype=float)


refernce
import numpy as np

# route_X columns:
# 0 package_class, 1 product_family, 2 declared_sensitivity,
# 3 carrier_type, 4 planned_duration, 5 route_risk
# step_X columns:
# 0 duration, 1 reported_temperature, 2 humidity, 3 load_fraction,
# 4 delay_flag, 5 vibration, 6 scan_gap, 7 relative_position
# step_code columns:
# 0 phase_code, 1 handling_code, 2 sensor_code

_SENSOR_BIAS_EST = np.array([0.0, -0.7, 1.2, 1.8], dtype=float)
_PHASE_BIAS_EST = np.array([0.0, 0.6, -0.5, 0.0, 0.3], dtype=float)


def _onehot_int(x, k):
    out = np.zeros(k, dtype=float)
    ix = int(round(float(x)))
    if 0 <= ix < k:
        out[ix] = 1.0
    return out


def _safe_bounds(prod):
    prod = int(round(float(prod)))
    if prod == 0:
        return 8.0, 31.0, 20.0
    if prod == 1:
        return -2.0, 10.0, 5.0
    return -18.0, -1.5, -6.0


def _longest_duration_run(mask, dur):
    best = 0.0
    cur = 0.0
    for m, d in zip(mask, dur):
        if bool(m):
            cur += float(d)
            if cur > best:
                best = cur
        else:
            cur = 0.0
    return best


def _extract_features(step_X, step_code, route_X, route_offsets):
    step_X = np.asarray(step_X)
    step_code = np.asarray(step_code)
    route_X = np.asarray(route_X)
    route_offsets = np.asarray(route_offsets)
    n = len(route_offsets) - 1
    rows = []
    for i in range(n):
        a = int(route_offsets[i])
        b = int(route_offsets[i + 1])
        X = np.asarray(step_X[a:b], dtype=float)
        C = np.asarray(step_code[a:b], dtype=int)
        r = np.asarray(route_X[i], dtype=float)
        if X.shape[0] == 0:
            rows.append(np.zeros(640, dtype=float))
            continue
        dur = X[:, 0]
        temp = X[:, 1]
        hum = X[:, 2]
        load = X[:, 3]
        delay = X[:, 4]
        vib = X[:, 5]
        pos = X[:, 7]
        phase = np.clip(C[:, 0], 0, 4)
        hand = np.clip(C[:, 1], 0, 4)
        sensor = np.clip(C[:, 2], 0, 3)
        total = float(np.sum(dur)) + 1e-9
        w = dur / total
        pkg, prod, sens, carrier, planned, risk = r[:6]
        pkg_i = int(round(float(pkg)))
        prod_i = int(round(float(prod)))
        lo, hi, mid = _safe_bounds(prod_i)
        f = []
        f.extend([total, planned, planned - total, sens, risk, len(dur), float(np.mean(dur)), float(np.max(dur))])
        f.extend(_onehot_int(pkg, 4))
        f.extend(_onehot_int(prod, 3))
        f.extend(_onehot_int(carrier, 3))
        for arr in [temp, hum, load, delay, vib]:
            arr = np.asarray(arr, dtype=float)
            f.extend([float(np.mean(arr)), float(np.sum(w * arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))])
            f.extend(np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(float).tolist())
        f.extend([
            float(np.sum(dur * (temp - mid))),
            float(np.sum(dur * np.abs(temp - mid))),
            float(np.max(np.abs(temp - mid))),
            float(temp[0]),
            float(temp[-1]),
            float(np.mean(temp[:max(1, len(temp) // 3)])),
            float(np.mean(temp[-max(1, len(temp) // 3):])),
        ])
        for th_shift in [-4, -2, 0, 2, 4, 7, 10]:
            th = hi + th_shift
            excess = np.maximum(temp - th, 0.0)
            f.append(float(np.sum(dur * excess)))
            f.append(float(np.max(excess)))
        for th_shift in [-10, -7, -4, -2, 0, 2, 4]:
            th = lo + th_shift
            excess = np.maximum(th - temp, 0.0)
            f.append(float(np.sum(dur * excess)))
            f.append(float(np.max(excess)))
        for k in range(5):
            m = phase == k
            if np.any(m):
                f.extend([
                    float(np.sum(dur[m]) / total),
                    float(np.mean(temp[m])),
                    float(np.max(temp[m])),
                    float(np.sum(dur[m] * np.maximum(temp[m] - hi, 0.0))),
                ])
            else:
                f.extend([0.0, 0.0, 0.0, 0.0])
        for k in range(4):
            m = sensor == k
            if np.any(m):
                f.extend([float(np.sum(dur[m]) / total), float(np.mean(temp[m]))])
            else:
                f.extend([0.0, 0.0])
        for k in range(5):
            m = hand == k
            f.append(float(np.sum(dur[m]) / total) if np.any(m) else 0.0)
        for cut in [0.25, 0.5, 0.75]:
            m = pos <= cut
            f.extend([
                float(np.sum(dur[m] * np.maximum(temp[m] - hi, 0.0))) if np.any(m) else 0.0,
                float(np.sum(dur[m] * np.maximum(lo - temp[m], 0.0))) if np.any(m) else 0.0,
                float(np.mean(temp[m])) if np.any(m) else 0.0,
            ])
            m = pos >= cut
            f.extend([
                float(np.sum(dur[m] * np.maximum(temp[m] - hi, 0.0))) if np.any(m) else 0.0,
                float(np.sum(dur[m] * np.maximum(lo - temp[m], 0.0))) if np.any(m) else 0.0,
                float(np.mean(temp[m])) if np.any(m) else 0.0,
            ])
        deb = temp - _SENSOR_BIAS_EST[sensor] - _PHASE_BIAS_EST[phase]
        for tau in [0.45, 0.9, 1.6, 2.8, 4.8, 7.0]:
            core = mid
            dmg = 0.0
            mx = -1e9
            mn = 1e9
            swing = 0.0
            prev = core
            early = 0.0
            late_safe = 0.0
            cores = []
            for j in range(len(dur)):
                resp = 1.0 - np.exp(-dur[j] / (tau * (1.0 + 0.25 * load[j])))
                core = core + resp * (deb[j] - core)
                over = max(0.0, core - hi)
                under = max(0.0, lo - core)
                val = dur[j] * ((over / 5.0) ** 1.4 + (under / 5.0) ** 1.3)
                dmg += val
                mx = max(mx, core)
                mn = min(mn, core)
                swing += max(0.0, abs(core - prev) - 2.0)
                if pos[j] < 0.35:
                    early += val
                if pos[j] > 0.5 and lo + 1.0 < core < hi - 1.0:
                    late_safe += dur[j]
                prev = core
                cores.append(core)
            cores = np.asarray(cores, dtype=float)
            f.extend([
                float(dmg), float(mx), float(mn), float(core), float(swing),
                float(early), float(late_safe), float(early * np.log1p(late_safe)),
                float(np.sum(w * np.maximum(cores - hi, 0.0))),
                float(np.sum(w * np.maximum(lo - cores, 0.0))),
            ])
        above = temp > hi
        below = temp < lo
        f.extend([_longest_duration_run(above, dur), _longest_duration_run(below, dur)])
        f.extend([
            float(np.sum(dur * (pos - 0.5) * (temp - mid))),
            float(np.sum(dur * (pos - 0.5) * np.abs(temp - mid))),
        ])
        handoff = hand > 0
        f.extend([
            float(np.sum(dur * handoff * np.maximum(temp - hi, 0.0))),
            float(np.sum(dur * handoff * np.maximum(lo - temp, 0.0))),
            float(np.sum(dur * (phase == 1) * np.abs(temp - mid))),
        ])
        tau_pkg = [0.55, 1.2, 2.2, 4.0][min(max(pkg_i, 0), 3)] * (1.0 + 0.25 * float(np.mean(load)))
        for debias_scale in [0.7, 1.0, 1.25]:
            deb2 = temp - debias_scale * _SENSOR_BIAS_EST[sensor] - debias_scale * _PHASE_BIAS_EST[phase]
            for tau_mul in [0.75, 1.0, 1.45]:
                tau = tau_pkg * tau_mul
                core = mid
                damage_like = 0.0
                swing_like = 0.0
                direct_like = 0.0
                hand_like = 0.0
                early_like = 0.0
                late_safe_like = 0.0
                late_shock_like = 0.0
                post_recovery_like = 0.0
                safe_clock_like = 0.0
                max_over = 0.0
                max_under = 0.0
                prev = core
                for j in range(len(dur)):
                    resp = 1.0 - np.exp(-dur[j] / (tau * (1.0 + 0.35 * load[j])))
                    if hand[j] in (1, 2) and pkg_i == 0:
                        resp = min(0.95, resp * 1.45)
                    core = core + resp * (deb2[j] - core)
                    over = max(0.0, core - hi)
                    under = max(0.0, lo - core)
                    hum_mult = 1.0 + 0.004 * max(0.0, hum[j] - 60.0) * (1.0 if core > hi - 1.0 else 0.0)
                    val = dur[j] * ((over / 5.0) ** 1.55 + 0.80 * (under / 4.5) ** 1.45) * hum_mult
                    damage_like += val
                    max_over = max(max_over, over)
                    max_under = max(max_under, under)
                    was_safe_like = (lo + 1.0 < prev < hi - 1.0)
                    jump_like = max(0.0, abs(core - prev) - 2.4)
                    swing_like += jump_like
                    if was_safe_like:
                        safe_clock_like += dur[j]
                    if (over > 0.5 or under > 0.5) and safe_clock_like > 1.5 and pos[j] > 0.35:
                        if pkg_i == 0:
                            shock_mult = 1.25
                        elif pkg_i == 1:
                            shock_mult = 0.55
                        elif pkg_i == 2:
                            shock_mult = 0.95
                        else:
                            shock_mult = 1.20
                        late_shock_like += val * np.log1p(safe_clock_like) * shock_mult
                        safe_clock_like = 0.0
                    raw_exc = max(0.0, abs(deb2[j] - mid) - max(7.0, (hi - lo) * 0.55)) / 9.0
                    if pkg_i == 0:
                        direct_mult = 1.0
                    elif pkg_i == 1:
                        direct_mult = 0.30
                    else:
                        direct_mult = 0.10
                    direct_like += (raw_exc ** 1.55) * dur[j] * direct_mult
                    hand_like += ((hand[j] == 4) * 0.18 + (hand[j] == 1) * 0.08 + (hand[j] == 2) * 0.10) * (1.0 + 0.5 * val)
                    if pos[j] < 0.35:
                        early_like += val
                    if pos[j] > 0.5 and lo + 1.0 < core < hi - 1.0:
                        late_safe_like += dur[j]
                    if pos[j] > 0.45 and val > 0.0:
                        post_recovery_like += val * pos[j]
                    prev = core
                f.extend([
                    float(damage_like), float(direct_like), float(swing_like), float(hand_like),
                    float(early_like), float(late_safe_like), float(early_like * np.log1p(late_safe_like)),
                    float(late_shock_like), float(post_recovery_like),
                    float(max_over), float(max_under), float(damage_like * sens),
                    float((damage_like + direct_like + swing_like) * sens),
                    float((damage_like + direct_like + swing_like + 0.35 * late_shock_like + 0.15 * post_recovery_like) * sens),
                ])
        base = np.asarray(f[:min(80, len(f))], dtype=float)
        f.extend((base * (pkg_i == 0)).tolist())
        f.extend((base * (pkg_i == 3)).tolist())
        f.extend((base * (prod_i == 2)).tolist())
        f.extend((base * sens).tolist())
        rows.append(np.asarray(f, dtype=float))
    F = np.vstack(rows) if rows else np.zeros((0, 640), dtype=float)
    F[~np.isfinite(F)] = 0.0
    return F


def fit_quality_model(train_step_X, train_step_code, train_route_X, train_route_offsets, train_y):
    X = _extract_features(train_step_X, train_step_code, train_route_X, train_route_offsets)
    y = np.asarray(train_y, dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError('one target is required for each route')
    mu = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Xz = (X - mu) / scale
    y_mean = float(y.mean())
    yc = y - y_mean
    alpha = 60.0
    A = Xz.T @ Xz
    A.flat[:: A.shape[0] + 1] += alpha
    b = Xz.T @ yc
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return {'mu': mu, 'scale': scale, 'beta': beta, 'y_mean': y_mean}


def predict_quality_score(step_X, step_code, route_X, route_offsets, params):
    X = _extract_features(step_X, step_code, route_X, route_offsets)
    mu = np.asarray(params['mu'], dtype=float)
    scale = np.asarray(params['scale'], dtype=float)
    beta = np.asarray(params['beta'], dtype=float)
    y_mean = float(params['y_mean'])
    pred = y_mean + ((X - mu) / scale) @ beta
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred[~np.isfinite(pred)] = y_mean
    return pred

hidden
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np

SLICE_NAMES = [
    'normal_routes',
    'short_spike_routes',
    'sustained_moderate_routes',
    'order_early_routes',
    'order_late_routes',
    'rare_package_routes',
    'sensor_phase_shift_routes',
    'long_recovery_routes',
]

# overall, then the eight hidden slices above
GOOD_RMSE = np.array([20.0, 4.4, 5.5, 27.0, 22.0, 16.5, 25.0, 7.2, 40.0], dtype=float)
BAD_RMSE = np.array([60.0, 18.0, 20.0, 90.0, 80.0, 70.0, 70.0, 16.0, 130.0], dtype=float)

HIDDEN_N_ROUTES = 1200
HIDDEN_SEED = 10413



def _load_module(path):
    spec = importlib.util.spec_from_file_location('candidate_solve', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rmse(y, pred):
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _score_rmse(value, good, bad):
    if not math.isfinite(value):
        return 0.0
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(((bad - value) / (bad - good)) ** 1.5)


def _compute_reward(y, pred, slice_id):
    overall_rmse = _rmse(y, pred)
    overall_reward = _score_rmse(overall_rmse, GOOD_RMSE[0], BAD_RMSE[0])
    slice_metrics = {}
    slice_rewards = []
    for k, name in enumerate(SLICE_NAMES):
        mask = np.asarray(slice_id) == k
        if not np.any(mask):
            r = 0.0
            sr = 0.0
            n = 0
        else:
            r = _rmse(y[mask], pred[mask])
            sr = _score_rmse(r, GOOD_RMSE[k + 1], BAD_RMSE[k + 1])
            n = int(np.sum(mask))
        slice_rewards.append(sr)
        slice_metrics[name] = {'rmse': r, 'reward': sr, 'n': n}
    slice_rewards = np.asarray(slice_rewards, dtype=float)
    slice_geom = float(np.exp(np.mean(np.log(np.clip(slice_rewards, 1e-4, 1.0)))))
    reward = float(0.40 * overall_reward + 0.60 * slice_geom)
    reward = max(0.0, min(1.0, reward))
    return reward, {
        'overall_rmse': overall_rmse,
        'overall_reward': overall_reward,
        'slice_geomean_reward': slice_geom,
        'slice_metrics': slice_metrics,
    }


def evaluate(solve_path=None, app_dir=None, output_dir=None):
    here = Path(__file__).resolve().parent
    app = Path(app_dir or os.environ.get('APP_DIR', '/app'))
    path = Path(solve_path or os.environ.get('SOLVE_PATH', app / 'solve.py'))
    out = Path(output_dir or os.environ.get('OUTPUT_DIR', '/logs/verifier'))
    out.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        metrics = {'failure': f'solve.py not found at {path}'}
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (out / 'reward.txt').write_text('0.0\n')
        return 0.0, metrics

    try:
        solve = _load_module(path)
        if not hasattr(solve, 'fit_quality_model') or not hasattr(solve, 'predict_quality_score'):
            raise AttributeError('fit_quality_model and predict_quality_score are required')

        train = np.load(app / 'train_data.npz')
        gen = _load_module(here / 'generator.py')
        hidden = gen.generate_routes(HIDDEN_N_ROUTES, seed=HIDDEN_SEED, split='hidden')

        train_args = [
            np.array(train['train_step_X'], copy=True),
            np.array(train['train_step_code'], copy=True),
            np.array(train['train_route_X'], copy=True),
            np.array(train['train_route_offsets'], copy=True),
            np.array(train['train_y'], copy=True),
        ]
        train_before = [a.copy() for a in train_args]
        params = solve.fit_quality_model(*train_args)
        for got, before in zip(train_args, train_before):
            if not np.array_equal(got, before):
                raise ValueError('fit_quality_model mutated an input array')

        pred_args = [
            np.array(hidden['step_X'], copy=True),
            np.array(hidden['step_code'], copy=True),
            np.array(hidden['route_X'], copy=True),
            np.array(hidden['route_offsets'], copy=True),
        ]
        pred_before = [a.copy() for a in pred_args]
        pred1 = np.asarray(solve.predict_quality_score(*pred_args, params), dtype=float).reshape(-1)
        pred2 = np.asarray(solve.predict_quality_score(*pred_args, params), dtype=float).reshape(-1)

        n_routes = len(hidden['route_offsets']) - 1
        if pred1.shape != (n_routes,):
            raise ValueError(f'prediction shape {pred1.shape} does not match ({n_routes},)')
        if not np.all(np.isfinite(pred1)):
            raise ValueError('predictions must be finite')
        if not np.allclose(pred1, pred2, rtol=1e-9, atol=1e-9):
            raise ValueError('predictions are not deterministic')
        for got, before in zip(pred_args, pred_before):
            if not np.array_equal(got, before):
                raise ValueError('predict_quality_score mutated an input array')

        y = np.asarray(hidden['y'], dtype=float).reshape(-1)
        slice_id = np.asarray(hidden['slice_id'], dtype=int).reshape(-1)
        reward, metrics = _compute_reward(y, pred1, slice_id)
        metrics['prediction_min'] = float(np.min(pred1))
        metrics['prediction_max'] = float(np.max(pred1))
        metrics['prediction_mean'] = float(np.mean(pred1))
    except Exception as exc:
        reward = 0.0
        metrics = {'failure': type(exc).__name__ + ': ' + str(exc)}

    (out / 'reward.txt').write_text(f'{reward:.12f}\n')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return reward, metrics


if __name__ == '__main__':
    reward, metrics = evaluate()
    print(json.dumps({'reward': reward, **metrics}, indent=2, sort_keys=True))


generator
import numpy as np
from dataclasses import dataclass

PHASES=5; HANDLINGS=5; SENSORS=4
# route_X cols: package_class, product_family, declared_sensitivity, carrier_type, planned_duration, route_risk

def rng_choice(rng, probs):
    return rng.choice(len(probs), p=np.array(probs)/np.sum(probs))

def generate_routes(n, seed=0, split='train'):
    rng=np.random.default_rng(seed)
    step_Xs=[]; step_codes=[]; offsets=[0]; route_X=[]; ys=[]; slices=[]
    # slice probs per split
    slice_names=['normal','short_spike','sustained_moderate','order_early','order_late','rare_package','sensor_phase_shift','long_recovery']
    probs=np.array([0.20,0.15,0.16,0.14,0.14,0.10,0.05,0.06])
    if split=='hidden':
        probs=np.array([0.15,0.15,0.14,0.16,0.16,0.12,0.06,0.06])
    probs=probs/probs.sum()
    safe_mid={0:20.0,1:5.0,2:-6.0}
    safe_lo={0:8.0,1:-2.0,2:-18.0}
    safe_hi={0:31.0,1:10.0,2:-1.5}
    # package response tau: thin responds quickly (less protection), heavy slow but long recovery
    tau_base=np.array([0.55,1.2,2.2,4.0])
    # product family distribution: hidden more rare frozen ambient? 
    for i in range(n):
        sl=rng.choice(len(slice_names), p=probs)
        # rare slice force rare combos often
        if sl==5:
            pkg=rng.choice([2,3], p=[0.35,0.65])
            prod=rng.choice([1,2], p=[0.35,0.65])
        else:
            pkg=rng.choice(4, p=[0.35,0.30,0.25,0.10])
            prod=rng.choice(3, p=[0.45,0.35,0.20])
        sens=(0.65+0.35*prod+0.10*pkg+rng.normal(0,0.08))
        sens=np.clip(sens,0.5,1.7)
        carrier=rng.choice(3,p=[0.45,0.35,0.20])
        risk=rng.beta(2,5)
        L=int(rng.integers(8,26))
        if sl in [2,7]: L=int(rng.integers(16,36))
        if sl in [3,4,5]: L=int(rng.integers(12,32))
        # phase sequence, mostly monotone route phases with repeats
        # boundaries roughly warehouse/dock/truck/air/local
        phases=[]
        for j in range(L):
            t=j/(L-1)
            if t<0.12: ph=0
            elif t<0.25: ph=1
            elif t<0.72: ph=2 if carrier!=2 else rng.choice([2,3],p=[0.4,0.6])
            elif t<0.85: ph=1
            else: ph=4
            if rng.random()<0.08: ph=int(rng.integers(0,5))
            phases.append(ph)
        phases=np.array(phases)
        # durations
        durations=rng.gamma(1.15,1.2,size=L)+0.10
        if rng.random()<0.35:
            # nuisance segmentation: a few very short scans mixed with coarser records
            m = rng.random(L) < rng.uniform(0.10,0.22)
            durations[m] *= rng.uniform(0.20,0.55)
            durations[~m] *= rng.uniform(0.95,1.25)
        if sl==1: durations*=rng.uniform(0.45,0.9)
        if sl in [2,7]: durations*=rng.uniform(1.0,1.75)
        # normalize planned duration plus noise
        planned=float(durations.sum()*rng.uniform(0.9,1.12))
        mid=safe_mid[prod]; lo=safe_lo[prod]; hi=safe_hi[prod]
        # true temp profile near safe mid plus ambient by phase and route risk
        true_temp=mid + rng.normal(0,1.4,size=L)
        # phase ambient biases: dock/truck/air/local vary
        phase_effect=np.array([0.0, 3.0, 4.5, -1.5, 2.0])
        true_temp += phase_effect[phases]*(0.2+0.8*risk)
        # handling
        handling=np.zeros(L,dtype=int)
        for j,ph in enumerate(phases):
            pp=[0.72,0.10,0.08,0.06,0.04]
            if ph==1: pp=[0.42,0.25,0.22,0.07,0.04]
            if rng.random()<0.14+risk*0.18: handling[j]=rng.choice(5,p=pp)
            else: handling[j]=0
        # route type patterns
        if sl==1: # short spike: 1-2 short high temp spikes or low for frozen
            k=int(rng.integers(1,3)); idx=rng.choice(L,size=k,replace=False)
            for j in idx:
                sign=1 if prod!=2 or rng.random()<0.65 else -1
                amp=rng.uniform(8,17)*(1 if sign>0 else -1)
                true_temp[j]+=amp
                durations[j]*=rng.uniform(0.12,0.45)
                handling[j]=rng.choice([1,2,3])
        elif sl==2: # sustained moderate warm/cold exposure
            start=int(rng.integers(2,max(3,L//2)))
            end=min(L,start+int(rng.integers(max(4,L//5), max(5,L//2))))
            sign=1 if prod!=2 or rng.random()<0.72 else -1
            true_temp[start:end]+=rng.uniform(4.0,8.0)*(1 if sign>0 else -1)
            durations[start:end]*=rng.uniform(1.1,2.0)
            handling[start:end]=np.maximum(handling[start:end],3)
        elif sl in [3,4]: # order pair: same extrema but early vs late affects due to accumulation/recovery
            # create three blocks: high stress block, safe recovery block, mild stress block
            high=rng.uniform(8,15) * (1 if prod!=2 or rng.random()<0.7 else -1)
            mild=rng.uniform(3.5,7.0) * (1 if high>0 else -1)
            block=max(2,L//7)
            if sl==3:
                hi_start=1; mild_start=L//2
            else:
                mild_start=1; hi_start=L//2
            true_temp[hi_start:hi_start+block]+=high
            durations[hi_start:hi_start+block]*=rng.uniform(0.45,1.05)
            true_temp[mild_start:mild_start+block*2]+=mild
            durations[mild_start:mild_start+block*2]*=rng.uniform(1.35,2.15)
            handling[hi_start:hi_start+block]=np.maximum(handling[hi_start:hi_start+block],1)
        elif sl==5: # rare package/product, dynamics are subtle; moderate exposure + package lag
            st=int(rng.integers(1,L//2))
            en=min(L,st+int(rng.integers(4, max(5,L//2))))
            true_temp[st:en]+=rng.uniform(3,9)*(1 if prod!=2 or rng.random()<0.55 else -1)
            durations[en//2:en]*=rng.uniform(1.4,2.2)
        elif sl==6: # sensor phase shift: reported readings biased by phase/sensor, true profile milder/has phase-specific offset
            true_temp += np.where(phases==1, rng.normal(1.5,0.4,size=L), 0)
            true_temp += np.where(phases==2, rng.normal(-1.0,0.3,size=L), 0)
        elif sl==7: # early damage then long recovery/stabilization
            b=max(2,L//5)
            true_temp[:b]+=rng.uniform(7,14)*(1 if prod!=2 or rng.random()<0.65 else -1)
            durations[:b]*=rng.uniform(0.8,1.5)
            true_temp[b:]+=rng.normal(0,0.6,size=L-b) - (true_temp[:b].mean()-mid)*0.15
            durations[b:]*=rng.uniform(1.3,2.2)
        # sensors: codes and reported bias/lag/noise
        sensor=np.zeros(L,dtype=int)
        for j,ph in enumerate(phases):
            if ph==1: probs_s=[0.15,0.25,0.35,0.25]
            elif ph==2: probs_s=[0.25,0.55,0.15,0.05]
            else: probs_s=[0.35,0.35,0.15,0.15]
            sensor[j]=rng.choice(4,p=probs_s)
        # Sensor bias by sensor and phase. In hidden sensor_shift stronger but visible train includes some.
        sensor_bias=np.array([0.0,-0.7,1.4,2.2])
        phase_sensor_bias=(phases==1)*1.2*(sensor==2) + (phases==2)*(-0.8)*(sensor==1) + (phases==4)*0.9*(sensor==3)
        if sl==6: phase_sensor_bias += (phases==1)*2.0 - (phases==2)*1.1
        reported_temp=true_temp + sensor_bias[sensor] + phase_sensor_bias + rng.normal(0,0.55+0.2*sensor,size=L)
        humidity=np.clip(45 + 1.7*(true_temp-mid) + rng.normal(0,8,size=L) + 8*(phases==1), 15, 95)
        load=np.clip(rng.normal(0.6,0.18,size=L) + 0.12*(pkg==3), 0.1, 1.0)
        delay=(handling==3).astype(float)
        vibration=np.clip(rng.normal(0.25,0.13,size=L)+0.25*(handling==4)+0.12*(phases==1),0,1.5)
        scan_gap=np.clip(durations*rng.uniform(0.7,1.4,size=L)+rng.normal(0,0.2,size=L),0.05,None)
        # Damage with true temp and order/dynamics
        tau=tau_base[pkg]*(1+0.35*load.mean())
        core=mid+rng.normal(0,0.8)
        damage=0.0; prev_core=core; swing=0.0; early_damage=0.0; late_safe=0.0; late_shock=0.0; post_recovery_stress=0.0; safe_clock=0.0
        for j in range(L):
            # thin gets big handling exposure; heavy recovers slowly but protects short spike
            resp=1-np.exp(-durations[j]/(tau*(1+0.4*load[j])))
            if handling[j] in [1,2] and pkg==0:
                resp=min(0.95,resp*1.6)
            core=core + resp*(true_temp[j]-core)
            # humidity makes warm exposure worse for ambient/chilled
            hum_mult=1+0.006*max(0,humidity[j]-60)*(1 if core>hi-1 else 0)
            over=max(0,core-hi); under=max(0,lo-core)
            stress=(over/5.0)**1.55 + 0.85*(under/4.5)**1.45
            # short raw spike can still matter for thin/door/handoff
            direct=max(0,abs(true_temp[j]-mid)-max(7, (hi-lo)*0.55))/9.0
            direct=(direct**1.6)*durations[j]*(1.0 if pkg==0 else 0.35 if pkg==1 else 0.12)
            # rough handling damage amplified if thermal stress present
            hand=0.18*(handling[j]==4)+0.09*(handling[j]==1)+0.12*(handling[j]==2)
            hand *= (1+0.8*stress+0.3*delay[j])
            # cycle penalty from swings around safe band
            jump = max(0, abs(core-prev_core)-2.6)
            swing += jump * (0.04+0.02*sens)
            was_safe = (lo + 1.0 < prev_core < hi - 1.0)
            now_stress = (core > hi + 0.5) or (core < lo - 0.5)
            if was_safe:
                safe_clock += durations[j]
            if now_stress and safe_clock > 1.5 and j > L*0.35:
                # late stress after a settled period is handled differently by package class.
                late_shock += step_damage * np.log1p(safe_clock) * (1.35 if pkg==0 else 0.55 if pkg==1 else 0.95 if pkg==2 else 1.25)
                safe_clock = 0.0
            step_damage=sens*(durations[j]*stress*hum_mult*(1+0.12*(phase_effect[phases[j]]>0)) + direct) + hand
            # heavy/recovery: early damage sticks more if long warm recovery after it
            if j < L*0.35: early_damage += step_damage
            if (lo+1 < core < hi-1) and j > L*0.45: late_safe += durations[j]
            if j > L*0.45 and stress > 0.05:
                post_recovery_stress += step_damage * (pos[j] if 'pos' in locals() else j/max(1,L-1))
            damage += step_damage
            prev_core=core
        # non-reversible, but long safe recovery can stabilize a little, not erase
        recovery_credit=0.08*np.log1p(late_safe)*(1.3 if pkg in [1,2] else 0.7)
        damage = damage + 0.28*swing + 0.20*early_damage*np.log1p(late_safe)/(1+0.25*(pkg==2)) + 0.10*late_shock + 0.06*post_recovery_stress - recovery_credit
        damage += 0.4*risk + 0.12*max(0, planned - durations.sum())
        y=2.0 + 8.7*damage + rng.normal(0,1.15+0.55*risk)
        # step_X includes no true temp, only reported
        relpos=np.linspace(0,1,L)
        step_X=np.column_stack([durations, reported_temp, humidity, load, delay, vibration, scan_gap, relpos])
        code=np.column_stack([phases, handling, sensor])
        step_Xs.append(step_X.astype(np.float32)); step_codes.append(code.astype(np.int16)); route_X.append([pkg,prod,sens,carrier,planned,risk])
        ys.append(y); slices.append(sl); offsets.append(offsets[-1]+L)
    return {
        'step_X':np.vstack(step_Xs).astype(np.float32),
        'step_code':np.vstack(step_codes).astype(np.int16),
        'route_X':np.array(route_X,dtype=np.float32),
        'route_offsets':np.array(offsets,dtype=np.int64),
        'y':np.array(ys,dtype=np.float32),
        'slice_id':np.array(slices,dtype=np.int16),
        'slice_names':np.array(slice_names)
    }

# feature extraction

def onehot_int(x, k):
    arr=np.zeros(k,dtype=float); 
    if 0<=int(x)<k: arr[int(x)]=1.0
    return arr

SENSOR_BIAS_EST=np.array([0.0,-0.7,1.2,1.8])
PHASE_BIAS_EST=np.array([0.0,0.6,-0.5,0.0,0.3])

def safe_bounds(prod):
    if int(prod)==0: return 8.0,31.0,20.0
    if int(prod)==1: return -2.0,10.0,5.0
    return -18.0,-1.5,-6.0

def extract_features(data, mode='reference'):
    SX=data['step_X']; SC=data['step_code']; RX=data['route_X']; off=data['route_offsets']; n=len(off)-1
    feats=[]
    for i in range(n):
        a,b=off[i],off[i+1]
        X=SX[a:b]; C=SC[a:b]; r=RX[i]
        dur=X[:,0].astype(float); temp=X[:,1].astype(float); hum=X[:,2].astype(float); load=X[:,3].astype(float); delay=X[:,4].astype(float); vib=X[:,5].astype(float); pos=X[:,7].astype(float)
        phase=C[:,0].astype(int); hand=C[:,1].astype(int); sensor=C[:,2].astype(int)
        total=dur.sum(); w=dur/(total+1e-9)
        pkg,prod,sens,carrier,planned,risk=r
        lo,hi,mid=safe_bounds(prod)
        f=[]
        # route meta: raw and onehots
        f.extend([total, planned, planned-total, sens, risk, len(dur), dur.mean(), dur.max()])
        f.extend(onehot_int(pkg,4)); f.extend(onehot_int(prod,3)); f.extend(onehot_int(carrier,3))
        # generic aggregates
        for arr in [temp, hum, load, delay, vib]:
            f.extend([np.mean(arr), np.sum(w*arr), np.std(arr), np.min(arr), np.max(arr)])
            f.extend(np.quantile(arr,[0.1,0.25,0.5,0.75,0.9]).tolist())
        f.extend([np.sum(dur*(temp-mid)), np.sum(dur*np.abs(temp-mid)), np.max(np.abs(temp-mid)), temp[0], temp[-1], np.mean(temp[:max(1,len(temp)//3)]), np.mean(temp[-max(1,len(temp)//3):])])
        # above threshold raw exposures
        for th_shift in [-4,-2,0,2,4,7,10]:
            th=hi+th_shift
            f.append(np.sum(dur*np.maximum(temp-th,0)))
            f.append(np.max(np.maximum(temp-th,0)))
        for th_shift in [-10,-7,-4,-2,0,2,4]:
            th=lo+th_shift
            f.append(np.sum(dur*np.maximum(th-temp,0)))
            f.append(np.max(np.maximum(th-temp,0)))
        # phase/sensor/handling stats (mostly aggregate; in rich mode enough)
        for k in range(5):
            m=phase==k
            f.extend([dur[m].sum()/total if m.any() else 0, temp[m].mean() if m.any() else 0, temp[m].max() if m.any() else 0, np.sum(dur[m]*np.maximum(temp[m]-hi,0)) if m.any() else 0])
        for k in range(4):
            m=sensor==k
            f.extend([dur[m].sum()/total if m.any() else 0, temp[m].mean() if m.any() else 0])
        for k in range(5):
            m=hand==k
            f.append(dur[m].sum()/total if m.any() else 0)
        # First/last part aggregates included in rich maybe
        for cut in [0.25,0.5,0.75]:
            m=pos<=cut; f.extend([np.sum(dur[m]*np.maximum(temp[m]-hi,0)), np.sum(dur[m]*np.maximum(lo-temp[m],0)), temp[m].mean() if m.any() else 0])
            m=pos>=cut; f.extend([np.sum(dur[m]*np.maximum(temp[m]-hi,0)), np.sum(dur[m]*np.maximum(lo-temp[m],0)), temp[m].mean() if m.any() else 0])
        if mode in ['sequence','reference']:
            # de-biased temp (rough)
            deb=temp-SENSOR_BIAS_EST[np.clip(sensor,0,3)]-PHASE_BIAS_EST[np.clip(phase,0,4)]
            # dynamic filters
            for tau in ([0.45,0.9,1.6,2.8,4.8,7.0] if mode=='reference' else [0.9,2.0,4.0]):
                core=mid; dmg=0.0; mx=-1e9; mn=1e9; swing=0.0; prev=core; early=0.0; late_safe=0.0
                cores=[]
                for j in range(len(dur)):
                    resp=1-np.exp(-dur[j]/(tau*(1+0.25*load[j])))
                    core=core+resp*(deb[j]-core)
                    over=max(0,core-hi); under=max(0,lo-core)
                    val=dur[j]*((over/5)**1.4+(under/5)**1.3)
                    dmg+=val; mx=max(mx,core); mn=min(mn,core); swing+=max(0,abs(core-prev)-2.0)
                    if pos[j]<0.35: early+=val
                    if pos[j]>0.5 and lo+1<core<hi-1: late_safe+=dur[j]
                    prev=core; cores.append(core)
                cores=np.array(cores)
                f.extend([dmg,mx,mn,core,swing,early,late_safe,early*np.log1p(late_safe),np.sum(w*np.maximum(cores-hi,0)),np.sum(w*np.maximum(lo-cores,0))])
            # order-sensitive run features
            above=temp>hi; below=temp<lo
            def longest_run(mask):
                best=cur=0.0
                for m,d in zip(mask,dur):
                    if m: cur+=d; best=max(best,cur)
                    else: cur=0.0
                return best
            f.extend([longest_run(above),longest_run(below)])
            # signed slope and hot-after-handoff / stress after safe recovery
            f.extend([np.sum(dur*(pos-0.5)*(temp-mid)), np.sum(dur*(pos-0.5)*np.abs(temp-mid))])
            handoff=hand>0
            f.extend([np.sum(dur*handoff*np.maximum(temp-hi,0)), np.sum(dur*handoff*np.maximum(lo-temp,0)), np.sum(dur*(phase==1)*np.abs(temp-mid))])

        if mode=='reference':
                # package-aware damage summaries (closer to the mechanism but still fitted from data)
                tau_pkg = [0.55, 1.2, 2.2, 4.0][int(pkg)] * (1 + 0.25*np.mean(load))
                for debias_scale in [0.7, 1.0, 1.25]:
                    deb2 = temp - debias_scale*SENSOR_BIAS_EST[np.clip(sensor,0,3)] - debias_scale*PHASE_BIAS_EST[np.clip(phase,0,4)]
                    for tau_mul in [0.75, 1.0, 1.45]:
                        tau = tau_pkg * tau_mul
                        core = mid
                        damage_like = 0.0; swing_like=0.0; direct_like=0.0; hand_like=0.0; early_like=0.0; late_safe_like=0.0; late_shock_like=0.0; post_recovery_like=0.0; safe_clock_like=0.0; max_over=0.0; max_under=0.0
                        prev=core
                        for j in range(len(dur)):
                            resp = 1 - np.exp(-dur[j] / (tau*(1+0.35*load[j])))
                            if hand[j] in [1,2] and int(pkg)==0:
                                resp = min(0.95, resp*1.45)
                            core = core + resp*(deb2[j]-core)
                            over = max(0.0, core-hi); under=max(0.0, lo-core)
                            hum_mult = 1 + 0.004*max(0.0, hum[j]-60)*(1.0 if core>hi-1 else 0.0)
                            val = dur[j]*((over/5.0)**1.55 + 0.80*(under/4.5)**1.45)*hum_mult
                            damage_like += val
                            max_over=max(max_over,over); max_under=max(max_under,under)
                            jump=max(0.0, abs(core-prev)-2.4)
                            was_safe_like = (lo + 1.0 < prev < hi - 1.0)
                            swing_like += jump
                            if was_safe_like:
                                safe_clock_like += dur[j]
                            if (over > 0.5 or under > 0.5) and safe_clock_like > 1.5 and pos[j] > 0.35:
                                shock_mult = 1.25 if int(pkg)==0 else 0.55 if int(pkg)==1 else 0.95 if int(pkg)==2 else 1.20
                                late_shock_like += val * np.log1p(safe_clock_like) * shock_mult
                                safe_clock_like = 0.0
                            raw_exc=max(0.0, abs(deb2[j]-mid)-max(7.0, (hi-lo)*0.55))/9.0
                            direct_like += (raw_exc**1.55)*dur[j]*(1.0 if int(pkg)==0 else 0.30 if int(pkg)==1 else 0.10)
                            hand_like += ((hand[j]==4)*0.18 + (hand[j]==1)*0.08 + (hand[j]==2)*0.10)*(1+0.5*val)
                            if pos[j] < 0.35: early_like += val
                            if pos[j] > 0.5 and (lo+1 < core < hi-1): late_safe_like += dur[j]
                            if pos[j] > 0.45 and val > 0.0:
                                post_recovery_like += val * pos[j]
                            prev=core
                        f.extend([damage_like, direct_like, swing_like, hand_like, early_like, late_safe_like, early_like*np.log1p(late_safe_like), late_shock_like, post_recovery_like, max_over, max_under, damage_like*sens, (damage_like+direct_like+swing_like)*sens, (damage_like+direct_like+swing_like+0.35*late_shock_like+0.15*post_recovery_like)*sens])

        if mode=='reference':
            # explicit package/product interactions (multiply last chunk maybe too many)
            base=np.array(f[:min(80,len(f))])
            # interactions with package/product/sens
            f.extend((base*(pkg==0)).tolist()); f.extend((base*(pkg==3)).tolist()); f.extend((base*(prod==2)).tolist()); f.extend((base*sens).tolist())
        feats.append(np.asarray(f,dtype=float))
    # pad to max same by mode
    return np.vstack(feats)

if __name__=='__main__':
    train=generate_routes(3000,1,'train'); hidden=generate_routes(1600,2,'hidden')
    print(train['step_X'].shape, train['route_X'].shape, train['y'].mean(), train['y'].std(), hidden['y'].mean(), hidden['y'].std())
    for mode in ['rich','sequence','reference']:
        F=extract_features(train,mode)
        print(mode,F.shape, np.isfinite(F).all())
