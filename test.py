import importlib.util
import json
import math
import os
import sys
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
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_candidate_in_process(path, train_payload, inputs_payload):
    """Local fallback for authors who run this file outside the benchmark sandbox.

    The verifier path uses sandbox_util.run_eval when that helper is available.
    This fallback is kept so calibration can still be reproduced in a plain local
    checkout where sandbox_util is not installed.
    """
    solve = _load_module(path)
    if not hasattr(solve, 'fit_quality_model') or not hasattr(solve, 'predict_quality_score'):
        raise AttributeError('fit_quality_model and predict_quality_score are required')

    train_args = [np.array(a, copy=True) for a in train_payload]
    train_before = [a.copy() for a in train_args]
    params = solve.fit_quality_model(*train_args)
    for got, before in zip(train_args, train_before):
        if not np.array_equal(got, before):
            raise ValueError('fit_quality_model mutated an input array')

    preds = []
    for item in inputs_payload:
        pred_args = [np.array(a, copy=True) for a in item]
        pred_before = [a.copy() for a in pred_args]
        pred = solve.predict_quality_score(*pred_args, params)
        for got, before in zip(pred_args, pred_before):
            if not np.array_equal(got, before):
                raise ValueError('predict_quality_score mutated an input array')
        preds.append(np.asarray(pred, dtype=float))
    return {'preds': preds}


def _run_candidate(path, payload):
    """Run candidate solution through the benchmark sandbox when available."""
    try:
        import sandbox_util  # provided by the verifier runtime
    except ImportError:
        return _run_candidate_in_process(path, payload['train'], payload['inputs'])
    return sandbox_util.run_eval(str(path), payload)


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
        train = np.load(app / 'train_data.npz')
        gen = _load_module(here / 'generator.py')
        hidden = gen.generate_routes(HIDDEN_N_ROUTES, seed=HIDDEN_SEED, split='hidden')

        train_payload = (
            np.array(train['train_step_X'], copy=True),
            np.array(train['train_step_code'], copy=True),
            np.array(train['train_route_X'], copy=True),
            np.array(train['train_route_offsets'], copy=True),
            np.array(train['train_y'], copy=True),
        )
        hidden_inputs = (
            np.array(hidden['step_X'], copy=True),
            np.array(hidden['step_code'], copy=True),
            np.array(hidden['route_X'], copy=True),
            np.array(hidden['route_offsets'], copy=True),
        )
        payload = {
            'train': train_payload,
            # The duplicate input asks the sandboxed runner to predict the same
            # hidden set twice, so hidden_eval can still enforce determinism
            # without importing the candidate module in-process.
            'inputs': [hidden_inputs, hidden_inputs],
        }
        result = _run_candidate(path, payload)
        if not isinstance(result, dict) or 'preds' not in result:
            raise ValueError('sandbox runner must return a dict with key "preds"')
        preds = result['preds']
        if len(preds) != 2:
            raise ValueError('sandbox runner returned the wrong number of prediction arrays')
        pred1 = np.asarray(preds[0], dtype=float).reshape(-1)
        pred2 = np.asarray(preds[1], dtype=float).reshape(-1)

        n_routes = len(hidden['route_offsets']) - 1
        if pred1.shape != (n_routes,):
            raise ValueError(f'prediction shape {pred1.shape} does not match ({n_routes},)')
        if pred2.shape != (n_routes,):
            raise ValueError(f'second prediction shape {pred2.shape} does not match ({n_routes},)')
        if not np.all(np.isfinite(pred1)) or not np.all(np.isfinite(pred2)):
            raise ValueError('predictions must be finite')
        if not np.allclose(pred1, pred2, rtol=1e-9, atol=1e-9):
            raise ValueError('predictions are not deterministic')

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
    solution = sys.argv[1] if len(sys.argv) > 1 else None
    app_dir = sys.argv[2] if len(sys.argv) > 2 else None
    reward, metrics = evaluate(solve_path=solution, app_dir=app_dir)
    print(json.dumps({'reward': reward, **metrics}, indent=2, sort_keys=True))

