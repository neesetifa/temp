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

Dependency note: grading runs in the base environment only. Do not rely on packages installed during exploration or in a local/public run. Your solve.py should import only the Python standard library plus numpy, scipy, and scikit-learn; packages such as pandas are not guaranteed to be available during grading.


agent call
  #!/usr/bin/env python3
"""Sandbox-side agent runner for invoice_line_reconciliation.

The root verifier writes plain .npz inputs, this unprivileged subprocess loads the
candidate once, runs fit + predict in-process, and writes only plain .npz outputs.
No params or pickled objects cross the sandbox boundary.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

REQUIRED = ("fit_invoice_model", "predict_net_line_cost")


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


def _dependency_detail(exc: BaseException) -> str:
    name = getattr(exc, "name", None)
    if name:
        return f"{name}: {exc}"
    return str(exc)


def check(path: str | os.PathLike) -> bool:
    try:
        mod = _load(path)
        return all(callable(getattr(mod, name, None)) for name in REQUIRED)
    except Exception:
        return False


def _arrays_equal(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _save_error(out_path: str | os.PathLike, code: str, detail: str = "") -> None:
    np.savez(
        out_path,
        n=np.array(0, dtype=np.int64),
        error=np.array(str(code)),
        detail=np.array(str(detail)[:1200]),
    )


def run_eval(solve_path: str | os.PathLike, in_path: str | os.PathLike, out_path: str | os.PathLike) -> int:
    try:
        with np.load(in_path, allow_pickle=False) as z:
            train_line_X = np.asarray(z["train_line_X"], dtype=np.float64)
            train_line_code = np.asarray(z["train_line_code"], dtype=np.int64)
            train_invoice_X = np.asarray(z["train_invoice_X"], dtype=np.float64)
            train_invoice_code = np.asarray(z["train_invoice_code"], dtype=np.int64)
            train_invoice_offsets = np.asarray(z["train_invoice_offsets"], dtype=np.int64)
            train_y = np.asarray(z["train_y"], dtype=np.float64)
            line_X = np.asarray(z["line_X"], dtype=np.float64)
            line_code = np.asarray(z["line_code"], dtype=np.int64)
            invoice_X = np.asarray(z["invoice_X"], dtype=np.float64)
            invoice_code = np.asarray(z["invoice_code"], dtype=np.int64)
            invoice_offsets = np.asarray(z["invoice_offsets"], dtype=np.int64)
        try:
            mod = _load(solve_path)
        except (ModuleNotFoundError, ImportError) as exc:
            _save_error(out_path, "dependency_error", _dependency_detail(exc))
            return 0
        except Exception:
            _save_error(out_path, "runtime_error", traceback.format_exc())
            return 0

        if not all(callable(getattr(mod, name, None)) for name in REQUIRED):
            _save_error(out_path, "api_missing", "required fit_invoice_model/predict_net_line_cost not found")
            return 0

        fit_args = [
            train_line_X.copy(),
            train_line_code.copy(),
            train_invoice_X.copy(),
            train_invoice_code.copy(),
            train_invoice_offsets.copy(),
            train_y.copy(),
        ]
        fit_before = [x.copy() for x in fit_args]
        params = mod.fit_invoice_model(*fit_args)
        if not all(_arrays_equal(a, b) for a, b in zip(fit_args, fit_before)):
            _save_error(out_path, "input_mutation_in_fit", "fit_invoice_model modified one or more input arrays")
            return 0

        pred_args = [
            line_X.copy(),
            line_code.copy(),
            invoice_X.copy(),
            invoice_code.copy(),
            invoice_offsets.copy(),
        ]
        pred_before = [x.copy() for x in pred_args]
        pred1 = np.asarray(mod.predict_net_line_cost(*pred_args, params), dtype=np.float64)
        if not all(_arrays_equal(a, b) for a, b in zip(pred_args, pred_before)):
            _save_error(out_path, "input_mutation_in_predict", "predict_net_line_cost modified one or more input arrays")
            return 0

        pred2_args = [line_X.copy(), line_code.copy(), invoice_X.copy(), invoice_code.copy(), invoice_offsets.copy()]
        pred2 = np.asarray(mod.predict_net_line_cost(*pred2_args, params), dtype=np.float64)
        if pred1.shape != pred2.shape or not np.allclose(pred1, pred2, rtol=0, atol=1e-10, equal_nan=True):
            _save_error(out_path, "nondeterministic_prediction", "two predict calls with the same params differed")
            return 0
        np.savez(out_path, n=np.array(1, dtype=np.int64), p0=pred1.astype(np.float64))
        return 0
    except Exception:  # pragma: no cover - verifier diagnostic path
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

hidden eval
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

try:
    from .generator import make_hidden, make_train_public
    from .sandbox_utils import run_eval
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from generator import make_hidden, make_train_public
    from sandbox_utils import run_eval


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
SLICE_BAD["discount"] = 0.95
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


def _payload(train, hidden):
    return {
        "train_line_X": train.line_X,
        "train_line_code": train.line_code,
        "train_invoice_X": train.invoice_X,
        "train_invoice_code": train.invoice_code,
        "train_invoice_offsets": train.invoice_offsets,
        "train_y": train.y,
        "line_X": hidden.line_X,
        "line_code": hidden.line_code,
        "invoice_X": hidden.invoice_X,
        "invoice_code": hidden.invoice_code,
        "invoice_offsets": hidden.invoice_offsets,
    }


def evaluate(solve_path: str | Path, write_dir: str | Path | None = None):
    solve_path = Path(solve_path)
    train, _ = make_train_public()
    hidden = make_hidden()
    try:
        out = run_eval(solve_path, _payload(train, hidden))
    except Exception as exc:
        return _finish(0.0, {"error": "runtime_error", "detail": str(exc)[:1200]}, write_dir)
    if "error" in out:
        return _finish(0.0, {"error": out.get("error"), "detail": out.get("detail", "")}, write_dir)
    pred1 = np.asarray(out["preds"][0], dtype=float)
    if pred1.shape != hidden.y.shape:
        return _finish(0.0, {"error": "wrong_shape", "shape": list(pred1.shape), "expected": list(hidden.y.shape)}, write_dir)
    if not np.all(np.isfinite(pred1)):
        return _finish(0.0, {"error": "nonfinite_prediction"}, write_dir)
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

test smoke
from __future__ import annotations

import numpy as np

from _paths import app_data_path, app_solve_path


def test_candidate_file_exists():
    assert app_solve_path().exists(), "solve.py not found"


def test_visible_data_loadable():
    for name in ("train_data.npz", "public_eval.npz"):
        path = app_data_path(name)
        assert path.exists(), f"{name} not found"
        with np.load(path, allow_pickle=False) as z:
            assert "line_X" in z.files and "invoice_offsets" in z.files, f"{name} missing core arrays"

test sh

#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
mkdir -p /sandbox/io

# Make the sandbox runner available outside /tests before /tests is locked down.
if [ -f /tests/agent_call.py ]; then
  cp /tests/agent_call.py /sandbox/agent_call.py
elif [ -f "$(dirname "$0")/agent_call.py" ]; then
  cp "$(dirname "$0")/agent_call.py" /sandbox/agent_call.py
fi
chmod 755 /sandbox/agent_call.py 2>/dev/null || true
chmod 777 /sandbox/io 2>/dev/null || true

START_SEC=$(date +%s)

# — Smoke check —
set +e
pytest /tests/test_smoke.py -v --tb=short 2> /logs/verifier/smoke_stderr.txt
SMOKE_EXIT=$?
set -e
if [ $SMOKE_EXIT -ne 0 ]; then
  echo pipeline_broken > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# — Schema validation —
set +e
pytest /tests/test_schema.py -v --tb=short 2>> /logs/verifier/smoke_stderr.txt
SCHEMA_EXIT=$?
set -e
if [ $SCHEMA_EXIT -ne 0 ]; then
  echo schema_violation > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# Lock verifier code away from the untrusted nobody subprocess. Root pytest can
# still read /tests; the candidate subprocess should not be able to.
chmod 700 /tests 2>/dev/null || true

# — Sandbox/security regression —
set +e
pytest /tests/test_security.py -v --tb=short >> /logs/verifier/security_stdout.txt 2>&1
SECURITY_EXIT=$?
set -e
if [ $SECURITY_EXIT -ne 0 ]; then
  echo reward_hack_open > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# — Baseline floor —
set +e
pytest /tests/test_baseline.py -v --tb=short >> /logs/verifier/baseline_stdout.txt 2>&1
BASELINE_EXIT=$?
set -e
if [ $BASELINE_EXIT -ne 0 ]; then
  python3 - <<'PYFAIL'
import json
from pathlib import Path
m = Path('/logs/verifier/metrics.json')
mode = 'below_baseline'
if m.exists():
    try:
        data = json.loads(m.read_text())
        if isinstance(data, dict) and data.get('error'):
            mode = str(data.get('error'))
    except Exception:
        pass
Path('/logs/verifier/failure_mode.txt').write_text(mode + '\n')
PYFAIL
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# — Main continuous scoring —
set +e
pytest /tests/test_main.py \
  --json-report --json-report-file=/logs/verifier/ctrf.json \
  -v --tb=short \
  2> /logs/verifier/stderr.txt
EXIT=$?
set -e

if [ $EXIT -ne 0 ]; then
  python3 /tests/categorize_failure.py /logs/verifier/stderr.txt \
    /logs/verifier/failure_mode.txt 2>/dev/null \
    || echo unknown > /logs/verifier/failure_mode.txt
  echo 0 > /logs/verifier/reward.txt
else
  # test_main.py writes continuous reward.txt and metrics.json.
  if [ ! -f /logs/verifier/reward.txt ]; then
    echo missing_reward > /logs/verifier/failure_mode.txt
    echo 0 > /logs/verifier/reward.txt
  fi
  # — Holdout / slice hygiene (informational) —
  set +e
  timeout 60s pytest /tests/test_holdout.py -v --tb=short >> /logs/verifier/holdout_stdout.txt 2>&1 || true
  set -e
fi

END_SEC=$(date +%s)
echo $(( END_SEC - START_SEC )) > /logs/verifier/wall_clock_sec.txt
if [ -f /proc/self/status ]; then
  awk '/VmHWM/ {printf "%.0f\n", $2/1024}' /proc/self/status \
    > /logs/verifier/peak_memory_mb.txt 2>/dev/null || true
fi

exit 0

