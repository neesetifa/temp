test security
from __future__ import annotations

from pathlib import Path
import textwrap

from hidden_eval import evaluate


def test_evaluate_does_not_run_candidate_in_verifier_process(tmp_path):
    """Regression guard for the old same-process reward hack.

    If hidden_eval.evaluate() imports the candidate and calls predict in the
    verifier process, this solve can walk Python stack frames, find the local
    hidden dataset object, and return hidden.y for a perfect score. With the
    sandbox boundary, the candidate process only sees plain arrays and this
    probe falls back to zeros.
    """
    probe_dir = Path("/sandbox/io") if Path("/sandbox/io").exists() else tmp_path
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_dir.chmod(0o777)
    except Exception:
        pass
    try:
        tmp_path.chmod(0o755)
    except Exception:
        pass
    cheat = probe_dir / "solve_stack_probe.py"
    cheat.write_text(textwrap.dedent(
        r'''
        import inspect
        import numpy as np

        def fit_invoice_model(train_line_X, train_line_code, train_invoice_X,
                              train_invoice_code, train_invoice_offsets, train_y):
            return {"mean": float(np.mean(train_y))}

        def predict_net_line_cost(line_X, line_code, invoice_X, invoice_code,
                                  invoice_offsets, params):
            for frame_info in inspect.stack():
                hidden = frame_info.frame.f_locals.get("hidden")
                if hidden is not None and hasattr(hidden, "y"):
                    return np.asarray(hidden.y, dtype=float).copy()
            return np.zeros(line_X.shape[0], dtype=float)
        '''
    ))
    cheat.chmod(0o644)
    reward, metrics = evaluate(cheat, tmp_path / "logs")
    assert metrics.get("error") not in {"api_missing", "runtime_error"}, (
        "stack-probe solve did not actually execute; "
        f"reward={reward:.6f}, metrics={metrics}"
    )
    assert reward < 0.90, (
        "candidate appears to have run inside the verifier process; "
        f"stack-probe reward={reward:.6f}, metrics={metrics}"
    )

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

# — Reference anchor (informational) —
set +e
pytest /tests/test_reference.py -v --tb=short >> /logs/verifier/reference_stdout.txt 2>&1
set -e

# — Baseline floor —
set +e
pytest /tests/test_baseline.py -v --tb=short >> /logs/verifier/baseline_stdout.txt 2>&1
BASELINE_EXIT=$?
set -e
if [ $BASELINE_EXIT -ne 0 ]; then
  echo below_baseline > /logs/verifier/failure_mode.txt
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
  pytest /tests/test_holdout.py -v --tb=short >> /logs/verifier/holdout_stdout.txt 2>&1 || true
  set -e
fi

END_SEC=$(date +%s)
echo $(( END_SEC - START_SEC )) > /logs/verifier/wall_clock_sec.txt
if [ -f /proc/self/status ]; then
  awk '/VmHWM/ {printf "%.0f\n", $2/1024}' /proc/self/status \
    > /logs/verifier/peak_memory_mb.txt 2>/dev/null || true
fi

exit 0
hiddenfrom __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

try:
    from .generator import make_hidden, make_train_public
    from .sandbox_utils import run_eval, check_solution
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from generator import make_hidden, make_train_public
    from sandbox_utils import run_eval, check_solution


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
    if not check_solution(solve_path):
        return _finish(0.0, {"error": "api_missing"}, write_dir)
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

