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
        mod = _load(solve_path)
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
    except Exception as exc:  # pragma: no cover - verifier diagnostic path
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

sand box
"""Root-side sandbox helper for invoice_line_reconciliation.

When running in the platform verifier container as root, this helper executes the
candidate fit/predict path inside one unprivileged subprocess. Only plain .npz
arrays cross the boundary. The candidate never imports verifier code and cannot
read /tests after test.sh locks it down.

Local author fallback: when the platform sandbox is unavailable, run through the
same agent_call protocol in-process/subprocess without privilege dropping.
"""
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
REQUIRED = ("fit_invoice_model", "predict_net_line_cost")


def _sandbox_ready() -> bool:
    return os.path.exists(RUNNER) and getattr(os, "geteuid", lambda: 1000)() == 0


def _local_runner() -> str:
    return str(Path(__file__).resolve().parent / "agent_call.py")


def _demote():  # pragma: no cover - runs in child before exec
    try:
        # PR_SET_NO_NEW_PRIVS = 38. Prevent later privilege regain through exec.
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
        # If nobody is unavailable, subprocess still runs isolated from verifier
        # state by protocol; platform containers normally have this account.
        pass



def _nobody_command(args: list[str]) -> tuple[list[str], object | None]:
    """Return a command that runs as nobody without preexec when possible."""
    if shutil.which("runuser"):
        return ["runuser", "-u", "nobody", "--", *args], None
    if shutil.which("su"):
        # Fallback for minimal images with su but no runuser.
        quoted = " ".join(subprocess.list2cmdline([a]) for a in args)
        return ["su", "-s", "/bin/sh", "nobody", "-c", quoted], None
    return args, _demote

def _payload_to_npz(path: str | os.PathLike, payload: dict) -> None:
    np.savez(
        path,
        train_line_X=np.asarray(payload["train_line_X"], dtype=np.float64),
        train_line_code=np.asarray(payload["train_line_code"], dtype=np.int64),
        train_invoice_X=np.asarray(payload["train_invoice_X"], dtype=np.float64),
        train_invoice_code=np.asarray(payload["train_invoice_code"], dtype=np.int64),
        train_invoice_offsets=np.asarray(payload["train_invoice_offsets"], dtype=np.int64),
        train_y=np.asarray(payload["train_y"], dtype=np.float64),
        line_X=np.asarray(payload["line_X"], dtype=np.float64),
        line_code=np.asarray(payload["line_code"], dtype=np.int64),
        invoice_X=np.asarray(payload["invoice_X"], dtype=np.float64),
        invoice_code=np.asarray(payload["invoice_code"], dtype=np.int64),
        invoice_offsets=np.asarray(payload["invoice_offsets"], dtype=np.int64),
    )


def _read_output(out_path: str | os.PathLike) -> dict:
    with np.load(out_path, allow_pickle=False) as z:
        if "error" in z.files:
            return {"error": str(z["error"].item()), "detail": str(z.get("detail", ""))}
        n = int(z["n"])
        return {"preds": [np.asarray(z[f"p{i}"], dtype=np.float64) for i in range(n)]}


def run_eval(solve_path: str | os.PathLike, payload: dict, timeout: int = 180) -> dict:
    """Run candidate fit + predict and return {"preds": [array]} or {"error": ...}."""
    if not _sandbox_ready():
        # Author/local path. Use the same .npz protocol but avoid absolute /sandbox.
        import tempfile
        runner = _local_runner()
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in_eval.npz")
            out_path = os.path.join(td, "out_eval.npz")
            _payload_to_npz(in_path, payload)
            r = subprocess.run(
                [sys.executable, runner, str(solve_path), "eval", in_path, out_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if r.returncode != 0 or not os.path.exists(out_path):
                raise RuntimeError(f"local eval failed (rc={r.returncode}): {r.stderr[-800:]}")
            return _read_output(out_path)

    Path(IO_DIR).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(IO_DIR, 0o777)
    except Exception:
        pass
    _counter[0] += 1
    in_path = os.path.join(IO_DIR, f"in_eval_{_counter[0]}.npz")
    out_path = os.path.join(IO_DIR, f"out_eval_{_counter[0]}.npz")
    _payload_to_npz(in_path, payload)
    os.chmod(in_path, 0o644)
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd, preexec = _nobody_command([sys.executable, RUNNER, str(solve_path), "eval", in_path, out_path])
    r = subprocess.run(
        cmd,
        preexec_fn=preexec,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/sandbox",
    )
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"sandboxed eval failed (rc={r.returncode}): {r.stderr[-800:]}")
    return _read_output(out_path)


def check_solution(solve_path: str | os.PathLike, timeout: int = 30) -> bool:
    runner = RUNNER if _sandbox_ready() else _local_runner()
    try:
        if _sandbox_ready():
            cmd, preexec = _nobody_command([sys.executable, runner, str(solve_path), "check"])
        else:
            cmd, preexec = [sys.executable, runner, str(solve_path), "check"], None
        r = subprocess.run(
            cmd,
            preexec_fn=preexec,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/sandbox" if _sandbox_ready() else None,
        )
        return r.returncode == 0
    except Exception:
        return False

test smoke
from __future__ import annotations

import numpy as np

from _paths import app_data_path, app_solve_path
from sandbox_utils import check_solution


def test_candidate_file_exists():
    assert app_solve_path().exists(), "solve.py not found"


def test_required_functions_present():
    assert check_solution(app_solve_path()), "required fit_invoice_model / predict_net_line_cost functions not found"


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

test schema
from __future__ import annotations

import numpy as np

from _paths import app_data_path
from generator import N_INVOICE_CODE, N_INVOICE_X, N_LINE_CODE, N_LINE_X, make_hidden, make_train_public


def _check_dataset(d, *, has_y: bool, has_slices: bool = False):
    assert d["line_X"].ndim == 2 and d["line_X"].shape[1] == N_LINE_X
    assert d["line_code"].ndim == 2 and d["line_code"].shape[1] == N_LINE_CODE
    assert d["invoice_X"].ndim == 2 and d["invoice_X"].shape[1] == N_INVOICE_X
    assert d["invoice_code"].ndim == 2 and d["invoice_code"].shape[1] == N_INVOICE_CODE
    assert d["invoice_offsets"].ndim == 1
    assert d["invoice_offsets"][0] == 0
    assert d["invoice_offsets"][-1] == d["line_X"].shape[0]
    assert np.all(np.diff(d["invoice_offsets"]) > 0)
    assert d["line_code"].shape[0] == d["line_X"].shape[0]
    assert d["invoice_code"].shape[0] == d["invoice_X"].shape[0]
    assert d["invoice_offsets"].shape[0] == d["invoice_X"].shape[0] + 1
    assert np.all(np.isfinite(d["line_X"]))
    assert np.all(np.isfinite(d["invoice_X"]))
    if has_y:
        assert "y" in d
        assert d["y"].shape == (d["line_X"].shape[0],)
        assert np.all(np.isfinite(d["y"]))
    if has_slices:
        assert "slice_name" in d
        assert d["slice_name"].shape == (d["line_X"].shape[0],)


def test_agent_visible_schema():
    for name in ("train_data.npz", "public_eval.npz"):
        with np.load(app_data_path(name), allow_pickle=False) as z:
            _check_dataset(z, has_y=True)


def test_verifier_generated_schema():
    train, public = make_train_public()
    hidden = make_hidden()
    _check_dataset(train.as_npz_dict(include_y=True), has_y=True)
    _check_dataset(public.as_npz_dict(include_y=True), has_y=True)
    _check_dataset(hidden.as_npz_dict(include_y=True, include_slices=True), has_y=True, has_slices=True)


def test_train_public_hidden_are_distinct_sizes():
    train, public = make_train_public()
    hidden = make_hidden()
    assert train.line_X.shape[0] > public.line_X.shape[0] > 0
    assert hidden.line_X.shape[0] > public.line_X.shape[0] > 0

test main
from __future__ import annotations

from _paths import app_solve_path, verifier_log_dir
from hidden_eval import evaluate


def test_main_scoring_writes_reward_and_metrics():
    out_dir = verifier_log_dir()
    reward, metrics = evaluate(app_solve_path(), out_dir)
    assert (out_dir / "reward.txt").exists(), "reward.txt was not written"
    assert (out_dir / "metrics.json").exists(), "metrics.json was not written"
    assert 0.0 <= reward <= 1.0
    assert "reward" in metrics

test baseline
from __future__ import annotations

from _paths import app_solve_path, verifier_log_dir
from hidden_eval import evaluate


def test_candidate_not_below_minimal_floor():
    # This is a very low smoke-floor, not the final score. It catches broken
    # pipelines that return constants/garbage while leaving normal scoring to
    # test_main.py.
    reward, metrics = evaluate(app_solve_path(), verifier_log_dir() / "baseline_floor")
    assert reward >= 0.02, f"candidate below minimal baseline floor: reward={reward:.6f}, metrics={metrics}"
