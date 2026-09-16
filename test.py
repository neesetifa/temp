from __future__ import annotations

import json
import math
import os
from pathlib import Path
import pwd
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

import numpy as np

# Hidden-only code is imported in the privileged evaluator process. Submitted
# model code is NEVER imported in this process.
from generator import generate


HERE = Path(__file__).resolve().parent
DEFAULT_RUNNER = Path("/usr/local/libexec/cpr_model_runner.py")


def score_rmse(r):
    # Continuous anchors from frozen v0.1 calibration; valid weak models retain
    # gradient rather than collapsing to hard zero.
    BAD, GOOD = .225, .105
    return float(np.clip((BAD - r) / (BAD - GOOD), 0, 1))


def _runner_path() -> Path:
    override = os.environ.get("CPR_MODEL_RUNNER")
    if override:
        p = Path(override)
    elif DEFAULT_RUNNER.exists():
        p = DEFAULT_RUNNER
    else:
        # Development fallback.  The production tests image copies the runner
        # out of /tests before making /tests unreadable to the model user.
        p = HERE / "model_runner.py"
    if not p.is_file():
        raise RuntimeError(f"model runner not found: {p}")
    return p


def _sandbox_identity():
    # Production image creates modelrunner.  'nobody' is only a local-dev
    # fallback so the same tests can be exercised before image build.
    for name in ("modelrunner", "nobody"):
        try:
            pw = pwd.getpwnam(name)
            return pw.pw_uid, pw.pw_gid
        except KeyError:
            pass
    raise RuntimeError("no unprivileged sandbox account is available")


def _drop_privileges(uid: int, gid: int):
    def preexec():
        os.setsid()
        # Bound obvious resource abuse.  Values are intentionally generous for
        # the reference PyTorch model.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    return preexec


def _safe_env(workdir: Path, model_dir: Path) -> dict[str, str]:
    # Do not forward arbitrary evaluator environment variables to submitted code.
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        # Allows model.py to import helper modules the agent created beside it,
        # while deliberately excluding /tests.
        "PYTHONPATH": str(model_dir),
    }
    return env


def _run(cmd, *, uid, gid, cwd, env, timeout):
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            preexec_fn=_drop_privileges(uid, gid),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"submitted model timed out after {timeout}s") from exc
    if proc.returncode != 0:
        out = (proc.stdout or "")[-4000:]
        err = (proc.stderr or "")[-4000:]
        raise RuntimeError(
            f"submitted model process failed with code {proc.returncode}\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )


def run_model_sandboxed(
    *,
    model_py: str,
    train_path: str,
    hidden_inputs: dict[str, np.ndarray],
    fit_timeout: int = 240,
    predict_timeout: int = 120,
) -> np.ndarray:
    """Fit and predict in separate unprivileged subprocesses.

    Crucially, hidden labels and generator.py never enter either subprocess.
    """
    if os.geteuid() != 0:
        raise RuntimeError(
            "hidden evaluator must run as root so it can drop submitted model "
            "code to the unprivileged modelrunner account"
        )

    model_py = str(Path(model_py).resolve())
    train_path = str(Path(train_path).resolve())
    model_dir = Path(model_py).parent
    if not Path(model_py).is_file():
        raise RuntimeError(f"missing submitted model: {model_py}")
    if not Path(train_path).is_file():
        raise RuntimeError(f"missing training data: {train_path}")

    uid, gid = _sandbox_identity()
    runner = _runner_path().resolve()

    workdir = Path(tempfile.mkdtemp(prefix="cpr_agent_", dir="/tmp"))
    try:
        os.chown(workdir, uid, gid)
        os.chmod(workdir, 0o700)

        input_npz = workdir / "hidden_inputs.npz"
        model_artifact = workdir / "model.bin"
        output_npy = workdir / "predictions.npy"
        np.savez_compressed(input_npz, **hidden_inputs)
        os.chown(input_npz, uid, gid)
        os.chmod(input_npz, 0o600)

        # The submitted process must be able to traverse/read the model and train
        # paths. Production /app is world-readable by construction.
        env = _safe_env(workdir, model_dir)

        _run(
            [sys.executable, str(runner), "fit", model_py, train_path, str(model_artifact)],
            uid=uid,
            gid=gid,
            cwd=workdir,
            env=env,
            timeout=fit_timeout,
        )
        if not model_artifact.exists():
            raise RuntimeError("fit subprocess did not create model artifact")

        _run(
            [
                sys.executable,
                str(runner),
                "predict",
                model_py,
                str(input_npz),
                str(model_artifact),
                str(output_npy),
            ],
            uid=uid,
            gid=gid,
            cwd=workdir,
            env=env,
            timeout=predict_timeout,
        )
        if not output_npy.exists():
            raise RuntimeError("predict subprocess did not create predictions")
        return np.asarray(np.load(output_npy, allow_pickle=False), dtype=float)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def evaluate(model_py="/app/model.py"):
    try:
        # Generate labels only in this privileged evaluator process.
        d = generate(1200, 882341, labeled=True)
        hidden_inputs = {
            "cells": d["cells"],
            "offsets": d["offsets"],
            "treatment": d["treatment"],
            "case_meta": d["case_meta"],
        }
        p = run_model_sandboxed(
            model_py=model_py,
            train_path="/app/train.npz",
            hidden_inputs=hidden_inputs,
        )
        if p.shape != (len(d["y"]),) or not np.all(np.isfinite(p)):
            return {"reward": 0.0, "error": "invalid predictions"}

        err = p - d["y"]
        overall = float(np.sqrt(np.mean(err**2)))
        sizes = np.diff(d["offsets"])
        qnorm = np.linalg.norm(d["treatment"], axis=1)
        fam = d["case_meta"][:, 0]
        masks = {
            "small_population": sizes <= np.quantile(sizes, .30),
            "strong_treatment": qnorm >= np.quantile(qnorm, .70),
            "rare_family": fam >= 3,
        }
        metrics = {"rmse": overall}
        parts = [score_rmse(overall)]
        weights = [.70]
        for k, mask in masks.items():
            r = float(np.sqrt(np.mean(err[mask] ** 2)))
            metrics[k + "_rmse"] = r
            metrics[k + "_n"] = int(mask.sum())
            parts.append(score_rmse(r))
            weights.append(.10)
        reward = float(np.dot(parts, weights))
        return {"reward": reward, **metrics}
    except Exception as exc:
        return {"reward": 0.0, "error": f"evaluator failure: {type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))



model runner
"""Unprivileged runner for submitted model code.

This file deliberately contains no hidden-data generator or labels.  The hidden
evaluator launches it in a separate OS process after dropping privileges.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np


def _load_model(path: str):
    spec = importlib.util.spec_from_file_location("submitted_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model module: {path}")
    module = importlib.util.module_from_spec(spec)
    # exec_module is safe here because this process is the isolated,
    # unprivileged model runner.  It has no generator module or hidden labels.
    spec.loader.exec_module(module)
    for name in ("fit_model", "predict"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"model.py must define callable {name}()")
    return module


def do_fit(args) -> None:
    model = _load_model(args.model_py)
    model.fit_model(args.train_npz, args.model_artifact)
    if not Path(args.model_artifact).exists():
        raise RuntimeError("fit_model() did not create model_path")


def do_predict(args) -> None:
    model = _load_model(args.model_py)
    with np.load(args.input_npz, allow_pickle=False) as data:
        required = ("cells", "offsets", "treatment", "case_meta")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise RuntimeError(f"prediction input missing keys: {missing}")
        # Copy arrays so the submitted model receives ordinary ndarrays and no
        # handle to an npz archive containing anything else.
        arrays = {k: np.asarray(data[k]).copy() for k in required}
    pred = np.asarray(
        model.predict(
            arrays["cells"],
            arrays["offsets"],
            arrays["treatment"],
            arrays["case_meta"],
            args.model_artifact,
        ),
        dtype=np.float64,
    )
    np.save(args.output_npy, pred, allow_pickle=False)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("model_py")
    fit.add_argument("train_npz")
    fit.add_argument("model_artifact")

    pred = sub.add_parser("predict")
    pred.add_argument("model_py")
    pred.add_argument("input_npz")
    pred.add_argument("model_artifact")
    pred.add_argument("output_npy")

    args = p.parse_args()
    if args.cmd == "fit":
        do_fit(args)
    else:
        do_predict(args)


if __name__ == "__main__":
    main()


model
"""Starter model for Conditional Population Response.

Replace this baseline with your own implementation.  The hidden evaluator calls
fit_model() once and predict() on fresh unlabeled cases.
"""
from pathlib import Path
import numpy as np


def fit_model(train_path: str, model_path: str) -> None:
    """Fit a deliberately weak constant baseline."""
    with np.load(train_path) as data:
        mean_y = float(np.mean(data["y"]))
    Path(model_path).write_text(repr(mean_y), encoding="utf-8")


def predict(cells, offsets, treatment, case_meta, model_path: str) -> np.ndarray:
    """Return one finite prediction per case."""
    mean_y = float(Path(model_path).read_text(encoding="utf-8"))
    n_cases = len(np.asarray(offsets)) - 1
    return np.full(n_cases, mean_y, dtype=float)

test smoke
"""Fast API and sandbox smoke tests for conditional_population_response v0.1."""
from pathlib import Path
import os
import sys
import tempfile
import textwrap

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generator import generate
from hidden_eval import run_model_sandboxed, score_rmse


def _make_public_tempdir():
    td = tempfile.TemporaryDirectory(prefix="cpr_smoke_", dir="/tmp")
    os.chmod(td.name, 0o755)
    return td


def test_smoke_agent_api_end_to_end():
    train = generate(n_cases=24, seed=101, labeled=True)
    hidden = generate(n_cases=11, seed=202, labeled=True)

    assert train["offsets"].shape == (25,)
    assert hidden["offsets"].shape == (12,)
    assert train["treatment"].shape == (24, 5)
    assert hidden["treatment"].shape == (11, 5)
    assert train["case_meta"].shape == (24, 2)
    assert hidden["case_meta"].shape == (11, 2)
    assert train["cells"].shape[1] == hidden["cells"].shape[1] == 12

    model_source = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np

        def fit_model(train_path: str, model_path: str) -> None:
            with np.load(train_path) as d:
                mean_y = float(np.mean(d["y"]))
            Path(model_path).write_text(repr(mean_y), encoding="utf-8")

        def predict(cells, offsets, treatment, case_meta, model_path: str):
            mean_y = float(Path(model_path).read_text(encoding="utf-8"))
            return np.full(len(np.asarray(offsets)) - 1, mean_y, dtype=float)
        """
    )

    with _make_public_tempdir() as td_raw:
        td = Path(td_raw)
        train_path = td / "train.npz"
        agent_path = td / "model.py"
        np.savez_compressed(train_path, **train)
        agent_path.write_text(model_source, encoding="utf-8")
        os.chmod(train_path, 0o644)
        os.chmod(agent_path, 0o644)

        pred = run_model_sandboxed(
            model_py=str(agent_path),
            train_path=str(train_path),
            hidden_inputs={k: hidden[k] for k in ("cells", "offsets", "treatment", "case_meta")},
            fit_timeout=30,
            predict_timeout=30,
        )

    assert pred.shape == (11,)
    assert np.all(np.isfinite(pred))
    rmse = float(np.sqrt(np.mean((pred - hidden["y"]) ** 2)))
    assert np.isfinite(rmse) and rmse >= 0.0
    assert 0.0 <= score_rmse(rmse) <= 1.0


def test_submitted_model_cannot_read_hidden_generator():
    """Regression test for the old in-process exec_module leakage."""
    train = generate(n_cases=12, seed=404, labeled=True)
    hidden = generate(n_cases=5, seed=405, labeled=False)
    generator_path = (HERE / "generator.py").resolve()

    # Submitted code actively tries both the import path and direct file read.
    model_source = textwrap.dedent(
        f"""
        from pathlib import Path
        import numpy as np

        GENERATOR_PATH = {str(generator_path)!r}

        def fit_model(train_path, model_path):
            Path(model_path).write_text("ok", encoding="utf-8")

        def predict(cells, offsets, treatment, case_meta, model_path):
            leaked = False
            try:
                import generator  # noqa: F401
                leaked = True
            except Exception:
                pass
            try:
                Path(GENERATOR_PATH).read_text(encoding="utf-8")
                leaked = True
            except Exception:
                pass
            value = 12345.0 if leaked else 0.0
            return np.full(len(offsets)-1, value, dtype=float)
        """
    )

    with _make_public_tempdir() as td_raw:
        td = Path(td_raw)
        train_path = td / "train.npz"
        agent_path = td / "model.py"
        np.savez_compressed(train_path, **train)
        agent_path.write_text(model_source, encoding="utf-8")
        os.chmod(train_path, 0o644)
        os.chmod(agent_path, 0o644)

        pred = run_model_sandboxed(
            model_py=str(agent_path),
            train_path=str(train_path),
            hidden_inputs=hidden,
            fit_timeout=30,
            predict_timeout=30,
        )

    assert np.all(pred == 0.0), "submitted model was able to access hidden generator"


def test_smoke_generator_determinism_and_unlabeled_contract():
    a = generate(n_cases=7, seed=303, labeled=False)
    b = generate(n_cases=7, seed=303, labeled=False)
    assert "y" not in a and "y" not in b
    for key in ("cells", "offsets", "treatment", "case_meta"):
        assert np.array_equal(a[key], b[key])



docker
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    torch \
    pytest \
 && groupadd --gid 10001 modelrunner \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin modelrunner

# Put the harmless submitted-model runner outside the hidden evaluator tree.
COPY model_runner.py /usr/local/libexec/cpr_model_runner.py
RUN chown root:root /usr/local/libexec/cpr_model_runner.py \
 && chmod 0555 /usr/local/libexec/cpr_model_runner.py

WORKDIR /tests
COPY . /tests/

# hidden generator/evaluator/test sources are root-only. The evaluator itself
# runs as root and explicitly drops submitted model code to `modelrunner`.
RUN chown -R root:root /tests \
 && find /tests -type f -exec chmod 0600 {} \; \
 && chmod 0700 /tests \
 && chmod 0700 /tests/test.sh

USER root
CMD ["/tests/test.sh"]
