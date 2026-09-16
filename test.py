"""Fast end-to-end smoke test for conditional_population_response v0.1.

This test intentionally uses a tiny constant baseline.  It does not test task
quality or reference performance; it checks that the generator, agent API, model
loading, prediction contract, and continuous RMSE scoring can execute together.
"""

from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np


# When this file lives in tests/, generator.py and hidden_eval.py are siblings.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generator import generate
from hidden_eval import load_model, score_rmse


def test_smoke_agent_api_end_to_end():
    train = generate(n_cases=24, seed=101, labeled=True)
    hidden = generate(n_cases=11, seed=202, labeled=True)

    # Basic fixture contract.
    assert train["offsets"].shape == (25,)
    assert hidden["offsets"].shape == (12,)
    assert train["treatment"].shape == (24, 5)
    assert hidden["treatment"].shape == (11, 5)
    assert train["case_meta"].shape == (24, 2)
    assert hidden["case_meta"].shape == (11, 2)
    assert train["cells"].shape[1] == hidden["cells"].shape[1] == 12

    # A deliberately weak agent implementation is enough for a smoke test.
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
            n_cases = len(np.asarray(offsets)) - 1
            return np.full(n_cases, mean_y, dtype=float)
        """
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train_path = td / "train.npz"
        agent_path = td / "model.py"
        model_path = td / "model.bin"
        np.savez_compressed(train_path, **train)
        agent_path.write_text(model_source, encoding="utf-8")

        agent = load_model(str(agent_path))
        agent.fit_model(str(train_path), str(model_path))
        pred = np.asarray(
            agent.predict(
                hidden["cells"],
                hidden["offsets"],
                hidden["treatment"],
                hidden["case_meta"],
                str(model_path),
            ),
            dtype=float,
        )

    assert pred.shape == (11,)
    assert np.all(np.isfinite(pred))

    rmse = float(np.sqrt(np.mean((pred - hidden["y"]) ** 2)))
    reward = score_rmse(rmse)
    assert np.isfinite(rmse) and rmse >= 0.0
    assert 0.0 <= reward <= 1.0


def test_smoke_generator_determinism_and_unlabeled_contract():
    a = generate(n_cases=7, seed=303, labeled=False)
    b = generate(n_cases=7, seed=303, labeled=False)

    assert "y" not in a and "y" not in b
    for key in ("cells", "offsets", "treatment", "case_meta"):
        assert np.array_equal(a[key], b[key])

