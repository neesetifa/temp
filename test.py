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

## Grading environment

The grader runs in a **fresh image**, not the development environment. At grading time, **only the following three project files** are carried into `/app`:

- `/app/solve.py`
- `/app/train_data.npz`
- `/app/public_eval.npz`

Only the Python packages **preinstalled in the grading image** are available. No other project files from the development environment are carried over, including custom helper modules, saved models or checkpoints, caches, and generated artifacts. Do not rely on additional local files, installing packages, or network access.

Keep all code needed for fitting and predicting inside `app/solve.py`. If your method learns model state, construct that state inside `fit_dispute_model` using the supplied training arrays, return it through `params`, and use those returned parameters in `predict_dispute_outcome`. Do not expect a separately saved model file to persist into grading.

Predictions must be finite, deterministic, one-dimensional, and have shape `(n_cases,)`. Do not mutate input arrays.
