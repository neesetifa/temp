# Transaction dispute outcome

## Objective

Predict the final refund fraction for each dispute case, a continuous value between 0 and 1. The objective is to minimize root mean squared error (RMSE) between predicted and actual refund fractions.

The file `app/solve.py` contains a simple fallback for the model. Replace it with a deterministic implementation.

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

## Runtime and file availability

The solution runs in a fresh, isolated Python environment. The only project files available under `/app` are:

- `/app/solve.py`
- `/app/train_data.npz`
- `/app/public_eval.npz`

Only preinstalled Python packages are available. Other project files and development artifacts are not present, including custom helper modules, saved models, checkpoints, and caches. The solution must not require additional local files, package installation, or network access.

Keep all code needed for fitting and prediction in `app/solve.py`. Build any learned model state inside `fit_dispute_model` using the supplied training arrays, return it as `params`, and use those parameters in `predict_dispute_outcome`. Do not rely on separately saved model files.

Predictions must be finite, deterministic, one-dimensional, and have shape `(n_cases,)`. Do not mutate input arrays.
