# Inspection event risk

## Objective

Predict the future continuous risk value for each entity from its prior inspection-like event history. Larger risk values indicate more severe future outcomes. The objective is to minimize root mean squared error (RMSE) between predicted and actual risk values.

The current model in `app/solve.py` is a weak fallback. Replace it with a stronger deterministic model.

Implement these two functions in `app/solve.py`:

```python
def fit_inspection_model(
    train_entity_X,
    train_entity_code,
    train_event_time,
    train_event_code,
    train_event_value,
    train_entity_offsets,
    train_y,
):
    ...


def predict_inspection_risk(
    entity_X,
    entity_code,
    event_time,
    event_code,
    event_value,
    entity_offsets,
    params,
):
    ...
```

## Data

`entity_X` is a numeric matrix with one row per entity. `entity_code` is an integer-coded matrix with one row per entity. The event arrays are packed across all entities, and `entity_offsets[i]:entity_offsets[i+1]` gives the history rows for entity `i`.

`event_time` gives the event time relative to the prediction point. The histories are ordered within each entity. `event_code` gives the event type. `event_value` contains numeric measurements attached to each event.

`train_y` is the future risk value for each training entity. The `public_eval.npz` file includes labeled examples for local sanity checks.

## Runtime and file availability

The solution runs in a fresh, isolated Python environment. The only project files available under `/app` are:

- `/app/solve.py`
- `/app/train_data.npz`
- `/app/public_eval.npz`

Only preinstalled Python packages are available. Other project files and development artifacts are not present, including custom helper modules, saved models, checkpoints, and caches. The solution must not require additional local files, package installation, or network access.

Keep all code needed for fitting and prediction in `app/solve.py`. Build any learned model state inside `fit_inspection_model` using the supplied training arrays, return it as `params`, and use those parameters in `predict_inspection_risk`. Do not rely on separately saved model files.

## Notes

Simple averages, last-event rules, or category-only fallbacks are intentionally weak. Useful signal can come from event order, event type, numeric event values, recency, repeated patterns, gaps between events, and short-history fallbacks.

Your predictions must be a one-dimensional finite NumPy-compatible array of shape `(n_entities,)`. The functions must be deterministic and must not mutate any input arrays. Do not read files beyond the available inputs, use the network, or depend on external data.
