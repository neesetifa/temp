# Compact next-basket recommendation

The transaction history in `/app/train.csv` comes from a real online retail log. Each customer has a sequence of earlier orders; the final evaluation asks for the next basket for customers already present in that history.

Build `/output/predict.py`.

It will be called as:

```bash
python /output/predict.py <in.csv> <out.csv>
```

`<in.csv>` has one column:

```text
customer_id
```

Write `<out.csv>` with exactly one column named `prediction`. Each row must contain **exactly 10 distinct item IDs**, separated by single spaces, in recommendation order (best first). Item IDs must come from the item catalog visible in the training data. Output rows must correspond to input rows in the same order.

The complete `/output` artifact is limited to **64 KiB (65,536 bytes)**. This includes `predict.py` and every model, table, index, or other file placed under `/output`. Symlinks and other special files are not allowed.

The final evaluation does not provide the training CSV to the predictor. Anything needed at inference time must therefore be encoded inside the `/output` artifact. You can use `/app/train.csv` while developing the submission.

Training rows have these columns:

```text
customer_id,order_index,item_id,quantity,days_since_first,days_since_prev
```

Rows with the same `customer_id` and `order_index` belong to the same basket. IDs are opaque and should not be interpreted as original retailer identifiers.

The score is based on how highly the hidden next-basket items appear in each customer's top-10 list. A global-popularity recommender is only a floor; the useful part of the task is deciding what customer- and item-level information is worth keeping under the artifact-size limit.

Keep the submission deterministic. Reordering input rows must only reorder the corresponding outputs.
