environment/app/solve.py has the two functions to fill in:

fit_component_model(
    train_part_X,
    train_part_slot,
    train_case_offsets,
    train_y,
)

predict_component_score(
    part_X,
    part_slot,
    case_offsets,
    params,
)

Return one score for each case.

Rows are packed by case. For case i, use:

case_offsets[i] : case_offsets[i + 1]

Each row has a slot id. The columns line up, but a row from one slot is not interchangeable with a row from another slot.

The old code flattened each case by averaging its rows. That was good enough for easy cases, but it washed out some failures: a single strange row, a missing optional row, or two slots that looked fine alone but did not line up together.

Use the arrays passed into the functions. Do not read hidden files, use network calls, use outside data, or fit on public eval answers.
