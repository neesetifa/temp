#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Install the submitted/reference implementation at the path expected by the
# hidden evaluator.  Keep solve.sh limited to installation; fit_model() is
# invoked later by the evaluator on /app/train.npz.
cp "$SCRIPT_DIR/reference_solution.py" /app/model.py


FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    torch \
    pytest

WORKDIR /tests
COPY . /tests/
RUN chmod +x /tests/test.sh

CMD ["/tests/test.sh"]
