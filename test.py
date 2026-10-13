# Task

`/app/model.npz` is a small pretrained decoder-only language model (4 layers, 4 heads per layer, head dim 32, RoPE). `/app/kvmodel.py` is its numpy inference code. `/app/data/train_tokens.npy` holds 800 token sequences (512 tokens each) from the private corpus the model was trained on.

Write a KV-cache eviction policy in `/app/policy.py` so that the model behaves as closely as possible to its full-cache self under a small cache budget.

## How the policy is run

`TinyLM.run(tokens, policy, budget, chunk)` in `kvmodel.py` is exactly the loop the verifier uses:

- sequences are processed in chunks of **32** tokens;
- for each chunk and each layer (0..3), the chunk's queries attend to the entries still in that layer's cache plus the chunk itself (causal), the chunk's keys/values are appended, and then `policy.select(view)` returns, for every head, the indices of the entries to **keep**;
- after the last layer of every chunk, the total number of entries over all layers and heads must be **at most 96** (the full cache reaches 8192). Heads and layers may hold different numbers of entries.

The policy only selects entries; it cannot modify, merge or add keys/values. `view` is a `kvmodel.LayerView`, and its docstring lists every field. The policy is not passed token ids directly. Token identity is, however, recoverable from the cache contents with the shipped weights (for example, layer-0 keys and values depend only on the token), and using it is allowed.

Required interface (see the starter `/app/policy.py`):

```python
class EvictionPolicy:
    def __init__(self, config: dict): ...        # d, heads, layers, head_dim, vocab, chunk, seq_len, budget, app_dir
    def begin_sequence(self, seq_len: int): ...  # optional
    def select(self, view) -> list[np.ndarray]:  # one int index array (or bool mask) per head
```

## Scoring

The verifier runs hidden sequences twice, once with the full cache and once with your policy. On every position after the first chunk it compares the next-token distributions using KL(full || yours).

Positions are grouped by the role their token plays in the sequence:

- **answer**: positions whose next token can only be predicted by recalling a specific item stated much earlier in the same sequence;
- **filler**: positions inside ordinary running text (a text word followed by a text word);
- **other**: all remaining positions.

Each hidden shard is scored as 0.45 × answer KL + 0.20 × answer top-1 agreement + 0.20 × filler KL + 0.15 × other KL. Each KL term is the mean KL over that group's positions, mapped to [0, 1] on a log scale: score 0 is roughly where generic attention-statistics policies such as heavy-hitter or observation-window eviction land, and score 1 is an idealized upper reference that a policy is not expected to reach. Top-1 agreement is the fraction of answer positions where the most likely next token under your policy matches the full cache's, mapped linearly to [0, 1] between 0.5 and 1.0.

The reward is 0.4 × the in-distribution shard + 0.6 × a shifted shard whose workload statistics differ from the training corpus (for example, more items to remember and longer gaps before they are needed).

The reward is continuous in [0, 1]. `python /app/evaluate_local.py` gives a quick unweighted local estimate.

## Rules

- The verifier runs your policy in a separate, unprivileged process with no network. `select` sees only what is in `view`.
- Your policy may read any files you place under `/app` (for example, tables or parameters you precompute), but must not write files at evaluation time.
- The total policy compute for the hidden evaluation (about 7,700 `select` calls) must stay under 20 minutes on 4 CPU cores.
- Exceeding the budget, returning invalid indices, crashing or timing out gives reward 0.
