# Error-bounded KV-cache codec

Serving systems increasingly ship the prefill KV cache of a language model between machines instead of recomputing it. Build a codec that compresses these caches as far as possible while reconstructing every value within a fixed absolute error bound.

## Data in `/app`

**`/app/pool/`** holds 200 prefill KV caches of Qwen2.5-0.5B-Instruct. Each cache has two files:

- `<id>.kv.npy`: a `uint16` array of the raw bf16 bits, shape `(23, 2, T, 2, 64)`. The axes are layers 1–23 (layer 0 is excluded), [key, value], tokens, KV heads, and head dimension. Keys are stored after rotary position embedding.
- `<id>.pos.npy`: the `int64` token positions `0 … T-1`.

`manifest.json` lists each cache's id, source domain and T.

The requests are raw text with no chat template or fixed prefix. They come in equal shares from four domains: chat, Python code, math problems with solutions, and long encyclopedic documents. Each request is 512–3,072 tokens long, drawn uniformly.

**`/app/bounds.npy`** is a `(23, 2)` table of absolute error bounds, one per (layer, key/value). Each bound is 1% of that tensor's standard deviation over the first 60 pool caches.

**`/app/baseline/`** is a working codec and the reward anchor. It quantises every element independently and entropy-codes each channel with rANS under a static per-channel Laplace model, with no prediction of any kind. You may start from it:

```bash
cp /app/baseline/* /output/
```

**`/app/tools/evaluate_local.py`** runs the grader's checks and reward on held-out pool caches:

```bash
python /app/tools/evaluate_local.py --codec /output --n 25
```

## What to build

Build `/output/encode.py` and `/output/decode.py`. Each is called once for a whole batch of caches.

```bash
python /output/encode.py <in_dir> <out_dir>
python /output/decode.py <in_dir> <out_dir>
```

**Encode:**
- **Input.** `<in_dir>` contains `<id>.kv.npy` and `<id>.pos.npy` for every cache.
- **Output.** Write exactly one file `<out_dir>/<id>.bin` per cache, and nothing else.

**Decode:**
- **Input.** `<in_dir>` contains only `<id>.bin` and `<id>.pos.npy`.
- **Output.** Write exactly one `<out_dir>/<id>.npy` per cache: a C-order `float32` array with the original shape `(23, 2, T, 2, 64)`, saved with `numpy.save`.
- **The bound.** Every decoded element must satisfy `|decoded - original| <= bound[layer, key/value]`. The comparison is in float32, against the original bf16 value converted exactly to float32. NaN fails.

## Scoring

The grader runs your codec on **50 hidden caches**. They are built the same way as the pool, but from documents that do not appear in the pool.

- **CR** is the total raw bf16 bytes of all hidden caches divided by the total size of your `.bin` files.
- **CR_base** is the baseline codec's compression ratio on the same caches, recomputed by the grader. Its source is exactly `/app/baseline`.
- **Reward** is `clamp(log(CR / CR_base) / log(2), 0, 1)`. Matching the baseline scores 0. A codec twice as compact as the baseline scores 1.

**The reward is 0 if any of these fail:**
- Encoding the hidden caches twice, into fresh directories, gives byte-identical `.bin` files.
- Decoding produces exactly one valid array per cache, with every element within its bound.
- Each encode run and each decode run of the full hidden batch finishes within **600 seconds**.
- `/output` is at most **50 MB** in total (file bytes plus path names) and holds only regular files and directories.

## Environment and grading machine

- **Hardware.** The working session and grading run on the same machine type: **4 CPUs, 12 GB RAM, 10 GB disk, x86_64**, no GPU.
- **Disk.** The pool alone occupies about 4 GB, so avoid materialising float32 or float64 copies of the whole pool on disk.
- **Memory.** All encode or decode processes share the 12 GB.
- **Network.** None.
- **Decoder input.** `decode.py` receives only the `.bin` files and positions. It must not rely on any other file outside `/output`.
- **Threads.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are set to 1 when your scripts run. You may use up to 4 processes or threads yourself.
- **Libraries.** The same Python and libraries as this environment: numpy, scipy, numba, PyTorch (CPU), zstandard, and constriction (rANS and range coding).

## Rules

- **Self-contained.** Every table, model or other asset your codec needs must be inside `/output`, within the 50 MB limit.
- **Determinism.** Keep encoding deterministic. Floating-point work that the decoder must repeat exactly should run in the same order on both sides.
