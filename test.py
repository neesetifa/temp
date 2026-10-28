PROBLEM
LLM serving engines reuse the KV cache of shared prompt prefixes, in 512-token blocks. A request
reuses a block only if every block before it is cached too, so a hit is the longest fully cached
prefix. Memory holds a few thousand blocks: the engine must decide what to admit and evict.

The workloads come from real one-hour production traces of a commercial LLM service (the public
Mooncake FAST'25 release: a chat workload and a tool/agent workload). Each request has an arrival
timestamp (3 s grid), its input token count, and the chain of block hashes of its prompt; equal
hashes mean equal prefixes. No text is included. Development and graded traces are independent
~29-minute draws from a generator fitted to the real second half hour: it keeps the real shared
prompt prefixes and arrival patterns and resamples conversations from real ones, so no draw replays
any real request sequence. The agent gets 3 dev draws per workload; grading uses 2 secret draws per
workload at four capacities (about 460 to 3,750 blocks).

Write an online policy. For each request, in arrival order, the simulator computes the hit, then
calls the policy with the request and its hit length; the policy returns which of the request's
blocks to admit and which resident blocks to evict. The simulator owns the cache and enforces the
capacity; the policy never sees future requests. The policy may keep state and ship artifacts
(e.g. a model fitted on the dev draws).

REWARD
For each graded cell c (2 workloads x 2 draws x 4 capacities = 16):
  h_c   = hit blocks / total prompt blocks under the submitted policy
  L_c   = the same quantity under LRU (vLLM default; deeper blocks of a request evicted first)
  B_c   = the same quantity under Belady's offline optimum (furthest next use, bypass allowed,
          ties broken deeper-block-first; the resulting cache is always ancestor-closed, so B_c
          is an upper bound on any policy's prefix hits)
  g_c   = (h_c - L_c) / (B_c - L_c)
L_c and B_c are recomputed by the verifier at grading time on the same graded draw.
Graded draws: chat ~6,300 requests / ~140,000 prompt blocks; tool-agent ~12,300 requests /
~205,000 prompt blocks each.
G = mean of g_c over the 16 cells (g_c may be negative; it is not clipped per cell).
reward = clamp(G / 0.40, 0, 1)
Hard-gated to 0: any protocol violation (evicting a non-resident block, admitting a block not in
the current request, exceeding capacity after a step), a crash, exceeding the per-cell CPU budget, a
nondeterministic result (one cell is replayed twice and must match exactly), or a malformed return.

CONSTRAINTS
Interface: policy.py defines class Policy(capacity: int) with step(request: dict, hit_blocks: int) -> (admit: list[int], evict: list[int]).
The request dict has timestamp_ms, input_length, hash_ids; nothing else is provided (no output length).
The policy runs in a separate process fed one request at a time; it cannot see future requests or the graded trace file.
CPU only: 1 core, 4 GB RAM, 600 s CPU time per graded cell (enforced with RLIMIT_CPU, not wall clock).
Python 3 with numpy, scipy, scikit-learn available; no other packages.
Submission directory at most 50 MB, including any fitted artifacts.
The sandbox has internet access. The public source traces give no information about the secret
graded draws beyond what the dev draws show; downloading trace data is still forbidden by rubric.
The policy must be deterministic (fixed seeds) and must not depend on the wall clock.
Capacity is passed to the constructor; the same code must handle every capacity.
