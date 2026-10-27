PROBLEM
LLM serving engines reuse the KV cache of a shared prompt prefix across requests. The cache is
split into 512-token blocks, and a request can only reuse a block if every block before it in its
prefix is also cached, so a hit is the longest fully cached prefix. GPU memory holds only a few
thousand blocks, so the engine must decide which blocks to admit and which to evict.

You are given real one-hour production request traces from a commercial LLM service (the public
Mooncake FAST'25 release: a chat workload and a tool/agent workload). Each request has an arrival
timestamp (3 s granularity), input and output token counts, and the chain of block hashes of its
prompt; equal hashes mean equal prefixes. No text is included. You see the first 30 minutes of
each trace. Grading replays the remaining ~29 minutes of both traces, with block ids consistently
relabelled and timestamps rebased, at four cache capacities per trace (about 460 to 3,750 blocks).

Write an online policy. For each request, in arrival order, the simulator first computes the hit
against the current cache, then calls your policy with the request and its hit length. The policy
returns which of this request's blocks to admit and which resident blocks to evict. The simulator
owns the cache and enforces the capacity, and the policy never sees future requests. The policy
may carry state across requests and may ship artifacts (for example a model fitted on the visible
30 minutes).

REWARD
For each graded cell c (2 traces x 4 capacities):
  h_c   = hit blocks / total prompt blocks under the submitted policy
  L_c   = the same quantity under LRU (vLLM default; deeper blocks of a request evicted first)
  B_c   = the same quantity under Belady's offline optimum (furthest next use, bypass allowed)
  g_c   = (h_c - L_c) / (B_c - L_c)
L_c and B_c are recomputed by the verifier at grading time on the same graded segment.
G = mean of g_c over the 8 cells (g_c may be negative; it is not clipped per cell).
reward = clamp(G / 0.40, 0, 1)
Hard-gated to 0: any protocol violation (evicting a non-resident block, admitting a block not in
the current request, exceeding capacity after a step), a crash, a per-cell timeout, a
nondeterministic result (one cell is replayed twice and must match exactly), or a malformed return.

CONSTRAINTS
Interface: policy.py defines class Policy(capacity: int) with step(request: dict, hit_blocks: int) -> (admit: list[int], evict: list[int]).
The request dict has timestamp_ms, input_length, output_length, hash_ids; nothing else is provided.
The policy runs in a separate process fed one request at a time; it cannot see future requests or the graded trace file.
CPU only: 1 core, 4 GB RAM, 300 s wall clock per graded cell, no network.
Python 3 with numpy, scipy, scikit-learn available; no other packages.
Submission directory at most 50 MB, including any fitted artifacts.
The policy must be deterministic (fixed seeds) and must not depend on the wall clock.
Capacity is passed to the constructor; the same code must handle every capacity.
