PROBLEM
You design the rules of an allocation, and strategic parties play under them.

Six participants share nine divisible resources. Each resource has one unit of capacity,
which may be split arbitrarily. Participant i has a private type with three parts: a weight
vector W_i over the nine resources (non-negative, sums to 1), a scale v_i, and a curvature
exponent a_i drawn from {0.5, 1.0, 2.0}. Its true utility from a share vector x_i is

    u_i(x_i) = v_i * (W_i . x_i) ** a_i

so a_i = 0.5 is a participant with diminishing returns, a_i = 1.0 is additive, and a_i = 2.0
is a participant that needs concentration before the allocation is worth anything. The
curvature is part of the private type and is never revealed.

You submit a single deterministic function

    allocate(reports) -> X

where reports is a 6x9 array of claimed weight vectors and X is a 6x9 non-negative
allocation whose columns each sum to 1. You never see true types, only reports.

Participants do not report truthfully. Each one is driven by a frozen deterministic
best-response solver that searches the report space for the claim maximising its own true
utility under your submitted rule, holding the other participants' current reports fixed.
The solvers are iterated to a fixed point, so what your rule is scored on is the allocation
produced at the resulting equilibrium of reports, not at the truthful profile.

There is no money. You may not charge, transfer, or refund anything; the only instrument is
how capacity is divided as a function of the claims. Participants will exaggerate, and any
rule that simply maximises welfare on the reports it is handed invites them to.

Instances are drawn fresh at verify time from the same private generator and are not
available to you.

REWARD
reward = clamp((realised_welfare - 0.7585) / (0.9090 - 0.7585), 0, 1)

where realised_welfare is the utilitarian welfare of your rule's allocation computed on the
PARTICIPANTS' TRUE TYPES at the best-response equilibrium, divided by the first-best welfare
of the same instance, then averaged over the verify instances.

0.7585 is the measured score of the strongest free published rule under the same solver, a
random-priority serial dictatorship, recomputed at verify time against the same instances.
0.9090 is the score of utilitarian-maximisation-on-reports when the same participants report
truthfully, also recomputed at verify time; it is the same allocation problem with the
strategic constraint removed, and no rule facing strategic reporters is expected to reach it.

validity_gate: reward = 0 for the whole run if the submitted rule returns an array of the
wrong shape, returns any negative entry, returns any column that does not sum to 1 within
1e-6, raises, exceeds its per-call time limit, is non-deterministic across two calls on
identical reports, or inspects anything outside its reports argument. The gate is evaluated
before any welfare is computed and is never folded into the score as a penalty.

CONSTRAINTS
No transfers of any kind: no payments, no numeraire, no artificial currency that is
conserved, scored, or carried between instances.
The rule must be a pure deterministic function of its reports argument. No randomness
without a seed derived from the reports, no clocks, no files, no global state, no state
carried between instances.
The rule may not attempt to read true types, solver internals, or any verifier object.
Anonymity: relabelling the participants must relabel the allocation the same way.
Per-call time limit enforced by the harness; the rule is called once per best-response
iteration, so a slow rule fails the time gate rather than scoring poorly.
