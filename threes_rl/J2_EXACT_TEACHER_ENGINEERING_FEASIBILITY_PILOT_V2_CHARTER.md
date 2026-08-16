# J2 exact-teacher engineering feasibility pilot V2

Status: frozen outcome-free source/test charter. V2 `prepare`, `open`, and
`execute` remain on HOLD pending a separate research-lead authorization. This
charter does not authorize teacher loading or queries, J2 labels, distillation,
PPO, validation, development, confirmation, promotion, or scientific stream
reservation.

## 1. Scope and immutable history

V2 asks only whether the exact protected composite software incumbent can be
queried with deterministic eight-process throughput and round-synchronous
orchestration adequate for the accepted J2 caps. Human actions and assisted
session content remain forbidden. Pilot states, queries, and evidence are
permanently inadmissible as J2 labels.

The V1 pilot is an authoritative outcome-free HOLD because its frozen prefix
survival assumption was not total over natural normal-start ancestries. V2
binds and preserves these V1 identities byte-for-byte:

- charter `fb61f4cea16ecea6bc30f24f42b2b6a5ed483e3172428a2a20dc66bbc16a0de9`;
- runner `57264251bd49e78e4608c5a8adfb91c9a14bf524327a68f88104a9a541d14970`;
- tests `1754c8d89df5bd75541e6d302a1e905918e2c33f1022620a65f41eed4f1a9a85`;
- test evidence file
  `ec9235912266c0e18e7596d08cdca4468f2605888c6a6de1c165224c17e8aee3`
  and payload
  `d960d3a904572ffa2f2ea8df645e890c6048af399e7d6cdf32d33e1919691658`.

The V1 namespace must contain exactly that evidence file and no inventory,
lock, result, marker, terminal, retention, teacher load, or teacher query.
V2 does not edit, retry, reinterpret, or reuse any V1 ancestry or stream.

V2 also binds the accepted J2 teacher provenance, readiness lock/result,
protected incumbent policy/config/checkpoint/source identities, dashboard
association with record 263670, and top three
`263670/261369/258561`. Drift fails before V2 state generation or teacher
access.

The namespace is
`threes_rl/runs/forensics/j2_exact_teacher_feasibility_pilot_v2`.
The public commands are exactly `audit-zero-work`, `write-test-evidence`,
`prepare`, `open`, and `execute`. This source/test turn may invoke only the
first two.

## 2. Total engineering state authority

V2 has exactly 5,000 fresh normal-start whole ancestries. Engineering stream
IDs are:

- deck `253000000000..253000004999`;
- slot `254000000000..254000004999`; and
- exploration policy `255000000000..255000004999`.

These blocks are engineering-only and disjoint from all spent 213B-226B
authorities, J2 scientific 227B-249B authorities, and V1 250B-252B
engineering authorities. They are never scientific reservations.

For state index `i`, the target prefix is
`16 + ((73*i + 19) mod 160)` moves. Starting with `starter_tile=None`, a
local NumPy generator seeded by the exploration stream chooses uniformly
from legal actions in canonical simulator order.

Every root contributes exactly one live, naturally reachable, feature-only
state:

1. The initial normal-start state is the current retained live state.
2. A legal exploration move is applied through the exact simulator.
3. If the successor remains live and legal, it becomes the current retained
   state and increments `realized_prefix_steps`.
4. If the successor is terminal or has no legal action before the target is
   reached, the successor is discarded and the current live pre-terminal
   state is retained. `prefix_clamped` is true.
5. Otherwise the exact target-prefix live state is retained and
   `prefix_clamped` is false.

The manifest records target prefix, realized prefix, and the clamp boolean,
but no terminal state, termination action, action history, reason, score,
progression, future outcome, or policy outcome. A malformed initial state,
illegal reported move, restoration mismatch, duplicate state hash, duplicate
root/ancestry, or count mismatch fails closed. Natural early termination does
not fail and never causes replacement, dropping, retry, quota filling,
survival conditioning, or adaptive allocation.

Before any teacher import/load/query, zero-query `prepare` must create once,
reload, and hash-verify the complete 5,000-row inventory. It contains only
index, root/ancestry identity, stream IDs, worker ownership, target and
realized prefix counts, clamp boolean, feature-family name, exact current
state hash, and legal-action count. Full current state/context exists only
transiently and is regenerated exactly from sealed streams after `open`.

Target-prefix bands are fixed as `16-55`, `56-95`, `96-135`, and `136-175`.
The inventory reports natural counts and clamp counts/rates by target-prefix
band, by frozen J2 current-board feature family, and by the band-by-family
cross-tab. The families are `low_air`, `low_constrained`,
`mid_progression`, and `upper_progression`. These summaries expose
representativeness; no band or family is a quota, filter, or pass gate.

## 3. Exact teacher workloads

The teacher is the exact accepted composite protected incumbent and is loaded
only after a separately authorized `open`. Each query uses a deterministic
policy RNG derived from the sealed engineering policy stream. Actions remain
transient.

Warmup is exactly eight calls per loaded process. Every eight-process group
completes one explicit 64-call warmup barrier before measured dispatch;
subsequent measured rounds report zero warmups. Warmup is excluded from
steady timing. Cold process startup, teacher load, and warmup are reported
separately.

The two unchanged cost workloads are:

- central: inventory indices `0..511`, exactly 512 states;
- sensitivity: inventory indices `0..4999`, exactly 5,000 states.

Each runs once in one process and once in eight processes on the identical
ordered inventory. Eight-way ownership is `state_index mod 8`; there is no
work stealing. Merge order is canonical state-index order.

Persisted evidence may contain inventory/state hashes, worker ownership,
whole-workload or whole-256-state-round output digests, equality booleans,
timing summaries, counts, crash/tamper status, and resource metrics. It may
not contain an action, action array, Q/value vector, score, progression
outcome, policy outcome, or trajectory. Per-state and tiny-record output
digests are forbidden. Transient action vectors are zeroized and dropped
before writes.

## 4. Synchronous analogue

Inventory indices `0..4095` form exactly 16 ordered rounds of 256 states.
Eight persistent workers use modulo ownership. Round `r+1` cannot dispatch
until every round `r` result is received, checked against the transient serial
reference, canonically merged, and committed create-once.

Each compact round manifest binds input digest, worker counts, aggregate
output digest, equality, timing/resource summary, predecessor, and dispatch,
receive, and durable-commit chronology. Missing, duplicate, late,
cross-round, wrong-worker, shuffled, illegal, or source-drift records fail
closed. This is an orchestration analogue only: no PPO trajectory, reward,
return, advantage, student output, optimizer step, or checkpoint exists.

## 5. Timing, memory, and admission

V2 preserves the V1 timing and admission contract exactly. It reports wall
and child CPU time, calls/second, serial/eight-way speedup, per-call
median/p90/p99/max, startup/load/warmup, parent peak RSS, per-worker peak RSS,
summed worker peaks, maximum contemporaneous parent-plus-live-children RSS,
output bytes/files, and disk delta.

The central pretraining projection is exactly
`1.25 * ((10,240 * 512 * observed_p99_seconds) / (8 * 3600)
+ optimizer_fixture_hours)`. The maximum admissible p99 is solved before
timing. Aggregate throughput is descriptive only. The online projection uses
the measured 16-round barrier cost for exactly `4,096 * 512` calls plus the
inherited bounded-training projection. Both must fit 72 active hours.

Peak process footprint must be at most 24 GiB and at most 75% of preflight
available memory. Output delta must be at most 1 GiB; free disk must remain
above 100 GiB and the 120 GiB target is required at preflight. A frozen
16 MiB terminal/retention allowance is included in the sole preterminal hard
output check. No fallible conjunctive cap may run after terminal sealing.

The admission RSS sampler spans worker creation, source load, warmup, measured
queries, and shutdown, and must have nonzero samples. The 5,000-move
sensitivity keeps the same 25% margin and is diagnostic, not conjunctive.

## 6. Outcome-free power sizing

The exact accepted J2 eight-stratum common-OR implementation is unchanged
except for the predeclared N grid `2048, 3072, 4096, 6144, 8192`. Control
rates, couplings, gates, 0.5 correction, 768 datasets, 199 within-stratum
whole-root bootstraps, seed derivation, and linear quantiles remain exact.
The result reports worst-case power, MCSE, and the smallest grid N reaching
0.80. It may recommend only a later J2 amendment and cannot alter existing
authority.

## 7. Durability, tests, and decisions

Source/tests and immutable test evidence must pass before V2 `prepare`.
Inventory, preflight lock/result, and marker are separate create-once,
self-hashed boundaries. This turn must stop with no inventory, lock, result,
marker, teacher load/query, label, game, outcome, or scientific reservation.

One top-level job runs at nice 10 with eight bounded children, no other heavy
job, deterministic process environment, healthy ports 8765/8770, advisor,
dashboard, and top three. Human sessions remain opaque.

Tests cover total early-termination clamping, all-root retention, zero
replacement, zero survival conditioning, exact clamp accounting, inventory
regeneration, source/inventory drift, transient action retention, worker and
barrier failure modes, cost/power contracts, create-once behavior, terminal
precedence, and the impossibility of a post-terminal failure leaving READY.
Synthetic fixtures cannot satisfy real throughput evidence.

Terminal decisions remain:

- `READY_J2_FEASIBILITY_AMENDMENT_PREFLIGHT_V2` only when all throughput,
  orchestration, power, integrity, retention, and operational gates pass. It
  authorizes only research-lead review of a separate J2 amendment.
- `HOLD_J2_TEACHER_ENGINEERING_FEASIBILITY_V2` for inadequate cost, memory,
  storage, power, or an operational stop without evidence corruption.
- `KILL_J2_TEACHER_PILOT_V2_INTEGRITY` for teacher drift, equality failure,
  malformed orchestration, forbidden retained data, or immutable corruption.

V1 remains HOLD. V2 cannot promote, alter the incumbent/dashboard, or
authorize J2 scientific work.
