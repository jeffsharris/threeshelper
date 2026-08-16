# J2 exact-teacher engineering feasibility pilot

Status: frozen outcome-free engineering pilot charter. This charter does not
authorize J2 distillation, PPO, validation, development, confirmation,
promotion, or any J2 scientific stream reservation.

## 1. Scope and immutable parents

The pilot answers only whether the exact protected composite software
incumbent can be queried with deterministic eight-process throughput and
round-synchronous orchestration adequate for the already frozen J2 caps.
Human actions and assisted-session content remain forbidden. Pilot states,
queries, and evidence are permanently inadmissible as J2 labels.

The pilot binds the accepted J2 teacher-provenance artifact, J2 readiness
lock/result, protected incumbent policy/config/checkpoint/source identities,
dashboard association with record 263670, and top three
`263670/261369/258561`. Drift fails before inventory or teacher access.

The namespace is
`threes_rl/runs/forensics/j2_exact_teacher_feasibility_pilot_v1`.
No J2 future execution namespace may exist. The public commands are exactly
`audit-zero-work`, `write-test-evidence`, `prepare`, `open`, and `execute`.

## 2. Engineering state authority

The pilot uses 5,000 fresh normal-start whole ancestries. Engineering stream
IDs are:

- deck `250000000000..250000004999`;
- slot `251000000000..251000004999`; and
- exploration policy `252000000000..252000004999`.

These blocks are engineering-only, disjoint from the spent compact
213B-226B authorities and all J2 scientific 227B-249B authorities. They are
never scientific reservations.

For state index `i`, the target prefix is
`16 + ((73*i + 19) mod 160)` moves. Starting with `starter_tile=None`, a
local NumPy generator seeded by the exploration stream chooses uniformly
from legal actions in canonical simulator order. Natural termination before
the target prefix, a missing legal action, or any restoration mismatch fails
the complete inventory without replacement. No root is filtered, replaced,
or selected by score, progression, teacher action, or policy outcome.

During zero-query `prepare`, before `open` and before any teacher import/load
or query path is reachable, the runner must create once and reload an
immutable inventory manifest containing only index, ancestry/root identity,
stream IDs,
target and reached prefix counts, current feature-family name, exact current
state hash, and worker ownership. No trajectory, action history, score,
terminal score, future progression, policy outcome, or human content is
retained. Full current state/context exists only transiently in memory and is
regenerated from the manifest streams after `open`. Every state must be
reachable, nonterminal, legal, unique by state hash, and ancestry-disjoint
from all J2 authorities. The preflight terminal binds the complete inventory
file/payload identity; `open` cannot precede that terminal seal.

The natural feature families are the frozen J2 current-board families
`low_air`, `low_constrained`, `mid_progression`, and `upper_progression`.
Their natural counts/frequencies are reported. No family is a quota or filter.

## 3. Exact teacher workloads

The teacher is loaded through the accepted bound-incumbent loader. Each query
uses a fresh deterministic policy RNG derived from the sealed engineering
policy stream. Actions are transient.

Warmup is exactly eight calls per loaded process. Every eight-process group
completes one explicit 64-call warmup barrier before its first measured
dispatch; every subsequent measured round reports zero warmup calls. Warmup is
excluded from steady-state timing in every mode. Cold process startup,
teacher-load time, and warmup time are reported separately.

Two cost workloads are frozen:

- central: inventory indices `0..511`, exactly 512 unique states;
- sensitivity: inventory indices `0..4999`, exactly 5,000 unique states.

Each workload runs once in one process and once in eight processes on the
identical ordered inventory. Eight-way ownership is `state_index mod 8`.
Workers are single-threaded, do not steal work, and the parent merges only in
state-index order.

Persisted evidence may contain only inventory/state hashes, worker ownership,
whole-workload or whole-256-state-round output digests, exact equality
booleans, timing summaries, counts, crash/tamper status, and resource metrics.
It may not contain an
action, action array, Q/value vector, score, progression outcome, policy
outcome, or trajectory. Per-state and tiny-record output digests are forbidden.
Serial and worker action vectors are zeroized and dropped before any evidence
write.

## 4. Synchronous analogue

Inventory indices `0..4095` form exactly 16 ordered rounds of 256 states.
Eight persistent workers use the same modulo ownership. The parent sends no
round `r+1` work until all round `r` results are received, validated against
the serial reference, canonically merged, and sealed in a compact round
manifest.

Every round manifest binds its input-state digest, worker counts, ordered
output digest, equality boolean, timing/resource summary, and predecessor
manifest. It also binds dispatch, receive, and durable-commit chronology;
round `r+1` dispatch must be strictly later than round `r`'s create-once
commit. Missing, duplicate, late, cross-round, wrong-worker, shuffled,
illegal, or source-drift records fail closed. This is only an orchestration
analogue: no PPO trajectory, reward, return, advantage, student output,
optimizer step, or checkpoint exists.

## 5. Timing, memory, and admission

The pilot reports wall time, child CPU time, calls/second, serial/eight-way
speedup, per-call median/p90/p99/max, process startup/load time, parent peak
RSS, per-worker peak RSS, summed worker peaks, total peak footprint, output
bytes, file count, and disk delta. It separately samples and reports maximum
contemporaneous parent-plus-children RSS; the memory gate uses this sampled
quantity rather than the sum of independent lifetime peaks.

The central pretraining runtime projection is exactly
`1.25 * ((10,240 * 512 * observed_p99_seconds) / (8 * 3600)
+ optimizer_fixture_hours)`. Its maximum admissible p99 is derived by solving
that equation at the 72-hour cap before timing. Aggregate calls/second is
descriptive and a consistency safeguard only; it cannot substitute for the
p99 admission gate. The online projection uses the synchronous observed
per-call/barrier cost for exactly `4,096 * 512` calls plus the inherited J1
bounded-training projection. Both apply the frozen 25% margin and must fit
72 hours. Required calls/second is derived directly from those equations;
there is no post-result speedup floor.

Total peak process footprint must be at most 24 GiB and at most 75% of
preflight-available memory. Output delta must be at most 1 GiB, free disk must
remain above 100 GiB, and the 120 GiB target is required at preflight.
Central projected retained storage must remain within each 24 GiB phase cap.
The preterminal output check reserves a frozen 16 MiB allowance for terminal
and retention evidence. This preterminal delta plus allowance is the sole
conjunctive 1 GiB admission check. Any post-write exact footprint observation
is descriptive and cannot contradict an already sealed terminal decision.

The admission RSS sampler starts immediately after worker process creation and
remains live through source load, warmup, every measured query, and worker
shutdown. Admission uses its maximum contemporaneous parent-plus-children RSS
and requires a nonzero sample count. Per-round receive-window RSS remains
descriptive only.

The 5,000-state timing workload is real. Separately, the frozen 5,000-move
phase sensitivity is recomputed from measured throughput and reported with
the same 25% margin and cap booleans. As in accepted J2, that sensitivity is
diagnostic, not a conjunctive gate.

Real-throughput PASS requires exact serial/eight-way equality, matching
ordered digests, deterministic repeated subset digests, zero worker errors,
the central runtime/storage gates, and memory/disk gates. Synchronous PASS
requires all 16 barriers, exact reference equality, zero malformed records,
the online runtime/storage gates, and memory/disk gates.

## 6. Outcome-free power sizing

The exact accepted J2 eight-stratum common-OR simulation is run for the
predeclared N grid `2048, 3072, 4096, 6144, 8192`. Every N uses the unchanged
control-rate cells `0.02, 0.04, 0.08, 0.15`, couplings `0, 0.05, 0.10`,
point gate `>=0.90`, lower-95 gate `>0.50`, 0.5 edge correction, 768 datasets,
199 within-stratum whole-root bootstraps, frozen seed derivation, and linear
quantiles. It reports worst-case power and MCSE and the smallest grid N with
power at least 0.80. This can recommend only a later J2 amendment.

## 7. Durability and failure policy

Before real calls, source/tests and immutable test evidence must pass, then a
zero-query preflight lock/result and zero-query marker must seal. The inventory
must seal before the first teacher call. All immutable JSON uses create-once
exact-byte writes and self hashes.

One top-level job runs at nice 10 with eight bounded child workers, no other
heavy job, deterministic process environment, and healthy ports 8765/8770,
advisor, dashboard, and top three. Human sessions stay opaque. Worker crashes
or partial scientific-looking output do not trigger retries.

Tests cover worker crash, missing/duplicate/shuffled/late/cross-round records,
source and inventory drift, output-digest tamper, illegal action,
non-reachable state, family reporting, label non-retention, restart identity,
create-once behavior, and one-heavy-job enforcement. Synthetic tests never
count as real throughput or synchronous evidence.

The terminal decisions are:

- `READY_J2_FEASIBILITY_AMENDMENT_PREFLIGHT` only when throughput,
  synchronous orchestration, power sizing, integrity, retention, and
  operations pass. It authorizes only research-lead review of a separately
  frozen J2 readiness amendment.
- `HOLD_J2_TEACHER_ENGINEERING_FEASIBILITY` for inadequate cost, memory,
  storage, power, or an operational stop without evidence corruption.
- `KILL_J2_TEACHER_PILOT_INTEGRITY` for teacher drift, equality failure,
  malformed orchestration, retained forbidden data, or immutable evidence
  corruption. This kills this pilot implementation, not the J2 hypothesis.

No pilot terminal authorizes J2 labels, distillation, validation, PPO,
development, confirmation, promotion, incumbent change, or dashboard change.
