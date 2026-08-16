# G1-R Quality-Diversity Acquisition Family Proposal

Date drafted: 2026-07-25

Status: proposal only. Execution is not authorized. This document does not
authorize implementation, an action-signature admission run, normal-start
generation, labels, fitting, policy evaluation, or dashboard changes.

## Trigger and Scientific Boundary

The immutable G1-R pilot-v1 preflight stopped before game generation because
six nominal policy specifications formed only four genuine action families:

1. `corner2`;
2. hand-built `expectimax2`;
3. parent MC1000 / student1 / full incumbent as one alias component;
4. replay-calibration n-tuple depth 2.

The frozen `2%` overall and nonzero-in-each-stratum action-disagreement rule is
unchanged. The remedy must add a materially different behavior generator, not
rename a checkpoint, vary a temperature, split one policy by seed, or count
descendants of one reset as independent roots.

## Proposed Family: Static-Archive QD One-Ply

The proposed family is `g1r_qd_static_archive_oneply_v1`.

It plays complete simulator-valid games from a fresh normal reset. It never
starts from, splices, or continues a retained state. Each completed reset
contributes at most one G1 state and one ancestry, even if it visits both exact
rungs.

### Frozen natural-state archive

Before implementation, construct one root-capped static archive from the
already excluded A2 natural-state inventory. The archive is diagnostic and
policy-context input only; it contributes no G1 root. Select at most one state
per old ancestry by deterministic SHA argmin, without source role, future
milestone, score, final outcome, incumbent action, or human action.

Before extracting the descriptor, make a working board and remove exactly one
free starter tile with value `1536`: remove `(0,0)` when it contains `1536`;
otherwise remove the row-major-smallest `1536`; if none exists, use no removal
and set starter index to missing. Row-major index is `4*r+c`.

Positive working-board cells are sorted by `(tile value descending, row-major
index ascending)`. The first is the built-max cell; the second is the
second-largest built cell and may have the same value as the first. This makes
duplicate targets explicit. When fewer than one/two positive built cells
exist, the corresponding cell index is `16`. The built-max value is `0` when
absent. Manhattan distance is `|r1-r2|+|c1-c2|`; it is `6` when either cell is
missing.

Each archived state maps to this ordered descriptor:

1. built-max band excluding the fixed starter: `<384`, `384`, `768`, `1536`,
   `3072+`;
2. built-max cell index `0..15`;
3. second-largest built tile cell index `0..15`;
4. Manhattan distance between those two cells, with `6` when absent;
5. equal-support connected-component count, capped at `4`;
6. target/support adjacency-edge count, capped at `4`;
7. empty-count bin `0`, `1`, `2`, `3`, `4+`;
8. legal-action count `1..4`;
9. top-row monotonic-violation count, capped at `3`;
10. left-column monotonic-violation count, capped at `3`;
11. fixed-starter cell index `0..15`, with `16` when absent;
12. anchor-integrity bit;
13. visible-preview kind `blue/red/gray/bonus`;
14. large-pending bit.

Definitions are exact:

- Built-max band uses value `0..383`, `384..767`, `768..1535`,
  `1536..3071`, or `3072+`.
- Support levels are the exact positive powers-of-two ladder
  `{built_max/2, built_max/4, built_max/8}` whose values are integers `>=3`.
  For each level independently, count 4-neighbor (up/down/left/right)
  connected components of cells equal to that level; sum levels and cap at 4.
  Diagonals never connect.
- Target/support adjacency counts undirected 4-neighbor edges with one endpoint
  equal to built max and the other equal to any support level above. Count each
  board edge once and cap at 4. The removed free starter is never a target.
- Convert board values to Threes ranks with `0 -> 0`, `1/2 -> 1`, `3 -> 2`,
  and each doubling adding one. Top-row violations are
  `sum(c=0..2, 1[rank(0,c) < rank(0,c+1)])`; left-column violations are
  `sum(r=0..2, 1[rank(r,0) < rank(r+1,0)])`. Zeros participate as rank zero.
- Anchor integrity is one exactly when the removed free starter was at `(0,0)`
  and both monotonic-violation counts are zero; otherwise it is zero.
- Preview category is exactly the simulator `Preview.kind` in ordered set
  `blue, red, gray, bonus`; any other/missing value is a hard schema error.
- Large pending is exactly `int(bool(state.large_pending))`. There is no
  probability threshold or inferred pending state.

The archive cell key is the exact 14-value descriptor tuple. Each selected old
root contributes one count to exactly one cell; duplicate descriptors from
different roots increase that integer count, while copied frames/replays never
do. Empty cells have count zero. Freeze the ordered descriptor schema, missing
conventions, source manifest, complete sorted `(cell_key,count)` table, and
archive SHA before any policy action is evaluated. The archive never updates
from generated games, so the behavior family remains fixed.

Distance is a fixed mixed metric. Categorical coordinates are built-max band,
built-max cell, second-largest cell, starter cell, anchor bit, preview
category, and pending bit; each contributes Hamming distance `0` or `1`.
Ordered coordinates are Manhattan distance, component count, adjacency count,
empty bin, legal count, top violations, and left violations; they contribute
absolute difference divided respectively by `6,4,4,4,3,3,3`. The total is the
unweighted sum of all 14 contributions, in `[0,14]`. Numeric subtraction is
never applied to a cell index, band label, preview label, or bit.

For a descriptor not present in the archive, nearest-cell distance is the
minimum mixed distance over occupied cells. Exact distance ties choose the
lexicographically smallest 14-value cell key using each categorical order
listed above and natural ascending order for integers. An empty archive is a
hard preflight error. Exact-cell occupancy always uses the candidate
descriptor's own count, not the nearest cell's count.

### Frozen action objective

At each decision:

1. enumerate every legal base move;
2. enumerate exact visible-preview value and legal insertion-slot outcomes
   using current simulator probabilities, without a second player decision;
3. for each outcome, compute the frozen parent-MC1000 afterstate value and the
   descriptor above;
4. define `quality(a)` as the exact expected parent afterstate value;
5. for each outcome descriptor `x`, define
   `novelty(x) = 1/(1+archive_count[x]) + nearest_distance(x)/14`, and define
   `novelty(a)` as its exact spawn-probability-weighted expectation;
6. convert quality and novelty to ordinal ranks among legal actions, rank `0`
   best;
7. select the action minimizing `quality_rank + novelty_rank`;
8. break ties by higher quality, then higher novelty, then simulator action
   order `up, down, left, right`.

Equal rank weight is part of the algorithm, not a tunable coefficient. There
is no rollout, learned bonus, score target, support-rank sweep, temperature,
human label, or cross-game archive adaptation.

## Deterministic Stream Contract

If later authorized, reserve a new collision-audited namespace:

- logical/root: `45_000_000_000 + game`;
- deck: `46_000_000_000 + game`;
- slot: `47_000_000_000 + game`;
- policy/tie: `48_000_000_000 + game`.

All streams use `split_exogenous_v1`. The policy is deterministic, but the
policy stream remains recorded for evaluator compatibility. The complete
manifest, historical collision union, archive hash, implementation hash,
parent checkpoint payload hashes, and output directory must be immutable
before generation.

## Outcome-Free Admission Gate

Implementation, if separately authorized, stops before generation unless all
of these pass:

- exact replay/state round trip and no mutation;
- exact action determinism and save/reload-equivalent archive;
- the same frozen `32 pre1536 + 32 pre3072` panel used by pilot-v1;
- pairwise action disagreement against each of the four admitted components is
  at least `2%` overall and nonzero in both strata;
- the signature is unique and does not transitively alias an existing
  component;
- action-only runtime passes both the absolute and relative frozen timing gate
  below;
- compact projected storage remains below `4 GiB` for the first 120 games and
  free disk remains above `120 GiB`;
- ports `8765/8770`, dashboard record `263670`, protected top-three replays,
  policy payload hashes, and all stream collisions pass.

No generated trajectory or final score is inspected at this gate.

### Frozen action-latency schedule

Use the same 64-state admission panel in its frozen order. Load both policies
before timing. For every state, call the QD candidate and incumbent once
untimed to populate stable read-only model pages, then clear per-action search
caches. Run five timed passes. Within each pass, alternate call order by
`(pass_index + state_index) mod 2`; clear the policy's documented transient
search caches immediately before each timed call. Measure only one complete
action decision with `time.perf_counter_ns`; checkpoint loading, panel loading,
and JSON serialization are outside the interval. Pin one process, no worker
pool, nice priority at least 10, and record thermal/power state when available.

Candidate latency must satisfy all absolute limits:

- median `<=75 ms`;
- p90 `<=150 ms`;
- p99 `<=250 ms`;
- maximum `<=500 ms`.

It must also satisfy median `<=1.0x` and p90 `<=1.5x` of the interleaved
incumbent depth-2 calls. Absolute limits are primary, so an unusually cheap
incumbent denominator cannot recreate the C1 ratio ambiguity. Any action
mismatch across the five candidate repeats is a determinism failure.

## Conditional Acquisition Gate

Only a separately authorized PASS of the admission gate may open a pilot:

- exactly 20 complete fresh-root games for the QD family and 20 for each of the
  four previously admitted representatives, with one or two nice-priority
  workers;
- at most one selected exact-rung state per ancestry;
- compact results and exact-rung source replays only;
- no all-action h40 labels, model fit, policy comparison, or dashboard claim.

The existing final requirement remains `256 train + 96 validation + 512 test`
roots, with exact stratum targets, ten roots in every stratum-role cell per
partition, at least five genuine families overall, at least three test
families, and no family above `40%`. Five equal-capable families make the family
cap algebraically feasible at `20%` each; this is only an outcome-free
feasibility argument, not evidence that the QD family reaches either rung.
Actual family-by-stratum yields and the deterministic allocator decide whether
acquisition continues.

After every complete pilot or balanced acquisition round, compute this frozen
budget-feasibility projection using acquisition yields only:

1. For each genuine family `f`, let `n_f` be completed fresh roots and let
   `k_f,1536`, `k_f,3072`, and `k_f,any` be root-capped counts reaching the
   exact pre1536 rung, exact pre3072 rung, or either rung.
2. For each count use the 90% Wilson lower bound with
   `z=1.6448536269514722`:
   `center=(p+z^2/(2n))/(1+z^2/n)` and
   `half=z*sqrt(p*(1-p)/n+z^2/(4n^2))/(1+z^2/n)`;
   `L=max(0,center-half)`, with `L=0` when `n=0`.
3. Let `B=12000-sum_f(n_f)`. Allocate `floor(B/F)` projected attempts to
   every admitted family and the first `B mod F` residual attempts in frozen
   family order.
4. Project each family/stratum as
   `k_f,s + floor(projected_attempts_f * L_f,s)`. Sum families.
5. Feasibility requires projected pre1536 roots `>=432`, projected pre3072
   roots `>=432`, and projected any-rung unique roots `>=864`. Report every
   family rate, Wilson bound, attempt allocation, and contribution.

This projection does not establish role-cell or family-cap feasibility and
cannot produce READY; the actual deterministic allocator remains authoritative.
It only prevents spending the 12,000-game budget when even conservative
stratum/union counts cannot reach the frozen corpus. A failing projection yields
`HOLD_QD_YIELD_PROJECTION`; its confidence level or formula may not be tuned.

## Runtime and Kill Rules

- Admission implementation plus action panel: at most 4 wall-hours and
  `512 MiB`.
- Conditional pilot: at most 12 active wall-hours and `4 GiB`.
- One heavy process, one or two frozen workers, nice priority at least `10`.
- `KILL_QD_ALIAS` if action distinctness fails.
- `KILL_QD_COST` if the action-only runtime gate fails.
- `HOLD_QD_RUNG_SCARCITY` if, after 200 complete QD roots, it contributes no
  exact pre1536 state.
- If it contributes no exact pre3072 state after 200 roots, mark it unable to
  supply pre3072, matching the parent charter. Continue only if at least three
  other genuine pre3072 suppliers remain and the frozen yield projection plus
  actual allocator can still satisfy the test-family and 40% caps; otherwise
  `HOLD_QD_RUNG_SCARCITY`.
- `HOLD_QD_YIELD_PROJECTION` whenever the frozen 12,000-game conservative
  projection fails for pre1536, pre3072, or any-rung unique roots.
- `HOLD_QD_RUNG_SCARCITY` if the five-family deterministic partition remains
  infeasible at the 12,000-game limit despite a prior projection pass.
- No objective, descriptor, rank weight, archive source, threshold, or stream
  may be tuned from generated yields.

## Ranked Alternatives if QD Is Not Admitted

1. **Compiled exact one-ply parent actor.** Cheap and likely behaviorally
   distinct from depth-2 parent, but it reuses the same value representation
   and may not add enough source diversity.
2. **Exact hand-built corner depth 3.** More likely to reach high rungs and
   behaviorally differ, but prior search evidence makes its runtime cost
   unattractive for thousands of acquisition games.
3. **Stop self-generated family acquisition and request genuinely independent
   human or external-policy roots.** Methodologically strongest diversity, but
   it violates the current preference to exhaust self-learning first and
   depends on external input.

The proposed static-archive family ranks first because it is deterministic,
cheap, source-independent at generation time, and structurally different from
all four admitted policies without claiming to improve gameplay.
