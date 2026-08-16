# G2 Scale-Equivariant Relational Hazard Proposal

Date frozen: 2026-07-25

Status: authoritative outcome-free proposal. G2 corpus statistics, labels,
rollouts, model fits, candidate actions, and policy outcomes may not be opened
before this file is frozen and hashed.

## Decision Context

The sealed G1-R QD5 pilot remains
`HOLD_G1R_AFTER_PILOT_V2_QD5_SEAL`. Its ancestry-unique Wilson projection
passes pre1536 availability but projects only `27` pre3072 roots against the
required `432`. Routine continuation of that exact two-rung acquisition design
is therefore `KILL_G1R_EXACT_RUNG_CONTINUATION`.

This is an acquisition-design decision, not a policy failure and not evidence
against relational hazard modeling. G2 asks a materially different question:

> Can one scale-equivariant representation learn the local support transition
> at target scale T from naturally abundant earlier rungs and transfer upward
> to the untouched 1536-to-3072 transition?

G2 does not reinterpret R1b, H0, A2, R2a, C1, QD-v1, S3, or the QD5 pilot.
QD-v2 remains an acquisition diagnostic family, not an incumbent.

## Scientific Contract

The normalized transitions are:

- `pre768`: built maximum excluding the fixed starter is exactly `384`;
  target scale `T=768`.
- `pre1536`: built maximum excluding the fixed starter is exactly `768`;
  target scale `T=1536`.
- `pre3072_transfer`: built maximum excluding the fixed starter is exactly
  `1536`; target scale `T=3072`.

`pre768` and `pre1536` may supply training and development evidence.
`pre3072_transfer` is never pooled into fitting, normalization, calibration,
feature selection, or threshold choice. It is untouched upward-transfer
evidence only. All states from one canonical fresh-root ancestry remain in one
partition, even when that replay visits multiple scales.

Human, assisted-human, partial, restart, continuation, replay-start,
synthetic, and imported states are forbidden from ordinary G2 evidence.
Human actions are never labels.

## Frozen Model

G2 has one primary model family:

- grouped-binomial discrete-time logistic hazard;
- one row for each root/action/horizon interval ending at h10, h20, or h40;
- event likelihood only while the trajectory remains at risk;
- terminal trajectories without the target are right-censored at terminal;
- ancestry-equal loss weight, split equally across legal actions and at-risk
  interval rows;
- deterministic train-only standardization for columns marked continuous;
- unpenalized intercept and horizon indicators;
- L2 penalty `lambda=1.0` on every other coefficient;
- deterministic L-BFGS, maximum `500` iterations, gradient tolerance `1e-8`;
- one fixed Platt calibration fit on development roots only, with slope
  constrained positive and no regularization;
- no architecture, feature, penalty, seed, checkpoint, or calibrator sweep.

The primary endpoint is stage-appropriate milestone event/censoring likelihood.
Score is not a target. Survival, anchor integrity, and geometry are later
causal safeguards, not weighted reward terms.

If labels are later authorized, every legal first action is forced and then
the frozen incumbent depth-2 policy continues. One h40 CRN path provides h10,
h20, and h40 event/censoring readouts. Exactly eight replicates per root/action
are split into descriptive stream blocks `A=4` and `B=4`. Shared deck and slot
uniform streams are mapped independently over each arm's valid insertion
positions. No recorded action is a target.

## Scale And Orientation Contract

For a tile `v >= 3`, `level(v)=log2(v/3)` and target-relative rank is
`clip(level(v)-level(T), -6, +1)`. Tiles `1` and `2` use fixed small-tile
sentinels below `-6`; empty cells are absent, not rank zero. The fixed starter
cell is excluded from target-relative rank summaries and support graphs but is
retained as an anchor-position relation.

All support identities are relative to `T`:

- parent: `T/2`;
- child: `T/4`;
- grandchild: `T/8`.

The feature calculation uses the deterministic post-swipe, pre-spawn
afterstate. It consumes neither deck nor slot RNG. The only board symmetry
compatible with the fixed top-left starter is main-diagonal transpose.
Canonical orientation is the lexicographically smaller of:

1. normalized board plus action under identity; and
2. normalized transposed board plus action mapped
   `up<->left`, `down<->right`.

Ties choose identity. Feature vectors must be exactly invariant under applying
that transpose/action map and under multiplying every non-starter tile,
preview value, preview candidate, and `T` by two on crafted valid fixtures.
Canonicalization never mutates the input state.

Pair selection is lexicographic over
`(Chebyshev distance, Manhattan distance, row1, col1, row2, col2)` after sorting
the two coordinates. Missing pairs use distance `1`, false indicators, and
zero counts after normalization. Graph connectivity uses four-neighbor edges;
diagonal touch is reported separately with eight-neighbor-minus-four-neighbor
edges. Line blockers count occupied cells strictly between aligned endpoints.

## Frozen 64-Column Schema

The implementation emits this exact ordered schema, formulas, bounds,
missing conventions, and normalization mask in a hashed manifest. All outputs
must be unique, finite, and in `[0,1]`.

Columns `0..23` are horizon, canonical action, global afterstate, and compact
preview/deck context:

`h10`, `h20`, `h40`, `action_up`, `action_down`, `action_left`,
`action_right`, `empty_fraction`, `legal_mobility_fraction`,
`moved_cell_fraction`, `merge_count_fraction`, `insertion_lane_fraction`,
`preview_is_small`, `preview_is_bonus`, `preview_relative_rank`,
`preview_candidate_fraction`, `p_plus_next`, `large_pending`,
`distance_to_forced_plus`, `small_bag_entropy`, `small_bag_position`,
`span_position`, `parent_count_fraction`, `child_count_fraction`.

Columns `24..63` are target-relative relational support geometry:

`grandchild_count_fraction`, `parent_pair_exists`,
`parent_pair_chebyshev`, `parent_pair_manhattan`,
`parent_pair_same_row`, `parent_pair_same_column`,
`parent_pair_diagonal_touch`, `parent_pair_line_clear`,
`parent_pair_blockers`, `parent_pair_action_aligned`,
`parent_pair_min_anchor_distance`, `parent_pair_max_anchor_distance`,
`parent_pair_axis_imbalance`, `support_graph_node_fraction`,
`support_graph_component_fraction`, `support_graph_edge4_fraction`,
`support_graph_diagonal_edge_fraction`, `support_parent_adj4_fraction`,
`support_parent_adj8_fraction`, `parent_neighborhood_occupied_fraction`,
`parent_neighborhood_support_fraction`, `top_row_monotonic_violations`,
`left_column_monotonic_violations`, `anchor_integrity`,
`high_tile_displacement_fraction`, `insertion_near_parent_fraction`,
`insertion_near_support_fraction`, `insertion_anchor_distance`,
`clear_merge_path_fraction`, `blocked_merge_path_fraction`,
`support_ladder_gap`, `max_relative_rank`, `second_relative_rank`,
`largest_parent_component_fraction`, `largest_support_component_fraction`,
`board_relative_rank_mean`, `board_relative_rank_spread`,
`parent_orientation_faces_anchor`, `support_mass_fraction`,
`legal_mobility_delta`.

The implementation manifest is authoritative for exact arithmetic. Changing a
name, order, formula, bound, missing convention, normalization bit,
canonicalization rule, or schema version changes the schema hash and
invalidates the preflight.

## Outcome-Free Corpus Contract

The preflight inventories only existing completed normal-start machine replays
from the retained A2 source catalog. It may inspect replay completion,
provenance, board/context state, and file metadata, but may not inspect or
report final score, per-game score, recorded action quality, future milestone
outcome, or favorable trajectory rank.

For every source:

- bind path, byte size, and SHA-256;
- require genuine fresh replay and root origins plus reset invariants;
- exclude human and all non-natural origins;
- deduplicate replay copies and aliases by canonical whole ancestry;
- restore board, preview, tile-cycle counters, pending state, starter, move
  count, and legal actions exactly;
- select at most one state per ancestry per normalized scale by SHA-256 argmin;
- report which roots naturally visit one, two, or all three scales without
  using later success.

Historical overlaps are disclosed by source. Any ancestry sealed as untouched
confirmation in R1b, S3, G1, or another prior policy gate is diagnostic-only
and cannot enter G2 train, development, or transfer. Other prior training or
development overlap is reported and may enter only train/development, never
untouched transfer.

Partition order is fixed:

1. remove forbidden and diagnostic-only roots from ordinary evidence;
2. assign every eligible root with a `pre3072_transfer` state to untouched
   transfer, withholding its earlier-scale states;
3. assign remaining roots by SHA-256 namespace
   `G2-dev-v1|root`, with the lowest 20% within each genuine family to
   development and the rest to training;
4. do not split, rebalance, or reassign roots after availability is known.

Family-balanced fit weights cap each family's effective training loss at
`40%` without duplicating roots. Natural root counts and effective counts are
both reported.

## Readiness Gates

Representation integrity is a hard gate:

- exactly 64 unique finite columns in order;
- deterministic schema and implementation hashes;
- exact state and provenance reconstruction;
- no input or RNG mutation;
- orientation-canonical round trip;
- target-scale invariance on crafted fixtures;
- missing/pair/graph/path conventions pass focused tests.

Failure of representation integrity yields
`KILL_G2_REPRESENTATION_PREFLIGHT`.

Data readiness requires:

- training: at least `240` unique roots, at least `100` roots at each training
  scale, and at least five genuine families;
- development: at least `60` unique roots, at least `24` roots at each
  training scale, and at least three genuine families;
- untouched transfer: at least `96` unique pre3072 roots, at least three
  genuine families, no family above `50%`, and zero prior untouched-root
  overlap;
- zero cross-partition ancestry, source-alias, and protected-stream collision;
- feature missing/nonfinite rate zero in every scale and partition;
- free disk above `100 GiB`.

The same root may contribute both training scales only within its one assigned
train or development partition. Each root counts once for ancestry totals.

## Prospective Power And MDE

The transfer power calculation is frozen before G2 outcomes:

- base h40 transfer event probability `0.04`;
- beta-binomial root intraclass correlation `0.15`;
- eight paired repeats per compared action;
- selector activity/disagreement fraction `0.30`;
- inactive roots are exact structural-zero pairs;
- target strata-standardized policy odds ratio `1.75`;
- two-sided ancestry-cluster 95% interval;
- `10,000` deterministic simulations per candidate design;
- candidate root counts `96, 128, 192, 256, 384, 512`;
- report power and the smallest detectable OR on grid
  `1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00`;
- minimum viable design has at least `80%` power for OR `1.75` and MDE no
  worse than OR `2.00`.

These are prospective assumptions, not observed G2 outcomes. If available
untouched transfer roots cannot meet the minimum viable design, the decision
is `HOLD_G2_DATA_OR_POWER`, never a scientific failure.

## Later Offline And Causal Gates

These gates are preregistered for continuity but are not authorized now.

Predictive transfer requires on untouched roots:

- calibrated log loss and Brier better than the root-equal constant-hazard
  baseline with a 95% root-bootstrap interval excluding zero;
- calibration intercept absolute value `<=0.15`, slope in `[0.80,1.20]`, and
  ECE `<=0.05`;
- positive root-equal action-rank correlation;
- same predictive direction in every family with at least 12 roots;
- no single family supplies more than 50% of aggregate improvement.

Before transfer outcomes open, frozen model actions must disagree with the
incumbent on at least 20 roots across at least three families and preserve the
prospective 80% power under observed structural-zero activity. Otherwise the
branch holds inactive with outcomes sealed.

Causal action selection requires a root-cluster 95% interval above OR `1`,
point OR at least `1.25`, upward-transfer direction consistent with both
earlier rungs, and no material survival (`-2 pp`), anchor (`-3 pp`), or family
concentration harm. Underpowered late-rung evidence yields HOLD, never failure
or promotion. A pass authorizes only a separately frozen low-margin reranker
gate, not normal-start promotion.

## Authorized Preflight Decisions

The no-outcome preflight must seal exactly one:

- `READY_G2_RELATIONAL_HAZARD_LABEL_PREFLIGHT`;
- `HOLD_G2_DATA_OR_POWER`;
- `KILL_G2_REPRESENTATION_PREFLIGHT`.

READY authorizes only a later separately frozen small paired-label engineering
block. It does not authorize fitting, candidate policy evaluation, normal-start
development, confirmation, incumbent change, or dashboard change.
