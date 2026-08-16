# G3 Scale-Transfer Bootstrap Charter

Date frozen: 2026-07-25

Status: authoritative outcome-free charter. This file must be hashed before
any G3 label availability count, label value, model fit, transfer prediction,
candidate action, rollout outcome, or score outcome is opened.

## Decision Context

The sealed G2 fresh-transfer acquisition is spent after exactly `640` complete
normal-start games for each of corner2, hand-built expectimax2, and the current
phaseblend incumbent. It retained `12/1/19` clean natural
`pre3072_transfer` roots against frozen quotas `32/32/32`.

The acquisition result remains
`HOLD_G2_AFTER_FRESH_TRANSFER_ACQUISITION_SEAL`. Repeating or scaling the same
routine acquisition channel is `KILL_G2_ROUTINE_REACQUISITION`. This is an
acquisition decision, not a solver-quality result.

G3 asks a different question:

> Can the already frozen scale-equivariant relational hazard representation
> learn action-conditioned milestone structure from abundant earlier rungs,
> transfer directionally to the 32 untouched late-rung roots, and thereby
> justify one full-policy hazard reranker whose repeated exposure may improve
> normal-start play and create new late-rung source diversity?

G3 does not reinterpret R1b/C, H0, A2, R2a, C1, S3, G1, G1-R, QD-v1, QD-v2,
or either G2 hold. Human and recorded actions remain forbidden targets.

## Immutable G2 Inputs

G3 inherits the G2 representation and ordinary corpus without modification:

- proposal:
  `threes_rl/G2_SCALE_EQUIVARIANT_RELATIONAL_HAZARD_PROPOSAL.md`,
  file SHA-256
  `43b413c1a8145a25750009cc3048bbda6127a44cfccbf72c7d1710e1e6027099`;
- feature implementation:
  `threes_rl/g2_scale_relational_hazard.py`, file SHA-256
  `9ffaa45dd36b633cdae10110fdaefc8cd27053ab3f0216ddb3f1886ea625af8a`;
- exact 64-column schema SHA-256
  `6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e`;
- G2 outcome-free preflight:
  `threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/G2_PREFLIGHT.json`,
  file SHA-256
  `2e1084f2a0673935866839e89765d3d1a31a2c2348e99c01edc9abc2405f05cc`;
- G2 root manifest:
  `threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/G2_ROOT_MANIFEST.json`,
  file SHA-256
  `60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca`,
  canonical payload SHA-256
  `15ecb9d52ae66e938952a07a8c3d6ef3f2d39b0dd1ef3ecb3c1e4e6fcab031ce`;
- incumbent policy file:
  `threes_rl/current_incumbent_policy.txt`, file SHA-256
  `d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4`.

Any byte-level mismatch in the inherited feature implementation, schema,
proposal, root manifest, state reconstruction, simulator, incumbent spec, or
continuation semantics is an integrity failure. G3 may not repair it by
silently regenerating G2 inputs.

## Frozen Ordinary Partitions

The ordinary roots and selected states are exactly the rows already marked
`train` and `development` in the immutable G2 root manifest. G3 may not
repartition, downsample, add roots, move roots between roles, select a
different frame, or split an ancestry.

Canonical partition hashes use ASCII JSON with sorted object keys and compact
separators. Rows are sorted by
`(root_cluster, scale, record_id)` before hashing.

### Training

- selected state-scale records: `550`;
- unique whole ancestries: `283`;
- scale rows: `283 pre768`, `267 pre1536`;
- canonical full-record SHA-256:
  `5858ed61befcd521d3f70ba496d2c7bf2782541e295e9d90bd23897dae77fceb`;
- ordered record-ID SHA-256:
  `cff3567780ab8fd21cd812c1ba7d3addd244bf25c0be941706b7d4401e716db9`;
- ordered root-ID SHA-256:
  `ea8b66cb91dcbffefcf03ca8c20cb1a0366ac8b146c252badefd2880f55fb55a`.

Training roots by frozen behavior family are `20 corner2_lineage`,
`8 expectimax_baseline`, `20 legacy_learned_lineage`,
`3 phaseblend_cheap_lineage`, and `232 phaseblend_incumbent_lineage`.

### Development

- selected state-scale records: `133`;
- unique whole ancestries: `69`;
- scale rows: `69 pre768`, `64 pre1536`;
- canonical full-record SHA-256:
  `6a210761eaa832f6413516776a0237859eea7cc23987ecc57779d133f8470619`;
- ordered record-ID SHA-256:
  `a6ed4d1266d888c0f431e43823f99e9e45865df9980c0b424a76bbef291e36a8`;
- ordered root-ID SHA-256:
  `318da48e46ca93ed0efaae2b1ae30ea5ba520a4dd7ef23a7d25b760a8e713d8f`.

Development roots by frozen behavior family are `5 corner2_lineage`,
`2 expectimax_baseline`, `5 legacy_learned_lineage`, and
`57 phaseblend_incumbent_lineage`.

All roots from one ancestry remain in one partition at every scale. Training
uses deterministic family-balanced loss weights: every present family
receives total weight `1/F`; every ancestry in family `f` receives
`1/(F*n_f)`; legal actions and at-risk interval rows split that ancestry's
weight equally. No root is duplicated. The maximum effective family share is
therefore `20%`, below the inherited `40%` cap.

## Frozen Transfer Panel

The transfer panel is exactly the 32 retained natural roots in:

`threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1/G2_TRANSFER_ACQUISITION_RESULT.json`

The result file SHA-256 is
`7b862377546b35c8c53967eedd39edb736c5db039d262f65048da4c47774ca74`;
its canonical payload SHA-256 is
`a464287ea64a9cac11971cbec9ba45731291c9bc9dacfdbe472ee6661895cee4`.
The retained-source manifest SHA-256 is
`e689accbd2f5f7a869112efb884de1a1ef80d78ab61a75183352749d8c7daba9`.

The panel has `12` corner2, `1` expectimax2, and `19` phaseblend-incumbent
roots. These roots are immutable, diagnostic-only, and excluded from:

- fitting and feature normalization;
- Platt calibration;
- penalty, checkpoint, threshold, or model selection;
- reranker threshold choice;
- family robustness or powered promotion claims.

The single expectimax2 root is descriptive only. Results must be reported by
family and overall, with the `59.375%` incumbent-family concentration explicit.
N=32 is underpowered for modest effects and can justify only directional
credibility within the sensitivity reported by the prospective power audit.

## Exact Label Contract

If labels are later authorized, every selected state-scale record uses this
single contract:

1. Restore the exact board, visible preview, candidate bundle, small-bag
   counts and position, span position, pending state, move count, starter,
   deck stream state, and slot stream state from the frozen source.
2. Enumerate every legal first action in canonical order
   `up, down, left, right`.
3. For each action and replicate `0..7`, force that first action, including
   exact visible-tile insertion, then continue with the frozen current
   depth-2 phaseblend incumbent.
4. Run one path through h40. Read event/censoring checkpoints at h10, h20, and
   h40 from that same path. Do not launch separate horizon paths.
5. The event is first attainment of the state-scale target:
   `768` for pre768, `1536` for pre1536, and `3072` for
   pre3072_transfer, always excluding the fixed starter.
6. The discrete intervals are `(0,10]`, `(10,20]`, and `(20,40]`. A target
   event removes later rows from risk. Terminal without target is
   right-censored at the exact terminal move. A live h40 endpoint is
   right-censored at h40.
7. Labels contain only event/censoring sufficient statistics and provenance.
   Score is not a target and is not opened for G3 fitting.
8. Replicates `0..3` are descriptive block A and `4..7` block B. Blocks are
   not a conjunction gate.
9. Within each root/replicate, all legal-action arms share logical and deck
   exogenous tapes. They share slot uniform variates mapped independently over
   each arm's legal insertion slots. Policy tie streams are shared but may not
   leak one arm's action into another.

The reserved G3 label namespaces are:

- logical: `57_000_000_000`;
- deck: `58_000_000_000`;
- slot: `59_000_000_000`;
- policy: `60_000_000_000`.

Record ordinals are all training rows, then development rows, then transfer
rows, each in the frozen order above. For record ordinal `r`, canonical action
ordinal `a`, and replicate `j`, each stream ID is
`base + 32*r + 8*a + j`. Illegal actions reserve no row and consume no stream.
The eventual label manifest must be frozen and collision-checked before any
stream is consumed.

Every ancestry has equal total metric weight. Within a root, selected
state-scale records split root weight equally, legal actions split the
state-scale weight equally, and at-risk interval rows split the action weight
equally. Training additionally applies the frozen family-balanced factor
above. Trials from the eight replicates remain binomial trials; they are not
treated as eight independent roots.

Recorded actions, human actions, source ranking, final score, future recorded
milestones, and replay success status never enter labels, weights, fitting, or
selection.

## Legacy Label Reuse

An existing artifact is reusable only if a machine-readable metadata manifest
proves all of the following exactly for every reused path:

- G2 root-manifest file and payload hashes;
- record ID, root ancestry, source replay hash, source state hash, state SHA,
  scale, target, action, replicate, and all four stream IDs;
- every legal first action is present;
- h10/h20/h40 are checkpoints from one h40 path;
- the forced-first-action and terminal-censoring semantics above;
- the exact frozen incumbent policy file and all component hashes;
- depth-2 continuation, chance semantics, chance limit, preview/deck
  mechanics, shared-deck and shared-uniform slot coupling;
- deterministic completion and collision audit.

Artifacts lacking this sidecar are missing, even if filenames, boards, seeds,
or aggregate counts look similar. G3 may inventory metadata and provenance
only. It may not open or report a legacy label value to decide compatibility.
No value-based reinterpretation or partial semantic matching is allowed.

## Frozen Model And Calibration

G3 uses exactly the G2 model:

- the inherited 64 input columns, unchanged;
- grouped-binomial discrete-time logistic hazard;
- unpenalized intercept and horizon indicators;
- L2 penalty `lambda=1.0` on all other coefficients;
- train-only standardization for inherited continuous columns;
- deterministic L-BFGS, maximum `500` iterations, gradient tolerance `1e-8`;
- no architecture, feature, penalty, optimizer, seed, or checkpoint sweep.

The base model fits training roots only. The train-only constant baseline is
the ancestry- and family-weighted event hazard for each of the three time
intervals, pooled across pre768 and pre1536. Development comparisons use the
uncalibrated base model so fitting the calibrator cannot make development
prediction look better.

After all development predictive diagnostics are computed, one positive-slope
Platt calibrator is fit once on all development rows with no regularization.
That frozen calibrator is then applied to transfer predictions. Transfer roots
never alter the base model or calibrator.

There is one final coefficient vector and one calibrator. No checkpoint
selection, seed selection, transfer adaptation, or post-outcome threshold
change is allowed.

## Predictive And Directional Gates

All uncertainty uses a deterministic whole-ancestry bootstrap with `10,000`
draws and seed `2_026_072_530`. Root-equal and family-balanced metrics are both
reported. Legal actions, horizons, and repeats never replace the ancestry as
the inference unit.

For a root, log loss and Brier are averaged across its selected scales, legal
actions, at-risk intervals, and eight binomial trials before roots are
aggregated. Legal-action rank correlation is the root-equal mean Spearman
correlation between predicted h40 target hazard and empirical h40 action
hazard; roots with fewer than two distinct predicted and observed action
values are reported as uninformative and excluded from the correlation
denominator, never assigned zero.

### Earlier-Scale Development Gate

All conditions must pass:

- root-equal log-loss improvement over the train-only constant is positive,
  with bootstrap 95% lower bound above zero;
- root-equal Brier improvement is positive, with bootstrap 95% lower bound
  above zero;
- pre768 and pre1536 each have positive point improvement on both metrics;
- overall legal-action rank correlation is positive, with bootstrap 95% lower
  bound above zero;
- each scale's rank-correlation point estimate is nonnegative;
- no family with at least five development roots has both worse log loss and
  worse Brier than the constant baseline;
- serialization, finite prediction, state/provenance, weighting, and frozen
  incumbent nonmutation checks pass.

Failure here kills the exact G3 bootstrap fit as
`KILL_G3_EARLIER_SCALE_PREDICTION`. It does not open transfer outcomes.

### Transfer Direction Gate

Transfer may open exactly once only after the earlier-scale gate passes and
the model, calibrator, coefficient hash, action predictions, and activity
manifest are immutable.

Because N=32 is diagnostic, all conditions are directional:

- overall calibrated log-loss and Brier point improvements over the
  train-only constant are both positive;
- overall legal-action rank correlation is positive;
- corner2 and incumbent families must not each show a joint reversal in which
  log loss, Brier, and rank direction are all worse; expectimax2 is
  descriptive only;
- calibration has finite positive slope, absolute intercept at most `0.50`,
  and root-equal ECE at most `0.15`;
- the model's h40 argmax differs from the incumbent on at least `6/32` roots,
  including at least one corner2 and one incumbent root;
- log-loss and Brier improvement signs are positive at pre768, pre1536, and
  pre3072_transfer, establishing earlier-to-later directional consistency;
- no single root supplies more than `10%` of aggregate metric improvement.

Passing yields `READY_G3_FULL_POLICY_RERANKER_PREREGISTRATION`, not a policy
or capability result. A null or ambiguous transfer diagnostic yields
`HOLD_G3_TRANSFER_INCONCLUSIVE`, especially when the frozen N=32 MDE shows
that the unresolved effect is below sensitivity. An integrity failure yields
`KILL_G3_TRANSFER_INTEGRITY`. No N=32 result establishes powered family
robustness or promotion.

## N=32 Prospective Sensitivity

The no-outcome preflight must compute N=32 power with the exact G2 mechanism:

- h40 base event probability `0.04`;
- beta-binomial root ICC `0.15`;
- eight paired repeats;
- model/incumbent activity `0.30`, with inactive roots structural zeroes;
- two-sided ancestry-cluster 95% interval;
- pass event requires lower interval above zero and estimated policy OR at
  least `1.25`;
- `10,000` deterministic simulations;
- the G2 heterogeneous-beta active-root OR calibration;
- simulation seed `2_026_072_508 + 32*100 + round(100*OR)`;
- OR grid
  `1.25,1.50,1.75,2.00,2.25,2.50,3.00,4.00,5.00,6.00,8.00,10.00,15.00,20.00,30.00`.

The preflight reports power for every grid row and the smallest OR with at
least `80%` full-gate power. That OR is the transfer MDE. Transfer evidence is
informative only for effects at or above this MDE. Smaller effects remain
unresolved and produce HOLD, never scientific failure or promotion.

## Conditional Full-Policy Reranker

This section freezes continuity only. No reranker construction or policy
outcome is authorized by the G3 preflight.

If both predictive gates pass, the one permitted candidate uses:

- current frozen depth-2 incumbent values and action;
- the calibrated model's h40 hazard for every legal action;
- stages with current built maximum excluding starter exactly
  `384`, `768`, or `1536`, mapped to targets `768`, `1536`, or `3072`;
- model action = highest calibrated h40 hazard, with ties in canonical action
  order `up, down, left, right`;
- eligibility only when at least two actions are legal, model action differs
  from incumbent, calibrated hazard gain is at least `0.025`, and the
  incumbent normalized value gap to the model action is at most `0.02`, where
  normalized gap is
  `(Q_incumbent-Q_model)/max(1,abs(Q_incumbent))`;
- incumbent action on every ineligible move;
- reranker action on every eligible move for the full trajectory.

There is no one-move wrapper, online fitting, hazard bonus added to score,
auxiliary weighted reward, threshold sweep, or outcome-driven exception.

The first future causal test, if separately authorized, is `256` fresh paired
normal-start games on a manifest frozen before outcomes. Primary endpoint is
paired final score-minus-starter. Secondary endpoints are median, lower
decile, moves/survival, P1536, P3072, maximum score as diagnostic only, and
anchor/corner integrity. Development passes only if paired mean-score 95% CI
is above zero, P3072 difference is at least `-2 pp`, lower decile and survival
show no material regression, and no new corner failure appears.

A development pass may open exactly one separately sealed `512`-game
normal-start confirmation block under the same candidate. Promotion requires
the same score-CI, P3072, lower-tail, survival, and corner gates on
confirmation. Only confirmation may change the incumbent or dashboard.

## Outcome-Free Preflight

The currently authorized G3 preflight may:

- verify every immutable hash and selected-state reconstruction;
- build exact train, development, and transfer manifests;
- inventory metadata-only legacy label schemas and provenance;
- count compatible and missing root/action/h40 paths without reading values;
- build and collision-check the reserved stream manifest without consuming it;
- compute N=32 prospective power/MDE;
- project one-worker nice-10 runtime and compact storage;
- run focused and relevant regression tests;
- verify disk, services, dashboard, top-three, and no heavy contention.

It may not generate a label, consume a reserved stream, fit a model, open a
label value, inspect score or action outcomes, build a reranker, evaluate a
policy, run a continuation, change the incumbent, or change the dashboard.

The preflight seals exactly one:

- `READY_G3_BOOTSTRAP_LABELS`: immutable inputs, coverage manifest, stream
  reservation, projected cost, services, and storage all pass;
- `HOLD_G3_LABEL_COVERAGE_OR_COST`: contract is intact but compatible
  coverage, runtime, storage, disk, or service readiness is insufficient;
- `KILL_G3_PREFLIGHT_INTEGRITY`: any immutable input, provenance, partition,
  state, schema, collision, or no-outcome boundary fails.

READY authorizes only a later separately approved label and fitting execution.
It does not authorize labels now, transfer outcomes, reranker construction,
normal-start evaluation, C2, human training-ground work, incumbent change, or
promotion.
