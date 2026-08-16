# Human H2 Preview/Cycle Context Diagnostic

Status: frozen after H0 `KILL` and before H2 computation on 2026-07-11.

## Scope

H2 is a bounded representation diagnostic. It does not test a candidate policy,
open a normal-start evaluation block, authorize fitting, alter the incumbent,
or affect the dashboard. Human replays supply exact simulator-valid state
coverage only. Recorded human actions and H0 outcomes are not labels or
selection inputs.

The question is whether a board-only leaf aliases preview/tile-cycle contexts
that produce stable, decision-relevant continuation differences even when the
board and current inserted tile are held fixed.

## Frozen Inputs

- Root source:
  `runs/forensics/human_h0/human_h0_root_manifest_20260710.json`.
- Frozen incumbent: `current_incumbent_policy.txt`, unchanged from H0.
- All 48 H0 boards remain valid for preview-support enumeration.
- Cycle-rollout targets are exactly offsets 20 and 3 from each of the six H0
  ancestry clusters: 12 boards, two per ancestry. This uses only frozen offset
  and ancestry fields.
- No target is counted as an independent human game. Analysis clusters by the
  six source ancestries.

## Part A: Preview-Only Search Sensitivity

For each of the 48 boards, hold board and tile-cycle counters fixed and
enumerate every current preview with positive probability under
`ThreesSim.preview_options`. Each preview is therefore supported by the exact
cycle state. Evaluate all legal root action values with the frozen depth-2
incumbent.

Report:

- action changes relative to the observed preview;
- top-two action-margin range across supported previews;
- best-value and per-action value ranges;
- small-preview versus bonus-preview effects;
- results by ancestry and H0 role.

This is a control: depth-2 search is already expected to use the visible
preview. It does not by itself justify a context-aware leaf.

## Part B: Cycle-Only Search and Continuation Sensitivity

Freeze four within-ancestry donor pairs without policy values or outcomes:

1. require an ancestry to contain both a positive-plus and zero-plus context;
2. choose at most one pair per ancestry;
3. rank ancestries by their maximum plus probability, then ancestry ID;
4. select the maximum-plus root in that ancestry, tie-breaking by root ID;
5. select the zero-plus root with minimum mechanics-only cycle distance to the
   high-plus root, tie-breaking by root ID.

Cycle distance uses normalized small-bag position, span position, bag counts,
pending flag, and small-seen count. For each donor pair, choose one small
preview supported by both cycles that maximizes the minimum exact support
probability; ties are lexical. Holding that preview fixed ensures the immediate
inserted value is identical in low/high arms. Only exact cycle counters differ.

For every combination of 12 target boards and four donor pairs:

- transplant the low/high cycle state and common preview onto the same target
  board and move count;
- verify preview support, exact round trip, identical legal actions, and
  simulator validity;
- record frozen-incumbent depth-2 action values and selected action;
- run 16 paired continuations per arm, blocks A/B of eight, through h20 and
  read h10 from the same trajectory;
- share new deck, mapped-slot-uniform, and policy stream IDs across low/high
  arms; all IDs must be disjoint from H0, D0-D2, C, pre-C diagnostics, and
  original human streams.

Endpoints at h10/h20 are score delta, survival, first non-starter 1536/3072,
empty count, and anchor preservation. Store compact metrics and a fixed audit
for one target/donor pair only.

## Analysis

- The signed high-plus-minus-zero-plus contrast has no globally preferred
  direction; context may help one board and hurt another.
- Compute each target/donor pair's paired mean contrast and A/B block contrast.
- Report action-flip rate and normalized top-two-margin change from deterministic
  search.
- For score, first-1536, and survival, report median absolute h20 pair effect,
  A/B sign agreement, ancestry-cluster summaries, and a sign-flip null interval
  for the median absolute effect.
- An effect is block-informative when combined absolute score difference is at
  least 500 points, milestone difference at least 2 percentage points, or
  survival difference at least 2 percentage points. Require at least 12
  informative target/donor pairs for a stability claim.

## Frozen Decision Rule

`CONTEXT_MATERIAL` requires both:

1. decision sensitivity: cycle-only selected-action flip rate at least `15%`,
   or at least `25%` of target/donor pairs change the normalized top-two margin
   by at least `1%`; and
2. stable outcome sensitivity on h20: at least one endpoint has A/B sign
   agreement at least `70%` over at least 12 informative pairs, and either
   median absolute score effect is at least `1,000`, median absolute
   first-1536 effect is at least `5 pp`, or median absolute survival effect is
   at least `5 pp`; its sign-flip null 95th percentile must be below the
   observed median absolute effect.

Otherwise the result is `CONTEXT_WEAK_OR_INCONCLUSIVE`.

Neither result changes policy. If material, write a proposal for one small
stage-aware context residual using explicit preview/bundle probabilities,
`P(plus next)`, full bonus-value distribution, small-bag probabilities,
bag/span positions, and pending status. Do not train it without a separate
preregistration and stop/go note. If weak, hold context fitting and propose a
different representation/search direction.
