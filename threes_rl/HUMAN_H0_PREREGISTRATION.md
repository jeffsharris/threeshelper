# Human H0 Action-Conditioned Diagnostic

Status: frozen before rollout outcome inspection on 2026-07-10. The
disagreement bookkeeping clarification below was recorded after 100 task-count
updates but before reading any rollout outcome row.
Clarification lock:
`runs/forensics/human_h0/human_h0_prereg_clarification_lock_20260710.json`.

## Scientific Scope

H0 is a development/mechanism diagnostic. Human games are not held-out policy
evaluation, action agreement is not a label, and no H0 result can update the
incumbent or dashboard. R1b remains permanently unpromoted after sealed C.

## Frozen Corpus

- Corpus hash manifest:
  `runs/forensics/human_h0/human_corpus_manifest_20260710.json`.
- Root manifest:
  `runs/forensics/human_h0/human_h0_root_manifest_20260710.json`.
- Six substantial independent ancestry clusters: one 3072 success and five
  built-768 failures.
- Success frames are outcome-selected offsets `40,30,20,15,10,5,3,1` before
  first built 1536 at frame 289. This includes frame 286.
- Each success frame has one progress/geometry reference in each failure
  ancestry.
  Matching uses only current score/move progress, support mass, top-edge ranks,
  empties, snake geometry, visible preview/candidates, exact small-bag state,
  plus probability, span position, and pending status. It uses no incumbent
  values or rollout outcomes.
- Frames are correlated. The analysis cluster is the six source ancestries,
  never the 48 frames or rollout replicates.
- Failure-reference match distance is reported continuously and no root is
  discarded after outcomes. Across the frozen 40 references, median distance
  is `0.877`, p90 is `1.267`, maximum is `1.728`, and `15/40` exceed `1.0`.
  These are development geometry references, not exchangeable matched controls.
- Incumbent action and `recorded_disagrees` were frozen for every root before
  rollout. There are 18 informative disagreement roots: four on the success
  ancestry and 14 across the five failure ancestries (`5,1,4,3,1`). The other
  30 roots are same-action cases and therefore structural zero contrasts.

## Frozen Rollouts

- Every legal first action at every root is forced once per replicate, then the
  frozen current incumbent controls moves 2 through 40.
- 64 replicates per root/action, split into preregistered blocks A/B of 32.
- One trajectory supplies h10/h20/h40 checkpoints.
- All legal-action arms for a root/replicate share the same deck stream and the
  split-stream shared-uniform slot mapping. Continuation policy streams are
  also shared through independent generator instances.
- New stream namespace: `threes-human-h0-20260710`; logical IDs start at
  `7,000,000`. All IDs must be disjoint from D0-D2, C, pre-C diagnostics, and
  original human session streams.
- Outputs are compact metrics. Full frames are retained only for replicate 0
  of frame 286 and one frozen matched failure root.

## Endpoints

At h10/h20/h40:

- first non-starter 1536 and 3072 hazard;
- survival and score delta from the root;
- empty count;
- top-left anchor preservation;
- top-edge rank-mass delta and descending top-edge preservation.

Primary comparison is recorded human action minus incumbent-selected action on
the 18 informative disagreement roots only.
All-action means and ranks are secondary. Frame 286 is always reported as a
development case study and never as an independent promotion claim.

## Analysis

- Pair actions within root/replicate using common exogenous streams.
- For the primary contrast, average disagreement roots within each ancestry,
  then bootstrap the six ancestry clusters. Also report blocks A/B separately.
- Report same-action roots separately as uninformative structural zeroes. They
  remain in all-action ranking and coverage summaries but never enter the
  recorded-action gate statistics.
- Report success-window root directions, but do not treat eight correlated
  success frames as eight independent games.
- Report failure ancestries separately to detect reversal.

## H0 Gate

`CONTINUE` the human-guided branch only when all hold:

1. At least four success-window disagreement roots have positive h40 score
   advantage and positive roots outnumber negative roots.
2. Across the informative disagreement roots in the five failure ancestries,
   recorded-action h40 score and first-1536 point differences are both
   nonnegative. All five ancestries must have at least one disagreement;
   otherwise generalization is unavailable and the result is `HOLD`, not pass.
3. Either the six-cluster h40 score CI excludes zero positively, or h40
   first-1536 lift is at least `+5 pp`.
4. H40 survival difference is at least `-2 pp`.

`KILL` direct human-action supervision when the success-window h40 score point
is negative with nonpositive milestone lift, or when failure-ancestry score has
a wholly negative cluster interval, or survival is below `-2 pp` with a wholly
negative interval.

Otherwise `HOLD` imitation/action-prior fitting and proceed to the bounded H2
preview/cycle context-sensitivity proposal. No rule may be changed after H0
outcomes are read.
