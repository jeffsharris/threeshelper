# R1b Pre-C Congested-Window Diagnostic

Status: completed on 2026-07-10 with decision `NEUTRAL`; no pre-C block fired.

## Corpus

- Source artifact: `runs/forensics/restart_program/r1b_pre_c_independence_audit_20260710.json`.
- Exactly 21 fresh-root ancestries absent from every R1b restart update and
  R1b normal-start seed, and disjoint from D0-D2 and sealed-C identifiers.
- One state per ancestry, selected without outcome data: the earliest legal
  current state with the free starter masked, built max exactly 768, and at
  least one 384 support tile. Ties break by frozen record ID.
- Board, preview, tile-cycle/deck state, move count, and starter tile must
  round-trip exactly through the simulator before execution.
- The selected source-family mix is reported, not hidden or reweighted.

## Streams And Arms

- Horizons: `10`, `20`, and `40` moves from the exact start state.
- Two preregistered stream blocks, `A` and `B`, with eight repeats per root in
  each block: 16 paired continuations per root and 336 paths per policy arm.
- Namespace: `threes-r1b-pre-c-20260710`; logical IDs start at `6,000,000`.
- Every deck, slot, and policy stream ID must be internally unique and
  disjoint from D0, D1, D2, and C before execution.
- Arms: frozen current incumbent versus R1b at exactly 5,000. No action
  selection, training, tuning, or intermediate analysis.

## Endpoints

- Sentinel: paired difference in probability of first non-starter 1536 by h40.
- Secondary hazard readouts: the same endpoint at h10 and h20.
- Safeguards: paired score gain from the start state and survival after each
  horizon.
- Analysis unit: root ancestry. Average repeats within each root, then use a
  root-cluster bootstrap over the 21 roots. Report blocks A and B separately.

## Interpretation

- `BLOCKS_PRE_C`: the h40 upper 95% root-bootstrap interval is below zero for
  first-1536 hazard, score gain, or survival.
- `SUPPORTS_PRE_C`: no blocking result; h40 first-1536 point lift is positive,
  h40 score point lift is nonnegative, and h40 survival point lift is at least
  `-2 pp`.
- `NEUTRAL`: neither rule above. A neutral diagnostic does not erase D2's
  direct positive P3072 endpoint; it means this small failure-root slice did
  not localize the gain.

This diagnostic cannot promote R1b, open C, change the incumbent, or update the
dashboard. Its only purpose is the separate pre-C decision.

## Outcome

- h10 first-1536: `1/336` in both arms; difference `0.00 pp`.
- h20 first-1536: incumbent `9/336`, R1b `12/336`; difference `+0.89 pp`,
  root-bootstrap CI `[-0.60 pp, +2.38 pp]`. Both blocks were positive.
- h40 first-1536: `30/336` in both arms; difference `0.00 pp`, CI
  `[-2.38 pp, +2.38 pp]`. Blocks were equal and opposite at `+2.38 pp` and
  `-2.38 pp`.
- h40 score-gain difference: `-324.08`, CI
  `[-2,495.47, +1,761.65]`; h40 survival difference `+0.60 pp`, CI
  `[-5.95 pp, +6.25 pp]`.
- Frozen decision: `NEUTRAL`. No h40 upper interval is below zero, so the
  diagnostic does not block confirmation. It also does not independently
  localize the D2 gain to this 21-root failure slice.
