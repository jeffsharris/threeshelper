# State Information Audit

Status date: 2026-07-10

Purpose: verify which upcoming-tile and tile-cycle information is represented,
recorded, restored, consumed by search, and available to learned value models.

## Summary

| Layer | Current preview | Bonus candidates | Bag/cycle position | Plus probability | Result |
| --- | --- | --- | --- | --- | --- |
| Simulator state | yes | yes | yes | yes | pass |
| Human-play replay | yes | yes | yes | derivable exactly | pass |
| Replay restoration | yes | yes | yes | derivable exactly | pass |
| Depth-2 expectimax | yes | yes | yes | yes within search horizon | pass |
| N-tuple value leaf | no | no | no | no | representation gap |
| One-step TD target | current spawn only | current candidates only | counters carried but unused by leaf | next preview marginalized | representation gap |

## Simulator State

`SimState` contains:

- `preview.kind`, `preview.value`, and `preview.candidates`;
- `small_counts`, the remaining red/blue/gray counts in the current 12-small
  bag;
- `small_pos`, the position inside that small-tile bag;
- `small_seen_total`, the total number of small previews since game start;
- `span_small_pos`, the position inside the post-delay large-tile span;
- `large_pending`, whether a bonus preview is currently eligible to appear;
- `max_tile`, which determines whether bonuses are enabled and which three-tile
  bonus windows are legal.

The reverse-engineered schedule currently implemented is:

- no bonus before `21` small previews;
- no bonus while `max_tile < 48`;
- once a bonus is pending, probability is
  `1 / (21 - span_small_pos)`, rising from `1/21` to `1`;
- bonus support is a three-value window drawn from powers of two beginning at
  `6`, capped by approximately one quarter of the current maximum tile.

The schedule has previously matched tracker `TileCycle` behavior in lock-step
tests.

## Human-Play Recording

Every accepted move in `datasets/human_play/<session>/replay.json` stores:

- the complete board before and after the decision;
- the exact preview consumed by the move;
- all three candidates when the preview is a bonus;
- the complete next preview;
- all tile-cycle counters above;
- legal actions and legal mask;
- inserted value, insertion slot, and all eligible slots;
- split deck and slot RNG stream IDs;
- human provenance and decision time.

The replay loader reconstructs `SimState` with the same preview candidates and
cycle counters. Regression tests cover both ordinary and bonus previews.

## Search Consumption

Expectimax state/cache keys contain the board, exact preview, all bonus
candidates, remaining small counts, bag position, total small count, span
position, pending flag, maximum tile, and depth.

At depth 2, search:

1. branches over the exact current preview, including all three bonus values;
2. consumes the preview and updates the cycle counters;
3. computes the distribution over the next preview from those counters;
4. makes the next decision using that realized preview;
5. terminates at a board-only learned leaf.

Thus current search uses preview/cycle information explicitly inside its search
horizon.

## Learned-Value Gap

The n-tuple APIs are `value(board)` and `update(board, target)`. Stage selection
is also board-only. Two states with identical boards but different previews or
cycle positions therefore share the same learned value.

`expected_afterstate_target(...)` calls simulator transition expansion with
`include_next_preview=False`. It uses the current preview to model the immediate
inserted tile, but the next preview is replaced by an ignored placeholder and
the board-only post-spawn value cannot use the carried cycle counters.

Monte Carlo and n-step training retain the correct trajectory outcomes, but
their updates are still indexed only by board tuples. Context is averaged into
the same weights rather than represented explicitly.

## Research Consequence

The pipeline is data-ready for context-aware learning: no new recording format
is needed. The missing work is a model/training change, not more state capture.

The next context experiment should remain bounded:

1. measure action/value sensitivity on matched boards under simulator-valid
   preview and cycle counterfactuals;
2. if material, add a small stage-aware context residual rather than crossing
   every n-tuple table with the full context space;
3. generate TD targets with the next-preview distribution enabled;
4. evaluate on fresh paired normal-start streams with raw-score and tail
   safeguards.

## 2026-07-11 H2 Context Result

The bounded same-board test is complete and returns `CONTEXT_MATERIAL`.

- Supported current-preview variants changed the depth-2 action in `98/269`
  cases, confirming the expected visible-preview sensitivity already handled
  inside search.
- Holding board and current small preview fixed while changing only exact
  cycle state changed the selected action in `3/48` cases but moved the
  normalized top-two margin by at least `1%` in `34/48`.
- Cycle-only h20 score effects were stable and material: median absolute
  `3,400.59`, permutation null95 `2,155.43`, with A/B sign agreement in
  `39/46` informative cases.
- High-plus contexts increased score and promotion rates but reduced survival
  and anchor preservation. This rejects a simple scalar plus bonus and supports
  a distribution/tail-aware context residual proposal.

See `HUMAN_H2_CONTEXT_RESIDUAL_PROPOSAL.md`. No model was fit and no policy or
dashboard decision changed.
