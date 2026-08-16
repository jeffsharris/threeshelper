# Human-play review: 2026-07-10

## Scope

Ten completed exact-simulator human games were recorded, plus one empty active
session. The first two games were treated as calibration based on the player's
report. Four other sub-10k earned-score games remain retained as failure/error
evidence but are not proposed as positive demonstrations.

The primary analysis set contains six independent games earning at least
23,808 points beyond the fixed 1536 starter:

| Earned score | Moves | Highest built tile excluding starter |
| ---: | ---: | ---: |
| 36,312 | 370 | 768 |
| 30,261 | 297 | 768 |
| 23,808 | 290 | 768 |
| 36,105 | 316 | 768 |
| 28,734 | 277 | 768 |
| 121,638 | 327 | 3072 |

Thus the batch supplies one successful `1536 -> 3072` ancestry and five
substantial 768-level failure controls. It does not yet supply enough
independent promotion successes for direct supervised policy fitting or a
standalone evaluation claim.

## Strongest new evidence

The successful game is
`human_20260710_222838_7fc7a34249b1cd70`.

For about 40 moves before promotion, the player maintained a descending top
edge anchored by the free starter:

```text
1536  768  384  96/192
```

while accumulating the second 768 below it. The final transition was:

- move 288: adjacent `768, 768`;
- move 289: adjacent `1536, 1536`;
- move 290: merged `3072`.

This is exact simulator-valid evidence through the support rung that incumbent
self-play reaches rarely. Every frame retains the visible preview, exact plus
bundle, small-bag counters, plus-span counters, legal actions, and split RNG
provenance.

## Incumbent comparison

Action agreement was computed only as a diagnostic. The incumbent's own value
ranking is not ground truth.

Across the six substantial games at built max 384 or higher:

- 1,188 decisions;
- exact incumbent-action agreement: 744/1,188 (62.6%);
- human action in incumbent top two: 1,063/1,188 (89.5%).

The 3072 game was the least incumbent-like substantial game:

- exact agreement: 152/264 (57.6%);
- top-two agreement: 228/264 (86.4%).

Disagreement concentrated in the successful promotion window:

| Window before new 1536 | Exact match | In top two |
| --- | ---: | ---: |
| h40 | 19/40 | 31/40 |
| h20 | 10/20 | 16/20 |
| h10 | 5/10 | 8/10 |

At frame 286, the incumbent strongly preferred `up`, ranked the human `right`
third, and assigned it a normalized value gap of 0.649. The recorded human path
then produced the second 1536 three moves later and 3072 on the following move.
This is the highest-signal candidate for action-conditioned h10/h20/h40 paired
rollouts. It is not, by itself, proof that `right` was optimal.

After 3072, agreement rose to 25/37 (67.6%) and top-two agreement to 32/37
(86.5%). On the terminal decision, only `down` and `left` were legal and the
incumbent assigned both a value of zero. The recorded loss therefore cannot be
confidently attributed to a uniquely bad final move using the current leaf.

## Recommended use

1. Preserve all exact replays, but do not give every game equal positive-label
   weight. Treat the early/very short runs as calibration or failure evidence.
2. Freeze h10/h20/h40 roots from the successful game's pre-1536 window,
   especially frame 286, plus progress/geometry-matched roots from the five
   768 failures.
3. For every legal first action, run paired shared-deck action-conditioned
   continuations and measure first-1536/3072 hazard, survival, and score. Do not
   use incumbent action ranking or recorded action agreement as the label.
4. Use the human frames as a small, genuinely new restart family for a future
   context-aware value or adaptive-expectimax branch. Keep held-out normal-start
   evaluation separate.
5. Do not train generic behavior cloning from this batch. One successful
   ancestry is vulnerable to winner selection, correlated-frame inflation, and
   copying human mistakes.
6. Add a post-game quality annotation to the recorder (`good`, `mistakes`,
   `calibration/discard`) so future intake can use the player's own assessment
   without inferring quality from score.

## Decision

The human batch is scientifically useful and changes the next-data priority,
but it does not justify an immediate policy update. The highest-value next
experiment is a small preregistered action-conditioned promotion-window audit
on the human success and matched human failures. Training remains held until
that diagnostic is explicitly authorized and specified without using its
outcomes for tuning.
