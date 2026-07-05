# Threes RL — Research Plan V2 (post-review)

Reviewer read: RESULTS.md, PROGRESS.md, ML_FINDINGS.md (2026-07-05 round).
Human calibration: the player routinely scores **500k–900k** and has reached
12288 once. Best agent so far: 77k (expectimax3 probe). This plan reorients
the work around closing that qualitative gap.

---

## 1. The key fact everyone should stare at

Every policy evaluated so far — random included — shows:

```
">=1536": 1.0     (the free starter tile)
">=3072": 0.0     (never, not once, for any policy, any seed)
```

Mean scores decompose as 59,049 (starter) + 500–18,000 of small change.
**No agent has ever built a single tile beyond the starter.** The human
does so routinely (500k ⇒ a 6144 or 3072+1536+… board). So:

- We are not "somewhat below" expectimax2's level — expectimax2 itself is
  qualitatively not playing the real game yet. Its hand-built leaf evaluator
  (score + empties + monotonicity) is score-greedy and myopic; it harvests
  quick merges and dies at ~108 moves.
- **Imitating expectimax2 (BC, DAgger) inherits this ceiling.** DAgger toward
  a 66k teacher cannot reach 500k. Do not invest further there until the
  teacher itself is strong. ML_FINDINGS' DAgger recommendation is hereby
  superseded.
- The starter tile also poisons the headline metric: 95% of reported score is
  a constant. All future tables must report **score_minus_starter** and
  **max tile excluding the starter**; the single most informative progress
  number is now `P(build ≥ 3072)` (currently 0.000 for everything).

## 2. Answer to the information-design question

Q: expose more internal state, go AlphaZero, or make the model learn the
rules itself?

**Expose everything, and go "AlphaZero-lite" — but the right variant.**

- *Rules-from-scratch (pixels / hidden bag)* buys nothing: it spends compute
  rediscovering things we have already validated to 100% against live play.
  Nobody hides the rules from AlphaZero either — it gets the exact move
  generator. What AlphaZero removes is human *strategy* priors, not state.
- *Full AlphaZero/MuZero machinery* is overkill in the wrong dimension:
  MuZero learns a model (ours is exact and validated — the crown jewel of
  this repo), and MCTS is designed for adversarial branching. Threes is a
  single-player stochastic game with small chance branching and a **known
  preview** — the textbook-correct and empirically dominant recipe (from the
  2048 literature: Szubert & Jaśkowski's TD n-tuple agents and Jaśkowski's
  follow-ups reached superhuman play with exactly this) is:

  > **TD-learned afterstate value function + shallow expectimax over the
  > exact chance model.**

  That *is* the AlphaZero idea (learned evaluator + search + self-play
  improvement loop), minus the parts our problem doesn't need.
- Information exposure: the current "full" observation (bag counts, span
  state, preview, candidates) is right and stays. The bigger upgrade is
  giving the *search* the exact simulator (it already has it) and giving the
  *value function* the board — the question "should the net get more state"
  mostly dissolves, because planning consumes the state explicitly.

## 3. The centerpiece: TD afterstate n-tuple learner

New files: `threes_rl/ntuple.py`, `threes_rl/train_td.py`. **Numpy + stdlib
only** (no torch). This is also where the big-RAM machine genuinely matters:
the value function is a set of large lookup tables (hundreds of MB to a few
GB), and training is pure CPU at very high move throughput.

### 3.1 Value representation

- Afterstate = board immediately after the shift+merge, **before** the spawn.
- `V(board) = Σ_i LUT_i[index_i(board)]`, summed over n-tuple patterns and
  all 8 board symmetries (D4 group; actions permute correspondingly —
  implement `SYMMETRIES` as (cell-permutation, action-permutation) pairs and
  unit-test round-trips).
- Rank alphabet: 16 (0..15 as in sim.py). Start with this tuple set:
  - all 4 rows, all 4 columns (4-tuples): 16^4 = 65,536 entries each
  - all nine 2×2 squares: 65,536 each
  - the four 2×3 rectangles along each edge (6-tuples): 16^6 ≈ 16.8M each
  Total ≈ 4×16.8M×4bytes ≈ 270MB as float32 — fine on this machine. If
  results warrant, grow to the classic 8×6-tuple set (~1.1GB).
- Bag/schedule conditioning: none in V initially. The *search* handles the
  bag exactly. (Optional later: small additive tables keyed on
  (preview_kind, span_bucket).)

### 3.2 Training (self-play TD)

- Reward r_t = score_board(s_{t+1}) − score_board(s_t). This telescopes to
  exactly final score (γ=1, undiscounted) — dense AND unbiased; the sparse-
  vs-shaped dilemma from ML_FINDINGS does not exist for this objective.
- TD(0) on afterstates: after playing a→(afterstate A, spawn, next move's
  afterstate A'): `LUT += α/n_tuples * (r + V(A') − V(A))` on A's indices.
  Use TC (temporal coherence) auto-stepsizes if plain α is unstable;
  α start: 0.1.
- Action selection while learning: 1-ply expectation — for each legal a,
  E[V] over spawn slots (uniform) and spawn value (known preview; for bonus
  previews uniform over the 3 candidates), pick argmax. ε-greedy not needed
  (chance provides exploration); optimistic init (e.g., +
  a few thousand) encourages survival early.
- Throughput target: ≥ 5,000 moves/s single process (index computation is a
  few dozen int ops/move). Parallelism: N worker processes self-playing with
  periodic table merge (average deltas), or just run one fast process first.
- Checkpoint = memory-mapped .npy per LUT + JSON meta. Resumable.

### 3.3 Gates for this phase (evaluate on seeds 1000:1200, starter 1536,
greedy 1-ply-expectation action selection, no search)

- G1: mean moves ≥ 300 (currently best is 142)
- G2: `P(max tile excl. starter ≥ 1536) ≥ 0.5`
- G3: mean score_minus_starter ≥ 100k (currently ≤ 18k)

If G1–G3 hit, this is already beyond every existing baseline by a wide
margin. Historical 2048 results suggest they are reachable within days of
CPU self-play.

## 4. Search amplification (after the value net is real)

- `expectimax_learned.py`: depth-adaptive expectimax (d2 normally, d3+ when
  empty cells ≤ 4 or a bonus preview is pending) with V at the leaves,
  afterstate transposition cache keyed on (board bytes, depth).
- This becomes: (a) the strongest eval-time agent, (b) the **teacher** for
  any future distillation (the live-companion hint wants a fast policy), and
  (c) the actor for the next self-play data round — retrain V on games played
  by the search agent (expert-iteration loop, i.e. the AlphaZero improvement
  cycle). Iterate 2–3 rounds and measure each.
- Only if this plateaus clearly below the human anchor, escalate to a deep
  value net (CNN over 4×4×16 one-hot, torch/MPS) trained on the same TD
  targets — architecture switch, same algorithm. PPO stays parked.

## 5. Hygiene and diagnostics to add alongside

1. Eval output: add `score_minus_starter`, `max_tile_excl_starter`,
   move-count percentiles, and death forensics (final board + last 20 moves
   for the median and worst 5 seeds; classify: corner trap / bonus clog /
   bag starvation).
2. Starter curriculum: train and eval with starter ∈ {None, 96, 384, 1536}
   mixed (eval reports per-starter rows). Guards against overfitting to the
   1536 start and exercises the early game the current device never shows.
3. Sim-difficulty sanity check (cheap insurance): add
   `python -m threes_rl.play` — a 20-line curses/stdin interactive loop so a
   human can play the simulator directly. If the sim feels unfairly hard
   (e.g., bonus cadence wrong at high max_tile), that's a model bug the
   transition tests cannot catch. One 10-minute human session is enough.
4. Keep the 200-seed protocol, but headline tables now lead with
   score_minus_starter and `P(≥3072)`.

## 6. Explicit de-prioritizations (with reasons)

- **DAgger / BC toward expectimax2** — teacher ceiling (66k) is the problem,
  not distribution shift. Revisit distillation only with the search+V teacher,
  and only for inference-speed reasons (companion hints).
- **PPO (any reward shaping)** — dominated by TD-afterstate for this class of
  game; the shaping dilemma is moot given §3.2's exact telescoping reward.
- **MuZero/stochastic MCTS** — we own an exact validated model; learning one
  is pure overhead.
- **More RAM for the imitation pipeline** — correct observation in
  ML_FINDINGS (it was CPU-bound); but RAM *is* the right resource for
  n-tuple tables + transposition caches, which is the new center of gravity.

## 7. Suggested execution order for the next work session

1. §5.1 eval metrics + §5.3 interactive sanity check (half day, do first —
   every later result reads through these).
2. §3 ntuple + train_td + first overnight self-play run; check G1–G3.
3. §4 learned-leaf expectimax; re-eval; expert-iteration round 2.
4. Report in RESULTS.md with the new headline metrics; update ML_FINDINGS
   with what actually happened.
