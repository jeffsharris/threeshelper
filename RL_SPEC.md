# Threes RL Environment — Implementation Spec

This is a complete, self-contained specification for building a reinforcement-learning
environment and self-play training stack for Threes on top of this repository's
validated game model. It is written for an implementing agent that will work
autonomously for a long time. Follow it milestone by milestone, in order. Every
milestone has a definition of done and a verification command. Do not skip
validation milestones to get to training faster — the value of this project is
that the simulator provably matches the real game as observed through the
tracker.

---

## 0. Ground rules (read first, re-read when unsure)

1. **Do not modify any existing module.** `state_hunt.py`, `window_stream.py`,
   `tracker_runtime.py`, `live_debug_server.py`, `mirroring_control.py`,
   `preview_detector.py`, `gray_hog.py` are production tracker code. You may
   *import* them (notably `state_hunt` and `window_stream.TileCycle`) as test
   oracles. If you believe one has a bug, write it in `PROGRESS.md` and code
   around it; do not fix it in place.
2. All new code lives in a new package directory `threes_rl/` plus tests in
   `tests/test_rl_*.py` (the existing suite uses stdlib `unittest`, discovered
   via `python -m unittest discover -s tests`; keep that working — your tests
   must not require torch unless guarded with a skip).
3. Python: use the repo venv (`.venv/bin/python`, Python 3.12). New
   dependencies: append to `requirements.in`, then
   `uv pip compile requirements.in -o requirements.lock.txt && uv pip sync
   requirements.lock.txt`. Note `uv.toml` applies a one-week package age gate;
   if a pin fails, choose an older version. Expected additions: `torch`,
   `gymnasium`. The **core simulator must depend only on numpy + stdlib** so
   the tracker can later import it without pulling in torch.
4. Keep a `threes_rl/PROGRESS.md` journal: timestamped entries; per-milestone
   status; measured numbers (never "works", always the number); blockers and
   how you resolved them. Keep a `threes_rl/RESULTS.md` with the evaluation
   tables described in §9. These two files are how the human validates your
   work later.
5. Determinism everywhere: every stochastic component takes a
   `numpy.random.Generator`. Same seed ⇒ identical trajectory, identical
   training data order (training itself may be nondeterministic on GPU; eval
   must be deterministic).
6. Run the **entire** existing test suite after every milestone
   (`.venv/bin/python -m unittest discover -s tests`). It must stay green.

---

## 1. Complete game rules (normative)

The simulator implements exactly these rules. Where a rule was reverse-
engineered from live play, the matching reference implementation in this repo
is cited — the simulator must agree with those references, and §6 defines the
tests that prove it.

### 1.1 Board and tiles

- 4×4 grid. Cell values: `0` (empty), `1` (blue), `2` (red), `3, 6, 12, 24, 48,
  96, 192, 384, 768, 1536, 3072, 6144, 12288`. Rank encoding: rank 0 = empty,
  1 = 1, 2 = 2, and rank k = value `3 * 2**(k-3)` for k ≥ 3 (rank 3 = "3" …
  rank 14 = 6144, rank 15 = terminal 12288).

### 1.2 Merging

- `1 + 2 → 3` (either order). `1+1` and `2+2` do **not** merge.
- `n + n → 2n` for n ≥ 3.
- `6144 + 6144 → 12288` is legal and immediately ends the game; no new tile
  is inserted after that terminal merge. `12288` tiles do not merge further.
- Reference: `state_hunt.can_merge`, `state_hunt.merge_value`.

### 1.3 Move mechanics (this is Threes, not 2048)

A swipe in direction d shifts each line (row for left/right, column for
up/down) **one step at most**, processed from the wall the tiles move toward,
outward:

- For each line, iterate positions from index 1 (adjacent to the destination
  wall) to index 3. A tile at index i moves to i−1 if i−1 is empty, or merges
  into i−1 if mergeable — with the constraints that a cell that received a
  *moved* tile this move cannot also be merged into, and a cell that received
  a *merge* this move cannot receive another merge. Tiles behind a vacated
  cell slide into it (the cascade falls out of the iteration order).
- A move is **legal** iff at least one line changed.
- **Do not rewrite this logic.** Reference implementation:
  `state_hunt.advance_line_toward_start` + `state_hunt.simulate_base_move`.
  Copy the algorithm into `threes_rl/sim.py` (numpy-friendly port is fine) —
  and property-test equality against the original on random boards (§6.1).

### 1.4 Tile insertion (spawn)

After a legal move in direction d:

- Eligible slots: for every line that changed, the trailing-edge cell (the
  cell on the edge the tiles moved *away* from), if it is empty after the
  shift. (Swipe right ⇒ column 0 of each changed row; swipe up ⇒ row 3 of each
  changed column; etc.) Reference: the `eligible_positions` returned by
  `state_hunt.simulate_base_move`.
- The new tile is placed on one eligible slot chosen **uniformly at random**.
- The new tile's value is determined by the current preview (§1.5).

### 1.5 Preview and the tile bag

The player always sees a preview of the next tile. The preview is drawn
*before* the move and is inserted *by* the move; a new preview is drawn after.

- **Small-tile bag**: 12 tiles = 4×blue(1), 4×red(2), 4×gray(3), drawn without
  replacement; refilled when empty. Reference: `window_stream.TileCycle`
  (`small_counts`, `small_pos`, `SMALL_BAG_SIZE = 12`).
- **Bonus tiles** ("large_candidates"): unlocked once the board's max tile is
  ≥ 48. Scheduling (empirical model, validated in live play — must match
  `TileCycle` *exactly*, see §6.2):
  - No bonus can appear among the first `LARGE_DELAY_PREVIEWS = 21` small
    previews of a game.
  - After that, previews are organized in spans of `LARGE_SPAN_SMALLS = 20`
    small previews; each span contains exactly **one** bonus preview, at a
    position uniformly distributed among the 21 possible slots. (Equivalently:
    given the bonus hasn't appeared yet and k smalls of the span have passed,
    P(next preview is the bonus) = 1/(21−k). Implement generatively by
    sampling the slot at span start; the conditional probabilities then match
    `TileCycle.large_probability` automatically.)
  - Bonus previews do not consume from the small bag.
  - If max tile < 48 when a span's bonus slot comes up, no bonus appears
    (model this identically to `TileCycle`: `large_probability() == 0` when
    `max_tile < 48`; the pending bonus stays pending until eligibility).
- **Bonus value**: support = `6, 12, 24, …, max_tile // 4` (min cap 6; see
  `TileCycle.bonus_values`). The preview shows a *window* of 3 consecutive
  support values chosen uniformly among all such windows
  (`TileCycle.bonus_windows`); the inserted value is uniform among the 3 shown.
  If the support has fewer than 3 values, no bonus is possible (matches
  `bonus_windows()` returning empty and `preview_possible` requiring
  `bonus_values()` non-empty).
- The environment state must track bag/schedule with **the same fields** as
  `TileCycle` (`small_counts`, `small_pos`, `small_seen_total`,
  `span_small_pos`, `large_pending`, `max_tile`) or expose a lossless
  conversion to a `TileCycle.snapshot()` tuple — §6.2's tests depend on it.
  Counting semantics are insert-driven exactly as the tracker's: a preview is
  "consumed" from the bag when drawn as preview (generative direction), which
  corresponds one-to-one with the tracker counting it at insertion.

### 1.6 Fresh-game start (data-grounded)

From 40 recorded fresh games (`datasets/*/*/events.jsonl`, `game_start`
events):

- The board starts with exactly **9 tiles**: 8 small tiles + 1 big
  **starter tile**. On the recorded device the starter is always **1536**
  (progress-linked). Make it a constructor parameter
  `starter_tile: int | None = 1536`; `None` means no starter (plain 8-small
  start) — train the headline agent with 1536 to match the real device, but
  keep the option.
- The 8 smalls + the initial visible preview are the first 9 draws of the
  first bag (reference: `window_stream.seed_tile_cycle_from_initial_state`,
  which reconstructs exactly this. Observed per-color counts on the 8 board
  tiles never exceed 4 ✓).
- Tile positions: place the 9 board tiles uniformly at random on distinct
  cells (recorded starts show no positional pattern worth modeling; note this
  assumption in the docstring).
- The starter tile counts toward `max_tile` (with a 1536 starter, bonus
  eligibility is active from the start and the initial bonus support is 6…384).

### 1.7 Game over and score

- Game over ⇔ the board contains the terminal 12288 tile, or no direction
  produces a legal move.
- Score (standard Threes scoring): empty/1/2 score 0; a tile of value
  `v = 3 * 2**k` scores `3**(k+1)` (3→3, 6→9, 12→27, 24→81, …, 1536→59049,
  6144→531441, 12288→1594323). Total score = sum over board tiles at game end.
  Implement `score_board(board) -> int` and `score_tile(value) -> int`.

---

## 2. Package layout

```
threes_rl/
  __init__.py          # exports ThreesSim, ThreesEnv, version
  sim.py               # core simulator: numpy + stdlib ONLY
  env.py               # Gymnasium wrapper (imports gymnasium)
  obs.py               # observation encoders (numpy only)
  baselines.py         # random / greedy policies
  expectimax.py        # depth-limited expectimax with bag-aware chance nodes
  train_ppo.py         # PPO training entrypoint (torch)
  eval.py              # evaluation harness (works for any policy)
  bench.py             # throughput benchmark
  PROGRESS.md
  RESULTS.md
tests/
  test_rl_sim_rules.py     # §6.1
  test_rl_sim_schedule.py  # §6.2
  test_rl_sim_replay.py    # §6.3
  test_rl_env_api.py       # §6.4
```

---

## 3. Core simulator API (`threes_rl/sim.py`)

```python
Direction = int  # 0=up, 1=down, 2=left, 3=right  (order of mirroring_control.DIRECTIONS — verify at import time in tests, not by hardcoding elsewhere)

@dataclass
class Preview:
    kind: str                 # "blue" | "red" | "gray" | "bonus"
    value: Optional[int]      # 1 / 2 / 3 for smalls; None for bonus (hidden until insert)
    candidates: tuple[int, ...] = ()   # the 3 shown values when kind == "bonus"

@dataclass
class SimState:               # immutable-by-convention; step returns a new one or mutates a copy
    board: np.ndarray         # (4,4) int32 values (0 empty)
    preview: Preview
    small_counts: dict[str, int]
    small_pos: int
    small_seen_total: int
    span_small_pos: int
    large_pending: bool
    bonus_slot: Optional[int] # sampled slot of the pending bonus within the current span (generative twin of large_pending)
    max_tile: int
    move_count: int
    game_over: bool

class ThreesSim:
    def __init__(self, rng: np.random.Generator, starter_tile: Optional[int] = 1536): ...
    def reset(self) -> SimState: ...
    def legal_actions(self, state) -> list[int]: ...          # directions with ≥1 changed line
    def step(self, state, action: int) -> tuple[SimState, StepInfo]: ...
    def tile_cycle_snapshot(self, state) -> tuple: ...        # lossless TileCycle.snapshot() equivalent

@dataclass
class StepInfo:
    moved: bool               # False iff action was illegal (state unchanged)
    inserted_value: int
    inserted_pos: tuple[int, int]
    merged_score_delta: int   # score_board(after) - score_board(before)
    eligible_positions: list[tuple[int, int]]
```

Notes:

- `step` on an illegal action returns `moved=False` and the unchanged state —
  never raises. Callers that want strictness use `legal_actions`.
- Keep `step` allocation-light; target ≥ 10,000 steps/s single process on the
  dev machine (M-series). Measure with `bench.py` (§8). If plain Python dicts
  are the bottleneck, switch `small_counts` to a length-3 int array internally.
- Board tokens ↔ tracker tokens: provide
  `board_to_tokens(board) -> list[list[str]]` mapping 0→`·`, 1→`🟦`, 2→`🟥`,
  v≥3→`str(v)` (see `state_hunt.values_to_board`) so tracker oracles can be
  called directly in tests.

---

## 4. Environment API (`threes_rl/env.py`)

Gymnasium `Env` subclass:

- `action_space = Discrete(4)`.
- `observation_space`: `Dict` or flattened `Box` — build from `obs.py`
  encoders (see below). Default encoder: `"full"`.
- `reset(seed=...)` seeds the internal Generator.
- `step(a)`: if `a` illegal → reward `ILLEGAL_PENALTY = -1.0`, state unchanged,
  not terminal (but expose `info["legal_mask"]` every step so masked agents
  never hit this).
- Reward (default): `"final_score"` = 0 per non-terminal legal step and full
  board score at terminal. This is the headline training objective because
  early dense merge rewards can teach premature high-tile merges. Provide
  alternatives behind a constructor arg `reward_mode`: `"score_delta"`,
  `"merge_delta"`, `"log_score_delta"` (`log1p` of delta), `"survival"` (+1 per
  legal move). Log which mode every training run used.
- `info` must always contain: `legal_mask` (4-bool), `score`, `max_tile`,
  `move_count`, and at terminal `final_score`.

Observation encoders (`obs.py`), all pure numpy:

- `"full"` (default, the honest card-counter info set — everything a perfect
  human player could know, which is exactly what the live companion tracks):
  - board one-hot: 4×4×16 (ranks 0–15)
  - preview kind one-hot (4) + bonus-candidate multi-hot over ranks (16)
  - bag counts / 4 (3 floats), span progress `span_small_pos/20` (1),
    `large_pending` (1), smalls-until-bonus-possible normalized (1)
  - flattened `Box` size = 4*4*16 + 4 + 16 + 6 = 282.
- `"board_only"`: board one-hot + preview only (ablation).

Design intent: the RL agent conditions on bag state — the same information the
companion computes live. A strong learned policy is therefore directly usable
as a move hinter later.

---

## 5. Design decisions already made (do not relitigate)

| Question | Decision |
| --- | --- |
| Bonus insert value | sampled uniform among the 3 shown candidates at insertion time |
| Spawn slot | uniform among eligible slots |
| Starter tile | constructor param, default 1536 |
| Illegal action | no-op + −1 reward; mask provided |
| Bag semantics | must round-trip through `TileCycle.snapshot()` |
| Terminal tile | 6144+6144→12288, terminal, scored as 1,594,323 |
| RL algorithm | PPO with invalid-action masking (masked categorical) |
| Network | MLP: obs → 512 → 512 → {policy 4, value 1}, ReLU; try small CNN later only if MLP plateaus |
| Framework | torch (MPS if available, else CPU); gymnasium API |
| Eval seeds | fixed list `range(1000, 1200)` (200 games), greedy/argmax policy |

---

## 6. Validation suite (the heart of the project)

### 6.1 Rule equivalence vs tracker oracle (`test_rl_sim_rules.py`)

- Property test: 20,000 random boards (uniform random ranks incl. empties,
  plus adversarial cases: full boards, single-tile lines, merge chains) ×
  4 directions — the sim's line result and eligible slots must equal
  `state_hunt.simulate_base_move` on the token board, converted both ways.
  Acceptance: **zero** disagreements.
- Every simulated `step` transition, replayed as tokens through
  `state_hunt.validate_transition(before_tokens, direction, preview_label,
  after_tokens)`, must be `valid=True`. Run 50,000 steps across ≥ 200 episodes
  (mixed starter settings). Acceptance: zero invalid. (Preview label mapping:
  1→"blue", 2→"red", 3→"gray", bonus→"large_candidates" =
  `state_hunt.label_for_insert_value`.)
- `score_tile`: exact table check for all 12 values.

### 6.2 Schedule equivalence vs TileCycle (`test_rl_sim_schedule.py`)

- Lock-step test: run 100 episodes; at every step, feed the *insert-derived
  label* into a real `window_stream.TileCycle` (via
  `preview_check_from_snapshot` semantics or direct `update()` calls) and
  assert the sim's `tile_cycle_snapshot()` equals the TileCycle snapshot at
  every move. Acceptance: exact equality, every step.
- Statistical test: over ≥ 200,000 preview draws, (a) each 12-bag contains
  exactly 4/4/4; (b) with max_tile ≥ 48 throughout, bonus gaps: first bonus at
  small-preview index 21+U{0..20}, subsequent gaps consistent with one bonus
  per 20-small span (χ² goodness-of-fit p > 0.001 against the uniform-slot
  model); (c) empirical `P(next=bonus | span position k)` within ±10% relative
  of `1/(21−k)` where sample count ≥ 1,000.

### 6.3 Real-game replay (`test_rl_sim_replay.py`)

- For every `observed_move` event in `datasets/*/*/events.jsonl` with a valid
  single-step `transition_check` (same filter as
  `evaluate_preview_corpus.collect_pairs` uses for session consistency):
  parse `before_board` tokens → values, apply the recorded `direction` with
  the sim's move mechanics, and assert the recorded `after_board` equals one
  of the sim's reachable outcomes (shifted board + recorded inserted value at
  the recorded position, which must be in the sim's eligible set).
  Acceptance: ≥ **99%** of such events reproduce (report the exact number and
  list any failures in PROGRESS.md — a failure here means a real rule gap and
  is the single most important thing to investigate, not paper over).

### 6.4 Env API (`test_rl_env_api.py`)

- Gymnasium API conformance (`check_env` from gymnasium.utils if available,
  guarded import), determinism (two envs, same seed, random-legal policy →
  identical trajectories for 1,000 steps), observation bounds, mask
  correctness (masked actions are exactly the non-changing directions),
  illegal-step semantics, reward-mode switching.

---

## 7. Baselines (`baselines.py`, `expectimax.py`)

1. `RandomPolicy` — uniform over legal actions.
2. `GreedyPolicy` — 1-ply: pick the legal action maximizing
   `score_delta + W_EMPTY * empty_cells_after` (set `W_EMPTY = 1.0`
   initially; record what you use).
3. `ExpectimaxPolicy(depth)` — depth-limited expectimax over the **exact
   spawn model**: chance nodes enumerate (slot × value) with true
   probabilities — slot uniform over eligible; value from the *known preview*
   (deterministic for smalls; uniform over the 3 candidates for a bonus
   preview); the *next* preview distribution from the bag counts +
   bonus-schedule state (this is where bag counting pays off).
   Evaluation function for leaves — start with:
   `score_board + 2.0 * empty_cells + monotonicity_bonus`, where
   monotonicity_bonus rewards the max tile being in a corner with a
   descending gradient (standard Threes/2048 heuristic; document your exact
   formula). Depth 2 must run ≥ 50 moves/s; depth 3 is allowed to be slow —
   eval it on 50 seeds instead of 200 if needed.
   Cache line-shift results (`functools.lru_cache` on line tuples) — this is
   the standard trick and makes depth 3 feasible.
4. Interface: `policy(state: SimState, sim: ThreesSim, rng) -> action`, shared
   by `eval.py`.

---

## 8. Benchmarks and training

### `bench.py`
Prints steps/s for: raw sim (random legal policy), env with `"full"` obs, and
expectimax-d2 moves/s. Record results in RESULTS.md. Targets: sim ≥ 10k
steps/s; env ≥ 5k steps/s. If missed by >2×, profile and fix before training.

### `train_ppo.py`
- Vectorized envs: `gymnasium.vector.SyncVectorEnv` first; switch to
  `AsyncVectorEnv`/custom batching only if throughput-bound (measure first).
- PPO with invalid-action masking (add −inf to masked logits before softmax).
- Starting hyperparameters (log every run's full config as JSON next to its
  checkpoints): 64 envs × 128 rollout steps, γ=1.0, GAE λ=0.95, lr 3e-4
  (linear decay), clip 0.2, entropy 0.01, value coef 0.5, minibatch 4096,
  4 epochs/update, obs `"full"`, reward `"final_score"`, starter 1536.
- Checkpoint every 1M env steps to `threes_rl/runs/<run_name>/`; training must
  be resumable from a checkpoint (test this once deliberately).
- Metrics JSONL per update: env steps, mean episode score, mean moves, max
  tile histogram (fractions reaching ≥192/≥384/≥768/≥1536-merged), losses,
  entropy. A tiny `plot_metrics.py` (matplotlib, saves PNG) is enough — no
  tensorboard requirement.
- Budget guidance: an M-series laptop should sustain ≥ 2k env-steps/s
  end-to-end with this net; 20M steps ≈ 3h. Plan runs accordingly and leave
  the best long run going while you work on other milestones.

### `eval.py`
`python -m threes_rl.eval --policy {random,greedy,expectimax2,expectimax3,ppo:<ckpt>} [--seeds 1000:1200] [--starter 1536]`
→ prints and appends to RESULTS.md: mean/median/p90 final score, mean moves,
max-tile distribution table, and per-seed CSV under `threes_rl/runs/eval/`.
Policies are evaluated on the SAME seed list; the env draws its own RNG from
the seed so all policies face identical start states (chance after that
diverges — that's fine and standard).

---

## 9. Milestones

**M0 — Scaffolding + core sim.**
Package skeleton, `sim.py` complete, §6.1 tests passing.
Verify: `.venv/bin/python -m unittest tests.test_rl_sim_rules -v`.

**M1 — Schedule + replay validation.**
§6.2 exact lock-step + statistics pass; §6.3 replay ≥ 99%.
Verify: `.venv/bin/python -m unittest tests.test_rl_sim_schedule tests.test_rl_sim_replay -v`.
Record replay percentage and any failing events in PROGRESS.md.

**M2 — Env + bench + random baseline.**
§6.4 passes; bench targets met; `eval.py` runs RandomPolicy on the 200-seed
suite; RESULTS.md gets its first table.
Verify: `.venv/bin/python -m threes_rl.bench && .venv/bin/python -m threes_rl.eval --policy random`.

**M3 — Classical baselines.**
Greedy + Expectimax d2 (200 seeds) and d3 (≥ 50 seeds) in RESULTS.md.
Sanity expectations (verify, don't assume): random ≪ greedy ≪ expectimax;
expectimax-d3 should regularly build 384+ tiles. If greedy beats expectimax,
there is a bug — stop and find it.

**M4 — PPO ≥ Greedy.**
A trained checkpoint whose 200-seed mean final score ≥ GreedyPolicy's.
Record the full config, training curve PNG, and eval table.

**M5 — PPO vs Expectimax-d2.**
Target: meet or beat expectimax-d2's mean score. If after ~3 serious runs
(different reward modes / lr / entropy) it still loses, write an analysis in
RESULTS.md (where does it die? max-tile histogram vs expectimax; typical death
boards) and stop tuning — that analysis is a deliverable, not a failure.

**M6 (stretch) — Imitation warm-start.**
Generate 200k expectimax-d2 state→action pairs, pretrain the policy net
(cross-entropy), then fine-tune with PPO. Compare against M4/M5 runs.

**M7 (stretch, design-only) — Companion hint bridge.**
A function `suggest_move(board_tokens, tile_cycle_snapshot, preview_label) ->
{direction, win_probs}` wrapping the best available policy, importable without
torch (expectimax fallback). Do NOT wire it into `live_debug_server.py` — just
make it importable and unit-tested; the human will integrate it.

---

## 10. Final deliverables checklist (what the human will check)

- [ ] `threes_rl/` package as specced; core sim imports only numpy/stdlib
- [ ] All four `test_rl_*` test files pass; full repo suite still green
- [ ] §6.3 replay acceptance ≥ 99% with the number recorded
- [ ] `bench.py` numbers recorded; sim ≥ 10k steps/s
- [ ] RESULTS.md: one table with random / greedy / expectimax-d2 /
      expectimax-d3 / best-PPO on identical seed suites, plus max-tile
      distribution columns and the exact command to reproduce each row
- [ ] Best PPO checkpoint committed path + resumability demonstrated once
- [ ] PROGRESS.md tells the story: every milestone dated, every surprise noted
- [ ] No diffs outside `threes_rl/`, `tests/test_rl_*.py`, `requirements*.txt`
      (and this file, if you correct it — corrections must be flagged in
      PROGRESS.md)
