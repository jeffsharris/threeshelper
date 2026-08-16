# Threes RL Program Handoff

Status date: 2026-08-15

This is the current entry point for a new training/research agent. Historical
details remain in `EXPERIMENT_LOG.md`, decisions in
`CURRENT_DECISION_LEDGER.md`, and immutable local-artifact rules in
`ARTIFACT_RETENTION.md`.

## Executive State

- There is no active or authorized training run.
- The protected software incumbent remains
  `ntuple_phaseblend_expectimax2`, specified exactly in
  `current_incumbent_policy.txt`.
- The protected dashboard top three remain `263670 / 261369 / 258561`.
- No learned neural checkpoint from J1/J2 is authorized for evaluation or
  reuse.
- The latest course boundary is
  `HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM`. V3A1 repaired the prior
  post-seal reproducibility issue, but launch headroom was only `2.481632 GiB`
  above the hard 100-GiB floor. It needed another `2.518368 GiB` to satisfy the
  frozen 5-GiB cushion. That recovery was never opened or executed.
- The 2026-08-15 cleanup restored free disk to `124.95 GiB`, but does not
  reopen or reinterpret that spent HOLD. Cleanup details are in
  `STORAGE_CLEANUP_20260815.md`; the full current allocation is in
  `STORAGE_AUDIT_20260815.md`.
- The local `threes_rl/runs/` tree is evidence/data, not source. It is ignored
  by Git except for the small historical baselines already tracked. Do not
  delete anything under `runs/forensics/` without a reviewed retention
  manifest.

## What Worked

### Simulator and harness

- Move mechanics, tile-cycle behavior, preview/deck state, legal actions,
  scoring, deterministic streams, replay reconstruction, and the terminal
  `6144 + 6144 -> 12288` rule have broad test coverage.
- Normal starts are supported with `starter_tile=None`; historical corrected
  starts with the fixed top-left `1536` remain available for old comparisons.
- Evaluation can use deterministic root/stream manifests, paired common-random
  streams, whole-ancestry partitions, retained replays, and score/progression
  summaries.
- The later execution surfaces added create-once artifacts, phase barriers,
  exact resume, authenticated commit chains, root-equal PPO weighting, bounded
  rolling storage, retirement manifests, and fail-closed service/resource
  guards. These engineering components are reusable even though the associated
  scientific branches did not promote.

### Incumbent construction

The strongest deployed actor came from classical/search-guided learning:

1. A `corner2`-generated MC1000 n-tuple parent.
2. A student n-step/temporal-coherence table blended at `0.25` in all phases.
3. A replay-calibration table blended at `0.05` from midgame.
4. An endgame action-label sidecar blended at `0.10` in endgame.
5. Depth-2 expectimax over the composite value function.

This produced the `263670` record and a much stronger actor than random,
greedy, shallow imitation, or early pure TD. The exact four local model paths
are in `current_incumbent_policy.txt` and must remain together.

### Scientific/engineering lessons that held up

- Strong-actor trajectories were materially better than short pure self-play
  for n-tuple value learning.
- Root ancestry, not frame count, is the effective sample unit.
- Paired shared-stream evaluation and untouched confirmation prevented several
  attractive development effects from being mistaken for promotion evidence.
- The exact-teacher engineering pilot passed serial/8-process action equality,
  deterministic merge/barrier checks, transient-label retention checks, and
  real throughput/memory measurement. Its power audit selected `6144` paired
  roots for the full-policy fidelity gate.
- J1d completed a full from-scratch run with clean execution integrity:
  `16384` roots, `780/780` optimizer steps, `64/64` canonical metric commits,
  and zero abandoned attempts. This proved that the whole-game PPO harness can
  run deterministically end to end.

## What Did Not Work

### Search, selectors, and designated-pair courses

- First-action selectors, selective rollout, and the tested MCTS/UCT variants
  did not produce stable paired endpoint gains.
- Exact/adaptive depth-3 search was behaviorally active but missed its frozen
  runtime gates; optimized variants still failed tail-latency admission.
- O1-O6 designated-pair/stage/competing-risk courses encountered support,
  representation, label-domain, or orchestration gates before producing a
  promotable policy. Their negative results do not justify weakening their
  preregistered gates.
- Positive local continuation conversion rates were conditioned on rare
  successful states and did not establish normal-start capability.

### Neural training

- Sparse PPO from scratch had poor credit assignment and could collapse toward
  short games.
- J1d's full whole-game PPO run improved the actor metric slightly but failed
  the frozen learning-sanity gate:
  - final-four mean log score `5.5082024177` versus first-four `5.4267635090`;
  - final legal entropy `1.0578262871` nats;
  - value MSE `3.413235e-5` versus zero-predictor `6.973465e-6` (about `4.90x`
    worse);
  - only `1/3` auxiliary Brier scores beat prevalence baselines.
  The checkpoint is quarantined and must not be evaluated or reused.
- J1c was killed by metric-authentication reduction-order disagreement, not by
  a scientific result. J1d repaired that exact problem, then produced the clean
  learning-sanity HOLD above.
- J2's incumbent-distillation design passed outcome-free readiness only after
  increasing closed-loop fidelity validation from `2048` to `6144` pairs. It
  never completed distillation.
- J2A1 V2 collected `3048/14336` exact-teacher roots, then stopped because the
  72-hour cap incorrectly summed eight workers' CPU time instead of top-level
  wall time. No optimizer step, checkpoint, family gate, mechanism gate, or
  student fidelity outcome was opened. Those roots and streams are spent and
  protected.
- V3 corrected wall-time accounting and projected Stage A at `42.6028h` point
  / `50.9891h` conservative, but the V3 surface's post-seal test chronology was
  non-reproducible. V3A1 fixed that chronology and then honestly held on disk
  headroom. Neither V3 nor V3A1 executed science.

### General failure pattern

The project invested heavily in immutable orchestration because long runs had
several real serialization, ownership, timing, and accounting defects. That
work made the evidence trustworthy, but the scientific bottleneck remains:
from-scratch PPO did not bootstrap a useful critic, while teacher-distilled PPO
never reached training. A new agent should reuse the simulator and evaluation
infrastructure, but is not obliged to continue the exact J2A1 course.

## Locked and Quarantined Material

- Do not reuse J1c or J1d checkpoints, optimizer state, roots, transitions, or
  streams.
- Do not inspect or reuse O3/O5 protected episode bodies or quarantined
  checkpoints.
- Do not retry J2A1 V2 in place or re-query its `3048` completed roots.
- The killed R1 and unpromoted R1b learned tables were pruned under a reviewed
  manifest; their compact attribution and confirmation evidence remains.
- Do not use human actions as labels. Human-session content remains outside the
  machine-only scientific program.
- Do not promote from a mechanism, continuation, development, or high-score
  result alone. Promotion requires a separately sealed paired normal-start
  confirmation.

These restrictions protect the meaning of the existing evidence. A genuinely
new course should allocate fresh root/stream authorities rather than laundering
spent evidence through a new name.

## How to Run the Harness

### Environment

```bash
cd /path/to/threeshelper
uv venv
uv pip sync requirements.lock.txt
```

The local checkout already has `.venv`; it is ignored by Git.

### Core verification

Run the artifact-independent simulator/evaluator tests first:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_rl_sim_rules.py \
  tests/test_rl_sim_schedule.py \
  tests/test_rl_expectimax.py \
  tests/test_rl_ntuple.py \
  tests/test_rl_eval_metrics.py
```

Run the simulator/search benchmark:

```bash
PYTHONPATH=. .venv/bin/python -m threes_rl.bench
```

Some historical governance tests intentionally require the local hash-bound
forensic tree. A fresh clone can run core tests without it; copying or deleting
forensic artifacts changes which historical tests are applicable.

### Evaluate code-only baselines

Normal-start (`starter_tile=None`) smoke:

```bash
PYTHONPATH=. .venv/bin/python -m threes_rl.eval \
  --policy corner2 \
  --starter none \
  --seeds 1000:1010 \
  --no-append
```

Fixed-starter historical comparison:

```bash
PYTHONPATH=. .venv/bin/python -m threes_rl.eval \
  --policy expectimax2 \
  --starter 1536 \
  --seeds 1000:1010 \
  --no-append
```

### Evaluate the protected incumbent

This requires the four local run directories referenced by
`current_incumbent_policy.txt`. With those artifacts present:

```bash
POLICY="$(tail -n 1 threes_rl/current_incumbent_policy.txt)"
PYTHONPATH=. .venv/bin/python -m threes_rl.eval \
  --policy "$POLICY" \
  --starter none \
  --seeds 2000:2010 \
  --jobs 1 \
  --no-append
```

Use a new, explicit artifact directory for retained eval output; do not write
into a sealed forensic namespace.

### Current V3A1 audit surface

V3A1 is a held readiness package, not a launch command. Its safe command is:

```bash
PYTHONPATH=. .venv/bin/python \
  -m threes_rl.j2a1_distillation_fidelity_recovery_execution_surface_v3a1 \
  audit-post-seal
```

Do not infer execution authorization from a passing audit. The authoritative
result remains HOLD.

## Recommended Next-Agent Starting Point

1. Re-run the core simulator tests and a small normal-start baseline.
2. Read this file, `CURRENT_DECISION_LEDGER.md`, and the final J1d/J2A1 entries
   in `ARTIFACT_RETENTION.md`; use the full experiment log only for disputed
   history.
3. Treat the current incumbent as the teacher/control, not as proof that its
   own self-play distribution is sufficient.
4. Propose one materially different training regime with fresh authorities and
   a cheap outcome-free feasibility test. Good candidates include a simpler
   value target, a stronger critic bootstrap, or a search-improvement loop that
   first proves the teacher improves full-policy outcomes.
5. Keep whole ancestries in one partition and preregister sustained full-policy
   paired development plus a fresh confirmation. Do not optimize a one-action
   proxy and call it gameplay improvement.

The engineering harness is substantially stronger than the latest learner.
That is the useful inheritance: trust the simulator, manifests, paired
evaluation, and failure accounting; challenge the training objective.

## Handoff Verification

Verification on 2026-08-15:

- `compileall` over `threes_rl` and `tests`: PASS.
- Artifact-independent simulator/evaluator selection: `102 passed`.
- Current V3A1 focused suite: `43 passed`.
- Git object integrity after cleanup: PASS, zero garbage.
- Workspace cleanup: `15.01 GiB` reclaimed and `124.95 GiB` free; see
  `STORAGE_CLEANUP_20260815.md`.
- Unfiltered historical suite: `1804 passed`, `69 failed`, `1 skipped`.
  The failures are expected chronology/operational incompatibilities in old
  tests: they assert that later-created immutable namespaces are still absent,
  or require `nice`/localhost service access unavailable inside the test
  sandbox. Do not report the unfiltered suite as globally green. Run the
  focused source suites or the documented historical deselection sets instead.
