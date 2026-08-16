# K1 Root-Diverse Support Audit Charter

Status: frozen before opening K1-v1 replay state content.

Date: 2026-07-26

## Purpose

This compact, outcome-free audit asks whether immutable existing evidence can
support a materially different K2 fresh engineering design with exactly one
natural trigger state per independent whole-game root. K1-v1 is spent and may
not be rerun, extended, retimed, or reinterpreted. Its marker and 73B-76B
streams remain spent.

The audit cannot compile code, time a policy, consume a stream, generate a
game, produce a policy outcome, or authorize K2 execution.

## Immutable Inputs

Primary inputs are the sealed files under
`threes_rl/runs/forensics/k1_compiled_kernel_v1`:

- `K1_PREFLIGHT_LOCK.json`
- `K1_EXECUTION_OPENED.json`
- `K1_TERMINAL_RESULT.json`
- `K1_STREAM_MANIFEST.json`
- `K1_POLICY_LOCK.json`
- `K1_EXCLUSION_MANIFEST.json`
- `completed_games.jsonl`
- retained `source_replays/*.json`
- retained `selected_states/*.json`

The audit must hash every input it reads. C1, C2, and K1 artifacts are
read-only. The C2 untouched runtime partition and absent K1 fresh timing
partition may be checked only for unopened status.

Alternative-family evidence is limited to existing immutable action-signature
or family-admission records and existing normal-start source/corpus metadata.
Human, partial, restart, continuation, synthetic, score-selected, and policy
outcome sources are excluded.

## K1 Eligibility Predicate

For each retained natural replay frame:

1. restore the exact simulator state and prove a round trip;
2. require a live state with at least one legal action;
3. require a non-starter built maximum for which the frozen K1 milestone
   function is defined;
4. require either the unchanged low-empty trigger or unchanged low-margin
   trigger from K1-v1.

The frozen incumbent depth-2 computation may be executed only to reduce the
low-margin test to a boolean. The audit must not serialize or report action
values, actions, margins, timings, scores, future milestones, or any policy
outcome. K1/C1 depth-3 values are never computed.

Every completed game is one whole-root ancestry. A root without an immutable
retained replay is `unobservable`, not zero-support. No missing replay may be
recreated or reacquired.

## Frozen Statistics

For each behavior family report:

- attempted/completed/retained/unobservable independent roots;
- observed retained roots with at least 1, 2, 3, and 4 qualifying states;
- the qualifying-state-count distribution over retained roots;
- one earliest qualifying frame identity per retained root, consisting only
  of root ancestry, frame index, and state hash;
- the observed lower bound and the maximum possible bound after treating all
  unobservable roots as unknown.

The deterministic one-state-per-root proposal selects:

`argmin SHA256("K2-root-state-v1"|root|frame_index|state_hash)`

over all qualifying states in that root. No score, outcome, action, value,
margin, timing, or future information participates.

## Alternative Families

An alternative collector counts as genuinely distinct only when immutable
action-signature evidence shows at least 2% overall disagreement and nonzero
disagreement in both frozen strata against every family in the selected
three-family slate.

Pre-existing trigger support counts only when immutable normal-start
whole-root metadata provides an exact reconstructable natural state that
passes the same K1 predicate. Merely naming a checkpoint, replay family, exact
rung, or action signature is not support evidence. Prior untouched
confirmation/gate roots remain diagnostic-only and cannot be proposed as K2
gate states.

## Decision

Seal `READY_K2_ROOT_DIVERSE_PROPOSAL` only if all conditions pass:

- at least three genuinely distinct behavior families;
- at least 12 observed independent roots with one or more qualifying states
  in every selected family; 16 per family is the preferred target and is
  reported descriptively;
- no family exceeds 40% of proposed roots;
- exact source, ancestry, state, stream, and prior-overlap integrity;
- a deterministic one-state-per-root manifest can be constructed without
  using a prior untouched gate or any unobservable root.

If observed immutable support cannot meet the 12-root floor, missing content
cannot be imputed, or alternative evidence lacks exact K1-compatible trigger
support, seal `KILL_EXACT_DEPTH3_PROGRAM`.

A READY seal may contain a proposal only: target 16 roots per family with a
hard minimum of 12, one state per root, the unchanged K1 exactness/runtime
gates, and a recommended fresh stream namespace subject to a future complete
collision audit. It never authorizes compilation, acquisition, timing, or
policy evaluation.

## Forbidden Outputs

The audit artifact must attest zero new games, streams, compilations, timings,
depth-3 values, policy outcomes, scores inspected, actions inspected, labels,
models, incumbent changes, dashboard changes, and human-action use.
