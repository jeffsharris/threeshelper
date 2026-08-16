# G1-R Natural Root Acquisition Charter

Date frozen: 2026-07-25

Status: sole active branch after the outcome-free
`HOLD_G1_DATA_PREFLIGHT`. This charter authorizes replay/state acquisition
only. It does not authorize all-action h40 labels, model fitting, test outcome
opening, policy evaluation, or dashboard claims.

## Motivation and Immutable Inputs

The authoritative retained-corpus audit is
`runs/forensics/g1_relational/G1_EXISTING_CORPUS_PREFLIGHT_V5_AUTHORITATIVE.json`
(SHA-256 recorded in the experiment log). After excluding all prior roots plus
both sides of the S3 inventory, it found zero eligible natural roots.

The corrected G1 power contract requires:

- train: `256` independent roots;
- validation: `96` independent roots;
- untouched test: `512` independent roots;
- total: `864` roots, at most one state per whole ancestry;
- test: `256` pre-1536 and `256` pre-3072 roots;
- assumed selector activity `30%`, eight future action-label repeats;
- policy-level common OR `1.50`, implied active-root OR `7.20265`;
- median simulated OR `1.50257`, log calibration error `0.00171`;
- influence and 199-bootstrap complete-PASS power both above `99%`.

All G1 feature, model, exclusion, and untouched-test locks remain unchanged.
The dashboard record remains `263670`.

## Fresh Normal-Start Slate

Every generated game starts from `ThreesSim.reset()` with starter `1536` at
the top-left. Restarts, continuation states, human actions, source replay
starts, and synthetic boards are forbidden.

Six nominal frozen policy specifications enter an outcome-free behavioral
distinctness audit:

1. `g1r_corner2`: `corner2`;
2. `g1r_expectimax2`: `expectimax2`;
3. `g1r_parent_mc1000`:
   `ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`;
4. `g1r_student1`:
   `ntuple_expectimax2:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest`;
5. `g1r_replaycal`:
   `ntuple_expectimax2:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest`;
6. `g1r_incumbent_depth2`: the exact single active line in
   `current_incumbent_policy.txt`.

Before generation, all six policies are loaded and every referenced checkpoint
file (metadata/configuration and table payloads) plus the policy implementation
sources are hashed. Their actions are then evaluated, without continuations or
score outcomes, on a fixed diagnostic panel of `32 pre1536 + 32 pre3072`
naturally reachable states selected from prior excluded evidence. The panel is
root-capped, exact-restored, balanced by stratum, and hashed before actions.
Every legal action is evaluated with the policy's frozen calculation; exact
maximum ties choose the lowest simulator action index.

Two nominal policies are behavior aliases if their action disagreement is
below `2%` overall or zero in either stratum. Alias edges are transitively
collapsed, and the earliest policy in the ordered slate is the deterministic
representative. Generation requires at least five genuine components. It uses
one representative per component; checkpoint names alone never establish a
family. The distinctness panel and its action signatures are diagnostic only.

These resulting component names are acquisition-family identities, not aliases
for old family holdouts. No genuine family may contribute more than `40%` of
any eventual G1 partition.

## Streams and Ancestry

Family index is `0..5`; game index starts at zero. Stream IDs are:

- logical/root seed: `41_000_000_000 + family*1_000_000 + game`;
- deck: `42_000_000_000 + family*1_000_000 + game`;
- slot: `43_000_000_000 + family*1_000_000 + game`;
- policy/tie: `44_000_000_000 + family*1_000_000 + game`.

Before generation, hash the complete requested stream manifest and prove zero
collision with the reviewed union of JSON/JSONL/CSV provenance and stream
artifacts under `threes_rl/runs`, including eval manifests, eval artifacts,
replays, continuations, forensics, diagnostics, and completed training-run
manifests. Logical seeds are checked against historical `seed`, `root_seed`,
`source_seed`, and canonical `fresh:<seed>:1536` values. The union's matching
source files, byte sizes, hashes, and key counts are frozen in preflight. The
active human session is not read.

All games use split evaluator `split_exogenous_v1`. The canonical ancestry is
the new logical seed plus the genuine family and reset invariant. Replay copies
never create new roots.

## Deterministic Extraction

Only completed games are inspected. A game is eligible if it contains at least
one live exact-rung frame:

- `pre1536`: built maximum excluding the fixed starter equals `768`;
- `pre3072`: built maximum excluding the fixed starter equals `1536`.

For each root/rung, choose the frame with the lexicographically smallest
`SHA256("G1R-state-v1"|root|stratum|frame|state_hash)`. The final allocator
uses at most one selected state per root, with these frozen targets:

- untouched test first: `256 pre1536 + 256 pre3072`;
- validation second: `48 pre1536 + 48 pre3072`;
- train third: `128 pre1536 + 128 pre3072`.

Within each partition, allocate in stratum order `pre1536, pre3072` and role
order `source_success_window, source_control`. First fill ten roots in each
stratum-role cell. Then fill the remaining stratum target. At each choice,
select the eligible candidate with the lowest current genuine-family count,
then the lexicographically smallest
`SHA256("G1R-allocate-v1"|partition|root|stratum|role|record_id)`.
The per-partition family cap is `floor(0.40 * partition_size)`. A selected root
is removed from every later cell and partition. This ordering and every tie are
fixed; `864` pooled roots alone are never a stop condition.

READY requires the exact `256/96/512` partition sizes and stratum targets,
at least ten roots in every stratum-role cell in every partition, at least five
genuine families overall, at least three test families, the `40%` cap in every
partition, one state per root, and zero partition/source/root/stream collision.
The resulting compact manifest and assignment hash are mandatory. State
payloads must round-trip board, score, preview, cycle/deck, pending, starter,
move count, legal mask, and game-over status exactly.

Historical role is frozen from the same completed source replay:

- `source_success_window`: direct milestone occurs in the next 40 recorded
  moves;
- `source_control`: it does not.

Role is coverage metadata only and may not enter G1 features, action labels, or
weights. Preserve at least ten roots in every stratum-role cell of every final
partition.

## Acquisition Schedule

1. Preflight hashes policies, streams, extraction code, storage, and services.
2. Pilot exactly `20` completed games per genuine representative (at most
   `120` total), in frozen representative order.
3. If at least five families are simulator-valid, continue in balanced rounds
   of `100` completed games per viable family.
4. After each full round, rebuild a root-capped candidate manifest and attempt
   the frozen `256/96/512` whole-root partition. No all-action outcome exists.
5. Stop `READY_G1R_ROOTS` only when an actual hashed partition passes every G1
   stratum-role/family/provenance/collision rule.

Prefer roots over repeats. Acquisition does not stop on pooled record count.

## Scarcity and Remedy Rules

- Report family-by-stratum yield after each full round. A family with zero
  pre-3072 roots after `200` completed games is declared unable to supply that
  stratum; it is not renamed or split into aliases.
- Not every overall family must supply pre3072. The final corpus still requires
  at least five genuine families overall, at least three genuine test families,
  every frozen stratum-role cell, and the per-partition cap. Scarcity stops only
  when the deterministic partition is infeasible within the bounded slate, or
  fewer than three genuine pre3072 suppliers make the test rule impossible.
- The only preauthorized remedy is a new, separately hashed normal-start
  quality-diversity acquisition policy family that returns one complete
  trajectory per fresh root. It may archive/search internally but may not
  start from a retained state or contribute more than one ancestry sample per
  reset. Its objective and runtime gate must be frozen before generation.
- Do not use human games, policy aliases, stochastic temperature variants, or
  descendants of one reset to manufacture family/root counts.

## Runtime, Storage, and Stop Rules

- Pilot budget: at most `12` active wall-hours and `4 GiB`.
- Full fixed-slate budget: at most `12,000` completed games, `72` wall-hours,
  and `20 GiB`.
- Retain compact results for every game. Retain complete source replays only
  for roots with an exact eligible rung; those replays are provenance
  protected. Do not render HTML or retain periodic model/checkpoint payloads.
- One heavy process at a time. The process runs at verified nice priority
  `>=10` with one or two frozen workers; worker children inherit that priority.
  Games are processed in deterministic chunks of at most
  eight, and each completed result is fsync-checkpointed before the next chunk.
  A crash may replay only the incomplete chunk; duplicate checkpoint rows are
  rejected.
- Pause below `100 GiB` free; target `120 GiB`.
- Check disk, active runtime, output bytes, dashboard port `8765`, advisor
  health port `8770`, dashboard record `263670`, and protected top-three
  existence before and between chunks. Health reads only `/api/health`; it
  never reads, modifies, or interrupts an active human session.
- A simulator, stream, provenance, replay-resume, or state-round-trip failure
  stops generation until the same frozen engineering contract is repaired.
- Partial games and active recorder sessions are never inspected or modified.

The immutable preflight is bound to one resolved output directory and freezes
the worker count at one or two. A later larger lock in that same directory must
prove every completed row is an exact stream, policy-spec, and genuine-family
subset. A lock cannot be used in another directory. Before the first new game
of every acquisition/resume invocation, the current policy payloads are
rehashed against the frozen policy lock and the complete historical stream
collision audit is rerun.

The immutable preflight records this charter hash, the authoritative G1 V5
preflight hash, acquisition implementation and focused-test hashes, resolved
incumbent policy hash, complete policy artifact manifests, diagnostic panel and
action-signature hashes, historical collision-union sources, stream manifest,
round-trip fixture, free disk, service truth, and dashboard/top-three truth.
Pilot and later-round preflight locks have distinct filenames and may not be
overwritten.

Immediately before `READY_G1R_ROOTS`, every retained replay is reopened, its
file hash is rechecked, and every selected frame is exact-round-tripped and
matched to its frozen state hash and payload. Any mismatch yields
`HOLD_G1R_INTEGRITY`, never READY.

## Decisions

- `CONTINUE_G1R`: a complete round is valid but the actual partition is not
  ready; continue within the frozen game/time/storage budgets.
- `READY_G1R_ROOTS`: freeze the whole-root partition and return to the original
  G1 train/validation label sequence. No test labels yet.
- `HOLD_G1R_FAMILY_SCARCITY`: freeze evidence and open only the separately
  preregistered quality-diversity remedy.
- `HOLD_G1R_BUDGET`: stop without relaxing roots, families, roles, or power.

No acquisition score or replay is normal-start capability evidence or
dashboard eligible.
