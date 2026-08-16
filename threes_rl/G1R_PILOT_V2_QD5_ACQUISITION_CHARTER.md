# G1-R Pilot-v2 QD5 Acquisition Charter

Date frozen: 2026-07-25

This authoritative revision supersedes the pre-amendment charter at SHA-256
`06ae8fa29edee9d5e86a6af7e0f63a330fa4cde7d4a0fb8d1f6c2d67ef0daebf`.
That hash remains immutable evidence. The sole amendment makes the pilot yield
projection ancestry-unique across strata.

## Status

This charter authorizes implementation, tests, and one immutable no-game
preflight for a fresh five-family G1-R acquisition pilot. It does not authorize
pilot execution, normal-start game generation, acquisition-stream consumption,
action labels, h40 outcomes, model fitting, continuations, score or policy
outcome inspection, incumbent changes, or dashboard changes.

The output identity is:

`threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5`

The original `threes_rl/g1r_acquire.py`,
`tests/test_rl_g1r_acquire.py`, and complete `pilot_v1` directory are immutable
inputs and must remain byte-for-byte unchanged.

S3 remains `HOLD_UNDERPOWERED_PREFLIGHT`, not a utility failure. Preserve:

- `S3_POWER_PREFLIGHT_V2_SEALED.json`, SHA-256
  `4dabd5325dcbbc5234c4e015eccbd4d5f4706be9fefa54fd5220d8720b1fc345`;
- `S3_PROVENANCE_SEAL_V2.json`, SHA-256
  `5326f25b50ad33b4e00eb5ca7180468d3a243917075d15d377a1511b04867949`.

C2 remains held and exact depth-3 utility remains unknown.

## Frozen Family Slate

The exact order, names, and policies are:

1. `g1r_corner2`: `corner2`.
2. `g1r_expectimax2`: `expectimax2`.
3. `g1r_parent_mc1000`: `ntuple_expectimax2:` plus
   `threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`.
   This represents the spent parent/student/incumbent alias component only.
4. `g1r_replaycal`: `ntuple_expectimax2:` plus
   `threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest`.
5. `g1r_qd_static_archive_oneply_v2_terminal_schema`: the sealed policy bundle
   at
   `threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema/policy`.

QD-v2 is admitted only as an acquisition behavior family by the conjunction of
immutable `READY_QD_FAMILY_ADMISSION` and
`READY_QD_STORAGE_ADMISSION`. It is not a gameplay incumbent.

The preflight binds every resolved policy spec, policy/source implementation
file, checkpoint file, QD archive/source/policy payload, QD execution lock,
admission marker/result, supplemental storage audit, and their file/canonical
payload hashes.

## Action-only Identity Audit

Use only the complete immutable pilot-v1 preflight panel:

- `32` pre1536 and `32` pre3072 natural states;
- panel SHA-256
  `b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d`.

Recompute each policy's deterministic action twice per state without timing.
Reference policies use the frozen exact value and lowest-action-index tie rule.
QD uses its frozen ordinal quality-rank plus novelty-rank decision and action
priority. State/context must remain unchanged. The five signature hashes and
all pairwise overall/pre1536/pre3072 rates must exactly reproduce the accepted
pilot-v1 and QD-v2 admission artifacts. Every pair must remain at least `2%`
different overall and nonzero in both strata.

Do not read any partial pilot-v1 acquisition file or rerun either QD admission.

## Fresh Pilot Streams

The exact four bases are:

- logical seed: `49_000_000_000`;
- deck stream: `50_000_000_000`;
- slot stream: `51_000_000_000`;
- policy stream: `52_000_000_000`.

For family index `f` and game index `g`, where `f in 0..4` and `g in 0..19`,
the stream offset is `f * 1_000_000 + g`. The immutable requested manifest has
exactly `100` rows in frozen family order, then game-index order.

Before sealing preflight, scan the complete historical JSON/JSONL/CSV union
beneath `threes_rl/runs`, including every prior lock, marker, result, replay,
eval artifact, continuation, diagnostic, and manifest. Logical seeds also
collide with historical `seed`, `root_seed`, `source_seed`, and canonical
`fresh:<seed>:1536`. Record every matched source file/hash/count and require
zero historical and internal collisions. No requested pilot stream is consumed
in preflight.

## Conditional Pilot Contract

A later, separately authorized pilot consists of exactly `20` complete
normal-start games per family, `100` total. It uses split exogenous reset
streams, one worker, process nice at least `10`, deterministic chunks no larger
than `8`, at most `12` active wall-hours, and at most `4 GiB`. Pause below
`100 GiB` free and target more than `120 GiB`.

Only completed fresh normal-start games are valid. Human, partial, restart,
continuation, replay-start, and synthetic roots are forbidden. There is no
within-family early stopping.

Policy comparison remains outcome-free. Inspect only:

- completion and simulator/provenance integrity;
- exact pre1536/pre3072 rung availability;
- whole-ancestry/root uniqueness;
- frozen `source_success_window` versus `source_control` coverage metadata;
- the preregistered conservative yield projection.

Never filter by score, final state, favorable replay, family, or file size.
Never tune a policy, QD archive/objective, threshold, or family allocation from
pilot yields. Before computing any yield count, pool every eligible pre1536 and
pre3072 candidate from one whole ancestry and select exactly one candidate by
the global lexicographic argmin:

`SHA256("G1R-pilot-v2-root-cap"|root|stratum|frame|state_hash)`.

This tie rule is applied across strata, not separately within each stratum. A
root that reaches both rungs contributes to only the selected stratum in the
pilot feasibility projection. Preserved candidates remain available to a later
final allocator under that allocator's separately frozen contract.

`source_success_window` means the direct next milestone occurs within the next
40 recorded moves; otherwise the metadata role is `source_control`. Role is
coverage metadata only and may not become a policy input, weight, or label.

## Frozen Yield Projection

After all 100 games complete, for each family `f` let `n_f` be complete roots
and let `k_f,1536`, `k_f,3072`, and `k_f,any` be counts after the global
cross-stratum root cap above. The executable conservation invariant is:

`k_f,1536 + k_f,3072 == k_f,any`

for every family. Any duplicate root, cross-stratum double count, or invariant
failure invalidates the projection.
Use the 90% Wilson lower bound with
`z=1.6448536269514722`:

`center=(p+z^2/(2n))/(1+z^2/n)`

`half=z*sqrt(p*(1-p)/n+z^2/(4n^2))/(1+z^2/n)`

`L=max(0,center-half)`, with `L=0` when `n=0`.

Let `B=12000-sum_f(n_f)`. Allocate `floor(B/5)` projected attempts per family
and the first `B mod 5` residual attempts in frozen family order. Project each
family/stratum count as `k_f,s + floor(projected_attempts_f * L_f,s)`.

The projection passes only when pre1536 is at least `432`, pre3072 is at least
`432`, and any-rung unique roots are at least `864`. This is a budget
feasibility check only. It cannot produce a G1 corpus, policy comparison,
label/model authorization, or promotion.

## No-game Preflight Gate

Prepare in a fresh staging directory and atomically promote only when all
checks pass:

- exact five-family order, loadability, complete artifact hashes, and accepted
  64-state signatures/pairwise rates;
- QD execution lock, admission marker/result, storage audit, exactness and
  deterministic policy bundle bindings;
- split exogenous reset fixture and exact state round-trip;
- exact `100`-row stream manifest/hash and zero complete-history collisions;
- frozen one worker, nice at least `10`, no competing heavy Threes process;
- accepted conservative storage projection below `4 GiB`;
- free disk strictly above `120 GiB`;
- ports `8765` and `8770`, advisor ready, dashboard record `263670`, and
  protected top-three replays healthy;
- S3 seals, pilot-v1, original acquisition implementation/tests, incumbent,
  and all QD-v1/v2 evidence unchanged;
- no active or completed human-session content read;
- focused pilot-v2 tests and relevant G1/S3/QD regressions passed.

The immutable lock records canonical payload hash, implementation/test/charter
hashes, exact test commands/counts, stream-union count/hash, policies,
signatures, disk/services, and explicit zero-work counters.

Any failed scientific or integrity check leaves a preserved staging HOLD. The
real output may never be silently overwritten or retried. A passing preflight
authorizes no games by itself.
