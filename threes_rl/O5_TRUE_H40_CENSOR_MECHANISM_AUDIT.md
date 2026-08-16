# O5 True-H40-Censor Mechanism Audit

Date: 2026-07-27

Decision: `HOLD_O5_TRAINING_DATA_SUPPORT` remains authoritative.

## Scope And Immutable Inputs

This audit is read-only and aggregate-only. It opened no episode array,
per-episode metadata, action, prediction, checkpoint payload, development or
untouched root, O3 artifact, or policy outcome.

Bound immutable identities:

- O5 V2 charter:
  `0b274979e388f5e0297c17d85264193caf2186794024b14d500f504ff7a7aede`
- O5 V2 runner:
  `37e0a20d2437f09ef7efe1073573f7f53c4ed8ae0267560192e7892164e956ea`
- O5 V2 tests:
  `5006d11984ce927d72729ef065ad3bfd3772750303d62208f37e4872e5ca7e27`
- Aggregate support file:
  `bd905b62f05c95c42dc36336f9133d6d80044687476be396161d43ada10a7a94`
- Checkpoint quarantine file:
  `96a5336f3a9c37dad56447ceedf9481cd39fe0d6f896effa5f47b07b9c461ece`
- Terminal result file:
  `74ac4ca9f375ff93e2fed5dfa5c2154a7b4fcc682654539e05cc67cc4a515e05`

## Frozen Semantics

`generate_episode()` initializes the terminal state as an administrative
h40 censor. For each of at most 40 exact simulator transitions,
`transition_status()` returns:

- `success` only for the safe pair-specific designated merge;
- `failure` for a designated or third-party merge that is not a safe success,
  invalid lineage, game over, anchor/air violation, or no legal action;
- `live` otherwise.

Only 40 consecutive `live` transitions reach the loop's administrative
`censor` branch. `_support_report()` counts a true h40 censor exactly when
`terminal_status == "censor"` and `terminal_move == 40`.

The source branches and counter are internally consistent. Frozen domain tests
exercise legal `live` successors, so immediate censor impossibility is not
encoded by the transition definition.

## Aggregate Evidence And Attribution

The sealed aggregate report contains exactly:

- 1,152 episodes;
- 188 safe-merge successes;
- 964 failures;
- 0 true h40 censors.

Because `188 + 964 == 1,152`, every trajectory reached one of the two
competing absorbing events by or at h40. There is no unclassified residue for
the administrative-censor category.

Classification:

- code or counter bug: not supported;
- structural impossibility in the source definition: not supported;
- complete early resolution in the selected root/task population: supported;
- further cause or timing decomposition: unavailable without forbidden
  per-episode metadata and therefore not attempted.

The failed quota treated administrative censor mass as mandatory even though
complete competing-event resolution is valid data. That makes the exact O5
support contract unsuitable for this observed event process. It does not make
the quarantined model usable and does not establish policy utility.

## Decision

O5 remains spent, immutable, and held. All checkpoints remain quarantined.
No threshold is relaxed, no O5 artifact is reopened, and no downstream gate
is authorized.

The material mechanism finding is:
`O5_CENSOR_MISS_IS_COMPLETE_COMPETING_EVENT_RESOLUTION`.

`CONTINUE=proposal-only O6 competing-risks preflight design`;
`HOLD=all labels/training/evaluation`; `KILL=false` for the broader
designated-pair hypothesis; `PROMOTE=false`.
