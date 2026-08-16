# J2A1 Distillation/Fidelity Execution Surface V2

Status: outcome-free orchestration repair and zero-work readiness only.

## 1. Immutable predecessors

V1 remains authoritative historical evidence and is forbidden for execution
reuse. V2 must not edit, overwrite, reinterpret, import scientific content
from, or execute any V1 artifact.

V2 binds these V1 source identities:

- charter:
  `dbe3470f67229c086f514de20efdd2daf074329df81ca66611895fecabef8f61`;
- runner:
  `b5435d6d5d0999b035220a6763646ee133b23f06e79f45456d9c5af083dfe8c1`;
- tests:
  `bb1c8fffa52dea332032447f60426addfbd0acaf2bc5453feb8620856062889d`.

V2 binds all eight files and payloads in
`j2a1_distillation_fidelity_execution_surface_readiness_v1`, including
readiness result file/payload
`a90d1600502264d42315e0806d7665be679e06111aacc006dc193e88baa97d22` /
`e118a58de2357ff4d870283cd46a2e23696aad42303427f31dc2e6befc3b9861`
and retention file/payload
`dd258ffbded154ec299d5b48368cd21c9d35e585464079d57009a0e960eb28eb` /
`ea339915bb05f604bf9cafc7e319120354df803acbe8834e8c88ff88442db854`.

The spent V1 authorization is preserved at
`j2a1_distillation_fidelity_execution_authorization_v1/`
`J2A1_DISTILLATION_FIDELITY_EXECUTION_AUTHORIZATION.json`, file/payload
`29ea95388165250b7b7f7db909698ec853101f85bf62e81445e23540879e576f` /
`b2f5792e16be8ee8e08109fdf894f6243ee7dae840706656ff349da0e71c277b`.

The V1 zero-work HOLD is binding:

- `seal-phase-lock` produced no artifact;
- the V1 execution root is absent;
- phase locks, markers, materializations, owners, reservations,
  consumptions, genesis commits, teacher loads/queries, labels, games,
  optimizer steps, checkpoints, fidelity reads, and human-session reads are
  all zero;
- the live top-three object was a tuple containing exactly
  `(263670, 261369, 258561)`;
- the frozen expected object was the list
  `[263670, 261369, 258561]`;
- raw tuple/list equality was false while normalized values and order were
  exact;
- every other operational guard passed.

## 2. Sole repair

V2 makes exactly one semantic-neutral repair. Before the execution
operational guard compares protected top-three values, it canonicalizes both
the live value and the immutable expected value with the same function.

The canonical function:

1. accepts only an exact `list` or exact `tuple`;
2. requires length exactly three;
3. requires each element to have exact type `int` (booleans are rejected);
4. requires three distinct scores;
5. returns an immutable three-integer tuple preserving order.

The guard passes only when the two canonical tuples are exactly equal. A score
mutation, order mutation, duplicate, wrong length, mapping, set, iterator,
string, bytes value, boolean, float, subclass ambiguity, or other type fails
closed.

V2 may change only namespacing, version/decision labels, immutable predecessor
bindings, and this comparator. It may not change source authority, model,
labels, metrics, gates, stage chronology, stream rows, root identities,
storage/runtime limits, worker topology, restart semantics, or scientific
behavior.

## 3. Separate V2 surface

The new files are:

- `threes_rl/J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2_CHARTER.md`;
- `threes_rl/j2a1_distillation_fidelity_execution_surface_v2.py`;
- `tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py`.

The zero-work readiness namespace is
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_execution_surface_readiness_v2`.
The future execution root is
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_execution_v2` and must remain absent.

The public commands remain exactly:

1. `audit-zero-work`
2. `write-test-evidence`
3. `prepare-readiness`
4. `seal-phase-lock`
5. `open`
6. `materialize`
7. `execute`

No V2 authorization artifact may be created in this turn. A future READY
permits only research-lead review of a separately hashed V2 authorization and
phase lock.

## 4. Inherited scientific contract

All V1 scientific contracts are inherited unchanged: 8,192 BC teacher roots,
6,144 paired validation roots, 14,336 teacher roots, 20,480 complete game
arms, 63,488 streams, exact 227B-235B A1 rows, eight fixed single-thread
teacher shards, canonical merge, four-family support gate before optimization,
eight BC epochs, untouched mechanism gate, 6,144 sustained student arms,
frozen score/common-OR fidelity gates, checkpoint quarantine, bounded storage,
72-hour/24-GiB caps, and no PPO/development/confirmation/promotion authority.

## 5. Required evidence

Focused tests must prove:

- equal tuple/list values and order pass;
- any score or order mutation fails;
- duplicate, length, and type ambiguity fail closed;
- the V1 production-shaped tuple/list case reproduces V1 rejection and V2
  acceptance;
- V1 source/readiness/authorization identities remain exact;
- the V1 and V2 execution roots remain absent;
- every scientific counter remains zero.

Final nice-10 evidence must include py-compile, focused V2, immutable V1/J2/J1
parents, the same broad non-science surface and documented deselections, and a
fresh operational audit. Any predecessor, source, namespace, zero-work,
service, process, disk, Torch, comparator, or test mismatch is a fail-closed
HOLD or integrity KILL by class.

## 6. Readiness decisions

The only terminal decisions are:

- `READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2`;
- `HOLD_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2`;
- `KILL_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2_INTEGRITY`.

`CONTINUE` means research-lead review of a future V2 authorization only.
`HOLD` covers all J2A1 execution, teacher work, labels, training, fidelity,
PPO, development, confirmation, and promotion. V1 remains killed for execution
reuse; the J2A1 hypothesis remains live. `PROMOTE=false`.
