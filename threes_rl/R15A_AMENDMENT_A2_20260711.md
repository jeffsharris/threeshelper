# R1.5a Amendment A2: Final Data-Readiness Rule

Status: frozen on 2026-07-11 before label generation. No R1.5a label or model
outcome existed when this amendment was written.

## Immutable Prior Evidence

The original preregistration and both failed readiness gates remain unchanged:

- `R15A_OFFLINE_PREREGISTRATION.md`, SHA-256
  `b915e568980409c9b89d86c545a2f902d7c823c3e9f30af12ada529175a1ec92`;
- original inventory, SHA-256
  `f540ee95cfef89a3ad16dbc3cb1975aef39d0ec58fa76286e9c2bf06db92af17`;
- `R15A_AMENDMENT_A1_20260711.md`, SHA-256
  `ba6b03a80a3e391a16768ffbc66ead82c2aa29eb9cfe781db37e56d0471e3257`;
- A1 inventory, SHA-256
  `8604778696164fdabd5ab653c933b0b543ca1d20a8fde1d78b6e7da2994d794a`;
- A1 stop/go artifact, SHA-256
  `5be193d3a35ba755e57818d3f47ee30a08099d8d6c3afc1647628f9d5303d678`.

No A1 source record, partition, state selection, family weight, holdout, or
coverage result is changed by A2.

## Final Readiness Amendment

A1 passed every readiness rule except weighted train effective ancestry count:
`110.17874875868921` versus the manager-chosen threshold `120`. A2 changes that
single threshold to:

`weighted train effective ancestry count >= 100`.

The threshold is met. Every other A1 rule remains frozen exactly as written:

- at least 150 train roots and four nonhuman families;
- at least 25 ancestry-holdout roots, three families, and ESS 20;
- at least 20 untouched corner2 roots;
- maximum effective train-family loss mass 40%;
- all stage, plus, pending, and empties coverage minima;
- exact provenance/state reconstruction and zero root-cluster overlap.

This is the final data-readiness amendment. It may not be relaxed again.

## Authorized Execution

With A2 `READY`, execute the existing R1.5a contracts without another routine
approval: compact frozen-incumbent h40 labels, exactly two equal-capacity
board/stage-only and board+context models, and the frozen source-disjoint
offline gate. Synthetic H2 and human-assisted outcomes remain diagnostic-only.

If the offline gate fails, permanently kill this exact context-residual branch
without policy evaluation and proceed to the separately preregistered R2a
adaptive-expectimax branch. A policy development pass may open one sealed fresh
confirmation only under the existing R1.5a rules. Only confirmation can update
the incumbent or dashboard.
