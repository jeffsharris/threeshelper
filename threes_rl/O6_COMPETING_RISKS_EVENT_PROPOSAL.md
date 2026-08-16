# O6 Competing-Risks Designated-Pair Event Proposal

Date: 2026-07-27

Status: proposal only. No O6 source scan, stream reservation, label, model,
rollout, or policy evaluation is authorized.

## Scientific Question

Can a fresh scale-relative, action-conditioned designated-pair controller
increase the safe pair-specific merge hazard relative to the competing
absorbing failure hazard under sustained h40 exposure?

O6 is a new branch. It does not repair, retry, continue, threshold-relax, or
use any O5 model, checkpoint, selected root, task stream, episode, or label.

## Event Likelihood

Use one fixed discrete-time competing-risks likelihood:

- cause 1: safe pair-specific designated merge;
- cause 2: third-party merge, lineage loss, terminal state, anchor/air
  violation, or no legal action;
- remaining risk mass: still live after the transition.

Administrative h40 censoring contributes survival likelihood when present but
has no minimum-count gate and is not a separately fitted class. Every
state-action row is risk-set masked until its first absorbing event or h40.
Human and behavior actions are never labels.

The exact feature/operator, head width, optimizer, regularization, exploration
schedule, event-time basis, checkpoint, and support thresholds must be frozen
once in a separate execution charter. O5 weights and checkpoints are forbidden.

## Outcome-Free P0

Before labels, P0 must:

1. Inventory only natural completed normal-start machine ancestries that are
   disjoint from every O3/O4/O5 selected root and all protected confirmation
   roots. Use at most one current-state root per whole ancestry.
2. Freeze train/development/untouched mechanism/confirmation partitions before
   labels. No family or ancestry may cross a partition.
3. Require at least four genuinely action-distinct policy families, no family
   above 40%, and deterministic near-balance across T48/T96/T192.
4. Report outcome-free current-state reachability by target, family, empties,
   legal-action count, pair separation, blocker density, and anchor/air state.
   Static geometry must not be called proof of future h40 survival.
5. Reserve a fresh four-stream namespace and prove zero collision with the
   complete historical union, including all O3/O4/O5 reservations.
6. Reproduce domain, lineage, pair-specific merge, safety, and simulator
   round-trip tests with zero label or policy-outcome access.

Failure of ancestry, family, target, geometry, provenance, collision, service,
disk, or partition support seals `HOLD_O6_DATA_PREFLIGHT`.

## Prospective Power Contract

The untouched sustained-policy mechanism gate uses paired CRN h40 exposure,
whole-root clustering, target/starting-geometry strata, and a
strata-standardized common odds ratio for safe merge versus competing failure.
Family and stream-block signs are descriptive.

Before labels, simulate the exact gate for candidate root counts
`N in {192, 256, 384, 512}` using 4,096 frozen Monte Carlo datasets and 4,096
whole-root bootstrap replicates. Use the opened O5 aggregate success base rate
`188/1,152` only as a prospective base-rate anchor, with a predeclared
conservative root intraclass-correlation sensitivity grid. Target truth is
OR1.50; the pass event is point common OR at least 1.25 and bootstrap lower
95% bound above 1.0. Freeze the smallest N with at least 80% full-pass power.
Report the smallest 80%-power MDE on the fixed grid
`{1.25, 1.35, 1.50, 1.75, 2.00}`. If no N through 512 is adequate, HOLD
outcome-free.

## Pre-Fit Support Gate

After any later fully frozen label job and before fitting, require:

- at least 40 safe merges and 40 competing failures;
- at least 6 safe merges at each target;
- at least 3 safe merges in each represented family;
- finite, domain-valid risk rows;
- at least two occupied event-time bands;
- root/family weights and ancestry partitions exactly as frozen.

No administrative-censor quota is permitted. A miss is data-support HOLD, not
representation KILL. A pass could authorize only one frozen model fit and one
untouched sustained-policy mechanism gate, never normal-start promotion.

`CONTINUE=outcome-free O6 charter/P0 proposal review`;
`HOLD=source scan, streams, labels, training, and all policy outcomes`;
`KILL=false`; `PROMOTE=false`.
