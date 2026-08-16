# R1.5a Amendment A1: Weighted Natural Target Mixture

Status: frozen on 2026-07-11 before amended inventory regeneration. No labels
or model outcomes have been generated or inspected.

## Preserved Failed Preflight

The original preregistration, inventory, and `HOLD_DATA` decision remain
immutable evidence:

- `R15A_OFFLINE_PREREGISTRATION.md`, SHA-256
  `b915e568980409c9b89d86c545a2f902d7c823c3e9f30af12ada529175a1ec92`;
- `runs/forensics/r15a_context/r15a_natural_state_inventory_20260711.json`,
  SHA-256
  `f540ee95cfef89a3ad16dbc3cb1975aef39d0ec58fa76286e9c2bf06db92af17`;
- `runs/forensics/r15a_context/R15A_PREFLIGHT_STOP_GO.json`.

A1 changes only source partitioning and fit weights based on source counts. It
does not change target definitions, model architecture, optimization, offline
metrics, or policy gates.

## Diagnosis

The original hard 40% root deletion discarded 280 train and 83 ancestry-
holdout roots despite 643 deduplicated fresh replays, 165,987 valid states,
zero provenance failures, and broad context coverage. Incumbent-distribution
states are the deployment target, not contamination. Dominance will therefore
be controlled by frozen fit weights and dual reporting rather than deletion.

## A1 Root Catalog And Partitions

Natural replay acceptance, exact state validation, behavioral-family
coalescing, context bins, state-cell decorrelation, synthetic exclusion, and
human diagnostic-only status are unchanged.

Root identity is canonical fresh root cluster, independent of replay copy or
checkpoint alias. Apply these rules before state selection:

1. Every `corner2_lineage` root cluster is untouched `family_holdout`.
2. Every human root cluster is `human_diagnostic` only.
3. Any non-corner trajectory sharing a canonical root cluster with corner2 is
   excluded from ordinary partitions.
4. When one remaining canonical root cluster appears under multiple nonhuman
   behavioral families, retain one deterministic representative ancestry: the
   trajectory with the most distinct eligible context cells, then family and
   ancestry ID lexical tie-breaks. This prevents cross-partition root aliases.
5. Within each remaining behavioral family, sort root clusters by
   `SHA256("A1-holdout" + root_cluster)`. For families with at least five roots,
   assign `ceil(20%)` to `ancestry_holdout`; for two to four roots assign one;
   singleton families remain train-only. All remaining roots are `train`.
6. Root clusters may occur in exactly one ordinary partition. Partition overlap
   must be zero.

State selection remains outcome-blind:

- at most eight states per root;
- at most one state per root/context cell
  `(stage, plus-bin, preview-bin, pending, empties-bin)`;
- deterministic rarity-first state order and root round-robin;
- state caps remain train 1024, ancestry holdout 256, whole-family holdout 256,
  human diagnostic 64.

## A1 Family-Balanced Fit Weights

Keep every selected independent train root. Freeze loss weights before labels:

1. Begin with equal mass per selected root ancestry.
2. If a family's raw root mass exceeds 40%, cap its total loss mass at exactly
   40%.
3. Redistribute remaining mass across uncapped families in proportion to their
   selected root counts. Repeat water-filling only if another family reaches
   the cap.
4. Divide each family mass equally across its roots and each root mass equally
   across its selected states.
5. Normalize state weights to sum to one. Never duplicate or oversample a root
   to manufacture effective sample size.

Report raw root/state family shares, effective weighted family shares, root
weight distribution, and weighted effective ancestry count
`1 / sum(root_weight^2)`.

Training uses these A1 family-capped state weights identically for both models.
Ordinary offline metrics are reported twice:

- natural root-balanced target mixture;
- family-balanced mixture with equal metric mass per represented family, equal
  roots within family, and equal states within root.

The context model must pass the ancestry-holdout primary gate under both
reportings and improve in the same direction on the untouched corner2 holdout.
If gains occur only in phaseblend/incumbent roots, the gate fails.

## A1 Readiness

Proceed only when all hold:

- at least 150 train root ancestries from at least four nonhuman families;
- train unweighted state-selection ESS and weighted loss ESS both at least 120;
- no train family has more than 40% effective loss weight;
- at least 25 ancestry-holdout roots from at least three families, with
  unweighted ESS at least 20;
- at least 20 untouched corner2 family-holdout roots;
- all four stages and all frozen plus/pending/empties marginals satisfy the
  original train and combined-holdout minimums;
- exact provenance/state failures and cross-partition root overlap are zero.

Do not relax these rules after seeing amended counts.

## Downstream Contract

If A1 readiness passes, all unchanged R1.5a contracts immediately apply:

- primary h40 multi-step residual target
  `score_0:40 + V_inc(live_s40) - V_inc(s0)`, terminal bootstrap zero;
- one h40 path supplying h10/h20/h40, 16 A/B replicates per natural state;
- frozen equal-capacity 3,796-parameter models and optimization schedule;
- compact resumable labels, no full trajectory corpus;
- ancestry and whole-family offline predictive/calibration/context gate;
- policy development and sealed fresh confirmation only after offline success.

Synthetic H2 swaps remain diagnostic-only. Human frames remain
diagnostic-only. No direct action labels, hand-written plus bonus, model sweep,
or dashboard claim is permitted.
