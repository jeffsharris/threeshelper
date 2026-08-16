# G3 E0 Label/Fit Execution Charter

Date frozen: 2026-07-25

Status: authoritative preregistration for one breadth-first, two-replicate
engineering screen. This charter authorizes implementation, tests, and one
zero-work preflight only. It does not authorize label execution, model fitting
on scientific labels, transfer outcomes, E1, policy evaluation, incumbent
change, dashboard change, or promotion.

No G3 label value, score outcome, selected continuation action, future
milestone outcome, fitted scientific model, transfer prediction, or transfer
outcome was opened before this file was frozen.

## Preserved Scientific Lineage

The following evidence is immutable:

- G3 base charter SHA-256
  `e216aa50737afee0d439e060cc9b1e1f24d2f552af4c3f0c8944470ff7a45fc1`;
- G3 stream amendment A1 SHA-256
  `baba72003934ef55a48383704e4c6b5738787d561a0d81f0ea09383f64122b94`;
- G3-v2 integrity amendment SHA-256
  `c60895f9f72c78d72481e0d3759f2a818c1165de5e2f8e0d1fe189dc85026aef`;
- G3-v2 READY preflight file SHA-256
  `052985f7e5c13797df43bfd074602169ff5c85618dd0f3db549720fec95f7d66`;
- G3-v2 READY preflight canonical payload SHA-256
  `185390797d66d9997faa801292a89a8a8b24967304204fec42b8fbb1852a93ab`;
- corrected untouchedness audit file SHA-256
  `8b2fd76e593db0b6af998aa5f49092a5dea285754111b891d3aed85ad80e51ed`;
- v1 record-manifest file/payload SHA-256
  `938e903f8d2fefb072af84ac19baf4977e4f4d93bf72e8af7acc174b6974b9ec`
  /
  `a78e2fd51ee20a7aeb23c71d9930c33561844357920f4808eeeaff653d49f759`;
- v1 stream-manifest file/payload SHA-256
  `bdbe562167f304327e52f0593f0958753e8afa949a7b38e15b357492faea5744`
  /
  `c2afc3c6fa26c1106a480c58189d9a9b4f9dcf99ac8b506d890ff3c330278caa`.

G3-v1 remains permanently `KILL_G3_PREFLIGHT_INTEGRITY` and spent. G3-v2
corrected only the localized namespace audit and sealed
`READY_G3_V2_BOOTSTRAP_LABELS`; neither artifact may be edited or rerun.
Every G2/G1/S3/QD lock and the current incumbent remain preserved.

## Frozen Corpus And Work

E0 uses the exact v1 manifests referenced above:

- `683` ordinary selected state-scale records from `352` whole ancestries:
  `550` train records / `283` train roots and
  `133` development records / `69` development roots;
- `32` untouched transfer-diagnostic records:
  `12` corner2, `1` expectimax2, and `19` phaseblend-incumbent roots;
- every legal first action in canonical order `up, down, left, right`;
- exactly replicates `0` and `1`;
- exactly `5,072` h40 paths:
  `3,902 train`, `944 development`, and `226 transfer_diagnostic`.

There is no root, record, action, family, scale, or replicate selection,
reassignment, replacement, downsampling, or outcome-dependent skip. One h40
path supplies h10, h20, and h40 event/censoring observations.

E0 reuses the unconsumed `57B/58B/59B/60B` stream reservation. For record
ordinal `r` and replicate `j in {0,1}`, each stream ID is
`base + 8*r + j`. Every legal action arm for the same record/replicate shares
the same logical, deck, slot-uniform, and policy-tie streams. Each arm gets a
fresh simulator and policy RNG initialized from that shared tape; slot
uniforms map independently over each arm's legal insertion positions. No arm
may consume or inherit another arm's state or policy action.

## Exact Label Semantics

For each path:

1. Restore the exact source board, visible preview or candidate bundle, small
   bag counts and position, total small count, span position, pending state,
   move count, starter tile, and all simulator context from the immutable
   record source.
2. Verify source replay/state hashes, root ancestry, state SHA, legal actions,
   and all three frozen feature-row hashes before rollout.
3. Force the manifest action for local move one. Continue subsequent moves
   with the current frozen depth-2 phaseblend incumbent identified by
   `current_incumbent_policy.txt` and all bound component hashes.
4. Stop at first attainment of the record target, terminal state, or local
   move 40. The target is `768` for `pre768`, `1536` for `pre1536`, and
   `3072` for `pre3072_transfer`, excluding the fixed starter.
5. Store only event/censoring sufficient statistics and immutable provenance:
   event move, terminal move, completed moves, stream IDs, root/record/action,
   target, and interval rows. Do not store score, a continuation-action trace,
   selected replay rank, recorded/human action, or favorable-state metadata.

Intervals are `(0,10]`, `(10,20]`, and `(20,40]`, with rows named h10, h20,
and h40. The exact arithmetic is:

- an interval is at risk only when no event or terminal occurred at or before
  its start;
- if first target attainment is in `(start,end]`, the row is observed with
  event `1`, and no later interval is at risk;
- if the trajectory is live and target-free through `end`, the row is
  observed with event `0`;
- if target-free terminal occurs strictly before `end`, the row is
  right-censored at the exact local terminal move, has no binary event value,
  and contributes no likelihood, Brier, calibration, or rank trial;
- target-free terminal exactly at `end` is an observed event-`0` row, and no
  later interval is at risk;
- if event and terminal coincide, the event takes precedence;
- rows after first event or terminal are absent, never imputed as zero.

The compact path is complete only when deterministic replay from the immutable
source and streams reproduces its sufficient statistics exactly.

Human, assisted-human, recorded, restart, replay-start, continuation, partial,
synthetic, and imported actions are never targets or ordinary evidence.

## Phase Isolation And Sealing

The exact future execution order is:

1. atomically seal `G3_E0_EXECUTION_OPENED.json` before the first E0 path;
2. generate and transactionally persist all train and development paths;
3. verify ordinary completion, provenance, replay determinism, and hashes;
4. fit the single frozen model on train only and evaluate uncalibrated
   development predictions;
5. fit one development-only Platt calibrator;
6. atomically seal `G3_E0_CHECKPOINT_SEALED.json`, binding model arrays,
   normalization, calibrator, optimizer state, ordinary labels, ordinary
   metrics, all thresholds, code, and manifests;
7. only if the ordinary development gate passes, compute transfer features,
   frozen model predictions, canonical model actions, and current incumbent
   actions without opening transfer labels;
8. atomically seal `G3_E0_TRANSFER_PREDICTIONS_SEALED.json`;
9. if the outcome-free activity gate fails, seal
   `HOLD_G3_E0_UNDERPOWERED_TRANSFER` without generating transfer labels;
10. otherwise generate all 226 transfer paths, evaluate the frozen
    diagnostics once, and seal the terminal decision.

Transfer source states may be hash/restoration checked in preflight, but no
transfer label database may be opened or created before both checkpoint and
prediction seals exist. Transfer roots never affect feature normalization,
the base fit, Platt calibration, checkpoint selection, thresholds, margin
choice, or model action.

## Frozen Model

The one model is the exact G2 64-feature grouped-binomial discrete-time
logistic hazard:

- feature schema SHA-256
  `6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e`;
- one intercept and 64 feature coefficients;
- horizon indicators are feature indices `0,1,2`;
- intercept and horizon coefficients are unpenalized;
- all other coefficients use L2 `lambda=1.0`;
- deterministic L-BFGS-B, maximum `500` iterations, gradient tolerance
  `1e-8`, no warm-start, retry, seed, checkpoint, or hyperparameter sweep;
- coefficients initialize at exact zero;
- sigmoid logits clip only for numerical evaluation to `[-40,40]`.

Columns marked `train_standardize=true` in the immutable G2 schema use the
family-balanced training mean and population standard deviation. A scale below
`1e-12` becomes `1`, and the centered column is therefore zero. All other
columns have stored mean `0` and scale `1`. Transfer and development rows
never enter normalization.

Training uses exactly five present ordinary families with equal total family
weight `1/F`. Within family `f`, each ancestry has weight `1/(F*n_f)`.
Within an ancestry, selected scale records split that weight equally; legal
actions split record weight equally; observed at-risk intervals split action
weight equally. A grouped row with `n` observed E0 trials uses event fraction
`k/n`, and its mean-binomial log loss is multiplied by the row weight, so two
replicates never count as two roots. Censored trials are excluded before this
split. Root-equal evaluation uses equal root weight with the same
record/action/observed-interval splitting and no family rebalance.

The train-only constant baseline has one hazard for each interval. It uses the
same family-balanced train weights and a Jeffreys numerical guard only when an
interval is degenerate:
`p=(weighted_events+0.5e-12)/(weighted_trials+1e-12)`, then clipped to
`[1e-9,1-1e-9]`.

The sole Platt calibrator is:

`calibrated_logit = intercept + exp(log_slope) * base_logit`

It initializes at intercept `0`, log-slope `0`, uses development root-equal
weights, no regularization, L-BFGS-B with the same `500`/`1e-8` settings, and
has no restart or alternative calibration. Calibration is stable only when
both optimizers report success, all arrays and predictions are finite, base
gradient infinity norm is at most `1e-4`, calibrated slope is in
`[0.05,20]`, absolute calibration intercept is at most `3.0`, and
development root-equal 10-bin equal-width ECE is at most `0.25`.

Serialization stores schema, source/task/label hashes, all 64 names,
normalization mask/mean/scale, coefficients, intercept, calibrator parameters,
constant hazards, optimizer summaries, and exact floating dtypes. Loading
rejects wrong width, schema, hashes, nonfinite values, nonpositive scales,
negative calibration slope, or changed scientific config.

## Metrics, Bootstrap, And Ties

Metrics use observed interval trials only:

- root-equal log loss and Brier, compared as
  `constant loss - model loss` so positive is improvement;
- the same point metrics separately for pre768 and pre1536;
- family reports for every development family;
- h40 cumulative target hazard
  `1-product(1-interval_hazard)` for each legal action;
- root/record-level Spearman action-rank correlation between predicted h40
  cumulative hazard and empirical event-by-h40 rate over the two replicates.

Spearman uses average ranks for exact ties. A state-scale record is
uninformative and excluded from rank aggregation when fewer than two legal
actions have distinct predicted values or fewer than two have distinct
empirical rates; it is reported, never assigned zero. Overall rank is the
root-equal mean across informative records. Ordinary rank evidence requires at
least `20` informative development records overall and at least `5` at each
training scale.

Whole-ancestry bootstrap uses `10,000` draws, seed `2_026_072_601`, and
percentile `[2.5,97.5]` intervals. Each draw samples development ancestry IDs
with replacement and carries every selected scale/action/interval row for the
drawn ancestry. The same draw computes log-loss improvement, Brier
improvement, and rank. Repeated rows from one sampled ancestry remain one
cluster. Transfer descriptive intervals, when labels open, use the same
method with seed `2_026_072_602`; they never turn N=32 into powered evidence.

The model action is the legal action with largest calibrated h40 cumulative
hazard. Values within absolute tolerance `1e-12` tie; ties choose canonical
order `up, down, left, right`. The incumbent action is generated once from the
frozen depth-2 policy with its manifest policy stream. Outcome-free transfer
activity is the number of roots where any selected transfer record's model
action differs from the incumbent action.

The frozen activity floor is exactly `6/32` roots, including at least one of
the 12 corner2 roots and at least one of the 19 incumbent roots. Expectimax2
N=1 is descriptive only. No threshold is adapted after predictions or labels.

## E0 Decisions

### Earlier-Scale Gate

Ordinary development passes only when all are true:

- overall root-equal log-loss and Brier point improvements are both positive;
- at least one of those two improvements has whole-root bootstrap 95% lower
  bound above zero;
- pre768 and pre1536 each have positive log-loss and Brier point improvement;
- overall action-rank correlation is positive;
- each scale's action-rank point estimate is nonnegative;
- rank-informative counts meet `20` overall and `5` per scale;
- no development family with at least five roots has both negative log-loss
  and negative Brier improvement;
- both optimizers, calibration bounds, finite predictions, serialization,
  provenance, isolation, and deterministic replay checks pass.

Failure of this adequately sized ordinary gate, model convergence, or
integrity seals `KILL_G3_BOOTSTRAP_PREDICTIVE`. Transfer label values remain
unopened.

### Transfer Direction Gate

After ordinary pass and outcome-free activity pass, transfer labels open once.
`READY_G3_E1_COMPLETION` requires all:

- pooled calibrated root-equal log-loss and Brier improvements are
  nonnegative;
- pooled action-rank correlation is positive;
- corner2 and incumbent, the two families with at least 12 roots, each have
  nonnegative log-loss improvement, nonnegative Brier improvement, and
  nonnegative action-rank direction when informative;
- the frozen activity floor above passes;
- every transfer path, provenance, stream, censoring, checkpoint, prediction,
  serialization, and no-mutation audit passes.

`READY_G3_E1_COMPLETION` is evidence only that completing the already frozen
six replicates is warranted. It is not promotion, policy utility, or dashboard
evidence.

If the earlier-scale gate passes but activity is too low, a transfer family is
uninformative, or any transfer direction condition is null, mixed, or negative
within the N=32 panel's OR `4.0` MDE, seal
`HOLD_G3_E0_UNDERPOWERED_TRANSFER`. This decision may occur before transfer
labels when activity is insufficient. It never becomes a permanent utility
kill.

The three E0 scientific terminal decisions are disjoint:

- `READY_G3_E1_COMPLETION`;
- `HOLD_G3_E0_UNDERPOWERED_TRANSFER`;
- `KILL_G3_BOOTSTRAP_PREDICTIVE`.

## Frozen E1 Continuation Identity

E1 remains unauthorized. If a separately frozen prospective E0 decision later
opens E1, it must:

- use every same frozen root and legal action at replicates `2..7`;
- generate all `15,216` remaining paths, with no E0-dependent selection;
- retain the same feature, label, CRN, continuation, weighting, optimizer,
  calibration, bootstrap, metric, and tie rules;
- refit the final eight-replicate model from exact zero on all train labels,
  rather than warm-starting E0;
- recalibrate once on all development labels;
- open the same transfer panel once under the final checkpoint.

The E0 gate may authorize all E1 or none. E0 outcomes may never alter E1
roots, actions, families, streams, features, penalty, thresholds, or model.

## Execution And Artifacts

The separate implementation surface is:

- runner/fitter/evaluator:
  `threes_rl/g3_e0_label_fit.py`;
- preflight:
  `threes_rl/g3_e0_preflight.py`;
- focused tests:
  `tests/test_rl_g3_e0_label_fit.py`;
- output:
  `threes_rl/runs/forensics/g3_e0_label_fit_v1`.

Labels persist transactionally in SQLite with immutable task IDs and unique
primary keys. Chunks contain at most `8` paths. Resume accepts only the same
preflight lock, open marker, command, source/task/stream hashes, code hashes,
policy hashes, jobs `1`, and nice priority. Already completed task rows are
replayed and hash-verified before skip; duplicate or divergent rows fail
closed. Keyboard/process interruption may resume. Scientific integrity,
hash, collision, service, storage, or schema failure seals a terminal error
under `KILL_G3_BOOTSTRAP_PREDICTIVE` and may not silently rerun.

The exact future execution command is:

`zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m threes_rl.g3_e0_label_fit execute --out-dir threes_rl/runs/forensics/g3_e0_label_fit_v1 --preflight-lock threes_rl/runs/forensics/g3_e0_label_fit_v1/preflight_lock.json --jobs 1'`

One process and one heavy job are allowed. Chunks check no competing heavy
process, services, output below `4 GiB`, active runtime below `18h`, free disk
above the hard `100 GiB` floor, and nice at least `10`. Target free disk is
`120 GiB`. Dashboard record and protected top three must remain
`263670/261369/258561`.

## Authorized Zero-Work Preflight

The current turn may only:

- hash-bind this charter, G3-v2 READY evidence, v1 manifests, G2 schema,
  incumbent components, simulator/evaluator/policy sources, implementation,
  tests, and test evidence;
- reconstruct every frozen record and legal-action feature row;
- build exact E0 ordinary/transfer task and stream manifests;
- verify shared-CRN coupling and collision-free reuse of the unconsumed
  reservation without consuming a stream;
- test label/censor arithmetic, weights, phase isolation, deterministic
  synthetic fitting, serialization, bootstrap, ties, gates, resume identity,
  and fail-closed sealing;
- project E0 runtime/storage and verify disk, process, services, dashboard,
  advisor, and protected top three;
- verify the output namespace contains no preexisting E0 marker, label
  database, model, prediction, or terminal result before atomically promoting
  the compact preflight lock.

The preflight seals exactly one:

- `READY_G3_E0_LABEL_FIT_EXECUTION`;
- `HOLD_G3_E0_COST_OR_COVERAGE`;
- `KILL_G3_E0_PREFLIGHT_INTEGRITY`.

READY authorizes only a later separate E0 execution message. This turn creates
zero E0 paths, consumed streams, labels, label values, scientific fits,
checkpoints, checkpoint predictions, transfer outcomes, rerankers,
normal-start policy outcomes, score claims, incumbent changes, dashboard
points, E1 work, C2 work, or human-training-ground work.
