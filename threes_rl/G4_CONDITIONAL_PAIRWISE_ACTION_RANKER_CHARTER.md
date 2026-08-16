# G4 Conditional Pairwise Action Ranker Charter

Date frozen: 2026-07-26

Status at freeze: `CONTINUE_G4_CONDITIONAL_PAIRWISE_PREFLIGHT`

## 1. Scientific question and permanent locks

G3 E0 showed that the frozen G2 scale-equivariant feature map predicts
between-state milestone difficulty but failed to rank legal actions within the
same state. G4 asks one narrower question:

> Can the exact frozen G2 features, used only as within-record legal-action
> differences on common-random-number tapes, predict which action attains the
> stage-appropriate milestone first?

The authoritative G3 terminal decision remains
`KILL_G3_BOOTSTRAP_PREDICTIVE`. G3 E0/E1 may not be rerun, extended,
reinterpreted, or reopened. The 32 transfer roots, transfer predictions, and
226 reserved transfer paths are forbidden inputs to G4. G4 may read only the
already spent ordinary train/development labels. Those labels and development
roots are exploratory mechanism evidence and can never confirm G4, authorize a
policy, change the incumbent, or change the dashboard.

No simulation, new label, score inspection, policy outcome, human action label,
or heavy process is authorized by this charter.

## 2. Immutable spent inputs

G4 binds these G3-v4 ordinary artifacts before reading any label value:

- terminal result:
  `runs/forensics/g3_e0_label_fit_v4/G3_E0_TERMINAL_RESULT.json`,
  file SHA-256
  `e7ca390f0c32ebb3a680235de02e12beb62f45b1050115e8c9a30a7a3ca0ddd1`;
- ordinary record manifest:
  `runs/forensics/g3_e0_label_fit_v4/E0_RECORD_MANIFEST.json`,
  file SHA-256
  `90a4f55ff29f51c0d6ac35375650258188b6961debd6cbcc546382762547d9d5`;
- task manifest:
  `runs/forensics/g3_e0_label_fit_v4/E0_TASK_MANIFEST.json`,
  file SHA-256
  `087fd68c71421c8402360a1c096b476cb1bf494de7d8c8f025e7e699bf97bd2f`;
- stream manifest:
  `runs/forensics/g3_e0_label_fit_v4/E0_STREAM_MANIFEST.json`,
  file SHA-256
  `e40b7dd3744dd0df04f621034894656568991291c17490e27e8c3a93e189ea05`;
- spent ordinary label database:
  `runs/forensics/g3_e0_label_fit_v4/ordinary_labels.sqlite3`,
  file SHA-256
  `d0954a91e84bc7a420d64e7294f40232c1ffcb692fab86d07425b138e063f820`.

The database must be opened read-only and immutable. Its WAL must be empty.
G4 must reject any record or task whose partition is `transfer`, any transfer
database or seal path, and any root/state token from the frozen G3 transfer
partition. The source artifacts may not be modified, checkpointed, vacuumed,
or copied into a writable SQLite database.

The feature contract is exactly the 64 ordered columns from
`g2_scale_relational_hazard.py`, source SHA-256
`9ffaa45dd36b633cdae10110fdaefc8cd27053ab3f0216ddb3f1886ea625af8a`.
The spent-label arithmetic is bound to `g3_e0_label_fit.py`, source SHA-256
`19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5`.
The G4 preflight records the G2 schema SHA and all source hashes again.

## 3. Comparable matched-pair construction

Canonical action order is `up`, `down`, `left`, `right`, using the simulator's
integer action IDs. For each ordinary partition independently:

1. Group path rows by exact
   `(record_id, interval_horizon, replicate)`.
2. Verify every action arm in a group has the same root, state hash, target,
   scale, logical seed, deck stream ID, slot stream ID, policy stream ID, and
   interval boundaries. This is the common-random-number tape contract.
3. An unordered legal-action pair is comparable only when both arms contain an
   interval row with `observed=true` and binary `event` in `{0,1}` for that
   exact interval.
4. Retain a comparable pair only when its outcomes are discordant. Concordant
   event/event and no-event/no-event pairs carry no conditional ordering
   information and are omitted. A censored or absent arm makes the pair
   non-comparable and it is omitted.
5. Order each retained pair by canonical action ID. Let `a` be the lower action
   and `b` the higher action. The response is `y=1` exactly when action `a`
   has event 1 and action `b` has event 0; otherwise `y=0`.
6. Reconstruct the exact frozen G2 feature vector at the interval end horizon
   for both actions without consuming RNG. The model input is
   `delta = x(a)-x(b)`. Reversing a pair must produce `-delta` and `1-y`.
7. Each source state and each action feature vector must reproduce its frozen
   hash and remain unmutated. All 64 deltas must be finite.

Intervals are `(0,10]`, `(10,20]`, and `(20,40]`, named `h10`, `h20`, and
`h40`. A single path may contribute at most one row per interval. Replicates 0
and 1 remain distinct matched tapes.

The three horizon indicator columns are identical within an action pair and
therefore must equal zero in every delta. This is intentional: G4 fits one
shared action-ordering function across h10/h20/h40, not three independently
tunable models.

## 4. Exact weighting

Weights are computed separately for train and development after pair
eligibility is frozen.

Let:

- `F` be the number of behavior families with at least one retained pair;
- `n_root(f)` be informative roots in family `f`;
- `n_record(f,r)` be informative records for root `r`;
- `n_unit(f,r,c)` be informative `(horizon,replicate)` units for record `c`;
- `n_pair(f,r,c,u)` be discordant pairs in unit `u`.

Every retained pair receives:

```
w = 1/F
    / n_root(f)
    / n_record(f,r)
    / n_unit(f,r,c)
    / n_pair(f,r,c,u)
```

Thus each family has equal total weight, roots split their family weight
equally, records split root weight equally, and prolific horizons, replicates,
or action pairs cannot dominate. We also report an unweighted natural-mixture
view and root-equal-without-family-balancing view, but the family-balanced
weights above are primary. No row duplication or minority oversampling is
allowed.

## 5. Frozen model and optimizer

The only fitted model is a no-intercept conditional logistic ranker:

```
P(a beats b | root,tape,interval) =
    sigmoid(beta dot ((x(a)-x(b))/scale))
```

- Width: exactly 64 coefficients and no intercept.
- Feature map: exact G2 feature deltas; no added, removed, selected, or
  post-result interaction columns.
- Standardization: train only. Means are fixed to zero to preserve
  antisymmetry. For columns whose G2 schema has `train_standardize=true`,
  `scale_j = sqrt(sum(w*delta_j^2)/sum(w))`; values below `1e-12` use 1.
  All other scales are exactly 1.
- Initialization: all-zero coefficients.
- Loss: family-balanced weighted binary log loss plus
  `0.5 * 1.0 * ||beta||^2`.
- Optimizer: SciPy L-BFGS-B, analytic gradient, `maxiter=500`,
  `gtol=1e-8`, one deterministic run.
- Calibration: none.
- Threshold: logit greater than zero selects canonical action `a`; less than
  zero selects `b`; exact zero is a prediction tie and scores 0.5
  concordance. No sign flip is permitted.
- Serialization: ordered feature names, schema hash, normalization scales,
  coefficients, source hashes, optimizer summary, and pair-manifest hash must
  round-trip exactly. Nonfinite/incompatible payloads fail closed.

No alternative L2 value, intercept, calibration, feature subset, horizon
model, optimizer, seed, checkpoint, or objective may be tried.

## 6. Spent-data preflight gate

The no-new-outcome preflight must report comparable, discordant, censored, and
concordant counts by partition, scale, family, root, interval, replicate, and
canonical action pair. It also reports informative-root effective sample size,
maximum raw pair share, family concentration, feature/hash integrity, source
immutability, and transfer-access rejection.

`READY_G4_SPENT_DIAGNOSTIC` requires all of:

- every immutable input and schema hash matches;
- zero transfer records, paths, predictions, or source-state reads;
- exact CRN/tape and interval consistency for every compared arm;
- at least 100 informative train roots and 32 informative development roots;
- at least four informative train families and three informative development
  families;
- at least 12 informative development roots in each of `pre768` and
  `pre1536`;
- at least 128 discordant development pairs;
- no one development root supplies more than 10% of raw discordant pairs;
- all 64-feature, reversal, zero-horizon-delta, no-mutation, and finite-value
  checks pass.

An intact corpus below a count/support threshold seals
`HOLD_G4_PAIRWISE_UNDERPOWERED`. A source, transfer barrier, schema, CRN,
feature, or arithmetic integrity failure seals
`KILL_G4_PAIRWISE_INFEASIBLE`. No model is fit after either decision.

## 7. One allowed spent diagnostic

If and only if the preflight is READY, fit the frozen model once on spent train
pairs and open the spent development pairs once. Development is explicitly
contaminated historical mechanism evidence.

Primary metrics use the frozen family/root-balanced weights:

- pairwise log-loss improvement: `log(2)-model_log_loss`;
- pairwise concordance: 1 for the correct sign, 0 for the wrong sign, and 0.5
  for an exact zero logit.

Report the same metrics by scale, interval, family, action pair, and natural
root mixture. Report coefficient and prediction finiteness, optimizer
convergence, reversal antisymmetry, effective root counts, and concentration.
Do not report source-game scores or replayed action outcomes.

Uncertainty is a deterministic stratified root bootstrap:

- seed `2026072604`;
- 10,000 replicates;
- resample roots with replacement within each behavior family;
- include all retained records/units/pairs for each sampled root multiplicity;
- recompute family-balanced metrics without refitting.

`SUPPORT_G4_PAIRWISE_MECHANISM_SPENT` requires:

- optimizer success, finite model, gradient infinity norm at most `1e-4`;
- primary development log-loss improvement point estimate greater than zero
  and bootstrap 95% lower bound greater than zero;
- primary development concordance point estimate greater than 0.5 and
  bootstrap 95% lower bound greater than 0.5;
- both `pre768` and `pre1536` have positive point log-loss improvement and
  concordance greater than 0.5;
- every development family with at least eight informative roots has
  nonnegative point log-loss improvement and concordance at least 0.5.

If an adequately supported primary metric is nonpositive with its upper 95%
bound at or below its null, seal `KILL_G4_PAIRWISE_MECHANISM_SPENT`. Otherwise
seal `HOLD_G4_PAIRWISE_MECHANISM_AMBIGUOUS`. In all cases the operational
terminal state is `HOLD_G4_AFTER_SPENT_DIAGNOSTIC`; no untouched label or
policy work follows automatically.

## 8. Prospective untouched evidence contract

G4 separately inventories, without reading outcomes, any completed
normal-start machine ancestry not present in:

- incumbent or learned-value training/restart manifests;
- R1/R1b/A2/R2a/C1/S3/G1/G2/G3 selector, label, diagnostic, development, or
  confirmation manifests;
- human, partial, restart, continuation, or synthetic sources;
- any sealed untouched confirmation block.

Inventory means provenance/root/family/source hashes only. No replay score,
recorded action, future milestone, or board-based favorable selection may be
read. Roots are grouped by whole ancestry and no root can cross partitions.
If no existing eligible roots remain, G4 reports that honestly and does not
reacquire routine roots.

Fresh future stream namespaces are reserved, not consumed, at:

- logical `61_000_000_000`;
- deck `62_000_000_000`;
- slot `63_000_000_000`;
- policy `64_000_000_000`.

The preflight hashes a prospective 512-root paired normal-start stream manifest
and proves zero collision against the complete historical stream union. This
reservation is only for a future full-policy utility experiment and does not
authorize it.

For any later untouched predictive prescreen, model action selections must be
frozen before labels. Outcome-free activity must include at least 48
model/incumbent disagreements, at least 16 in each training scale, at least four
genuine behavior families, no family above 40%, and exact ancestry separation.
Otherwise it holds before labels.

The prospective predictive power calculation treats each ancestry as one
Bernoulli concordance unit after root-level aggregation. It reports exact
two-sided alpha 0.05 binomial power against null concordance 0.5 for true
concordance values 0.60, 0.65, and 0.70 at 48, 64, 96, 128, 192, and 256
informative roots. A future design must have at least 80% power for
concordance 0.65 and preserve the activity/family/scale floors. Repeats or
action pairs do not increase the root count.

The sealed 32 G3 transfer roots remain forbidden and underpowered: even if a
future charter separately authorized them, 32 ancestry clusters cannot meet
the 48-disagreement floor, four-family floor, or the previously recorded
late-rung sensitivity (approximately OR 4.0 MDE). They cannot confirm G4.

## 9. Future utility rule

Even a positive untouched predictive prescreen would authorize only a new
preregistration. Any utility test must compare a frozen deterministic
low-margin pairwise reranker against the incumbent over complete, paired,
fresh-root normal-start trajectories with full-policy exposure. It may not use
a one-move h40 gate. Development and one sealed confirmation must remain
separate; only confirmation can promote or update the dashboard.

Human actions remain queries, never labels.

## 10. Operations and artifacts

The G4 namespace is
`runs/forensics/g4_conditional_pairwise_v1`. It stores only compact manifests,
the preflight, and, if READY, one model and one spent diagnostic report. It must
not contain simulations, replay trajectories, transfer artifacts, or score
outcomes.

One-heavy-job-at-a-time, 100 GiB hard free-space, 120 GiB target, healthy
dashboard/advisor services, incumbent and protected top-three truth, and
immutable failed-branch evidence remain mandatory. The G4 work is lightweight
and must not start a heavy process.

At every terminal point report:

- `CONTINUE`: only the single spent diagnostic while its READY gate is active;
- `HOLD`: all untouched labels, policy construction/evaluation, C2, human
  training ground, and confirmation;
- `KILL`: G3 permanently and any exact G4 mechanism candidate that fails its
  frozen diagnostic;
- `PROMOTE`: false.
