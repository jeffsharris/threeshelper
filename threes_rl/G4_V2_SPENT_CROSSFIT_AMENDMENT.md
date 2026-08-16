# G4 V2 Spent-Data Cross-Fit Amendment

Date frozen: 2026-07-26

Status at freeze: `CONTINUE_G4_V2_SPENT_CROSSFIT`

## 1. Scope and permanent locks

The exact G4-v1 preflight remains permanently
`KILL_G4_PAIRWISE_INFEASIBLE`. It may not be rerun, weakened, upgraded, or
reinterpreted. Its support stop is not evidence about the unfit conditional
pairwise mechanism.

G3 remains permanently `KILL_G3_BOOTSTRAP_PREDICTIVE`. G4-v2 may not access
the 32 G3 transfer records, transfer predictions, transfer roots, or 226
reserved transfer paths. It may not generate a simulation, label, score
analysis, policy action, policy outcome, incumbent change, or dashboard point.

G4-v2 is one explicitly exploratory, non-promotable, spent-data cross-fit. It
uses only the 727 ordinary discordant CRN-matched pairs identified and hashed
by G4-v1. Its development roots are already spent and contaminated. A positive
result can authorize only a separately reviewed outcome-free fresh-root
acquisition preflight.

## 2. Immutable v1 identity

The following v1 artifacts are immutable inputs:

- charter SHA-256
  `765992cc0af3fc7c9d10c88ed3e0436a2ec6bc3b989f776775fe86230b22247e`;
- implementation SHA-256
  `d7fef45bb9d976b7912f6e12cde052fbd81c73589a3eed5029e8e9a1b95d2c27`;
- test SHA-256
  `67d2b91a07946788547ad2b860429ed7569c625f99ebf10d66ae0f3642fd0416`;
- preflight file/payload SHA-256
  `bad6ca9542990144ae4d6872ef16781ec741bdb4c0584b5cf24e9783797155db` /
  `4c0bc125ff2e14094d0bc6d330f3796458572af5bcf0ac22e53f0c6a6822a40e`;
- pair-manifest file SHA-256
  `5acad327380b8cdc021a3299e085a06de656a0cacab4ba9984c59079db63602a`;
- pair-dataset SHA-256
  `ade1040d0f1bc56f58dfd0dc73004fa12f02d3caaae01880b08cf424d824484d`;
- ordinary G3 SQLite SHA-256
  `d0954a91e84bc7a420d64e7294f40232c1ffcb692fab86d07425b138e063f820`.

V2 must reconstruct the pair dataset from the read-only immutable ordinary
database and reproduce the exact pair-dataset SHA before fitting. It inherits
v1's comparable-row and censor semantics, canonical lower-action orientation,
64 ordered G2 features, no-intercept antisymmetric action deltas, L2 value
1.0, L-BFGS-B optimizer (`maxiter=500`, `gtol=1e-8`), no calibration, no
sign flip, no alternate objective, and family/root/record/unit/pair training
weights exactly. Every fold uses training-fold-only RMS scaling with zero
means.

## 3. Outcome-free five-fold assignment

The prefit lock is created before reopening any pair outcome. It reads only
ordinary record metadata from the frozen G2 manifest.

All ordinary records from one `root_cluster` stay in one fold. Each root is
assigned one metadata stratum:

```
(behavior_family, scale_signature)
```

where `scale_signature` is the sorted `+`-joined set of ordinary scales present
for that root: `pre768`, `pre1536`, or `pre1536+pre768`.

Within each stratum:

1. order roots by
   `SHA256("G4-v2-fivefold-v1" | behavior_family | scale_signature | root)`;
2. break an impossible hash tie by the literal root string;
3. assign ordered root `i` to fold `i mod 5`.

Folds are numbered 0 through 4. The complete 352-root assignment and its
canonical hash are sealed before pair outcomes reopen. Fold assignment never
uses event, censor, action, score, frame geometry, or model information.

Each fold model trains on pairs from the other four folds and predicts only
roots in its held-out fold. A root can receive no in-fold prediction. Fold
metrics are descriptive and are never a conjunction gate.

## 4. Prefit support and power

The v1 immutable aggregate support counts are the only pair-outcome summary
used by the prefit lock:

- 727 total discordant pairs: 552 train and 175 development;
- 185 informative roots: 146 train and 39 development;
- 162 informative pre768 roots: 126 train and 36 development;
- 45 informative pre1536 roots: 37 train and 8 development;
- combined informative roots by family:
  `phaseblend_incumbent_lineage=154`, `corner2_lineage=14`,
  `legacy_learned_lineage=12`, `expectimax_baseline=3`,
  `phaseblend_cheap_lineage=2`;
- conservative pooled maximum raw pair share is at most `13/727`,
  or `0.01788170564`.

The train/development roots must be disjoint in the ordinary metadata, so the
counts above add without aliasing.

Power is the exact two-sided alpha-0.05 binomial test of a root-direction rate
against chance 0.50. A root direction is 1 when its balanced mean OOF pair
concordance exceeds 0.50, 0 when below 0.50, and 0.5 on an exact tie. Solving
for 80% power before fitting gives these minimum detectable true rates:

- all 185 roots: `0.6058598391551271`;
- 162 pre768 roots: `0.6093592872055698`;
- 45 pre1536 roots: `0.7118975492370878`;
- incumbent-lineage 154 roots: `0.6141147187979789`;
- corner2-lineage 14 roots: `0.8883369662582361`;
- legacy-learned-lineage 12 roots: `0.8692843235703345`.

The last two family calculations are sensitivity disclosures, not family
conjunction thresholds.

The machine-readable prefit lock is READY only if:

- at least 128 informative roots overall;
- at least 32 informative roots in each scale;
- at least three families have at least eight informative roots;
- conservative maximum raw root pair share is at most 0.10;
- all ordinary root partitions are ancestry-disjoint;
- the five-fold metadata assignment has zero root leakage;
- all immutable v1/G3/G2 hashes and transfer barriers pass;
- tests, disk, process, services, incumbent, dashboard, and top-three checks
  pass.

Failure of a count/power gate seals
`HOLD_G4_V2_CROSSFIT_UNDERPOWERED` before fit. An integrity or transfer-barrier
failure also fails closed without fitting and is reported under that terminal
state with the integrity cause.

## 5. Frozen fit and OOF prediction

After a READY lock, one immutable execution marker is written and logged before
pair outcomes reopen. Exactly five models are fit, one per held-out fold, in
fold order 0 through 4. Each model uses only its four training folds. No model,
normalization, coefficient, or prediction may be selected or discarded.

For each held-out pair:

- compute the fold model logit on the canonical action delta;
- logit greater than zero predicts the lower canonical action;
- logit less than zero predicts the higher action;
- exact zero is a prediction tie worth 0.5 concordance;
- swapping the delta must negate the logit exactly within `1e-12`.

The complete OOF prediction table is ordered by the frozen v1 pair order and
sealed once. It stores compact pair identity, fold, label, logit,
concordance, and hashes; it contains no source-game score.

## 6. Evaluation weights, bootstrap, and terminal gate

Training uses inherited v1 family/root/record/unit/pair weights. The primary
OOF evaluation is natural root-balanced:

1. within each root, records receive equal total weight;
2. within each record, informative `(horizon,replicate)` units receive equal
   total weight;
3. within each unit, discordant action pairs receive equal total weight;
4. each informative root receives total weight `1/N`.

The primary point estimate is the mean root-direction value defined in section
4. Continuous balanced pair concordance is reported secondarily. The same
calculation is reported by scale, family, fold, horizon, and action pair.
Pooled marginal calibration is absent and cannot rescue a rank failure.

The primary 95% interval is a deterministic ancestry bootstrap:

- seed `2026072605`;
- 10,000 replicates;
- resample the 185 informative roots with replacement;
- retain each sampled root's frozen root-direction value;
- percentile bounds at 0.025 and 0.975;
- never refit inside the bootstrap.

Model stability requires all five optimizers to succeed, finite coefficients
and logits, gradient infinity norm at most `1e-4`, exact save/load predictions,
and zero root/fold leakage.

`READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT` requires all:

- primary root-direction point estimate at least the frozen material floor
  `0.6058598391551271`;
- primary root-bootstrap 95% lower bound greater than 0.50;
- pre768 and pre1536 root-direction point estimates each greater than 0.50;
- at least three families with at least eight informative roots each have
  root-direction point estimates greater than 0.50;
- all model, antisymmetry, source, pair, fold, and transfer-barrier checks pass.

With adequate prefit power, failure of any primary/direction condition seals
`KILL_G4_PAIRWISE_MECHANISM` permanently. No fold-sign conjunction is added.

## 7. One-shot orchestration and outputs

V2 uses the fresh namespace
`runs/forensics/g4_conditional_pairwise_v2`.

The command sequence is:

1. `prefit`: writes a fold manifest and immutable no-outcome execution lock;
2. `open`: validates the lock, writes one immutable execution marker, and
   exits before reading pair outcomes or fitting;
3. `execute`: requires that exact marker, reconstructs and verifies the 727
   pairs, fits five models, seals OOF predictions and one terminal result.

Once a marker exists, `open` may never run again. `execute` is deterministic
and may only resume incomplete fold artifacts under the same hashes; completed
folds and the terminal result are immutable. No alternate run is permitted.

Allowed terminal states are exactly:

- `HOLD_G4_V2_CROSSFIT_UNDERPOWERED`;
- `KILL_G4_PAIRWISE_MECHANISM`;
- `READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT`.

At every terminal:

- `CONTINUE` is false unless the terminal is READY and the research lead later
  authorizes a separate outcome-free acquisition preflight;
- `HOLD` covers all fresh acquisition execution, labels, policy work, C2,
  human training ground, and confirmation;
- `KILL` preserves G3 and v1 permanently and adds G4 pairwise mechanism only
  if the powered OOF direction gate fails;
- `PROMOTE=false`.
