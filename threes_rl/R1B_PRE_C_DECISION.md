# R1b Pre-C Decision

Status: closed. C confirmation completed once and failed the frozen primary
score criterion. R1b is not promoted; the incumbent and dashboard are unchanged.

## D2 Result

- Frozen block: 512 split-stream normal-start games, disjoint from D0, D1,
  and sealed C.
- Candidate mean score-minus-starter: `25,073.68`; incumbent: `20,567.11`.
- Paired mean lift: `+4,506.57`, 95% CI `[+1,176.73, +7,894.16]`.
- P3072: candidate `32/512`, incumbent `18/512`; difference `+2.73 pp`,
  95% CI `[+0.20 pp, +5.47 pp]`.
- Median difference: `+4,123.5`, 95% CI `[+868.5, +8,620.54]`.
- Lower-decile difference: `+1,455.0`, 95% CI
  `[-1,100.05, +3,220.21]`; no frozen material-regression rule fired.
- Moves difference: `+12.25`, 95% CI `[+3.26, +21.36]`.
- Changed games: `276` wins, `235` losses, one tie. P3072 transitions were
  `30` gains and `16` losses.

## Robustness

- Mean difference after removing the largest gain: `+4,168.93`; after
  removing the largest loss: `+4,851.41`.
- Symmetric trimming leaves `+4,513.78` after one result per tail,
  `+4,511.40` after five, and `+4,447.94` after ten.
- Candidate P5/P10 improved from `3,427.65 / 4,380` to
  `4,067.85 / 5,835`.
- The fixed tail audit examined 23 cases: the 12 largest paired losses plus
  new crossings below the incumbent P5. The below-P5 rate changed by
  `-2.73 pp`, 95% CI `[-5.08 pp, -0.59 pp]`; zero candidate-only terminal
  anchor losses and zero candidate-only terminal maximum-tile displacements
  were found. No tail/corner block fired.

## Required Next Decision

R1b at 5,000 is a promotion candidate, not the incumbent. Before any opening
of C, run only the already-required held-out h10/h20/h40 pre-promotion
diagnostic on retained, provenance-safe roots. Compare R1b with the incumbent
under paired split exogenous streams and report score gain, survival, and
first-nonstarter-1536 hazard at each horizon. Do not retrain, tune, add a new
development block, or inspect C while doing this diagnostic.

C, the dashboard, `current_incumbent_policy.txt`, R1.5, R2, and all unrelated
branches remain held until that diagnostic receives explicit authorization and
its result is reviewed.

## Independent Diagnostic Result

- Read-only provenance audit found 21 roots absent from all R1b restart
  updates and normal-start seeds and disjoint from D0-D2 and sealed-C IDs.
  Exact board, preview, and tile-cycle state round-tripped for every root.
- Selection used only the current state: earliest one-768 plus one-384 frame
  per unsampled ancestry. Eventual outcomes were not used; all 21 happened to
  be historical failures. Families were phaseblend incumbent `19`, expectimax
  baseline `1`, and TD student `1`.
- With 16 paired continuations per root, h40 first-1536 was `30/336` in both
  arms. Score difference was `-324.08`, CI
  `[-2,495.47, +1,761.65]`; survival was `+0.60 pp`, CI
  `[-5.95 pp, +6.25 pp]`. No frozen harm rule fired.
- h20 was directionally positive: `12/336` for R1b versus `9/336` incumbent,
  with both seed blocks positive. h10 was tied at `1/336`.
- Frozen interpretation: `NEUTRAL`. This small, failure-only, highly
  phaseblend-concentrated slice does not localize D2's gain, but it does not
  contradict or block it.

Recommendation: authorize exactly one evaluation on sealed C. D2 already
shows a direct normal-start P3072 improvement (`32/512` versus `18/512`, paired
CI above zero), which is stronger level-up evidence than the diagnostic's
short-horizon first-1536 proxy. Do not open C until that authorization is
recorded.

## Confirmation Outcome

- C was opened exactly once per arm after content-hashing the manifest, all
  four incumbent components, and the R1b checkpoint. Pre/post hashes matched.
- Incumbent versus R1b mean score-minus-starter: `21,131.63` versus
  `21,919.80`; paired difference `+788.18`, 95% CI
  `[-2,412.96, +4,021.11]`.
- P3072 was tied at `21/512` in each arm, with `18` gains and `18` losses.
- Median difference `+453`; lower-decile difference `-113.7`, CI
  `[-1,593.66, +1,467.09]`; moves `+3.89`, CI `[-4.70, +12.62]`.
- The fixed tail audit did not block: below-P5 rate `+0.98 pp`, CI
  `[-1.95 pp, +3.91 pp]`; one candidate-only final anchor loss and one
  candidate-only terminal max displacement, both below the threshold of three.
- Symmetric trimming of 1/5/10 observations per tail left positive means of
  `+792.59 / +783.89 / +661.17`, but trimming cannot replace the frozen paired
  score interval criterion.

Final decision: `CONFIRMATION FAILED / NO PROMOTION / HOLD FOR REVIEW` because
the score CI crosses zero. Do not reinterpret D2, rerun C, update the incumbent
or dashboard, or launch another branch without a new reviewed decision.
