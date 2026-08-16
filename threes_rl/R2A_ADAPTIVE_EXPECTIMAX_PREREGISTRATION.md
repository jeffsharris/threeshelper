# R2a Incumbent-Leaf Adaptive Expectimax Preregistration

Status: frozen on 2026-07-11 after R1.5a failed offline and before R2a
continuation outcomes.

## Historical Locks

- R1.5a/A2 is permanently killed before policy evaluation. Its natural labels,
  synthetic diagnostic labels, exact model pair, and offline gate are retained.
- The incumbent leaf and all four components remain frozen. No context model,
  UCT/MCTS, learned sidecar, human action label, or hand-written milestone
  reward enters R2a.
- R1b sealed C remains spent and unpromoted. Dashboard record remains 263,670.

## Single Search Configuration

Compute exact incumbent depth-2 action values first. Adaptive depth 3 is
eligible only when all hold:

- non-starter built max is at least 768 and below 3072;
- empty count is at most 3, or normalized depth-2 top-two margin is at most
  0.02.

The depth-3 search uses the exact same composite incumbent leaf, the existing
afterstate/state/action transposition caches, at most eight deterministic
chance representatives per non-leaf chance layer, and a 2,048 expanded-value-
node budget per decision. At budget exhaustion it falls back to the incumbent
one-ply post-spawn value. There is no wall-clock-dependent cutoff, so choices
remain deterministic; elapsed time is a gate diagnostic.

No trigger, budget, margin, empties, chance limit, or tie rule may be tuned.

## Frozen Root Prescreen

Source only exact naturally reachable states from A1 ancestry and corner2
whole-family holdouts. Training-partition and human states are excluded. Root
identity is canonical fresh ancestry; select at most one state per root.

Eligible boards are pre-1536 or pre-3072 (non-starter built max 768 or 1536).
Using the source replay only, annotate whether the recorded trajectory reaches
the next milestone within 10/20/40 moves. This annotation stratifies coverage
but is never an action label. Retain both promotion windows and controls that
do not promote within 40 moves.

Select at most 64 roots by deterministic round-robin over behavioral family,
target milestone, success/control role, congestion, margin, and hash. Cap any
family at 50% and report root/family/role counts. Freeze incumbent and adaptive
actions, trigger reasons, node counts/cutoffs, and selection runtimes before
rollouts.

Prescreen activity gate:

- at least 40 roots from at least three policy families;
- at least 8 changed actions and at least 10% decision activity;
- changed roots span at least three families;
- median adaptive decision runtime no more than 3x depth-2 and p90 no more
  than 5x.

Failure kills this exact R2a configuration without rollouts.

## Paired Continuation Gate

For each frozen root/action, force incumbent or adaptive first action, then use
the frozen depth-2 incumbent for all continuation moves. Use 16 A/B replicates,
one h40 path supplying h10/h20/h40, with low/high arms sharing disjoint deck,
slot, and policy streams. Store compact metrics and a capped replay audit.

Primary endpoint is paired h40 score difference, adaptive minus incumbent.
Secondary endpoints are target-milestone hazard, survival, anchor preservation,
empties, and moves. Same-action roots are structural zeroes and reported
separately. Inference clusters by canonical root.

Continue to one fresh normal-start development block only if all hold:

- at least 8 informative changed roots across at least three families;
- h40 score root-bootstrap 95% lower bound exceeds zero, or target-milestone
  lift is at least +3 percentage points with lower bound at least zero;
- A and B block score effects are both positive and milestone effects are both
  nonnegative;
- survival is no worse than -2 percentage points and anchor preservation no
  worse than -3 percentage points;
- no clear catastrophic root/corner failure.

Otherwise permanently kill R2a. A development pass would freeze one fresh
normal-start development block and one sealed confirmation block before reading
development outcomes. Only confirmation can promote or update the dashboard.
