# J1a Outcome-Free Cost/Power Amendment

Status: preregistered arithmetic and readiness amendment only. The authoritative
parent remains `HOLD_J1_IMPLEMENTATION_PREFLIGHT`; J1 is not scientifically
killed. This amendment cannot create an execution marker or authorize any
scientific work.

## 1. Immutable Parent

J1a binds, without modifying or reinterpreting:

- `J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md`;
- `J1_IMPLEMENTATION_READINESS_AUDIT.json`;
- `J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md`;
- `j1_joint_policy_value.py`;
- `test_rl_j1_joint_policy_value.py`; and
- the sealed J1 test evidence, protected denylist, runtime/storage projection,
  preflight lock, and terminal `HOLD_J1_IMPLEMENTATION_PREFLIGHT` result.

The complete J1 learning design is unchanged: one from-scratch 411,656
parameter policy/value/auxiliary model; 16,384 independent complete
normal-start roots with `starter_tile=None`; dense score-delta reward whose
episodic sum is final score minus start score; fixed machine-derived auxiliary
labels and coefficient; root-equal PPO losses; 64 rounds, four epochs per
round, one final candidate, and no sweep or checkpoint selection. The sealed
J1 source and test files and `train_ppo.py` are not edited.

## 2. Sole Design Change

Only prospective paired evaluation counts change:

| phase | paired roots | complete arms |
| --- | ---: | ---: |
| training | 16,384 unpaired roots | 16,384 |
| development | 896 | 1,792 |
| confirmation | 4,480 | 8,960 |
| total | | 27,136 |

Development and confirmation counts are multiples of 64. Candidate and
incumbent arms share logical/deck/slot streams within each pair and use unique
policy streams. Whole roots and ancestries are disjoint across train,
development, and confirmation. The inherited prospective bases 213B through
226B are unchanged; J1a uses only the first 896 or 4,480 rows of the applicable
parent ranges. No stream is reserved or consumed here.

## 3. Score Power

The primary score estimand remains the paired difference in
`log1p(max(final_score - start_score, 0))`. The conservative paired standard
deviation remains 1.25, the meaningful effect is a 7% geometric lift, alpha is
two-sided 0.05, and power is the prospective positive-direction rejection
probability:

`Phi(log(1.07)*sqrt(N)/1.25 - z_0.975)`.

The 80% power MDE is:

`exp((z_0.975 + z_0.80)*1.25/sqrt(N)) - 1`.

The parent arithmetic is reproduced with its exact constants
`z_0.975=1.959963984540054` and `z_0.80=0.8416212335729143`, and
`Phi(x)=0.5*(1+erf(x/sqrt(2)))`. The published N=1,024 and N=5,120 score
cells must match exactly before amended cells are accepted.

Confirmation admission requires power for 7% at least 0.95 and MDE strictly
below 0.055. Development values are reported as a permissive screen and cannot
KILL a target-sized null.

## 4. Progression Power

The progression estimand remains the eight-stream-stratum Mantel-Haenszel
common OR for P1536 with whole-root bootstrap. The exact prospective method is:

- eight equal strata;
- 768 simulated paired datasets per `(N, control rate, coupling, true OR)`
  cell;
- 199 whole-root bootstrap replicates per dataset;
- control rates `{0.02, 0.04, 0.08, 0.15}`;
- paired couplings `{0.00, 0.05, 0.10}`;
- true OR grid `{1.25, 1.50, 1.75, 2.00, 2.50, 3.00}`;
- full pass event: point common OR at least 1.25 and bootstrap lower 95% bound
  above 1.0; and
- the preserved accepted implementation
  `threes_rl.o2_online_option_preflight.simulate_capability_power`, source SHA
  `99e61f551d607e3b5b8457b7e76a17c8540f0e1d88afec3fa544296bdcd05fda`,
  called with `calibration_name="J1"`.

The preserved routine uses base seed `2026072621` and exact cell seed
`base + 100*N + round(10000*OR) + round(1000000*coupling) + 500000000`;
the final term is the routine's frozen non-`D0_D2` calibration offset.

The paired Bernoulli cell probabilities are the frozen mixture of the
independent joint distribution and the maximally shared monotone joint
distribution. Resampling the four paired root categories by a multinomial of
the stratum root count is the exact aggregate whole-root bootstrap for this
binary paired design. Zero numerator or denominator receives the same
Haldane-Anscombe 0.5 correction used by the accepted implementation basis.

For each OR, power is the minimum over all control rates and couplings.
Confirmation admission requires worst-case OR1.50 power at least 0.80. The MDE
is the first frozen OR-grid value with at least 0.80 worst-case power. A sealed
control P1536 rate below 2% remains `HOLD_INCONCLUSIVE`, never KILL.

Before any amended N=896 or N=4480 result is accepted, this exact source and
call contract must reproduce the published parent cells byte-for-number:

- N=1024 worst-case OR1.50 power `0.30078125`;
- N=1024 OR2.50 worst-case power `0.9453125` and grid MDE `2.50`; and
- N=5120 worst-case OR1.50 power `0.8854166666666666` and grid MDE `1.50`.

Any mismatch seals `HOLD_METHOD_REPRODUCTION`. J1a may not substitute an
approximation, another seed, an analytic interval, or a reduced draw count.

## 5. Frozen Cost Projection

J1a reuses the sealed parent
`J1_RUNTIME_STORAGE_PROJECTION.json` bytes and fixture timings. It may not
retime, change the incumbent fixture, or alter assumptions. The exact parent
formulas are rerun with the amended arm counts:

- planning case: 512 moves per complete arm;
- descriptive sensitivity: 5,000 moves per complete arm;
- safety multiplier: 1.25;
- training cap: 72 hours and 24 GiB;
- development cap: 24 hours and 8 GiB; and
- confirmation cap: 120 hours and 16 GiB.

Training must still pass its original caps. Development and confirmation must
pass their original caps and each must consume no more than 91% of its runtime
cap after the 25% margin. The 5,000-move projections include the same margin
and are descriptive, not conjunctive. Free disk must exceed the 100 GiB hard
floor and the 120 GiB target.

## 6. Estimands and Gates

Paired score-minus-starter and base-rate-aware common-OR progression remain the
primary estimands. Family and stream-block signs are descriptive. Maximum,
P95, and P99 are mandatory descriptive statistics and never conjunctive PASS
vetoes. A null may KILL only when the frozen design excludes the minimum
meaningful effect; otherwise it is `HOLD_INCONCLUSIVE`.

The amendment preflight validates exact parent file and payload hashes,
prospective count/range arithmetic, inherited denylist identity, no prospective
range overlap, J1a namespace freshness, deterministic power/schema arithmetic,
operational services/process/disk state, and zero-work counters. It performs no
broad corpus scan and parses no protected scientific payload.

Terminal precedence:

1. failure to reproduce the published parent power cells exactly:
   `HOLD_METHOD_REPRODUCTION`;
2. immutable identity, schema, or zero-work failure:
   `KILL_J1A_AMENDMENT_INTEGRITY`;
3. power, cost/headroom, or mutable operational failure:
   `HOLD_J1A_COST_POWER_AMENDMENT`;
4. all checks pass:
   `READY_J1A_COST_POWER_AMENDMENT`.

READY permits research-lead review and later construction of a separately
frozen J1 execution surface only. It does not authorize an execution marker,
reservation, game, label, optimizer step, checkpoint, holdout opening, policy
or score outcome, human-session read, incumbent/dashboard change, or
promotion.

## 7. Immutable Outputs

The fresh namespace is
`threes_rl/runs/forensics/j1a_cost_power_amendment_v1/`. It may contain only:

1. `J1A_TEST_EVIDENCE.json`;
2. `J1A_COST_POWER_ARITHMETIC.json`;
3. `J1A_PREFLIGHT_LOCK.json`; and
4. `J1A_PREFLIGHT_RESULT.json`.

There is no marker or execution command in this surface.
