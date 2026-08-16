# O1 Charter Amendment A2: Labels, Multiplicity, Streams, And P0 Seal

Date: 2026-07-26

This amendment is authoritative over:

- the base O1 charter at SHA-256
  `d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4`;
- Amendment A1 at SHA-256
  `712dada0815a696beb6040b15970515d454f9c7ddb1578e73fd98cc27a87955e`.

Both earlier files remain immutable. No O1 candidate replay content, rollout,
label, fit, prediction, score, action, or policy outcome was opened before A2.

## A2.1 Non-Fictional Successor Labels

Auxiliary successor labels always describe the actual observed state at the
exact relative h10, h20, or h40 checkpoint:

- `separated`, `diagonal_touching`, `adjacent`, or `merge_ready` is the exact
  selected-pair stage then observed at the fixed target scale;
- `merged_success` is used only when the normalized count of `2*T` is greater
  than its option-root count at that checkpoint.

If an option terminates before a checkpoint for safe non-merge stage success,
safe merge success, failure, or terminal state, that later auxiliary checkpoint
is masked out. No terminal state is propagated as a fictional absorbing
geometry label. Event-category supervision remains available for the
terminating decision sequence.

## A2.2 Exact Learning Multiplicity

The P0 allocator freezes exactly 240 train roots and 80 development roots.
The untouched-test count is the selected smallest power-passing N.

E0 learning, if later authorized, uses:

- four rounds;
- exactly two trajectories per train root per round;
- exactly `240 * 4 * 2 = 1,920` train option trajectories;
- at most 40 decisions per trajectory;
- no trajectory from development or untouched test in the replay buffer.

After the final round-4 checkpoint is sealed, development uses exactly eight
paired O1/incumbent trajectories per root. Untouched test, after all opening
barriers pass, also uses exactly eight paired trajectories per root. These
paired evaluations are diagnostics/mechanism evidence, never training data.

## A2.3 Exact Stream Derivation

Roots are ordered by their immutable root ID within each partition. Define:

- train trajectory code:
  `root_index*8 + round_index*2 + replicate`, with zero-based round
  `0..3` and replicate `0..1`;
- development trajectory code:
  `1_000_000 + root_index*8 + replicate`, replicate `0..7`;
- untouched-test trajectory code:
  `2_000_000 + root_index*8 + replicate`, replicate `0..7`.

For trajectory code `c`:

- logical stream ID is `77_000_000_000 + c`;
- deck stream ID is `78_000_000_000 + c`;
- slot stream ID is `79_000_000_000 + c`;
- O1 policy stream ID is `80_000_000_000 + 2*c`;
- paired incumbent policy stream ID is `80_000_000_000 + 2*c + 1`.

Training uses only the O1 policy stream. Paired development/test arms share
logical, deck, and slot IDs and have separate policy IDs. The slot stream uses
the simulator's shared-uniform legal-slot mapping after trajectory divergence.

Every policy decision consumes the next variate from that trajectory's policy
stream in move order. Round 1 maps one uniform variate over the sorted legal
action list. Rounds 2-4 use one variate for the epsilon decision and, only when
exploring, the next variate over sorted legal actions. Greedy ties are resolved
by lowest simulator action enum and consume no extra variate.

No stream ID is allocated per decision. The trajectory stream plus zero-based
decision index and draw index is the complete reproducible identity.

## A2.4 P0 Power And MDE Reporting

P0 computes OR-grid power for every candidate N. It selects the smallest N
whose OR 1.50 power is at least 0.80. The reported MDE is then the smallest
OR-grid value with at least 0.80 power at that selected N, not at any other
sample size.

If no candidate N passes OR 1.50, selected N and MDE are null and P0 must seal
`HOLD_O1_DATA_OR_POWER` without constructing a nominal untouched-test
partition.

## A2.5 Exact P0 Artifact And Hash Envelope

The output namespace is
`threes_rl/runs/forensics/o1_goal_conditioned_option_p0_v1`.

Before candidate content access, `prepare` writes exactly:

- `O1_P0_SOURCE_PATH_INVENTORY.json`;
- `O1_P0_EXCLUSION_MANIFEST.json`;
- `O1_P0_CONTENT_OPENED.json`.

After the marker, `scan` writes exactly:

- `O1_P0_ROOT_MANIFEST.json`;
- `O1_P0_RESULT.json`.

Every JSON object has:

- a fixed `version`;
- sorted JSON-compatible keys;
- `canonical_payload_sha256`, computed over the entire object after removing
  that field with UTF-8 ASCII-safe canonical JSON
  `sort_keys=True,separators=(",",":")`;
- a post-write read/verify check.

The result binds file SHA-256 and canonical payload SHA-256 for all four prior
artifacts, plus charter/A1/A2, implementation, tests, test evidence, geometry
schema, power table, selected-N/MDE, partition manifests, services, disk,
process, streams, forbidden-work counts, and terminal decision.

`prepare` and `scan` both use exclusive creation. A marker or result can never
be overwritten or rerun. A result with a failed canonical round trip is
invalid and fails closed. The P0 result schema version is
`o1_goal_conditioned_option_p0_result_v1_a2`.
