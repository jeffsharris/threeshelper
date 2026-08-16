# K1 Compiled Exact Kernel Charter Amendment A5

Status: frozen before implementation and before any fresh K1 game, compiled
fresh-gate library, timing, depth-3 result, policy outcome, or score inspection.

Date: 2026-07-26

## Trigger

Amendment A4 moved the fresh K1 stream namespace to 73B through 76B after the
original zero-work 69B through 72B reservation. The next zero-work preflight
failed because the generic history scanner treated K1's own immutable A4
design declaration and its resulting failed staging copy as external prior
stream use.

This is a source-classification defect. It is not evidence about the kernel,
search, timing, policy, or game outcomes. No game, stream, compilation, timing,
depth-3 result, policy outcome, or score was produced.

## Sole Engineering Change

The K1 collision scan must classify exactly two current-experiment namespaces
as immutable internal declarations rather than external prior use:

1. `threes_rl/runs/forensics/K1_TOOLCHAIN_DESIGN_PREFLIGHT_A4.json`
   - file SHA:
     `a37f6a97dbe436cea40de3ece059bf618413a2ace7aa8793e0415feabebc11bc`
   - file-manifest SHA:
     `6e806ed4d9a7c62b9d674228af5d113216fd8dd6f77d3caddbdfb143866e4799`
2. `threes_rl/runs/forensics/k1_compiled_kernel_v1.staging.11781`
   - failure-file SHA:
     `df86b21b16bfe8bb52d34b2060dba50e60a4e6092d092abfae85f97a177d5fac`
   - directory-manifest SHA:
     `56c7405477df5c093ff4831ab2dde6e38dad156dcb10d6c7d4d16302192f505e`

Both namespaces must be hash-bound and revalidated as zero-work before they
are excluded. The scanner must still inspect every other matching JSON,
JSONL, and CSV source under `threes_rl/runs`, including the earlier failed
69B through 72B staging artifacts. Any hash change, new unclassified K1
staging artifact, or external 73B through 76B collision fails closed.

The current staging directory used by the in-progress preflight remains
excluded by its exact resolved path, matching the existing scanner contract.

## Preserved Contract

The A4 73B through 76B stream namespace remains unchanged. All K1 model,
native source, compiler, flags, exactness semantics, family slate,
game/root/state counts, deterministic extraction, partitions, timing
schedule, runtime thresholds, one-shot sealing, resource bounds, and
non-promotable decision rules remain unchanged.

The runner, tests, test evidence, and zero-work preflight must be freshly
hashed after this amendment. Every earlier staging directory and evidence
file remains preserved.
