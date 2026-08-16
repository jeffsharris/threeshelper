# O4 P0 Amendment V2: Marker Serialization

Date: 2026-07-27

Status: authoritative narrow engineering amendment. The O4 scientific charter,
representation, source universe, exclusion set, role/family/target matrices,
stream rows, power contract, and decisions remain unchanged.

## V1 Evidence

The immutable V1 marker was written before source-content access. Its first
`run` invocation stopped in `_load_marker` because Python tuples in
`role_family_target_counts` had become JSON arrays on disk and were compared
to newly constructed tuples with raw Python equality. No source replay body,
support scan, allocation, stream, label, model, or outcome was opened.

V1 is preserved as `HOLD_O4_P0_V1_ORCHESTRATION`. Its marker, runner, test
evidence, and engineering HOLD are immutable and may not be reused for another
scan.

## Sole Change

V2 compares marker bindings through canonical JSON normalization:

```text
normalize(x) = json.loads(json.dumps(x, sort_keys=True))
```

Equivalently, canonical payload hashes must match. Tuple/list representation is
therefore treated identically while every key, value, order-bearing array, and
scalar remains bound. Unexpected changes still fail closed.

V2 uses a fresh runner, focused test, evidence file, output namespace, marker,
and terminal result. It imports V1's pure scientific audit functions under
their immutable source hash and changes no scientific computation.

## Governance

V2 must seal new test evidence, then a zero-content marker, before source
access. It may run the same one-shot outcome-free P0 exactly once. It seals the
same terminal scientific decisions:

- `READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT`;
- `HOLD_O4_DATA_SUPPORT`;
- `KILL_O4_REPRESENTATION_PREFLIGHT`.

Acquisition, training, labels, models, policy outcomes, incumbent changes, and
dashboard changes remain unauthorized.
