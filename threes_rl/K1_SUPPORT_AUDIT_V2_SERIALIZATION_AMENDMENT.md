# K1 Support Audit V2 Serialization Amendment

Date: 2026-07-26

## Scope

The immutable K1 support audit v1 scientific result is complete, but its
embedded canonical payload hash does not survive JSON serialization. The
in-memory histogram used integer keys; JSON converted those keys to strings
after the hash had been computed.

The v1 artifact remains byte-for-byte preserved:

- Path:
  `threes_rl/runs/forensics/k1_support_audit_v1/K1_SUPPORT_AUDIT.json`
- File SHA-256:
  `536fd76da79791e873eecf1ea72c90ca2f26c9adc73f341c8fdb779517a73111`
- Embedded v1 payload SHA-256:
  `949bd408635a29e25ebdb6ca61b7723b7f68dca4f21d17d63be1b5e9de8efc3f`
- Canonical SHA-256 of the v1 body after a JSON read:
  `171c0b09ac6e92c7a36f1efb42932ba1a58ce1f917c310de470f5ffcfef7833b`

## Frozen Repair

V2 is a serialization-only wrapper generated from the existing v1 JSON. It:

1. Reads only the exact hash-bound v1 result and the amendment/source/test
   files needed to seal provenance.
2. Removes the invalid v1 `canonical_payload_sha256` field and embeds every
   remaining v1 field unchanged as `scientific_payload`.
3. Records the exact canonical hash of that post-JSON scientific payload.
4. Adds a new outer canonical payload hash computed after JSON-compatible key
   normalization.
5. Refuses a source hash, source-body hash, embedded v1 hash, or decision that
   differs from the frozen identities above.
6. Writes once to a separate v2 output and refuses overwrite.

No replay, source state, action, score, timing, value, stream, model, or policy
outcome may be reopened or recomputed. All support counts, operational audits,
and the terminal `KILL_EXACT_DEPTH3_PROGRAM` decision remain unchanged.

V1 is retained as superseded serialization-defect evidence. V2 becomes the
authoritative integrity envelope for the already completed scientific audit.
