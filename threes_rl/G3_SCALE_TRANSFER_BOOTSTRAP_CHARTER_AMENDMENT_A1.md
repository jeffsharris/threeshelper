# G3 Scale-Transfer Bootstrap Charter Amendment A1

Date frozen: 2026-07-25

Status: authoritative outcome-free amendment to
`G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER.md`.

The original charter is preserved byte-for-byte at file SHA-256
`e216aa50737afee0d439e060cc9b1e1f24d2f552af4c3f0c8944470ff7a45fc1`.
No label value, model outcome, transfer outcome, candidate action, rollout
outcome, or score outcome was opened before this amendment.

## Corrected Stream Formula

The original Exact Label Contract correctly requires every legal-action arm
within one selected state-scale record and replicate to share the same
logical, deck, slot-uniform, and policy-tie exogenous tapes. Its later formula
incorrectly included the action ordinal, which would assign different tapes
to different actions.

This amendment replaces only that formula.

Record ordinals remain all training rows, then development rows, then transfer
rows, each in the frozen charter order. For record ordinal `r` and replicate
`j in 0..7`, each stream ID is:

`base + 8*r + j`

All legal first-action arms for that record and replicate reuse those same
four stream IDs. The forced first action is an arm identifier, not a stream
identifier. Stream IDs are unique across `(record ordinal, replicate, stream
kind)` and intentionally equal across legal actions within that unit.

The manifest must audit both properties:

- no unintended collision between different record/replicate units or stream
  kinds;
- exact intended equality across every legal-action arm within one
  record/replicate unit.

The reserved bases remain `57B/58B/59B/60B`. No other scientific, model,
feature, partition, label, gate, power, cost, or downstream-policy rule
changes.
