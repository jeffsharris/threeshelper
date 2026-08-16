# O1 Charter Amendment A3: Pre-Spawn Air Consistency

Date: 2026-07-26

This amendment is authoritative over:

- the base charter at SHA-256
  `d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4`;
- A1 at SHA-256
  `712dada0815a696beb6040b15970515d454f9c7ddb1578e73fd98cc27a87955e`;
- A2 at SHA-256
  `42bec962eecda69d83e9493d3c57645a869b6fc048cd26ad8d77d259c9cdef76`.

All earlier files remain immutable. No O1 candidate replay content, marker,
rollout, label, fit, prediction, score, action, or policy outcome was opened
before A3.

The sole semantic correction is:

- A full simulator state is air-safe exactly when it has at least two empty
  cells.
- A deterministic pre-spawn base afterstate is an unconditionally air-safe
  merge afterstate only when it has at least three empty cells.
- The mandatory insertion consumes exactly one eligible empty cell, so the
  resulting full state then has at least two empties for every legal slot and
  visible-tile value.
- A candidate action whose pre-spawn afterstate has only two empties is not
  `merge_ready`, even when it pair-specifically merges the selected target
  pair.

P0 tests must include otherwise-equivalent pair-specific merges whose
pre-spawn afterstates have exactly two versus exactly three empty cells and
must reject/accept them respectively. No preview-context exception or
post-result relaxation is permitted.
