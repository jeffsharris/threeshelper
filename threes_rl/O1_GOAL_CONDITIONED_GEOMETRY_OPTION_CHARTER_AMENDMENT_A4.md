# O1 Charter Amendment A4: Structural Minimum for Powered Test Design

Date: 2026-07-26

This amendment is authoritative over:

- the base charter at SHA-256
  `d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4`;
- A1 at SHA-256
  `712dada0815a696beb6040b15970515d454f9c7ddb1578e73fd98cc27a87955e`;
- A2 at SHA-256
  `42bec962eecda69d83e9493d3c57645a869b6fc048cd26ad8d77d259c9cdef76`;
- A3 at SHA-256
  `b5564af3af217c9e69fb88d40c1a3d8af140439819b7fa67bdfb687c80ad6d6a`.

All earlier files and their power rows remain immutable evidence. No O1
candidate replay content, marker, rollout, label, fit, prediction, score,
action, or policy outcome was opened before A4.

The outcome-free power calculation found that candidate `N=144` has
OR-1.50 power above 0.80. That result remains reported and is not
reinterpreted. It is structurally ineligible because equal allocation over
the four starting stages provides only 36 roots per stage, below the base
charter's minimum of 48.

The authoritative test-size rule is therefore:

1. Compute and retain the frozen common-OR power result for every candidate N.
2. Mark a candidate statistically eligible when OR-1.50 power is at least
   0.80.
3. Mark a candidate structurally eligible only when it is divisible by 12 and
   `N / 4 >= 48`, equivalently `N >= 192`.
4. Select the smallest candidate satisfying both conditions.
5. Compute the reported 80-percent-power MDE at that selected N.
6. The allocator must independently assert at least 48 untouched-test roots
   per starting stage, in addition to all existing 12-cell, family,
   provenance, and overlap requirements.

`N=144` is recorded as `power_pass=true`,
`structural_minimum_pass=false`, and `eligible_for_selection=false`.
Insufficient support at the selected structurally valid N is
`HOLD_O1_DATA_OR_POWER`, never a representation failure.
