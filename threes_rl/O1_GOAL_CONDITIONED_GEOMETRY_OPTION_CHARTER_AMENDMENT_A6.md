# O1 Charter Amendment A6: Post-A5 Evidence Transition

Date: 2026-07-26

This amendment is authoritative over:

- the base charter at SHA-256
  `d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4`;
- A1 at SHA-256
  `712dada0815a696beb6040b15970515d454f9c7ddb1578e73fd98cc27a87955e`;
- A2 at SHA-256
  `42bec962eecda69d83e9493d3c57645a869b6fc048cd26ad8d77d259c9cdef76`;
- A3 at SHA-256
  `b5564af3af217c9e69fb88d40c1a3d8af140439819b7fa67bdfb687c80ad6d6a`;
- A4 at SHA-256
  `01e8ec2270ea82610bc84b36f5ab8d8abf6dbbc47ad5f611b1e54c4e5b633cf6`;
- A5 at SHA-256
  `e7e83607994f71fa5818e7c781511f22c344792f3a46eaec97a84a395829cf8d`.

The pre-A5 test evidence remains immutable at:

- path `threes_rl/runs/forensics/o1_p0_test_evidence.json`;
- file SHA-256
  `3354f76d42bc69cc425297b3e2aaa720847aec4d4d1fd746637c44a3ff633734`;
- canonical payload SHA-256
  `1d08910bc8230a07d9ddccf53eb0f5631b955f2b597044e742caae8cd953135a`.

It is valid historical evidence for the pre-A5 runner/tests and is never
overwritten or reinterpreted as post-A5 coverage.

The first `prepare` attempt stopped before creating the O1 P0 output
directory, marker, source inventory, or exclusion manifest. It opened no
candidate replay content and consumed no stream. All historical collision
lists were empty; only the now-corrected paired-CRN internal uniqueness check
failed.

Post-A5 compile/test evidence must be written to the distinct immutable path:

`threes_rl/runs/forensics/o1_p0_test_evidence_a5.json`.

Its file and canonical payload hashes, together with the current runner/test
hashes and the preserved old evidence hashes, must be sealed in:

`threes_rl/runs/forensics/o1_p0_a6_evidence_transition.json`.

The preparation command must verify that transition payload against the live
runner, tests, A6 amendment, old evidence, and new evidence before opening the
candidate-content marker. The marker and result versions are `v1_a6`; their
scientific contract remains A1-A5 with A6 changing only evidence provenance
and orchestration. No candidate content may open until all bindings agree.
