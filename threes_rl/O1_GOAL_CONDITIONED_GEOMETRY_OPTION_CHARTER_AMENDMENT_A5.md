# O1 Charter Amendment A5: Paired-Arm Stream Collision Semantics

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
  `01e8ec2270ea82610bc84b36f5ab8d8abf6dbbc47ad5f611b1e54c4e5b633cf6`.

All earlier files remain immutable. The first P0 `prepare` command stopped
before creating the output directory or opening candidate replay content. Its
only failed check was an engineering false positive:
`internal_stream_ids_unique=false`. All four historical collision lists were
empty across 9,097 matched metadata sources.

The 77B/78B/79B/80B namespaces remain frozen. The exact internal stream
contract is:

1. Every training trajectory code has exactly one row and one logical, deck,
   slot, and policy stream ID.
2. Every development and untouched-test trajectory code has exactly two rows:
   O1 and incumbent.
3. The two paired rows have exactly equal logical, deck, and slot stream IDs.
4. Logical, deck, and slot IDs are unique across distinct trajectory codes.
5. Policy stream IDs are globally unique, including between paired arms.
6. The four numeric namespaces are mutually disjoint.
7. Every requested ID must have zero intersection with the complete
   historical union.

Only the deliberate logical/deck/slot equality within one paired trajectory
code is permitted. A missing arm, third arm, policy duplicate, duplicate
across trajectory codes, cross-namespace collision, or historical collision
fails closed.

This correction changes collision validation only. It does not change any
stream ID, trajectory, partition, model, power, support, scientific gate, or
outcome contract.
