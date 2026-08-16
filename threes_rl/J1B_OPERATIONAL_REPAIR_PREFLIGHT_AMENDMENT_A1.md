# J1b operational-repair preflight amendment A1

Status: frozen before the A1 implementation edit.

The first J1b test-evidence write created a valid immutable JSON artifact but
returned process status 2 because the CLI's generic return path required a
top-level `passes` field that test evidence did not contain. No readiness
prepare, phase lock, marker, owner, stream reservation/consumption, genesis,
game, optimizer step, checkpoint, or outcome occurred.

The pre-A1 evidence bytes are preserved at file SHA-256
`d2f6333bd4fdbe584fbf231141a24c01256dcc9ebe0f57c2691e19a8f046bddf`
and canonical payload SHA-256
`b462c0b46afaa478caeb66c622799eb1e7a533673439a89fe0e60650a448e25e`.
They must be moved without byte changes into the separate historical namespace
`threes_rl/runs/forensics/j1b_operational_repair_preseal_history_v1` and bound
by the final J1b readiness package. They are not authoritative test evidence.

The only A1 change is to make a successful immutable test-evidence write return
status 0 and expose an explicit `passes=true`, with a focused subprocess
regression. The J1b runtime repair, fresh stream partition, scientific
contract, decisions, and zero-work boundary do not change.
