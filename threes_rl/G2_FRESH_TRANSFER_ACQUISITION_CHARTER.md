# G2 Fresh Transfer Acquisition Charter

Status: frozen before any new availability measurement or game generation.

## Scientific Boundary

This is a source-acquisition design for the existing
`HOLD_G2_DATA_OR_POWER`. It is not a revival of G1-R exact-rung acquisition
and is not a policy comparison. A qualifying source is any completed natural
normal-start machine ancestry containing the first valid frozen G2
`pre3072_transfer` state. The collector policy is provenance only.

This charter authorizes implementation, tests, and one no-game preflight. It
does not authorize games, stream consumption, labels, rollouts, h10/h20/h40
outcomes, fitting, action analysis, score inspection, continuations, policy
evaluation, promotion, or dashboard changes.

## Frozen Collectors

The ordered slate is exactly:

1. `g2_transfer_corner2`: `corner2`
2. `g2_transfer_expectimax2`: `expectimax2`
3. `g2_transfer_phaseblend_incumbent`: the exact single non-comment line in
   `threes_rl/current_incumbent_policy.txt`

The immutable 64-state pilot-v1 action audit is the admission evidence. Its
signatures are:

- corner2: `4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043`
- expectimax2: `2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38`
- incumbent: `868a6337d932cc034a633272d10fea3fc733a3f542b49a13eda1c075371d1ccb`

Frozen disagreement rates are corner2/expectimax2 `0.59375` overall
(`0.78125/0.40625` by pre1536/pre3072), corner2/incumbent `0.53125`
(`0.59375/0.46875`), and expectimax2/incumbent `0.375`
(`0.5/0.25`). All three are distinct transitive components. Parent and student
remain members of the incumbent alias component and are not collectors.
Preflight must reproduce these exact signatures and rates without retiming.
Failure is `KILL_G2_ACQUISITION_PREFLIGHT`; no substitute family is allowed.

## Qualifying Root

A game is eligible only when all of the following hold:

- it is a completed, genuine fresh normal-start machine replay;
- its root has the fixed starter `1536`, move count begins at zero, and direct
  root/fresh-origin provenance is exact;
- it is not human, partial, restart, continuation, synthetic, or descended;
- replay and every retained state pass exact simulator restoration;
- one whole game is one ancestry and contributes at most one root.

Traverse frames in increasing `(frame.index, physical list index)`. The
qualifying state is the first live frame whose built maximum excluding the
initial fixed starter is exactly `1536`, whose stored legal-action names equal
the simulator's legal actions, and which has at least one legal action. This is
the frozen G2 `pre3072_transfer` definition. Ties in duplicate frame indices
use physical list index. No board hash, score, later promotion, geometry, or
policy action participates in selection.

Within each family, qualifying roots are retained in immutable stream-manifest
order until exactly `32` roots are retained. Later qualifying roots are logged
as completed but not retained. Quotas are independent and cannot be
reallocated.

## Streams And Schedule

Fresh reserved bases are:

- logical seed: `53_000_000_000`
- deck stream: `54_000_000_000`
- slot stream: `55_000_000_000`
- policy stream: `56_000_000_000`

For family index `f` and game index `g`, each stream is
`base + f*1_000_000 + g`. Preflight freezes all `3*640=1,920` rows and proves
internal and historical collision freedom without consuming a stream.

Execution, if separately authorized, uses one worker at nice `>=10` and
deterministic family round-robin chunks. Round `r` schedules family indices
`0,1,2` in order for game index `r`; persistence chunks contain at most six
games (two full rounds). A family at quota is skipped without reallocating its
manifest rows. No family may exceed `640` completed games.

Each attempt stores one compact completion row. A full replay and exact compact
qualifying state are retained only for the first 32 qualifying roots per
family. Completion rows may contain integrity, provenance, stream, runtime,
terminal/completion, and qualification fields, but no score, chosen action, or
policy comparison field.

## Frozen Feasibility And Budgets

The cap uses only pre-existing G2 source metadata. Those retained replays may
be selection-biased and are not unbiased yield estimates. Historical
pre3072/root counts were corner2 `13/28`, expectimax2 `2/13`, and incumbent
lineage `69/404`. One-sided 90% Wilson lower bounds are approximately
`0.34886`, `0.06575`, and `0.14814`; at 640 games these project approximately
`223`, `42`, and `95` roots. The weakest projection exceeds quota 32, so the
fixed cap is 640 per family. There is no adaptive Wilson stop and no favorable
partial-geometry stop.

Execution stops only when all quotas are `32/32/32` or when a family reaches
640 without quota, active wall time reaches 12 hours, output reaches 4 GiB,
free disk falls below 100 GiB, a competing heavy process appears, or required
services degrade. Target free disk is 120 GiB. Bounds are checked before every
game and after every at-most-six-game chunk. Any bound produces an immutable
HOLD; no cleanup or reinterpretation.

Worst-case preflight projection uses 1,920 compact rows, 96 retained full
replays/states, and the conservative maximum existing replay size from the
sealed QD-v2 storage inventory. It must be below 4 GiB. Runtime projection uses
the sealed QD5 pilot's total active runtime per completed game multiplied by
1,920 and must be below 12 hours; it is a capacity bound, not a performance
comparison.

## No-Game Preflight

The separate v1 transfer-acquisition runner, tests, output directory, and lock
must bind:

- this charter;
- the authoritative G2 proposal, feature implementation, preflight,
  preflight result, root manifest, schema hash, and test evidence;
- the immutable pilot-v1 signature panel and preflight lock;
- the exact incumbent file and every resolved checkpoint/config/weight source;
- simulator, evaluator, expectimax, n-tuple, provenance, replay, and runner
  sources;
- the complete historical stream/provenance union and its source hashes;
- the exact 1,920-row requested stream manifest;
- the output directory resolved as
  `threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1`.

Preflight must prove:

- exact three-family order, loadability, signatures, pairwise rates, and
  distinct components;
- G2 proposal/schema/preflight identities and extraction compatibility;
- first-state extraction determinism, complete fresh-root provenance,
  one-root-per-ancestry semantics, split-reset equality, and exact state
  round-trip;
- quota, cap, independent-family, stream-order, and round-robin behavior;
- zero requested/historical stream collisions and zero stream consumption;
- one worker, nice `>=10`, no competing heavy process;
- free disk above 120 GiB target and 100 GiB hard minimum;
- projected storage below 4 GiB and worst-case runtime below 12 hours;
- healthy ports 8765/8770 and advisor, dashboard record `263670`, protected
  top three `263670/261369/258561`, and no active human-session content read.

The immutable preflight decision is exactly one of:

- `READY_G2_FRESH_TRANSFER_ACQUISITION`
- `HOLD_G2_ACQUISITION_COST_OR_YIELD`
- `KILL_G2_ACQUISITION_PREFLIGHT`

READY authorizes only a later separately approved acquisition run. It never
authorizes labels, fitting, policy evaluation, promotion, or dashboard change.

## Retention

All G2, G1, S3, QD, R1b/C, incumbent, and protected replay evidence remains
immutable. Preparation uses a staging directory and atomically promotes only
after all checks pass. A failed staging directory is retained. The no-game
lock, manifests, hashes, tests, and decision are permanent compact evidence.
