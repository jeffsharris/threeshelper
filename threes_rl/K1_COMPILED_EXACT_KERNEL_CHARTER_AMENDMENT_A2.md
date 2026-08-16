# K1 Charter Amendment A2: Loadable Deterministic Mach-O UUID

Date frozen: 2026-07-26

The original K1 charter, design preflight, and A1 remain immutable.

The first A1 development library linked successfully but macOS `dlopen`
rejected it because `-Wl,-no_uuid` removes the Mach-O `LC_UUID` command.
Installed linker documentation states that the default UUID is derived from a
hash of output content and is the reproducible-build mode; only
`-random_uuid` is nondeterministic.

This amendment removes exactly `-Wl,-no_uuid`. It does not add
`-random_uuid`. Determinism remains a hard focused test: two builds from the
same frozen source/toolchain/flags must have identical full-file SHA-256.

Compiler, SDK root, every other flag, source, ABI, native functions, floating
point semantics, search behavior, corpus, streams, timing schedule, gates, and
terminal decisions remain unchanged. No loadable library, fresh stream,
timing, or outcome existed when A2 was frozen.
