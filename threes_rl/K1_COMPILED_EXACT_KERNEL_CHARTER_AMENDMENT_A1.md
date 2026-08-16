# K1 Charter Amendment A1: Explicit macOS SDK Root

Date frozen: 2026-07-26

The original K1 charter and toolchain design preflight remain immutable.

The first development build invoked the byte-locked Apple Clang binary with
the frozen optimization and floating-point flags. It failed before producing a
library because the direct compiler invocation did not provide a macOS SDK
root and the linker could not locate the system library.

This amendment changes only build orchestration:

- installed SDK:
  `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/
  Developer/SDKs/MacOSX26.5.sdk`;
- add `-isysroot` followed by that exact path immediately after
  `-dynamiclib`;
- bind the SDK path and its read-only inventory in every later build/preflight.

The compiler binary, all original optimization/floating-point/warning/linker
flags, C source design, native exports, semantic contract, corpus, streams,
tests, equivalence tolerances, runtime gate, and terminal decisions are
unchanged. This is not a compiler or flag sweep. No library, fresh stream,
timing, or outcome existed when A1 was frozen.
