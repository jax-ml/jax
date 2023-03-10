This directory contains LLVM
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests that verify
that JAX primitives can be lowered to MLIR.

These tests are intended to be a quick and easy-to-understand way to catch
regressions from changes due the MLIR Python bindings and from changes to the
various MLIR dialects used by JAX, without needing to run the full JAX test
suite.
