# Vendored KleidiAI subset

This directory contains the minimal KleidiAI source subset used by FlagTree's
direct W4A8 and W8A8 CPU runtimes.

- Upstream: <https://github.com/ARM-software/kleidiai>
- Revision: `a2abc4d3fb4669dcd30bcad8cd6fd4c232f64924`
- License: Apache-2.0; see `LICENSES/Apache-2.0.txt`
- Local modifications to files below `kai/`: none

The selected translation units implement BF16-to-dynamic-INT8 LHS packing,
QSI4/QSI8 RHS packing, dot-product GEMV, and i8mm GEMM. FlagTree's integration
sources live separately in `_native/kleidiai`; `kleidiai.py` compiles each
runtime to a content-addressed cache outside the source tree.
