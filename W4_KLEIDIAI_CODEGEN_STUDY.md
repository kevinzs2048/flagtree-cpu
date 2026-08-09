# W4 KleidiAI-layout code-generation study

## Result

This experiment asks a strict question: when ordinary Triton and KleidiAI
consume the same `qsi8d32p/qsi4c32p` packed activation and weight blobs, can
Triton-CPU generate the same class of Arm DotProd microkernel?

Yes.  The generated function contains no TLE_raw operation, runtime call, or
hand-written C/assembly body.  The kernel is expressed with `tl.load`, nibble
arithmetic, `tl.sum`, `tl.join`/reshape, floating-point arithmetic, and
`tl.store`.

Single-thread direct-call medians on CIX P1 Cortex-A720, pinned to CPU 0:

| K x N | Triton | KleidiAI v1.24 | Triton / KAI |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 37.328 us | 38.001 us | 0.982x |
| 1024 x 2048 | 78.873 us | 77.780 us | 1.014x |
| 1024 x 3072 | 108.980 us | 111.333 us | 0.979x |
| 1024 x 4096 | 145.394 us | 148.120 us | 0.982x |

Both functions receive the same 64-byte-aligned blobs produced by KAI's
official LHS and RHS packers.  Packing, Python, and launcher time are excluded.
Timed batches alternate execution order.  Outputs are bit-exact for every
tested shape.  The table selects the median-ratio process for each shape from
repeated fresh-process runs.  All observed ratios after the final scheduling
change are 0.970-1.022x: the supported claim is parity, with neither backend
consistently ahead across cache/uncore timing bands.

The same result holds for all five shapes used by Qwen3-4B decode:

| K x N | Triton | KleidiAI v1.24 | Triton / KAI |
| --- | ---: | ---: | ---: |
| 2560 x 1024 | 91.382 us | 90.801 us | 1.006x |
| 2560 x 4096 | 364.403 us | 361.781 us | 1.007x |
| 2560 x 9728 | 866.839 us | 860.001 us | 1.008x |
| 4096 x 2560 | 357.412 us | 359.230 us | 0.995x |
| 9728 x 2560 | 848.742 us | 850.576 us | 0.998x |

This layout is now connected to the real Qwen/llama.cpp path.  It reaches
5.668 tok/s for single-core Qwen3-4B Q4_0 decode, versus 4.820 tok/s for the
previous Triton layout and 5.534-5.536 tok/s for the independent KleidiAI
build.  Integration details and the end-to-end operator profile are in
`W4_KAI_LLAMA_E2E_RESULTS.md`.

## Mechanisms absorbed from KleidiAI

KAI stores one K32 group for four outputs as four FP16 scales followed by four
16-byte nibble vectors.  Its microkernel avoids explicit subtract-eight:

- shift the low nibble left by four;
- mask the high nibble with `0xf0`;
- interpret both as signed int8 and execute eight SDOT instructions;
- combine the two partial accumulators with ADDP;
- convert with `SCVTF ..., #4`, which divides the scaled integer result by 16;
- multiply the FP16 activation and weight scales while SDOT is in flight, then
  consume the dot result only in the final FP32 FMLA.

The Triton source now describes exactly that dataflow.  Pointer induction is
written explicitly so LLVM selects post-increment loads.  `UNROLL=1` is the
best choice for this register-heavy K32 body; U4 introduces stack traffic.

## Compiler changes

The Arm dot-product conversion pass recognizes four ordinary lowered forms:

1. a `4x4` signed int8 multiply plus row reduction and loop-carried add, and
   emits `llvm.aarch64.neon.sdot`;
2. the seed reduction before the first loop-carried add, so all eight dots in
   a K32 group use SDOT;
3. a `4x2` pair reduction and emits `llvm.aarch64.neon.addp`;
4. signed int32-to-float followed by an exact `1/16` scale and emits the fixed
   point form `SCVTF ..., #4` instead of `SCVTF` plus `FMUL`.

It also recognizes the Triton 3.7 `tl.join(x, x)` interleave/reshape graph over
an eight-byte activation load and emits a replicated 64-bit load.  The final
U1 body has eight static SDOT instructions, one ADDP, no stack load/store, and
no external call.

The Triton 3.7 migration was recompiled from a fresh cache rather than accepted
from an older object.  The final clean-build spot check at K=1024, N=3072 was
110.987 us versus KleidiAI's 108.576 us (1.0222x); other paired batches put
the same object slightly ahead.  It is bit-exact and retains the same
eight-SDOT/one-ADDP/zero-stack/zero-call invariant, so the defensible claim is
performance parity rather than a consistent lead.

Integer matching is enabled for AArch64 NEON+DotProd independently of BF16.
The SDOT rewrite itself does not require i8mm.  The current CIX AOT object is
nevertheless host-specific: LLVM selected an SVE `and z*.b` for nibble masking,
so that object must not be presented as a portable DotProd-only binary.
Recompiling and testing the source on a DotProd-only target remains required.

Q4 prefill has a separate fixed-width gate.  Its fused M4 kernel uses NEON
SMMLA and does not require SVE, so the compiler now exposes a
`w4-only + fixed-i8mm` mode for Apple M-series and other NEON-I8MM targets.
The exact packed-Q4 fusion is enabled automatically when CPU feature detection
reports DotProd+i8mm; `TRITON_CPU_FIXED_I8MM=0` disables it independently.
The broader generic NEON-i8mm pass remains opt-in.  FileCheck proves that the
fixed mode emits 64 NEON SMMLA sites while the DotProd-only mode leaves all
four dots untouched.  Performance still needs remeasurement on the M4 host
before making a cross-machine claim.

## What remains

The generated loop is now performance-equivalent to the hand-written KAI
microkernel.  KAI still has cleaner post-indexed activation address formation,
while LLVM is able to match its important arithmetic schedule.  Adding a new
opaque W4 intrinsic would provide little value.  Model-load packing and
automatic selection are implemented in the experimental llama.cpp adapter.
The useful next steps are a first-class Triton packed-layout contract and
multi-core/fusion work that shares data across QKV and gate/up projections.

The strict source and harness are:

- `benchmarks/bench_w4_kleidiai_layout_codegen.py`
- `benchmarks/cpp_wrapper/bench_w4_kleidiai_layout.cpp`

## Multi-row Q4 prefill

Decode parity does not imply prefill parity.  KleidiAI changes to a 16x4
i8mm microkernel for multi-row input, so a separate strict same-ABI
microbenchmark was added.  The final Triton kernel and KleidiAI receive the
same blobs from KAI's official LHS and RHS packers, including the FP16 scales
embedded before each K32 panel.  Both sides exclude packing and Python; the
Triton shared object is called directly for its complete grid.  The C++
harness compares every output element before timing, and the Python harness
also checks an independently constructed exact-ABI blob against an explicit
groupwise reference.  The same-blob C++ comparison is bit-exact.

The comparison was re-run with the production BF16 contract after auditing
the harness. Both functions consume the exact same blobs produced by the
official KleidiAI v1.24 packers. Triton writes BF16; KleidiAI writes FP32 and
that output is rounded to BF16 before comparison. Packing, Python and launcher
time are excluded, and timed batches alternate execution order.

| M, N=K=1024 | KleidiAI v1.24 | Triton production object | Triton / KAI |
| ---: | ---: | ---: | ---: |
| 4 | 73.435 us | 72.536 us | 0.988x |
| 8 | 148.696 us | 128.179 us | 0.862x |
| 12 | 220.631 us | 190.462 us | 0.863x |
| 16 | 239.090 us | 239.763 us | 1.003x |

Every row has zero BF16 ULP difference. Generated tail objects beat KAI's
shared tail handling, while the production M16 object is now within 0.3% of
KAI under the same official packed-blob and BF16-output contract.

The best ordinary Triton schedule uses four logical
`tl.dot([4,32], [32,4])` subtiles inside a loop carrying four FP32 output
panels.  The source reconstructs the logical operands from KAI's physical
`[K8, row, 8]` LHS and `[K8-segment, column, packed-K8]` RHS layouts with
ordinary load, reshape, transpose, nibble arithmetic and dot operations.  The
compiler recognizes that graph and consumes each 16-byte panel directly.  A
shift produces K[0:16], a mask produces K[16:32], and both feed SMMLA without
the eight ZIP1 nibble-expansion shuffles required by the earlier packing.
ZIP2 is still used to form logical output rows from SMMLA's 2x2 accumulators,
as it is in KAI.  Four temporary integer
accumulators are converted with fixed-point `SCVTF #4`, then the embedded LHS
and RHS scales are folded immediately into the FP32 output state.  No expanded
weight matrix or logical dot-result buffer is materialized.

These objects carry exactly one, two or three M4 panels.  They remove the
generic M<16 control flow and partial-store handling present in KAI's shared
16x4 function. The audited ratios are 0.978x, 0.863x and 0.862x for M4/M8/M12.
The production router pads a 1-15-row remainder to the next legal M4 shape;
M1-M3 decode uses a separate generated SDOT kernel.

The generated M16/N4 body has 64 static NEON SMMLA sites, sixteen fixed-point
conversions, no external call, and no logical dot left. The compiler now
loads and consumes one M4 row pair at a time only for the four-panel loop.
This shortens the integer-input live range enough for LLVM to keep all sixteen
FP32 accumulators fixed in registers: the loop-body spill/reload and 32
register moves disappear, assembly contracts from 482 to 442 lines, and the
matrix improves from 266.225 to 239.763 us. M4/M8/M12 keep their prior load
schedule and performance. Two M8 programs are now 5.6% slower on CIX;
`FLAGGEMS_ARM_Q4_M16_AS_M8=1` remains only as a cross-SoC diagnostic.
FileCheck and the assembly audit protect the four-dot fusion and native
KAI-panel matcher.

The result is not confined to N=K=1024. Strict same-blob direct comparisons
measure Triton/KAI at 1.003x for N3072/K1024, 1.005x for N1024/K3072, and
1.012x for N4096/K4096, all with zero BF16 ULP difference. Across those shapes
the single M16 program remains about 5.6% faster than two M8 programs.

Activation packing was audited separately. The active Triton pack reduces
four rows independently, performs one vector M4 scale division, and reloads
one K8 slice at a time for FRINTN quantization. It is ordinary Triton, has no
external call or folded spill/reload, and produces the same bytes as the
retired row kernel. Complete M4 panels use a predicate-free specialization;
only the last partial panel uses the masked form. Adjacent row pairs are
joined into 16-byte stores. At M16/K1024 direct SO calls improve 16.139 to
12.575 us (1.283x).

For a strict KAI comparison, source values are first rounded to BF16; Triton
receives those bits and KAI receives exactly representable FP32 values. Both
produce a byte-identical packed blob. KleidiAI v1.24 takes 7.553 us and Triton
12.577 us, so the remaining pack deficit is a real 1.665x rather than an input
contract artifact.

### Audit basis

The layout constants are derived from KAI's public packer contract rather than
guessed from one object file.  For `bl=32`, `mr=nr=4`, `kr=16`, `sr=2`, one
LHS M4/K32 block is four FP16 scales plus 128 int8 bytes (136 bytes), and one
RHS N4/K32 block is four FP16 scales plus 64 Q4 bytes (72 bytes).  The strict
C++ test calls the official KAI packers once, passes those exact pointers to
both implementations, rounds KAI FP32 output to BF16, and reports
`triton_max_bf16_ulp=0`.

The compiler matcher is deliberately narrower than a generic Q4 guess.  It
requires signed-int8 `4x32 * 32x4`, a zero integer accumulator, the exact
ordinary reshape/transpose/interleave graph, direct 128/64-byte physical
loads, and the `1/16 * lhs_scale * rhs_scale` loop epilogue.  A mismatched
shape or graph stays on the generic dot path.  The pass never inserts a KAI
symbol or runtime call; fresh-cache assembly auditing requires 64 SMMLA, no
residual dot, and no external call before an M16 result is accepted.  Separate
pass tests protect the one-panel M4 loop in full and fixed-NEON modes.

The whole-N and scalable-SVE experiments were rejected because they increased
stack pressure or repeated work. The successful foundation is the layout
contract that exposes the same 16-byte panels KAI's i8mm kernel consumes.
M8 splitting is now retained only as an opt-in cross-SoC diagnostic. Q4
decode and prefill are connected to the FlagGems/vLLM production router;
cross-SoC default selection still requires the prepared Mac measurements.

Reproduction entry points:

- `benchmarks/bench_w4_prefill_i8mm_codegen.py`
- `benchmarks/bench_w4_prefill_i8mm_direct_codegen.py`
- `benchmarks/bench_q4_m16_schedule_codegen.py`
- `benchmarks/bench_q4_lhs_pack_codegen.py`
- `benchmarks/cpp_wrapper/bench_kleidiai_w4_prefill.cpp`
- `benchmarks/cpp_wrapper/bench_q4_roundeven_aot.cpp`
- `benchmarks/cpp_wrapper/bench_q4_pack_kleidiai_bf16.cpp`
