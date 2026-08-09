# W8 KleidiAI code-generation study

## Scope and comparison rules

All measurements below are single-thread C++ microbenchmarks pinned to CPU 0
on the CIX P1 Cortex-A720.  Generated Triton functions are called directly
from their cached shared objects; Python and launcher time are excluded.
KleidiAI is v1.24.0.  Timed batches alternate the Triton-first and KAI-first
orders to reduce frequency and thermal bias.

There are two distinct experiments:

- The model-native Triton path uses BF16 input, symmetric RNE activation
  quantization, and a wider 32-output layout.  Its comparison with KAI is a
  performance reference, not an accuracy-equivalent result.
- The KAI-layout path consumes the exact same official `qai8dxp/qsi8cxp`
  packed blobs as KleidiAI, performs the same zero-point correction, scale,
  bias, clamp, and F32 output, and excludes packing on both sides.

## Strict same-blob result

The strict kernel is still ordinary Triton: `tl.load`, integer arithmetic,
`tl.sum`, `tl.join`/reshape, floating-point epilogue, and `tl.store`.  There is no
TLE_raw op, external declaration, or runtime call.  The source mirrors KAI's
physical two-accumulator layout; target-aware compiler patterns recognize the
ordinary multiply/reduction graph and emit SDOT and ADDP.

### Triton 3.7 migration revalidation

Triton 3.7 removed the old multidimensional `tl.cat(..., dim=0)` API.  A
mechanical migration to the new one-dimensional `tl.cat` was not valid: the
operation requires permission to reorder and no longer preserved the graph
recognized by the Arm backend.  The resulting kernel still produced correct
answers but emitted zero SDOT instructions and took 1698.52 us at
1024x3072, 21.60x KleidiAI.  This is exactly the kind of silent codegen
regression that an instruction audit must catch.

The source now expresses the same ordering with portable
`tl.join(...).permute(...).reshape(...)`.  The compiler recognizes the
ordinary signed-i8 multiply plus row reduction, replicated eight-byte
activation, and final pair reduction. Keeping the loaded activation as a 2x4
value through the join makes LLVM select `LD1R`; flattening it first selects
paired 64-bit loads plus lane-insertion moves. The active source emits SDOT,
LD1R, and ADDP without a frontend intrinsic. Latest same-blob direct-call
results from the final compiler build are:

| K x N | Auto schedule | Triton 3.7 | KleidiAI v1.24 | Triton / KAI |
| --- | --- | ---: | ---: | ---: |
| 1024 x 3072 | pointer U2 | 78.17 us | 79.64 us | 0.982x |
| 4096 x 2560 | pointer U2 | 265.72 us | 268.22 us | 0.991x |
| 2560 x 9728 | pointer U2 | 958.80 us | 1013.83 us | 0.946x |
| 9728 x 2560 | pointer U2 | 953.92 us | 1006.51 us | 0.948x |

Every active generated body contains 16 static SDOTs, one ADDP, ten LD1R
sites, zero lane-insertion moves, zero stack load/store, zero residual dot,
and zero external calls. Maximum absolute error is `2.3841858e-7`. Thus the
result covers both N-heavy and K-heavy model shapes and is not based on an
older cache or a hidden library call.

The retired flatten-before-join graph remains available only behind the
benchmark's `--flatten-k8` flag. Fresh-cache active/retired objects have
identical outputs; repeated alternating direct calls improve by 0.9-1.3%.
The C++ wrapper accepts the retired object as a reference and rejects any
non-bit-exact generated pair, making the instruction-level claim reproducible.

### Complete BF16 decode pipeline

The production contract starts from BF16, not a prepacked activation. An
ordinary Triton pack kernel now reproduces KleidiAI's asymmetric `qai8dxp`
min/max, zero-point, RNE quantization, and MR1/MR4 interleave. Its blobs are
byte-identical to the official BF16 packer at M1 and M16. The generated object
has three FCVTNS instructions, no stack traffic, and no external call. Pack
alone is slower than KAI (0.786 versus 0.604 us at M1/K1024), so the acceptance
test includes the full operator rather than hiding this cost.

| K x N | generated pack+matrix+BF16 | KleidiAI pack+matrix+BF16 | ratio |
| --- | ---: | ---: | ---: |
| 1024 x 3072 | 80.57-80.98 us | 81.15-81.26 us | 0.992-0.998x |
| 2560 x 9728 | 983.13-990.08 us | 1017.98-1021.92 us | 0.962-0.973x |
| 9728 x 2560 | 982.75-985.20 us | 1010.82-1012.32 us | 0.972-0.973x |

The C++ harness calls generated pack and matrix symbols directly. The KAI
side includes its BF16 pack, native matrix, and FP32-to-BF16 conversion.
Python and weight packing are excluded. LHS blobs and final BF16 outputs are
bit-exact at every shape. This establishes an end-to-end operator result, not
only a favorable matrix-only comparison.

Eighteen Qwen3 projection shapes are now built as a cache-keyed AOT bundle,
including every TP=1 fused QKV, fused gate/up, O and down projection for the
0.6B, 1.7B and 4B models. A thin vLLM dispatcher selects generated pipelines
only inside the measured thread-count envelope and retains KAI elsewhere.
With one thread, the same Python custom-op layer gives M1
generated/KAI latency
is 134.4-135.3/164.0-165.2 us at K1024/N3072, 1085.63/1165.89 us at
K2560/N9728, and 1074.74/1148.72 us at K9728/N2560. The wrapper-level gain is
larger because it also removes the former OpenMP-region and separate FP32
conversion overhead. It must not be presented as matrix-codegen gain.

### Complete BF16 M16 prefill pipeline

The M16 pack now follows KAI's vector reduction schedule without embedding
its C implementation. Eight FP32 extrema per row remain live through K and
are reduced only after the loop. Quantized values narrow to int16 before the
zero-point add and clamp. These ordinary Triton operations reduce the pack
body from 563 lines and 19 stack references to 451-465 lines with zero stack
access or calls. The generated blob matches the official KAI BF16 pack for
all 65,280 finite BF16 encodings. Pack latency is 1.93/7.48/22.24 us at
K=256/1024/3072, versus KAI's 2.06/7.93/23.52 us.

The matrix epilogue explicitly forms `lhs_scale * rhs_scale` before applying
it to the converted accumulator, matching KAI's floating-point association.
This removed rare final-BF16 tie differences at K4096. The following direct
C++ measurements include generated/KAI BF16 pack, matrix, and final BF16
output on both sides:

| K x N | Generated M16 | KleidiAI M16 | Generated / KAI |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 124.88 us | 126.79 us | 0.985x |
| 1024 x 3072 | 357.60 us | 361.90 us | 0.988x |
| 2560 x 4096 | 1149.24 us | 1146.56 us | 1.002x |
| 2560 x 9728 | 2642.87 us | 2651.65 us | 0.997x |
| 4096 x 2560 | 1343.25 us | 1365.70 us | 0.984x |
| 9728 x 2560 | 3011.92 us | 2988.35 us | 1.008x |

All LHS blobs and outputs are bit-exact. At the actual plugin custom-op layer,
generated/KAI ratios are 0.913x, 0.957x, 0.959x and 0.980x for
1024x3072, 2560x4096, 2560x9728 and 9728x2560 respectively. All 18 bundled
shapes have generated M16 objects; whether the plugin selects them is governed
by the end-to-end thread-count policy rather than object availability.

Real eager vLLM measurements on Qwen3-0.6B show why the router is conservative.
Generated/KAI decode throughput is 13.838/13.315 tok/s at one thread, but
17.867/18.319 at two threads, 19.143/19.346 at four, and 19.324/19.900 at
eight. The default cutoff is therefore one thread. Above it the plugin
installs the original KAI closure at model load, avoiding a per-layer fallback
branch; at eight threads this safe route gives 19.815 versus 19.810 tok/s in a
same-phase rerun. Multi-thread M16 codegen remains an explicit experiment.

The generated U4 loop body has 32 static SDOT sites, one ADDP in the output
epilogue, no stack load/store, and no external call.  For each K32 chunk it
uses four replicated 8-byte activation loads and eight 16-byte weight loads,
then carries two native `vector<4xi32>` accumulators.  Keeping these as two
one-dimensional vectors is important: a logical `vector<4x2xi32>` loop-carried
value is ABI-expanded and creates lane shuffles.

## Model-native W8 result

The compiler-generated Triton decode path is ordinary `tl.load`, `tl.dot`,
arithmetic, and `tl.store`.  It contains no TLE runtime call.  With
`BLOCK_N=32` and `UNROLL=2`, its hot loop has 16 static SDOT sites and no
stack spill.

| K x N | Triton GEMV | KleidiAI GEMV | Triton / KAI |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 24.954 us | 23.275 us | 1.072x |
| 1024 x 2048 | 50.593 us | 49.982 us | 1.012x |
| 1024 x 3072 | 76.927 us | 78.556 us | 0.979x |
| 1024 x 4096 | 102.576 us | 105.885 us | 0.969x |

KleidiAI's F32 activation quant/pack costs 0.520 us at K=1024. The earlier
Triton BF16 symmetric RNE quantizer cost 1.631 us because it expanded ties-to-
even into truncation, comparisons and selects. The standard CPU
`libdevice.rint` path now lowers to four vector `FRINTN` instructions and
costs 0.623 us in a C++ direct call. It is bit-exact with the former kernel,
has no external call or spill, and reduces the object from 295 to 211 assembly
lines. The matrix-only and older complete-pair table below is retained as the
measurement record; a current production pair uses the new quantizer.

| K x N | Triton direct pair | KAI quant/pack + GEMV | Triton / KAI |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 26.410 us | 23.795 us | 1.110x |
| 1024 x 2048 | 51.903 us | 50.502 us | 1.028x |
| 1024 x 3072 | 78.637 us | 79.076 us | 0.994x |
| 1024 x 4096 | 103.783 us | 106.405 us | 0.975x |

The crossover is expected.  KleidiAI processes four outputs at a time and
reloads the packed activation for every 1x4 output tile.  Triton's 32-output
layout loads four activation bytes once and feeds eight vector accumulators.
For a 32-output panel and 32 K values, both execute 64 SDOT operations and
load the same weight payload, but Triton issues 8 replicated activation
loads instead of KleidiAI's 32.  KleidiAI retains an advantage for small N
through its tightly scheduled 1x4 loop and faster, different quantizer.

## Tile and software-pipeline experiments

The output tile sweep at K=1024, N=3072 shows that register pressure, rather
than instruction selection, is the hard limit:

| BLOCK_N | GEMV | Static instructions | Stack references |
| ---: | ---: | ---: | ---: |
| 16 | 79.521 us | 64 | 0 |
| 32 | 76.861 us | 99 | 0 |
| 64 | 78.300 us | 169 | 0 |
| 128 | 440.151 us | 499 | 215 |

`BLOCK_N=32` keeps eight int32 vector accumulators live and leaves enough
registers for overlapping weight loads.  `BLOCK_N=128` crosses the register
budget and collapses under spills.  A backend should therefore use 32 as the
default internal decode panel on this target and retain 16/64 as autotune
candidates.

Triton 3.7 represents `tl.range(loop_unroll_factor=...)` with a TTIR loop
attribute.  The CPU pipeline did not run Triton's LoopUnroll pass, so the
hint was silently ignored even though NVIDIA and AMD run that pass.  The CPU
pipeline now runs it.  The generated assembly and direct microbenchmark are:

| UNROLL | Static SDOT sites | GEMV |
| ---: | ---: | ---: |
| 1 | 8 | 78.765 us |
| 2 | 16 | 76.351 us |
| 4 | 32 | 78.616 us |
| 8 | 64 | 81.642 us |

KleidiAI groups work in K=32 chunks, but copying that source-level unroll
factor is not correct for a wider 32-output tile.  `UNROLL=2` is the best
software pipeline here; deeper expansion increases code size and pressure.

The same effect applies to the strict four-output KAI layout. U2 remains the
model-shape default; deeper unrolling expands the loop body without improving
the bandwidth-bound long-K case. Explicit pointer induction reduces the U2
function from 306-307 to 275 assembly lines.

A fresh same-process pointer/split sweep overturned the earlier K-based
threshold. Pointer is 3.6-6.1% faster at K=256 through 3072, neutral at
K4096/N2560, and 1.8% faster at K2560/N9728; the long K9728 route was already
faster with pointer induction. `mode=auto` therefore selects pointer for the
full tested K=256..9728 range. This selector depends only on the measured
graph shape, not a CPU model or vendor name. Both variants retain the exact
KAI ABI and the same 16-SDOT/one-ADDP arithmetic contract.

## What KleidiAI does

The selected M=1 W8 kernel is
`qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod`, not an i8mm kernel.  For every 32
K values and four outputs its hot loop performs:

- eight 16-byte RHS loads;
- four replicated 8-byte LHS loads;
- eight SDOT operations into two partial accumulators;
- one horizontal pair-add after the K loop;
- packed RHS-sum zero-point correction, LHS/RHS scaling, bias, and clamp.

The LHS packed row contains K int8 values, an int32 negative zero point, and
a float scale.  Each four-output RHS block contains `4*K` int8 values,
four int32 reduction sums, four float scales, and four float biases.

For prefill, KleidiAI switches to its 16x4 NEON i8mm kernel.  On the same
1024x1024 W8 problem it is about 1.91x faster than KleidiAI's dot-product
GEMM path:

| M | DotProd | i8mm | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 208.354 us | 108.972 us | 1.912x |
| 32 | 417.017 us | 218.288 us | 1.910x |

This confirms the dispatch boundary: SDOT for single-token decode, i8mm for
multi-row prefill.

The compiler now generates the corresponding KAI-layout prefill path from
ordinary Triton as well.  This is a strict packed-data comparison: the
generated function and KAI consume the same packed LHS/RHS blobs, packing is
excluded, and maximum absolute error is `2.3841858e-7`.

| M x N x K | Generated Triton | KleidiAI i8mm | Triton / KAI |
| --- | ---: | ---: | ---: |
| 16 x 1024 x 512 | 59.443 us | 58.054 us | 1.024x |
| 16 x 1024 x 1024 | 110.55 us | 109.06 us | 1.014x |
| 16 x 1024 x 2048 | 211.769 us | 209.869 us | 1.009x |
| 16 x 1024 x 4096 | 516.603 us | 504.008 us | 1.025x |
| 32 x 1024 x 1024 | 220.876 us | 218.308 us | 1.012x |
| 16 x 4096 x 1024 | 438.788 us | 433.713 us | 1.012x |

The compiler recognizes the KAI physical M4/RHS panels and rewrites the
outer K loop to carry sixteen native SMMLA accumulators instead of rebuilding
a logical `16x4` vector on every iteration.  Metadata loads are contiguous;
the final body has 64 static SVE SMMLA sites, zero hot-loop stack traffic and
no external call.  Across repeated seeds at the base shape the ratio remains
1.014-1.016x, so the supported claim is KAI parity, not a generated-code lead.

The M16 route is no longer used blindly for very short prefill. The same
packed-layout matcher now accepts M4 and M8, and the loop rewrite can fuse an
M8 and M4 dot that share one RHS into a single M12 reduction loop. Production
routes M2-M4/M5-M8/M9-M12 through padded M4/M8/M12 respectively, and retains
M16 from M13 onward. At K1024,N1024, including activation packing, output
allocation and Python launch, the new schedule is 1.38-1.40x faster at M2-M4,
1.18-1.20x at M6-M8, and 1.12-1.14x at M9-M12 than padding to M16. The three
matrix objects have 16/32/48 SVE2 SMMLA instructions, no residual dot or
external call, and 0/2/0 folded spill/reload pairs.

The result is visible at model level. Qwen3-0.6B W8 with an eight-token prompt
improves from 130.76 to 100.24 ms in one process pair and from 130.50 to
97.75 ms in the reverse-order pair. That is 23.3-25.1% lower prefill latency;
all generated token IDs match. The production switch remains reversible via
`FLAGGEMS_ARM_W8_SHORT_PREFILL=0`.

## What the compiler can absorb

Already generated from ordinary Triton:

- packed int8 loads and SDOT selection from `tl.dot`;
- a rolled K loop, multiple independent accumulators, and BF16 epilogue;
- activation reuse across a 32-output panel;
- SVE2 i8mm SMMLA lowering for M/N/K multiples of eight at SVE VL=128;
- explicit loop-unroll hints, after adding the missing CPU TTIR pass.
- chained packed-dot recognition after loop unrolling, so every unrolled copy
  remains SMMLA instead of falling into generic vector expansion.

Compiler work still needed:

1. Add a target-aware decode-panel policy so a large logical output vector
   is internally strip-mined to 16/32/64 outputs before LLVM register
   allocation.  This prevents the `BLOCK_N=128` failure without requiring
   every kernel author to know the register budget.
2. Turn the exact KAI-layout decode and prefill recognizers into a declared
   packed-layout contract.  The compiler now handles both outer loop-carried
   forms, but it should not depend indefinitely on one frontend graph shape.
3. Generalize the new KAI-layout matcher into a declared CPU packed-layout
   contract instead of relying on one exact lowered graph.  The current
   implementation already emits SDOT/ADDP from ordinary Triton and has strict
   same-blob validation, but a first-class encoding would make it easier for
   frontends to select the layout.

The main conclusion is that KAI-class W8 decode and prefill code can be
generated from ordinary Triton.  The compiler now absorbs the physical
partial-accumulator mapping, replicated activation loads, SDOT, ADDP, SMMLA,
and K32 scheduling.  Remaining work is frontend layout selection, automatic
output-panel selection, and fusion that reuses activation preparation across
QKV or gate/up projections.

`benchmarks/bench_w8_kleidiai_layout_codegen.py --compile-only` can compile
and audit large shapes without allocating their full test tensors.  In split
and pointer modes it fails if the expected SDOT/ADDP instructions disappear
or stack traffic is introduced.  Its placeholder uses signed int8 pointers to
preserve the real KAI ABI and prevent an accidental unsigned-extension
fallback.
The `split` and `pointer` modes retain both address-generation variants for
reproducible comparison.
