# KleidiAI comparison on CIX P1

## Scope

This comparison was run on CIX P1 CD8160, pinned to Cortex-A720 CPU 0 with
one worker.  The llama.cpp checkout is commit `3b4fca1-dirty`, and the
KleidiAI source is v1.24.0.  The CPU reports 128-bit SVE, SVE2, DotProd, and
i8mm; it does not report SME.

The end-to-end comparison is Qwen3-4B Q4_0.  Both backends execute the same
model and the same blockwise-symmetric mathematical workload.  Two Triton
packing generations are retained in the report:

- the previous Triton path uses a wider custom layout with FP32 group scales;
- the new Triton path consumes KAI-compatible FP16-scale Q4/Q8 blocks;
- KleidiAI uses its qsi8d32p x qsi4c32p packed kernel.

Both Triton generations use ordinary AOT `tl.load`, integer arithmetic,
reductions, floating-point epilogues, and `tl.store`; neither calls a W4
runtime GEMV.

No result below compares KleidiAI with the earlier matching ACLE C ceiling.

## Route verification

The independent llama.cpp build was configured with
`GGML_CPU_KLEIDIAI=ON`, `GGML_CPU_REPACK=ON`, and
`GGML_CPU_ARM_ARCH=armv9-a+dotprod+i8mm+sve2`.

Verbose model loading reports:

```text
kleidiai: primary q4 kernel feature I8MM
kleidiai: primary q8 kernel feature I8MM
kleidiai: SME disabled
load_tensors: CPU_KLEIDIAI model buffer size = 1895.64 MiB
```

The I8MM label selects KleidiAI's combined GEMM/GEMV entry.  Single-token
decode has M=1, so its actual Q4 compute function is the entry's NEON DotProd
GEMV fallback:

```text
kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod
```

Its inner loop contains eight static `sdot` instructions.  It does not use
SMMLA for M=1.

## Qwen3-4B single-core decode diagnostic

The model is `/home/cix/models/Qwen3-4B/Qwen3-4B-Q4_0.gguf`.  Each result is
the mean of five 16-token decode repetitions with prompt processing disabled.
All paths are pinned to CPU 0 with strict affinity.

| Backend | Latency for 16 tokens | Throughput |
| --- | ---: | ---: |
| Native llama.cpp ARM repack | 3423.68 ms | 4.673 tok/s |
| Previous Triton W4 layout | 3319.65 ms | 4.820 tok/s |
| KleidiAI v1.24.0 | 2890.31-2891.08 ms | 5.534-5.536 tok/s |
| KAI-style ordinary Triton | 2805.53 ms | 5.703 tok/s |

The final Triton result also routes the model's four Q4_1 down projections
through a new compiler-generated physical-microtile kernel.  Its previous
implementation used the older dot-ready layout.

These numbers deliberately use one core for correlation with the per-shape
microbenchmarks.  They are not full-processor CIX P1 throughput.

## Qwen3-4B multi-core decode

The CIX P1 exposes eight Cortex-A720 cores and four Cortex-A520 cores.  The
following 32-token tests use only explicitly listed A720s and three
repetitions:

| A720 workers | Native llama.cpp | KleidiAI | Triton default |
| ---: | ---: | ---: | ---: |
| 4 | 14.290 tok/s | 13.979 tok/s | 14.859 tok/s |
| 6 | 15.273 tok/s | 15.054 tok/s | 16.045 tok/s |
| 8 | 15.573 tok/s | 15.039 tok/s | 15.599 tok/s |

These measurements use the default Triton route.  Thread-local cached-weight
lookup removes the old per-projection global mutex, and 16 dynamic N-ranges
per worker recover the former 8-core scheduler loss.  KleidiAI reaches 10.887
tok/s when all 12 cores are used because the four slower A520 cores create a
heterogeneous tail.  Thus 5.5-5.7 tok/s is the one-core diagnostic; the useful
device operating point for this model is roughly 15-16 tok/s.

Native, Triton, and KleidiAI produced identical eight-token greedy output for
the same test prompt.

## Per-shape Q4 GEMV

The KleidiAI numbers use its official benchmark harness and the exact M=1
kernel selected by llama.cpp.  The Triton numbers call the generated static
symbol through the native prepared wrapper.  Both sides start with packed
weights and packed/quantized activation data, so Python and model-load weight
packing are excluded.

Lower is better; values are median microseconds.

| K x N | Triton static | KleidiAI kernel | KAI latency reduction | Triton/KAI |
| --- | ---: | ---: | ---: | ---: |
| 2560 x 1024 | 115.35 | 90.82 | 21.26% | 1.270x |
| 2560 x 4096 | 481.32 | 362.64 | 24.66% | 1.327x |
| 2560 x 9728 | 1088.29 | 864.01 | 20.61% | 1.260x |
| 4096 x 2560 | 463.30 | 366.48 | 20.90% | 1.264x |
| 9728 x 2560 | 1081.58 | 853.88 | 21.05% | 1.267x |

The Triton activation prepare step, which unpacks canonical Q8_0 blocks into
the generated-kernel inputs, takes 0.10-0.31 us for these shapes.  It does not
explain the gap.

KleidiAI dynamically quantizes and packs its FP32 activation input.  That
cost is 6.33 us at K=2560, 10.12 us at K=4096, and 24.00 us at K=9728.
Even if this cost is added to KleidiAI while Triton is allowed to start from
an already quantized Q8_0 input, KleidiAI remains faster:

| K x N | Triton static | KAI quant/pack + kernel | KAI latency reduction |
| --- | ---: | ---: | ---: |
| 2560 x 1024 | 115.35 | 97.15 | 15.78% |
| 2560 x 4096 | 481.32 | 368.96 | 23.34% |
| 2560 x 9728 | 1088.29 | 870.33 | 20.03% |
| 4096 x 2560 | 463.30 | 376.60 | 18.71% |
| 9728 x 2560 | 1081.58 | 877.88 | 18.83% |

This second table is deliberately conservative in favor of Triton.  The
end-to-end result remains the authoritative comparison because llama.cpp
accounts for each backend's full activation conversion path.

## W8 reference point

KleidiAI's nearest W8 kernel is
`qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod`.  A new strict harness gives its
official LHS/RHS packed blobs to both the KAI kernel and an ordinary Triton
kernel, calls both symbols directly, and excludes packing.  At K=1024:

| N | Triton | KleidiAI | Triton / KAI |
| ---: | ---: | ---: | ---: |
| 1024 | 26.334 us | 23.931 us | 1.100x |
| 2048 | 53.242 us | 49.989 us | 1.065x |
| 3072 | 79.389 us | 78.587 us | 1.010x |
| 4096 | 104.827 us | 106.523 us | 0.984x |

The maximum absolute difference is `2.3841858e-7`.  The generated loop uses
KAI's two physical partial accumulators, replicated activation loads, SDOT,
and ADDP, with no stack traffic or external call.  This supersedes the older
non-equivalent comparison between KAI's asymmetric F32 path and Triton's
model-native symmetric BF16 path.  Details are in
`W8_KLEIDIAI_CODEGEN_STUDY.md`.

At the five production-sized Qwen projection shapes, the U2 generated
schedule remains bit-accurate and improves the same-layout comparison:

| K x N | Triton U2 | KleidiAI | Triton / KAI |
| --- | ---: | ---: | ---: |
| 2560 x 1024 | 63.727 us | 63.994 us | 0.996x |
| 2560 x 4096 | 263.208 us | 265.690 us | 0.991x |
| 2560 x 9728 | 963.214 us | 1018.168 us | 0.946x |
| 4096 x 2560 | 262.243 us | 264.644 us | 0.991x |
| 9728 x 2560 | 960.065 us | 1005.609 us | 0.955x |

The generated U2 body has 16 SDOTs, one ADDP, no spill, and no external
call.  This confirms that the KAI-class W8 lowering generalizes beyond the
earlier K=1024 study point.

KleidiAI does not provide direct RMSNorm, fused add + RMSNorm, RoPE, fused
QKV, or SwiGLU operators in this llama.cpp integration.  Its backend only
selects `GGML_OP_MUL_MAT`.  The Qwen end-to-end comparison therefore measures
KleidiAI's individual projection kernels, not a KleidiAI fused QKV or MLP.

## Strict W4 same-layout result

The earlier production-layout Triton W4 path was 1.26-1.33x slower than KAI
and explained the former end-to-end gap.  That result mixed two different
packing strategies.  The replacement kernel consumes the exact same KAI
`qsi8d32p/qsi4c32p` blobs and absorbs its signed-nibble schedule:

| K x N | Triton | KleidiAI | Triton / KAI |
| --- | ---: | ---: | ---: |
| 1024 x 1024 | 37.328 us | 38.001 us | 0.982x |
| 1024 x 2048 | 78.873 us | 77.780 us | 1.014x |
| 1024 x 3072 | 108.980 us | 111.333 us | 0.979x |
| 1024 x 4096 | 145.394 us | 148.120 us | 0.982x |

Outputs are bit-exact and packing is excluded.  Ordinary Triton operations
now lower to eight SDOTs, one ADDP, and fixed-point `SCVTF #4` per K32 group,
with no spill or runtime call.  This shows that the previous W4 gap was mainly
a layout and lowering problem, not a fundamental limit of generated LLVM.
Repeated process-level measurements after the final scale-scheduling change
span 0.970-1.016x as both kernels move between cache/uncore timing bands.  The
supported claim is parity, rather than a stable win for either implementation.

This kernel is now wired into the experimental Qwen/llama.cpp production
path.  Canonical Q4_0 weights are converted to the 72-byte KAI layout at model
load, while canonical Q8_0 activation blocks are consumed directly during
decode.  Across all five production Qwen shapes, the same-blob Triton/KAI
ratio is 0.995-1.008x and outputs are bit-exact.  The code-generation study is
in `W4_KLEIDIAI_CODEGEN_STUDY.md`; complete integration and profiling results
are in `W4_KAI_LLAMA_E2E_RESULTS.md`.

## Reproduction

The independent build is:

```bash
third_party/llama.cpp-w4/build-kleidiai/bin/llama-bench
```

The official microbenchmark build is:

```bash
artifacts/kleidiai-v1.24-bench-build/kleidiai_benchmark
```

The native Triton prepared-path benchmark is built from:

```text
benchmarks/cpp_wrapper/bench_triton_w4_prepared.cpp
```

The KleidiAI activation-pack benchmark is built from:

```text
benchmarks/cpp_wrapper/bench_kleidiai_lhs_pack.cpp
```

The exact same-blob W4/W8 microbenchmarks are:

```text
benchmarks/bench_w4_kleidiai_layout_codegen.py
benchmarks/cpp_wrapper/bench_w4_kleidiai_layout.cpp
benchmarks/bench_w8_kleidiai_layout_codegen.py
benchmarks/cpp_wrapper/bench_w8_kleidiai_layout.cpp
```
