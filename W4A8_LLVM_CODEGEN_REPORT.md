# W4A8 ordinary-Triton LLVM codegen and llama.cpp end-to-end report

## Result

This prototype replaces every transformer-layer W4 projection in the tested
Qwen3-4B Q4_0 GGUF during single-token, single-thread decode.  The kernels are
written with ordinary Triton `tl.load`, nibble arithmetic, and `tl.dot`.
There is no `TLE_raw` GEMV body and no call to a hand-written GEMV runtime.

On CIX P1 CD8160, pinned to CPU 0 with one llama.cpp worker:

| Qwen3-4B decode, 16 tokens | native llama.cpp | Triton W4 backend |
|---|---:|---:|
| latency | 3423.81 ms | 3320.10 ms |
| throughput | 4.673 tok/s | 4.819 tok/s |
| change | - | **3.03% lower latency / 3.12% higher throughput** |

Five repetitions were used for both results.  With warmup disabled, a
one-token run was 214.96 ms native and 207.86 ms with Triton, a 3.30%
reduction.  Greedy generation for `The capital of France is` produced the
same eight tokens on both paths:

```text
 Paris. The capital of Germany is Berlin
```

## Covered projections

The GGUF has 36 transformer layers.  The backend covers all seven main
projection classes, or 252 generated GEMVs per token:

| Projection | Quantization | K x N | Calls/token |
|---|---|---:|---:|
| attention Q | Q4_0 x Q8_0 | 2560 x 4096 | 36 |
| attention K/V | Q4_0 x Q8_0 | 2560 x 1024 | 72 |
| attention output | Q4_0 x Q8_0 | 4096 x 2560 | 36 |
| FFN gate/up | Q4_0 x Q8_0 | 2560 x 9728 | 72 |
| FFN down, layers 4-35 | Q4_0 x Q8_0 | 9728 x 2560 | 32 |
| FFN down, layers 0-3 | Q4_1 x Q8_1 | 9728 x 2560 | 4 |

The Q4_0 projections consume 156.67 ms/token on the native path.  The four
native Q4_1 down projections add 9.71 ms/token.  Together, the replaced
operators account for **77.69% of total native decode CPU time**.  The final
profile reduces them from 166.23 to 159.60 ms/token, or 1.042x.  The measured
end-to-end reduction is 6.48 ms/token.

Coverage is dispatch-dependent rather than a single unconditional number:

| Workers | Generated projections/token | Dispatch policy |
|---:|---:|---|
| 1-2 | 252/252 | all listed Q4_0 and Q4_1 projections |
| 4 | 4/252 | Q4_1 down projections; Q4_0 uses native repack |
| 8+ | 0/252 | native fallback |

Every listed shape has both generated static and range AOT entries.  The
policy above is a performance guard: it does not claim codegen coverage for
thread counts where the measured range path regresses.

## Per-operator profile

The following means are from 17 decode passes (one llama-bench warmup plus
16 measured tokens):

| Type and K x N | native | Triton | speedup |
|---|---:|---:|---:|
| Q4_0 2560 x 1024 | 117.489 us | 117.908 us | 0.996x |
| Q4_0 2560 x 4096 | 465.058 us | 461.046 us | 1.009x |
| Q4_0 2560 x 9728 | 1103.584 us | 1088.480 us | 1.014x |
| Q4_0 4096 x 2560 | 464.491 us | 459.189 us | 1.012x |
| Q4_0 9728 x 2560 | 1098.204 us | 1079.032 us | 1.018x |
| Q4_1 9728 x 2560 | 2426.190 us | 1271.351 us | **1.908x** |

Q4_0 is deliberately compared with llama.cpp's ARM repack implementation,
not only with the slower canonical `ggml_vec_dot` path.  Q4_1 is where the
compiler-generated kernel produces most of the end-to-end gain.

## Compiler path

The Q4_0 source kernel is
[`benchmarks/bench_w4a8_codegen.py`](benchmarks/bench_w4a8_codegen.py).
The Q4_1 affine variant is
[`benchmarks/bench_w4a8_q41_codegen.py`](benchmarks/bench_w4a8_q41_codegen.py).
Their important dataflow is:

```text
ordinary Triton
  tl.load packed nibble bytes
  -> (& 15)/(>> 4), subtract 8
  -> two tl.dot operations
  -> TritonCPU cpu::DotOp pair
  -> compiler W4 pair fusion
  -> LLVM aarch64.neon.sdot intrinsics
  -> AOT .so
```

The Q4_0 static entry has 76 instructions, the range entry 92, and both have
eight SDOT instructions in the rolled inner body and zero calls.  The Q4_1
static entry has 84 instructions, eight SDOTs, and zero calls.

Two codegen changes were decisive:

1. Model-load packing stores each 4-output x 4-K nibble tile directly in
   SDOT order.  The Triton source still sees a logical `[K,N]` matrix, while
   LLVM eliminates the inverse source-level transpose.
2. The four activation bytes are scalar-bitcast before vector broadcast.
   This changes AArch64 selection from `USHLL + MOV + UZP1` sequences to one
   `LD1R {v?.4s}` per activation word.

For 1024 x 1024, the dot-ready kernel before the broadcast fix measured
108.93 us; after the compiler fix it measured 72.81 us.  The earlier
row-major/transposing kernel was about 100.8 us.

## Hardware affinity and ARM portability

This CIX process reports `SVE_CNT=16`, so SVE is 128 bits.  For M=1 GEMV,
fixed-width NEON SDOT is therefore the appropriate instruction; spelling the
same work with `z` registers provides no width increase.  The existing
M>=8 path remains available for SVE2/i8mm SMMLA prefill.

The W4 fusion now also has a `w4-only` compiler mode.  On non-SVE AArch64
targets with DotProd, including Apple M-series, it performs only fixed-width
SDOT fusion and cannot emit SVE instructions.  Forcing that path on CIX with
the full SVE2 lowering disabled passed correctness, emitted eight SDOTs and
no runtime call, and measured 75.72 us at 1024 x 1024.  Actual M4 performance
still needs to be measured on the M4 host.

## llama.cpp integration

The experimental checkout is
`third_party/llama.cpp-w4`.  Q4_0 weights are packed and cached from the ARM
CPU_REPACK model-load hook, then decode consumes ggml's canonical Q8_0
activation blocks.  Q4_1 keeps canonical bytes for prompt/multi-token
fallback, but its Triton cache is also constructed during model loading;
decode consumes canonical Q8_1 blocks and applies the affine correction:

```text
d_w * d_x * dot(q_w - 8, q_x)
  + (m_w + 8*d_w) * (d_x * sum(q_x))
```

The adapter is
[`integrations/llama.cpp/triton_w4_backend.cpp`](integrations/llama.cpp/triton_w4_backend.cpp).
Both Q4_0 and Q4_1 backends have scalar-reference tests.  The Q4_1 real-shape
test reports maximum absolute error `1.74e-5`.

The bundle now carries two AOT entry points for each shape.  A constant-bound
whole-projection kernel is selected for one worker, so LLVM retains the
single-thread optimum.  A runtime-range kernel lets persistent llama.cpp
workers split output tiles without Python, JIT, OpenMP, or a hand-written
GEMV body.  Activation unpack and cache resolution happen once per worker
and projection, not once per stolen range.

Strict affinity is required for interpretable heterogeneous-core data.  The
final guarded backend produced:

| Workers and mask | native | guarded backend | result |
|---|---:|---:|---:|
| 1, CPU 0 | 3423.81 ms | 3320.10 ms | -3.03% latency |
| 2, CPUs 0-1 | 1768.50 ms | 1745.04 ms | -1.33% latency |
| 4, CPUs 0,1,6,7 | 1143.27 ms | 1131.62 ms | -1.02% latency |
| 8, CPUs 0,1,6-11 | 1012.34 ms | 1009.87 ms | parity; native hot path |

These are five-repetition means except the final fallback confirmation,
which used three repetitions.  On this CIX part, CPU capacities range from
279 to 1024.  Without `--cpu-strict 1`, worker migration causes large
variance and can obscure both gains and regressions.

The unguarded range experiment exposed a real remaining limit.  On the four
best cores, forcing every Q4_0 range kernel regressed end-to-end latency by
11.3%; on the eight best cores it regressed by 15.0%.  FP16 group scales
reduced those regressions to 3.9% and 5.2% by shrinking weight traffic from
0.625 to 0.5625 bytes/weight, but the extra `FCVTL` in every group erased the
single-core gain.  The FP16 artifacts are retained as experimental evidence,
but are not the default bundle.

## Current costs and next work

The prototype intentionally keeps llama.cpp's native packed weights for
prompt/multi-thread fallback and adds a second compiler layout.  Peak RSS
therefore rises from 4.434 GiB to 6.619 GiB, an additional 2.185 GiB.
Model preparation also increases by roughly 0.48 seconds.  These are the
largest current limitations.

Decode remains restricted to batch 1, with the guarded worker policies
listed above.  `GGML_TRITON_W4_FORCE_RANGE_THREADS=1` enables the unguarded
range path for compiler experiments.  The next highest-value work is:

1. pipeline FP16 scale conversion with the next group's SDOT work, or use a
   wider group microtile, so reduced traffic does not add exposed latency;
2. reuse or replace the native repack allocation to remove the 2.185 GiB
   duplicate-layout cost;
3. fuse gate/up traversal, and then evaluate a compiler-generated fused MLP;
4. extend the same packed-dot matcher to Q4_K and other common GGUF W4
   formats;
5. benchmark the `w4-only` path on Apple M4 and tune for its issue width and
   cache hierarchy.
