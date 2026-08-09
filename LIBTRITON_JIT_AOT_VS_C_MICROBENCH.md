# libtriton_jit AOT versus C microbenchmark

Date: 2026-08-05

## Question and scope

This study separates three effects that are often mixed together when a CPU
Triton operator is described as slow:

1. the generated AOT compute function, called through a cached function
   pointer;
2. the low-level `libtriton_jit` CPU kernel wrapper around that same function;
3. the higher-level `TritonJITFunction` argument and kernel lookup APIs.

Compilation and first-load time are excluded from all steady-state numbers.
The ordinary-Triton kernels contain no `TLE_raw` compute call.  The C side is
an `-O3` Arm ACLE implementation with the same BF16 boundaries and operation
semantics.  The W8 comparison uses the official KleidiAI SDOT microkernel as a
stronger hand-tuned native reference.

If two inputs to the backend are literally the same optimized MLIR, use the
same target features, and carry the same alignment/alias facts, their source
language is no longer observable and there is no inherent Triton penalty.  In
practice, independently lowered Triton and C programs rarely produce the same
IR shape, so the useful question is whether each frontend exposes an equally
good loop, layout, reduction, and vectorization problem to LLVM.

## Measurement setup

- CIX Arm64 board, Cortex-A720 CPU 11, process pinned with `taskset -c 11`
- one thread (`OMP_NUM_THREADS=1`)
- BF16 elementwise kernels: shape-specialized production P0 AOT bundle
- W8: `K=1024`, `N=3072`, exact KAI physical layout
- 15 rotating-order batches for BF16 kernels; 31 paired batches for W8
- all implementations timed on the same data addresses

## Results

Times are microseconds.  A ratio below 1 means Triton is faster.

| Operator | Direct Triton AOT | ACLE C / KAI | Direct / native | `libtriton_jit` | JIT wrapper / native |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSNorm, BF16, N=1024 | 0.937 | 0.791 | 1.185x | 1.054 | 1.333x |
| vLLM fused add+RMSNorm, BF16, N=1024 | 1.003 | 1.229 | 0.816x | 1.120 | 0.912x |
| RoPE Q16/KV8/D128, BF16 | 0.925 | 0.933 | 0.991x | 1.052 | 1.127x |
| SwiGLU, BF16, N=3072 | 10.304 | 10.755 | 0.958x | 10.437 | 0.970x |
| W8 KAI-layout GEMV, K1024/N3072 | 79.439 | 79.599 | 0.998x | about 79.3 | about 0.996x |

The four BF16 comparisons have zero output mismatches.  The W8 result passes
the same numerical contract as KAI with maximum absolute error
`2.3841858e-07`.  Its generated hot function has 32 static `SDOT`
instructions, no call, and no stack access.

Repeated paired W8 wrapper/direct runs put the signed difference between
`-0.085` and `+0.030` microseconds.  This is measurement noise, not a wrapper
speedup.  At a 79 microsecond compute duration the wrapper cost is below the
resolution of this test.

## Launch cost

An empty three-pointer CPU-ABI probe gives:

| Call boundary | Time |
| --- | ---: |
| Direct function pointer | 1.68 ns |
| low-level `TritonKernelImpl<CpuBackend>` | 58.71 ns |
| wrapper minus direct | 57.03 ns |

The current low-level wrapper caches `dlopen`/`dlsym`, uses typed fast paths for
common pointer-only signatures, and then invokes the same AOT symbol.  Its
fixed cost matters for a sub-microsecond kernel but is negligible for W8 GEMV.

The higher-level C++ API is more expensive.  On the two-program tiny add probe,
after compilation and loading:

| `libtriton_jit` API | Time |
| --- | ---: |
| `launch_with_raw_args` | 472 ns |
| generic tensor/argument API | 1359 ns |

These numbers include two CPU program calls.  They show that the chosen C++
entry point matters independently of generated compute quality.

## Conclusions

There is no general, inherent microbenchmark gap between Triton AOT and C.
With a hardware-appropriate layout and lowering, the generated function in
this sample ranges from 18% slower to 18% faster than equivalent ACLE C, and a
compute-heavy W8 kernel matches KleidiAI within measurement noise.  The result
is operator-specific rather than language-specific.

The impression that CPU Triton is generally slow is valid for unoptimized
paths, but it usually comes from one of these concrete causes:

- generic `tl.dot` or a layout that does not map to SDOT/SMMLA;
- large fully expanded vector SSA values and register spills;
- dynamic shapes preventing loop and tail specialization;
- weaker alignment/alias information than the C compiler sees;
- a different floating-point association or output contract;
- Python, custom-op, or high-level JIT dispatch around a very small kernel.

For acceptance testing, direct AOT versus native C should be the codegen gate,
and the selected `libtriton_jit` entry point should be a separate integration
gate.  Tiny elementwise kernels should be fused or called through a cached
low-level launcher; matrix kernels above roughly 10 microseconds normally
amortize that launcher cost.

## Reproduction

The BF16 AOT/native benchmark is built by:

```bash
./benchmarks/cpp_wrapper/build.sh
```

Its source is `benchmarks/cpp_wrapper/bench_norm_rope_aot.cpp`.  The W8/KAI
comparison, including the low-level `libtriton_jit` path, is in
`benchmarks/cpp_wrapper/bench_w8_kleidiai_layout.cpp`.  The standalone launch
probe is `benchmarks/cpp_wrapper/bench_wrapper_overhead.cpp`.
