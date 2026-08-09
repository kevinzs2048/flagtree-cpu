# SVE2 i8mm lowering

## Scope

This tree contains a descriptor-driven, rolled SVE2 i8mm lowering for
row-major `i8 x i8 -> i32` `tl.dot` on AArch64.  It replaces the previous
whole-tile SSA expansion, which selected `smmla` but generated a very large,
spill-heavy function.

The current production gate is deliberately narrow:

- AArch64 with both `sve2` and `i8mm`;
- process SVE vector length exactly 128 bits;
- direct, unmasked 2-D dot loads with a proven unit minor stride;
- M, N and K tile dimensions are positive multiples of 8.

Anything outside that set remains on the existing generic lowering.  The
128-bit vector-length check is required for correctness: this first slice
uses one 128-bit SMMLA segment per scalable vector and must not silently use
only the low segment on a wider-SVE system.

## Lowering shape

`ConvertMemoryOps` now keeps suitable dot operands as one
`memref.reinterpret_cast` descriptor with layout `strided<[?, 1]>`, instead
of splitting a tile into an independent memref and vector load for every row.
Scalar loop induction variables are forwarded while the surrounding tensor
pointer expression is scalarized, so this also works inside Triton's outer K
loop.

`ConvertDotToSVE2I8MM` emits a conventional 8x8x8 microkernel:

1. walk N in panels of 8;
2. pack each Kx8 B panel with an explicit three-level NEON TRN network into
   four 2x8 column-pair vectors;
3. walk M in blocks of 8;
4. keep sixteen 2x2 i32 accumulator vectors live;
5. walk K in steps of 8 and issue sixteen rolled SVE `smmla` operations;
6. write the 8x8 result block.

For a loop-carried Triton accumulator, the temporary C buffer is initialized
before the outer Triton K loop and read after it.  This removes the original
large vector loop-carried SSA value while preserving accumulation semantics.
For a non-loop-carried dot whose accumulator is a constant zero, the
microkernel now starts its sixteen SMMLA accumulators from
`llvm.mlir.zero`; it does not clear the temporary C buffer and load those
zeros back.

## Measured result

Host: Cortex-A720 performance core, Linux AArch64, SVE VL=128, one pinned
thread (`taskset -c 0`, `OMP_NUM_THREADS=1`).  Values below are medians of
repeated runs.

The most direct code-generation comparison uses one 32x32 Triton program and
calls the generated symbol and optimized ACLE C function from native C++.
Both receive row-major A/B and perform their own B packing:

| Block | Generated function | `libtriton_jit` CPU wrapper | Optimized ACLE C |
|---:|---:|---:|---:|
| 32x32x128 | 1.686 us | 1.737 us | 1.857 us |
| 32x32x256 | 3.049 us | 3.095 us | 3.277 us |
| 32x32x512 | 5.782 us | 5.824 us | 6.107 us |

The native generated function is 5-9% faster than the optimized C
microkernel in this measurement.  The C++ wrapper adds 41-50 ns per complete
one-program call.  The B transpose selects 16 static `trn1`/`trn2` sites and
no scalar byte-lane moves.

The wrapper is implemented as a CPU `BackendPolicy` for FlagOS
`libtriton_jit`'s `TritonKernelImpl`.  It loads the generated shared object
once, caches the symbol and call layout, and invokes the CPU ABI directly.
The upstream repository did not have a CPU backend at the tested commit
`d66d2fa`, so the policy is carried locally under
`third_party/libtriton_jit/include/triton_jit/backends/cpu_backend.h`.

An intentionally tiny generated load/add/store probe isolates dispatch from
kernel work:

| Probe path | Time |
|---|---:|
| Direct generated function pointer | 1.62 ns |
| C++ `TritonKernelImpl<CpuBackend>` | 48.59 ns |
| Full Python Triton launch | 13.20 us |

Thus the wrapper itself adds about 47 ns and removes more than 99% of the
Python launch cost.  Full Python timing for the i8mm blocks shows the same
roughly 22 us fixed gap:

| Block | Full Python launch | C++ wrapper | Optimized ACLE C |
|---:|---:|---:|---:|
| 32x32x128 | 23.498 us | 1.737 us | 1.857 us |
| 32x32x256 | 24.863 us | 3.095 us | 3.277 us |
| 32x32x512 | 28.011 us | 5.824 us | 6.107 us |

Using the original fixed 32x32 program shape exposes a second, independent
cost:

| M=N=K | Grid | Full Python | C++ wrapper | Optimized ACLE C |
|---:|---:|---:|---:|---:|
| 128 | 4x4 | 52.717 us / 79.56 GOPS | 31.041 us / 135.12 GOPS | 25.299 us / 165.79 GOPS |
| 256 | 8x8 | 354.846 us / 94.56 GOPS | 312.506 us / 107.37 GOPS | 191.484 us / 175.23 GOPS |

That 256-size gap is not launch overhead.  Independent 32x32
Triton programs repack and reread a B panel for every M program, while the C
function packs each B panel once and reuses it over all M blocks.

The existing lowering can express the same reuse without changing the ABI:
use `BLOCK_M=M`, `BLOCK_N=8`, and `BLOCK_K=K`.  One program then owns a
complete 8-column B panel and walks all M blocks after packing it once:

| M=N=K | Triton tile / grid | Full Python | C++ wrapper | Optimized ACLE C | Wrapper / C |
|---:|---:|---:|---:|---:|---:|
| 128 | 128x8 / 1x16 | 50.404 us / 83.21 GOPS | 24.759 us / 169.41 GOPS | 25.468 us / 164.69 GOPS | 0.972x |
| 256 | 256x8 / 1x32 | 240.740 us / 139.38 GOPS | 191.401 us / 175.31 GOPS | 190.066 us / 176.54 GOPS | 1.007x |

The 128 case is 2.8% faster than C and the 256 case is 0.7% slower.  The
remaining difference is run-to-run noise; native wrapper performance is now
at optimized-C parity.  Python still hides that result behind dispatch and
its generic grid launcher.

The earlier end-to-end comparison, retained for the lowering speedup, was:

| M=N=K | Previous SVE2, BK=16 | New, BK=16 | New, BK=K | Optimized ACLE C |
|---:|---:|---:|---:|---:|
| 128 | 42.96 GOPS | 64.90 GOPS | 78.75 GOPS | 162.03 GOPS |
| 256 | 53.87 GOPS | 90.59 GOPS | 94.20 GOPS | 177.61 GOPS |

At the original BK=16 configuration, the new lowering is 1.51x and 1.68x
faster than the archived SVE2 pass.  The C++ launcher plus a B-reuse-aware
tile exposes the optimized-C performance of the generated microkernel.

Assembly audit for the full-K kernel:

| | Previous pass | New pass |
|---|---:|---:|
| assembly lines | 4069 | 1353 |
| static `smmla` sites | 128 (expanded) | 16 (rolled) |
| all folded spills | 722 | 19 |
| SVE Z-register spills / reloads | 16 / 16 | 0 / 0 |

The remaining folded spills in the new function are GPR/callee-save traffic;
there is no Z-register spill/reload in the SMMLA K loop.

## Reproduce

The Python benchmark verifies every result against an int32 PyTorch matmul:

```bash
PYTHONPATH=/home/kevin/triton-opt-cpu/python \
OMP_NUM_THREADS=1 taskset -c 0 \
/home/kevin/venv-int8-clean/bin/python \
/home/kevin/triton-opt-cpu/benchmarks/bench_i8mm.py \
  --m 256 --n 256 --k 256 --bm 256 --bn 8 --bk 256
```

Build and run the independent ACLE C reference:

```bash
gcc -O3 -march=armv9-a+sve2+i8mm -msve-vector-bits=128 \
  /home/kevin/triton-opt-cpu/benchmarks/sve2_i8mm_c.c \
  -o /home/kevin/triton-opt-cpu/artifacts/sve2_i8mm_c
taskset -c 0 /home/kevin/triton-opt-cpu/artifacts/sve2_i8mm_c 256 256 256 300
```

Build the C++ wrapper benchmarks:

```bash
/home/kevin/triton-opt-cpu/benchmarks/cpp_wrapper/build.sh

taskset -c 0 /home/kevin/triton-opt-cpu/artifacts/bench_cpp_wrapper \
  /path/to/i8mm_kernel.so \
  /home/kevin/triton-opt-cpu/artifacts/libsve2_i8mm_c.neonpack.so \
  256 1 32 1500 256 8

taskset -c 0 /home/kevin/triton-opt-cpu/artifacts/bench_wrapper_overhead \
  /path/to/three_pointer_probe.so 5000000
```

Set `TRITON_CPU_DISABLE_SVE2_I8MM=1` to force and test the generic fallback.

## Validation

- the CPU extension and runtime libraries build successfully;
- direct pass regression: four rolled loops, sixteen SMMLA sites, no residual
  `triton_cpu.dot`, and no zero-buffer initialization for a direct zero-acc
  dot;
- randomized end-to-end correctness covered 8, 16 and 32 M/N blocks and
  8 through 128 K blocks;
- the disabled generic path remains correct;
- `git diff --check` passes.

The standalone CPU-only `triton-opt` target is now fixed as well.  Dialect and
test-pass registration follow the enabled backends, so a CPU-only build no
longer requires AMD/NVIDIA generated headers or libraries.  The direct MLIR
FileCheck suite, Python `libtriton` pass manager, fresh-cache JIT compilation
and native direct-call benchmarks all exercise the same current pass.
