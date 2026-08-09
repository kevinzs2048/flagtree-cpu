# CPU launch-overhead validation

These benchmarks use FlagOS `libtriton_jit`'s `BackendPolicy` and
`TritonKernelImpl` abstractions with the local CPU policy in
`third_party/libtriton_jit/include/triton_jit/backends/cpu_backend.h`.

The policy follows the Triton CPU ABI:

```text
kernel(runtime arguments..., pid_x, pid_y, pid_z, grid_x, grid_y, grid_z)
```

It performs `dlopen`/`dlsym` once and caches both the symbol and parsed call
layout. Common 3/4/6-pointer kernels and the decode-attention signature use
typed fast paths. Other scalar/pointer signatures use `libffi`.

Build:

```bash
./benchmarks/cpp_wrapper/build.sh
```

Measure an i8mm kernel against a direct function-pointer loop and the ACLE C
reference:

```bash
taskset -c 0 artifacts/bench_cpp_wrapper \
  /path/to/i8mm_kernel.so artifacts/libsve2_i8mm_c.neonpack.so \
  K GRID_X GRID_Y ITERS BLOCK_M BLOCK_N
```

For the row-major GEMM benchmark, `BLOCK_M=M`, `BLOCK_N=8`, and `BLOCK_K=K`
give the generated kernel the same packed-B reuse as the C reference.  For a
256x256x256 problem the wrapper arguments are therefore
`256 1 32 1500 256 8`.

Measure wrapper-only overhead with the generated `three_pointer_probe`:

```bash
taskset -c 0 artifacts/bench_wrapper_overhead \
  /path/to/three_pointer_probe.so 5000000
```

All timings exclude first-load/JIT work and report the median of nine batches.

## Generated Q4 pack and matrix

`bench_q4_generated_pipeline_aot` calls the ordinary-Triton Q4 LHS pack and
I8MM matrix symbols directly. Appending a second matrix object enables an
alternating-order compiler A/B while requiring bit-exact output:

```bash
taskset -c 11 artifacts/bench_q4_generated_pipeline_aot \
  /path/to/_pack_lhs_qsi8d32p_panel4_scalar_kernel.so \
  /path/to/candidate/_q4_prefill_i8mm_kai_kernel.so \
  12 1024 1024 500 21 300 \
  /path/to/reference/_q4_prefill_i8mm_kai_kernel.so
```

Both paths reuse the same KAI blobs and output buffers; no Python or runtime
compute implementation is included in the paired matrix timing.

## Fused W8 decode AOT kernel

`bench_fused_decode_aot` validates and times the compiler-generated packed
SDOT decode kernel through both a direct function pointer and FlagOS
`libtriton-jit`.  The kernel `.so` must be compiled for the same static
`K`, `N`, and `BLOCK_N` passed to the benchmark.

```bash
taskset -c 0 artifacts/bench_fused_decode_aot \
  /path/to/_tle_fused_bf16_gemv_kernel.so \
  python/triton/_C/libTritonCPURuntime.so \
  2048 1024 1024 1000 9
```

The benchmark builds both the legacy K-major SDOT pack and the Triton
N-blocked pack, verifies the generated BF16 result against a scalar
round-to-nearest reference, and reports direct AOT, `libtriton-jit`, and
legacy C timings using preallocated buffers.

For a same-process compiler A/B, pass the whole-kernel symbol and a second
generated object compiled in a fresh cache with
`TRITON_CPU_DISABLE_BF16_LANE_MAX=1`:

```bash
taskset -c 11 artifacts/bench_fused_decode_aot \
  /path/to/active/_tle_whole_bf16_gemv_kernel.so \
  python/triton/_C/libTritonCPURuntime.so \
  1024 1024 64 3000 21 _tle_whole_bf16_gemv_kernel \
  /path/to/reference/_tle_whole_bf16_gemv_kernel.so
```

The two direct symbols run in alternating order over identical inputs and
packed weights; complete BF16 outputs must match before timing.

## Fused W8 MLP AOT kernel

`bench_fused_mlp_aot` performs the same direct/wrapper/runtime comparison for
the compiler-generated gate + up + SwiGLU kernel:

```bash
taskset -c 0 artifacts/bench_fused_mlp_aot \
  /path/to/_fused_mlp_kernel.so \
  python/triton/_C/libTritonCPURuntime.so \
  1024 3072 512 1000 9
```

The generated kernel uses the same six-pointer Triton CPU ABI that an external
runtime (including llama.cpp) can call through `libtriton-jit`.

Appending a reference object compiled with
`TRITON_CPU_DISABLE_BF16_LANE_MAX=1` enables the same paired compiler A/B and
also checks a finite-BF16 edge-pattern input:

```bash
taskset -c 11 artifacts/bench_fused_mlp_aot \
  /path/to/active/_fused_mlp_kernel.so \
  python/triton/_C/libTritonCPURuntime.so \
  1024 3072 512 500 21 \
  /path/to/reference/_fused_mlp_kernel.so
```

## Decode attention AOT kernel

The attention benchmark accepts sequence length at runtime; head counts and
head dimension match Qwen3-0.6B's 16/8/128 configuration:

```bash
taskset -c 0 artifacts/bench_attention_aot \
  /path/to/_flash_attn_decode_codegen_kernel.so \
  python/triton/_C/libTritonCPURuntime.so 128 1000 9
```

## llama.cpp FP32 W8 decode ABI

`bench_f32_decode_aot` compares the generated FP32-input/FP32-output W8 kernel
with the shared-activation-quantization C path used by the ggml adapter:

```bash
taskset -c 0 artifacts/bench_f32_decode_aot \
  /path/to/_f32_w8_gemv_kernel.so \
  python/triton/_C/libTritonCPURuntime.so \
  1024 1024 512 2000 9
```
