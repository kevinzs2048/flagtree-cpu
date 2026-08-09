# ARM64 high-quality Triton CPU codegen report

Date: 2026-07-31

Platform: CIX P1/CD8180, AArch64, SVE2 + i8mm, SVE VL=128. Unless noted,
measurements use one pinned performance core (`taskset -c 0`) and one OpenMP
thread. Values are medians after warmup.

## Result

The hot W8 decode path no longer consists of Triton wrappers tail-calling
whole hand-written GEMV or MLP implementations. The current path is:

```text
Triton frontend
  -> TritonCPU typed op or ordinary Triton IR
  -> rolled SCF/vector loops
  -> LLVM AArch64 SDOT / vector FMA / exp
  -> generated kernel .so
```

The generated `.so` computation reaches hand-written C parity or exceeds it.
Python launch overhead remains material for very short CPU kernels, while the
C++ `libtriton-jit` wrapper reduces this overhead to 0.1-1.7 microseconds.

## What was changed

### Packed W8 decode GEMV

`cpu::SdotGemvFusedBf16Op` now expands into compiler-visible activation
quantization, block-major packed-weight loads, a rolled K loop, NEON SDOT
intrinsics, dequantization, and BF16 stores. FP32 input/output is selected from
the pointer types for ggml/llama.cpp.

The default generated LLIR has no `sdot_gemv_m1_fused_bf16` or
`sdot_gemv_blk_prequant_f32_range` call.

For the llama/ggml FP32 contract, a second two-stage path quantizes the
activation once in an ordinary Triton kernel and feeds
`cpu::SdotGemvPrequantOp`. That op lowers to the same compiler-generated
packed SDOT loop and lets all output blocks reuse the shared INT8 activation.
Its round-to-nearest-even operation uses the standard CPU `libdevice.rint`
path. This becomes MLIR `math.roundeven`, then LLVM's round-even intrinsic;
AArch64 selects `FRINTN`. The standalone quantizer `.so` has no SLEEF or
runtime dependency.

### Fused W8 MLP

`cpu::FusedMlpOp` now generates both gate and up SDOT accumulation banks,
dequantization, observable BF16 boundaries, SiLU, multiply, and BF16 output.
The generated LLIR has no `fused_mlp_bf16` call.

### Decode attention

The new M=1 attention kernel is expressed with ordinary Triton loads,
reductions, a runtime sequence loop, online softmax, and stores. It does not
use a TLE/raw attention leaf. `BLOCK_N=1` is intentional: larger KV blocks
materialized a `BLOCK_N x HEAD_DIM` tensor, produced a 9 KB stack frame and
thousands of spills. The rolled single-token loop cuts the static assembly
from 289 to 81 FMA instructions and reaches C performance.

### Ordinary `tl.dot`

Short prefill keeps a normal `tl.dot` path lowered to SVE2 i8mm `smmla`.
For M=1 row-major weights, instruction selection alone was not sufficient:
K=2048, N=6144 took about 1847 us versus 386-425 us for packed C. Online
4x4 transposition dominated. Decode therefore uses a compiler-visible packed
layout rather than claiming that any `tl.dot` lowering is automatically fast.

## Operator performance

### Direct AOT and C++ wrapper

| Operator and shape | Direct generated `.so` | C++ operator path | Hand-written C | C++/C |
|---|---:|---:|---:|---:|
| BF16 W8 GEMV, K=2048 N=1024 | 64.201 us | 64.424 us | 64.038 us | 1.006x |
| Fused W8 SwiGLU, K=1024 N=3072 | 192.927 us | 193.714 us | 214.862 us | **0.902x** |
| FP32 W8 GEMV, K=1024 N=1024 | 32.246 us | 32.454 us | 40.398 us | **0.803x** |
| FP32 W8 split, K=1024 N=3072 | 88.695 us | 90.479 us | 111.616 us | **0.811x** |
| Attention N=7, Hq/Hkv=16/8, D=128 | 8.378 us | 8.518 us | 9.237 us | **0.922x** |
| Attention N=128 | 135.130 us | 135.708 us | 144.194 us | **0.941x** |
| Attention N=512 | 531.760 us | 532.218 us | 567.394 us | **0.938x** |
| Attention N=2048 | 2137.900 us | 2142.010 us | 2265.110 us | **0.946x** |

The FP32 W8 rows are the llama.cpp-compatible contract. A separately built
operator backend `.so`, including its public C ABI, measures 32.707 us versus
40.666 us for the existing ggml/FlagGems C path, or **19.6% lower latency**.

### Python launcher observations

Python results are useful for end-to-end FlagGems behavior but must not be
used to judge the generated computation body in isolation:

- BF16 W8 GEMV K=2048, N=1024: about 87.5 us through Python versus about
  64.8 us through the C++ wrapper.
- Fused MLP K=1024, N=3072: 245.5 us through Python versus 193.0 us through
  the C++ wrapper.
- Model-size fused MLP K=2048, N=6144 still beats the old whole C runtime
  through Python: 1190.4 us versus 1309.1 us.

## Qwen3-0.6B end-to-end

Configuration: 6-token prompt, greedy decode 16 tokens, one pinned core.
The table uses three-run medians from the final compiler build.

| Mode | 16-token time | Throughput | Relative to W8 baseline | Token result |
|---|---:|---:|---:|---|
| W8 Linears, no model fusions | 1.4921 s | 10.7228 tok/s | baseline | reference |
| Exact codegen set | 1.4019 s | 11.4134 tok/s | **+6.44%** | identical |
| Exact set + codegen attention | 1.2660 s | 12.6385 tok/s | **+17.86%** | identical in this sample |

The exact set contains compiler-generated W8 Linears and fused MLP, normal
Triton RoPE/RMSNorm/fused-add-RMSNorm, short-prefill SVE2 i8mm, and Triton
vocabulary argmax.

Codegen attention uses a different online-softmax reduction order. Its
single-operator relative L2 error versus ATen is about 0.24%, so it remains an
explicit opt-in even though the tested 16-token sequence did not diverge.

## Triton coverage

Coverage is calculated from one synchronous decode step, including vocabulary
argmax. Optional `FLAGGEMS_PROFILE_RANGES=1` ranges account for generated CPU
shared-library execution, which torch.profiler otherwise omits. The metric is:

```text
sum(self CPU time of triton::* ranges) / total profiled self CPU time
```

| Configuration | Profile total | Triton time | Coverage |
|---|---:|---:|---:|
| W8 Linear baseline | 82.499 ms | 53.262 ms | 64.56% |
| Exact codegen set | 75.294 ms | 61.761 ms | 82.03% |
| Exact set + codegen attention | 72.366 ms | 63.696 ms | **88.02%** |

Detailed final coverage:

| Generated range | Calls/decode | Self time | Total share |
|---|---:|---:|---:|
| W8 decode SDOT | 141 | 36.216 ms | 50.04% |
| Fused W8 MLP | 28 | 12.870 ms | 17.78% |
| RMSNorm | 85 | 6.992 ms | 9.66% |
| Decode attention | 28 | 2.494 ms | 3.45% |
| Fused add + RMSNorm | 28 | 2.317 ms | 3.20% |
| RoPE Q+K | 28 | 2.262 ms | 3.13% |
| Vocabulary argmax | 1 | 0.546 ms | 0.75% |

The largest remaining cost is no longer a compute operator; it is aggregate
view/reshape/slice/allocation and Python dispatch overhead.

## Weight memory

The external-runtime K-major weight pack is no longer retained. Each W8 Linear
keeps only:

1. row-major `[K,N]` for prefill;
2. block-major packed W8 for compiler-generated decode.

Qwen3-0.6B RSS fell from about 4328 MiB to 3840 MiB, a 488 MiB reduction.
Live-quantization setup fell from about 5.86 s to 5.28 s.

## llama.cpp operator backend prototype

`integrations/llama.cpp` builds:

```text
artifacts/llama-triton-backend/libtriton_w8_backend.so
```

The C ABI supports:

- loading a statically specialized AOT W8 kernel;
- packing ggml `[N,K]` INT8 weights into the generated block layout;
- full-grid FP32 GEMV launch through libtriton-jit;
- direct output-block range launch for ggml's heterogeneous work-stealing
  threadpool;
- a two-stage ordinary-Triton quantizer plus compiler-generated prequantized
  SDOT variant with shared activation scratch.

The standalone test validates both launch modes bit-exactly against the current
ggml/FlagGems C implementation. The local llama.cpp tree is not modified:
its repository policy requires a human contributor to own and explain an
upstream integration. `flaggems_ggml_hook.inc` documents the reviewable call
site for the existing extra-buffer adapter.

The Qwen3-0.6B bundle allowlists five profitable projection shapes and carries
both fused and split AOT variants. Single-core split AOT speedups over C are
17.7% for 1024x1024, 15.1% for 1024x2048, 13.5% for 2048x1024, 17.5% for
1024x3072, and 15.8% for 3072x1024. A BLOCK_N sweep for 1024x3072 measured
99.0/95.2/89.3/93.2/92.9 us for 128/256/512/768/1024; 512 remains selected.

Persistent-threadpool measurements validate the range ABI rather than only a
serial microbenchmark:

| Shape and workers | Triton variant | Triton | ggml C | Triton/C |
|---|---|---:|---:|---:|
| 1024x1024, 2 | fused | 48.52 us | 55.35 us | 0.877x |
| 1024x2048, 2 | fused | 53.73 us | 61.84 us | 0.869x |
| 2048x1024, 2 | fused | 52.41 us | 61.26 us | 0.856x |
| 3072x1024, 2 | fused | 69.84 us | 79.29 us | 0.881x |
| 1024x3072, 4 | fused | 53.33 us | 63.00 us | 0.847x |
| 1024x3072, 4 | split | 58.62 us | 63.37 us | 0.925x |
| padded vocab 1024x152064, 4 | fused | 3612 us | 3926 us | 0.920x |

Fused is usually the better multi-worker choice because independent workers
parallelize their per-block activation quantization; split is strongest on one
core. The 1024x2048 fused path regresses to 1.082x with four workers, so the
adapter needs a measured `(K,N,threads)` allowlist. The logical 151936-column
vocabulary output additionally needs padded-output ownership before it is
enabled by default. These are deliberate fallbacks rather than inflated
Triton coverage.

## Reproduction

Correctness:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm taskset -c 0 \
  /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/validate_w8a8.py
```

Structured coverage:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  FLAGGEMS_VENDOR_NAME=arm FLAGGEMS_PROFILE_RANGES=1 \
  FLAGGEMS_ARM_ATTN_DECODE_IMPL=codegen taskset -c 0 \
  /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-attn-argmax --threads 1 \
  --warmup-tokens 2 --new-tokens 2 --repeats 1 \
  --prefill-repeats 1 --profile decode
```

llama.cpp-compatible operator library:

```bash
bash integrations/llama.cpp/build.sh
kernel_so=$(find artifacts/aot-f32-w8-k1024-n1024-bn512 \
  -name _f32_w8_gemv_kernel.so | head -1)
taskset -c 0 artifacts/llama-triton-backend/test_triton_w8_backend \
  "$(dirname "${kernel_so}")" \
  python/triton/_C/libTritonCPURuntime.so

taskset -c 0,1,10,11 \
  artifacts/llama-triton-backend/bench_triton_w8_split_threadpool \
  artifacts/llama-triton-backend/qwen3-0.6b \
  python/triton/_C/libTritonCPURuntime.so \
  1024 3072 512 4 1000 7
```

## Boundaries

- This work targets W8A8/W8 dynamic activation quantization, not BF16 model
  weights and not W4.
- Attention is opt-in because it is not bit-exact to ATen.
- AOT GEMV currently specializes K, N, and BLOCK_N. A deployment bundle needs
  one generated kernel for each unique projection shape.
- The operator backend has a tested ggml-compatible ABI and thread-range call,
  but has not been inserted into or benchmarked inside the llama.cpp binary.
- The persistent-thread harness mirrors ggml work stealing and validates
  multi-core scheduling, but full llama token throughput still needs a
  human-owned patch in the policy-restricted llama.cpp worktree.
