# Triton W4/W8 operator backends for llama.cpp

This directory contains external, optional operator libraries.  The
experimental llama.cpp checkout used for end-to-end validation is
`third_party/llama.cpp-w4`.

It now contains two independent backends:

- the original W8/FP32 decode prototype described below;
- packed Q4_0 x Q8_0 and Q4_1 x Q8_1 backends generated from ordinary
  Triton operations.  Their KAI-style physical-layout kernels use `tl.load`,
  nibble arithmetic, `tl.sum`, and `tl.cat`; the earlier Q4_1 `tl.dot` kernel
  remains available as a fallback.

Build the Qwen3-4B W4 AOT bundle and the adapters with:

```bash
bash integrations/llama.cpp/build_qwen3_4b_w4_bundle.sh
bash integrations/llama.cpp/build.sh
```

For Q4_0, the loader converts canonical ggml blocks into KAI's final
`qsi4c32p4x8` physical layout during model load: 72 bytes per K32 group and
four output channels.  Canonical ggml Q8_0 activation blocks already match the
required scale-plus-32-byte data layout and are consumed directly at decode.
The generated projection contains eight LLVM SDOT intrinsics, one ADDP, and
fixed-point `SCVTF #4` per K32 group, with no W4 GEMV runtime call.  Q4_1
extends the same N4/K32 microtile with four FP16 minimum values; this is a
project-specific KAI-style layout, not an official KleidiAI Q4_1 format.

Run the end-to-end Qwen3-4B experiment with:

```bash
GGML_TRITON_W4_BUNDLE="$PWD/artifacts/llama-triton-backend/qwen3-4b-w4" \
  taskset -c 0 \
  third_party/llama.cpp-w4/build-triton-w4/bin/llama-bench \
  -m /home/cix/models/Qwen3-4B/Qwen3-4B-Q4_0.gguf \
  -p 0 -n 16 -r 5 -t 1 -C 0x1 --cpu-strict 1
```

Set `GGML_TRITON_W4_LEGACY_LAYOUT=1` for an A/B run with the previous Q4_0
packing and generated kernel.  A bundle that does not contain the new
`_kai_w4_layout_split_kernel.so` symbol also falls back automatically.

`GGML_TRITON_W4_N=0` is a low-memory diagnostic mode that disables Q4_0
shape caches but retains the four Q4_1 down projections.  The full prototype
preserves native packed weights for fallback.  The new 72-byte layout reduces
the Q4_0 Triton weight cache by 10% relative to the previous 80-byte layout;
peak RSS in the Qwen3-4B test fell by about 211 MiB.  Detailed integration
results are in
[`W4_KAI_LLAMA_E2E_RESULTS.md`](../../W4_KAI_LLAMA_E2E_RESULTS.md).

Each shape contains a static whole-projection AOT entry and a runtime-range
entry.  The guarded default uses all Q4_0/Q4_1 generated projections with up
to eight workers and native kernels above that tested range.  Set
`GGML_TRITON_W4_FORCE_RANGE_THREADS=1` only when profiling the unguarded
compiler range path.

The current prototype targets ggml's decode `MUL_MAT` contract:

- input activation: FP32 `[K]`
- pre-quantized per-output-channel W8 weight: INT8 `[N,K]`
- output: FP32 `[N]`
- AOT specialization: static `K`, `N`, and output block size

`libtriton_w8_backend.so` exposes two bit-exact AOT variants:

- fused: each output block performs activation quantization and packed SDOT;
- split: one ordinary Triton quantization kernel produces shared INT8 scratch,
  then a compiler-lowered prequantized SDOT kernel consumes it.

Both expose full-grid and output-block range launches. The range forms let
ggml's heterogeneous threadpool distribute work without OpenMP. Neither GEMV
variant calls a hand-written GEMV runtime.

Build and validate:

```bash
bash integrations/llama.cpp/build.sh

kernel_so=$(find artifacts/aot-f32-w8-k1024-n1024-bn512 \
  -name _f32_w8_gemv_kernel.so | head -1)

taskset -c 0 artifacts/llama-triton-backend/test_triton_w8_backend \
  "$(dirname "${kernel_so}")" \
  python/triton/_C/libTritonCPURuntime.so
```

The test covers W8 packing, shared-library loading, full-grid dispatch,
threadpool-style block range dispatch, and bit-exact comparison with the
existing ggml/FlagGems FP32 SDOT path.

Generate a deterministic Qwen3-0.6B bundle containing both variants:

```bash
bash integrations/llama.cpp/build_qwen3_0_6b_bundle.sh
```

Single-core direct AOT results for the split variant:

| K x N | Triton | ggml C | Triton/C |
|---|---:|---:|---:|
| 1024 x 1024 | 33.01 us | 40.11 us | 0.823x |
| 1024 x 2048 | 62.05 us | 73.12 us | 0.849x |
| 2048 x 1024 | 63.84 us | 73.79 us | 0.865x |
| 1024 x 3072 | 88.69 us | 107.47 us | 0.825x |
| 3072 x 1024 | 92.97 us | 110.39 us | 0.842x |

For 1024 x 3072, the public split operator ABI measures 90.48 us versus
111.62 us for the C path (0.811x); the direct generated pair is 88.69 us.

Persistent-thread results show why dispatch must include thread count. For
1024 x 3072 on four pinned cores, fused Triton is 53.33 us, split Triton is
58.62 us, and ggml C is 63.00-63.37 us. For 1024 x 2048, two-core fused and
split are both about 13% faster than C, while four-core fused regresses 8.2%
because the grid is too small for that scheduling overhead.

The logical 151936-column vocabulary projection is padded to 152064 for the
tested kernels. Fused AOT is slower than C on one core, but a four-core
persistent-thread microbenchmark is 3612 us versus 3926 us for C. It is not in
the default bundle until the ggml adapter owns padded-output handling and a
thread-count-aware allowlist.

The ordinary Triton quantizer implements round-to-nearest-even with arithmetic
and integer conversion, so its generated `.so` has no SLEEF dependency.

`flaggems_ggml_hook.inc` shows the small call site needed in the existing
FlagGems ggml extra-buffer adapter. It is deliberately not an automatically
applied upstream patch: llama.cpp's repository policy requires a human
contributor to own and explain the integration.
