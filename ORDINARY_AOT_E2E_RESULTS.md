# Ordinary Triton AOT decode results on Arm

This report covers the Qwen3-0.6B W8 decode path built from the isolated
Triton-CPU 3.7.2 port in `ports/triton-cpu-3.7.2`. Measurements were collected
on the CIX AArch64 machine with one thread pinned to CPU 0. The generated
kernels use ordinary Triton operations and are loaded from their generated
shared objects through one small C++ dispatcher. The dispatcher only owns
scratch buffers and invokes exported kernel symbols; it contains no compute
implementation.

## Implemented routes

- BF16 activation quantization uses round-to-nearest-even and matches the
  existing model-visible TLE semantics.
- W8 GEMV is an ordinary `tl.dot` kernel lowered to packed Arm SDOT. A
  64-output tile is used for decoder projections; the bandwidth-bound
  vocabulary projection uses 32 outputs to avoid excess live accumulators.
- Q/K/V weights are concatenated so one quantization and one ordinary GEMV
  replace three projections.
- Gate/up use a register-light three-stage path: quantize once, one joined
  ordinary GEMV, then a separate ordinary Triton SwiGLU kernel. Keeping `exp`
  out of the dot kernel eliminates accumulator spills. The separate activation
  expresses the SLEEF-u10 polynomial as Triton arithmetic, so LLVM emits no
  external `exp` call.
- RMSNorm, fused residual-add RMSNorm, and RoPE are shape-specialized ordinary
  Triton kernels invoked directly from their generated shared objects.
- Every route is optional. For W8 decode, joined QKV, and gate/up SwiGLU, a
  missing bundle now falls back to the same ordinary Triton quantizer, SDOT,
  and activation kernels through JIT. Unsupported shapes retain the normal
  FlagGems composition.

No path above calls TLE_raw or a hand-written C compute function.

## Isolated results

Lower is better. All rows are median single-thread latency and bit-exact
against the comparison path.

| Operation and shape | Ordinary AOT | Previous path | Change |
| --- | ---: | ---: | ---: |
| FlagGems W8 Linear, 1024 -> 1024 | 49.23 us | 85.68 us whole-TLE | -42.5% |
| FlagGems W8 Linear, 1024 -> 2048 | 74.96 us | 110.17 us whole-TLE | -32.0% |
| FlagGems W8 Linear, 1024 -> 3072 | 103.27 us | 134.70 us whole-TLE | -23.3% |
| FlagGems W8 Linear, 2048 -> 1024 | 74.51 us | 108.72 us whole-TLE | -31.5% |
| FlagGems W8 Linear, 3072 -> 1024 | 108.07 us | 137.32 us whole-TLE | -21.3% |
| Fused QKV, 1024 -> 2048+1024+1024 | 144.80 us | 167.18 us whole-TLE | -13.4% |
| Gate/up SwiGLU, 1024 -> 3072 | 170.25 us | 204.99 us specialized TLE | -16.9% |
| SwiGLU activation only, N=3072 | 16.39 us inline codegen | 20.01 us external exp | -18.1% |
| Complete W8 gate/up+SwiGLU, 1024 -> 3072 | 168.11 us inline codegen | 171.86 us external exp | -2.2% |
| RMSNorm, M=1 N=1024 | 7.84 us | 34.61 us JIT wrapper | -77.3% |
| RMSNorm, M=16 N=128 | 8.41 us | 36.82 us JIT wrapper | -77.2% |
| RMSNorm, M=8 N=128 | 8.07 us | 34.46 us JIT wrapper | -76.6% |
| Fused add + RMSNorm, M=1 N=1024 | 3.59 us | 20.16 us JIT wrapper | -82.2% |
| RoPE Q16/KV8, D=128 | 4.00 us | 20.56 us JIT wrapper | -80.5% |

The direct RNE quantization plus 64-output W8 GEMV for K=1024,N=3072 takes
80.16 us, versus 80.61 us for a matching fused ACLE C implementation. The
generated GEMV has 32 static SDOT instructions, no int32 accumulator alloca,
no stack load/store, and no external compute call.

The two new SwiGLU rows are alternating-order, same-process measurements on
CPU 11. Both paths use the same quantizer and joined GEMV. The inline
activation is BF16 bit-exact, replaces four external calls with ordinary LLVM
arithmetic, and has eight stack references solely for callee-saved vector
register save/restore.

## End-to-end result

The benchmark generated 32 tokens with five measured repeats. Prefill is
included in each `generate` call but is not routed through the decode AOT
kernels.

| Qwen3-0.6B W8 configuration | Median generation | Throughput |
| --- | ---: | ---: |
| Previous TLE decode, no QKV fusion | 2.61996 s | 12.21 token/s |
| Ordinary AOT W8 + QKV + MLP + norms + RoPE | 2.15521 s | 14.85 token/s |

The final route is 21.6% faster. All 32 generated token IDs are identical;
prefill remains about 251-253 ms in both runs. A warmed decode profile records
28 fused-QKV AOT calls, 28 MLP AOT calls, 57 remaining W8 AOT calls, 85
RMSNorm AOT calls, 28 fused residual-norm AOT calls, and 28 RoPE AOT calls.
Triton-marked ranges account for 79.0% of profiled self CPU time after these
ranges have themselves been shortened.

After selecting the inline activation, a fresh four-token model check produced
the same `[264, 8544, 429, 702]` token sequence. This short run is a correctness
check, not a replacement for the 32-token throughput measurement above.

The vocabulary projection is bandwidth-bound. A same-process persistent-worker
microbenchmark partitions only the generated BN32 GEMV ranges: one Cortex-A720
takes 7198.62 us and two take 4043.79 us (1.78x). Four and six cores reach only
3872.29 and 3860.11 us, confirming that most available memory bandwidth is
already consumed by two cores. The optional two-worker C++ AOT path measures
7104.88 to 4001.35 us through the complete quantize+GEMV API and remains
bit-exact.

On the full 32-token benchmark, one core (`taskset -c 11`) measures 2.03495 s,
15.73 token/s. Two cores (`taskset -c 10-11`) with
`FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS=2` measure 1.96991 s, 16.24 token/s: 3.2%
higher throughput with identical 32 token IDs. This is an explicit extra-core
launch strategy, not a single-core codegen claim, and is disabled by default.

## Generated-object audit

The bundle build audits all 22 objects selected by the Qwen3 route: fourteen
W8 quantizer/GEMV objects, the MLP quantizer/GEMV/inline-SwiGLU objects, three
RMSNorm objects, fused add+RMSNorm, and RoPE. Every selected function has zero
call instructions. Quantizers contain four vector `FCVTNS` operations; GEMVs
contain the expected 32 SDOT sites, or 16 for the vocabulary BN32 schedule.
All matrix, norm, and RoPE objects have zero stack references. Inline SwiGLU
has eight bounded prologue/epilogue references and no loop-body spill.

## Rejected or bounded experiments

- A 64-output vocabulary tile regressed by 4.9%. A 32-output tile removes the
  regression and is approximately equal to whole-TLE (7.170 vs 7.174 ms).
  The ordinary JIT fallback now uses the same N>=32768 BN32 selector as AOT;
  a fresh alternating comparison measured 7.157 vs 7.533 ms for BN32/BN64.
- A dual-accumulator dot+SwiGLU kernel caused 21-51 stack accesses. Separating
  the activation from the joined GEMV is faster and spill-free in the dot
  stage.
- A 128-output vocabulary tile caused 215 stack accesses and regressed to
  21.6 ms.
- The ordinary Triton attention override is faster, but its accepted error is
  relative-L2 rather than bit-exact. It changed Qwen generation at token 14,
  so it is excluded from the result above and remains opt-in.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu
bash integrations/pytorch/build.sh
bash integrations/pytorch/build_qwen3_0_6b_bf16_w8_bundle.sh

FLAGGEMS_ARM_W8_AOT_BUNDLE=$PWD/artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8 \
FLAGGEMS_ARM_W8_AOT_LIBRARY=$PWD/artifacts/pytorch-triton-backend/libtriton_bf16_w8_backend.so \
OMP_NUM_THREADS=1 taskset -c 0 \
/home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-qkv-argmax --threads 1 \
  --warmup-tokens 4 --new-tokens 32 --repeats 5 --prefill-repeats 2

FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS=2 \
FLAGGEMS_ARM_W8_AOT_BUNDLE=$PWD/artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8 \
FLAGGEMS_ARM_W8_AOT_LIBRARY=$PWD/artifacts/pytorch-triton-backend/libtriton_bf16_w8_backend.so \
OMP_NUM_THREADS=1 taskset -c 10-11 \
/home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-qkv-argmax --threads 1 \
  --warmup-tokens 4 --new-tokens 32 --repeats 3 --prefill-repeats 1

/home/cix/venv-fep-e2e/bin/python \
  integrations/pytorch/audit_qwen3_w8_bundle.py \
  artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8
```

These are shape-specialized results on one 128-bit-SVE Arm machine. Other
vector lengths, CPU models, model shapes, thread counts, and sustained thermal
states require independent validation.
