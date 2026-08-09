# vLLM Triton end-to-end coverage on CIX

Date: 2026-08-04

## Production policy being measured

The current performance route is mode- and thread-aware. The test uses CIX
Arm64, eight PyTorch CPU threads, Qwen3-0.6B, a 12-token prompt, and 32 greedy
output tokens. Engine initialization and one warm-up request are excluded.

- Eager decode uses ordinary-Triton AOT for standalone RMSNorm, fused
  residual-add + RMSNorm, RoPE, and SwiGLU.
- Compiled mode retains the native PyTorch/vLLM expressions for those four
  small operations, because Inductor fuses them with their neighbors and an
  opaque AOT boundary regresses the full graph.
- The model still has 112 W8 Linear modules (four fused projections in each
  of 28 layers). At eight threads, the measured production cutoff selects the
  direct KleidiAI closures for these linears. Generated W8 is selected only at
  thread counts where it wins. Therefore W8 time is not counted as Triton in
  the eight-thread result below.

This policy deliberately measures the fastest validated end-to-end route,
not a forced maximum-Triton-coverage configuration.

## Operators actually replaced in eager decode

| vLLM operation | Function | Calls per decode token | Generated shape |
|---|---|---:|---|
| standalone RMSNorm | final normalization plus Q/K head normalization | 57 | `(1,1024)`, `(16,128)`, `(8,128)` |
| fused add + RMSNorm | update residual and normalize before attention/MLP | 56 | `(1,1024)` |
| RoPE Q/K | apply position-dependent rotation to Q and K | 28 | Q=`(16,128)`, K=`(8,128)` |
| SwiGLU | `SiLU(gate) * up` MLP activation | 28 | joined input `(1,6144)` |

That is 169 generated calls per decode token. For each measured 32-token
request, vLLM performs 31 timed decode steps, giving exactly 1,767 standalone
RMSNorm, 1,736 fused RMSNorm, 868 RoPE, and 868 SwiGLU calls. All eligible
calls used generated code and the native-fallback count was zero. Prefill
shapes stay on their original implementation, so TTFT is not optimized by
these decode-only routes.

The model-load W8 replacement inventory is separate:

| Module per decoder layer | Weight `[N,K]` | Eight-thread route |
|---|---:|---|
| fused Q/K/V projection | `[4096,1024]` | KleidiAI |
| attention output projection | `[1024,2048]` | KleidiAI |
| fused gate/up projection | `[6144,1024]` | KleidiAI |
| MLP down projection | `[1024,3072]` | KleidiAI |

## End-to-end result

Five unprofiled complete requests were measured after one warm-up. The two
runs used the same W8 route and differ only in the four eager P0 replacements.

| Eight-thread eager route | Median inference | Median decode | Median TTFT |
|---|---:|---:|---:|
| direct KleidiAI W8, native P0 ops | 1773.586 ms | 18.176 tok/s | 67.582 ms |
| direct KleidiAI W8, generated P0 ops | 1403.325 ms | 23.236 tok/s | 69.172 ms |
| change | **-20.88%** | **+27.84%** | +2.35% |

Both runs produced the same greedy token digest:
`6ed177097af805f0f879510e8eaf8694efe093dc6cf1560e158d2a72989343db`.

Compiled mode was also run end to end with the P0 integration enabled. It
produced the expected compiled digest
`ba930545728f4520f2f21b72726afb468bef7e13c650cbcadd559f70b6ca0204`
at 29.115 tok/s in the final smoke. Every P0 call counter was zero, proving
that the mode guard preserved the compiled graph instead of silently
inserting AOT calls.

## Timed Triton share

Three additional complete eager requests enabled timing instrumentation. The
denominator is vLLM's interval from `scheduled_ts` through `last_token_ts`, so
it includes both prefill and decode.

| Route | Calls, 3 requests | Generated launch | Complete Python/AOT op |
|---|---:|---:|---:|
| RMSNorm | 5,301 | 42.174 ms | 85.477 ms |
| fused add + RMSNorm | 5,208 | 35.518 ms | 47.231 ms |
| RoPE | 2,604 | 13.014 ms | 16.200 ms |
| SwiGLU | 2,604 | 35.693 ms | 71.213 ms |
| **total** | **15,717** | **126.399 ms** | **220.121 ms** |

Total inference time for those requests was 4084.197 ms. Generated launch
time therefore accounts for 3.095% of end-to-end inference, while the full
operator boundary (allocation, Python routing, and generated launch) accounts
for 5.390%. This is the current Triton CPU-time coverage for the fastest
eight-thread production route; it must not be combined with the older forced
W8 coverage figure.

## Code-generation and deployment gates

The release contains six copied, read-only ELF objects: three RMSNorm shapes,
one vLLM-semantics fused RMSNorm, one RoPE, and one SwiGLU. They are generated
from ordinary Triton kernels. There is no TLE_raw compute body and no external
GEMV, normalization, RoPE, or activation runtime call.

- RMSNorm, fused RMSNorm, and RoPE have zero function-level stack access and
  zero external calls.
- SwiGLU has zero external calls. Its eight stack references are only the
  prologue/epilogue saves and restores of `d8` through `d14`; the hot loop has
  no spill traffic.
- RoPE reads the position and selects the cache row inside generated code.
  This both matches current vLLM FP32-then-BF16 semantics and removes Python
  `item()`/slice overhead.
- Bundle loading validates format, launcher ABI, OS/architecture, required
  Arm features, an exact manifest, regular files, and every SHA-256 digest.
  The negative gate rejects a deliberately corrupted object.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu
bash integrations/vllm/build_p0_backend.sh
build_output="$(bash integrations/vllm/build_qwen3_p0_codegen_bundle.sh)"
p0_bundle="$(printf '%s\n' "${build_output}" | awk -F= '$1 == "bundle" {print $2}')"

env \
  PYTHONPATH=/home/kevin/triton-opt-cpu/ports/triton-cpu-3.7.2/python:/home/kevin/triton-opt-cpu/third_party/FlagGems/src \
  TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
  /home/kevin/venv-int8-clean/bin/python \
  benchmarks/bench_vllm_w8_codegen_e2e.py codegen \
  --bundle /path/to/exact/w8/bundle \
  --p0-codegen --p0-bundle "${p0_bundle}" \
  --eager --tokens 32 --rounds 5 --warmup 1 --threads 8
```

Add `--profile-triton-time` for coverage timing. Timing is disabled for normal
performance runs.
