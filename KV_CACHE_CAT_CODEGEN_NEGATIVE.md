# Dynamic KV-cache concat: rejected ordinary-Triton replacement

Date: 2026-08-08
Device: CIX P1, eight Cortex-A720 cores
Shape: B=1, Hkv=8, D=128, BF16, one new token

Qwen3 decode spends roughly 6 ms in 56 K/V DynamicCache concatenations at a
512-token context.  A fused ordinary-Triton prototype allocated both outputs
and copied K and V in one launch.  It was correct but substantially slower
than two ATen cats:

| Existing sequence | Two ATen cats | Fused Triton, rolled per head | Ratio |
| ---: | ---: | ---: | ---: |
| 128 | 19.21 us | 85.72 us | 4.46x |
| 512 | 31.44 us | 204.75 us | 6.51x |
| 1024 | 57.30 us | 345.64 us | 6.03x |
| 2048 | 145.58 us | 622.59 us | 4.28x |

A GPU-style program per vector tile was worse: 605.35 us at N=512.  Reducing
the grid to one rolled program per head removes most program scheduling but
still cannot compete with the optimized copy path.  Increasing the Triton
block to 1024 also causes compile-time and code-size growth and regresses the
copy further.

This is an important boundary: concat has no arithmetic to amortize Triton
CPU launch and loop machinery, while ATen can use direct parallel memory-copy
primitives.  The prototype is retained only as
`benchmarks/bench_kv_cache_concat_codegen.py`; it is not registered or used by
the production router.  Eliminating this cost requires a preallocated/cache
management design, not replacement with an elementwise Triton copy kernel.
