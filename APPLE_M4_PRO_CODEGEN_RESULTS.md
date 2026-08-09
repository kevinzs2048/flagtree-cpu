# Apple M4 Pro fixed-width Arm codegen validation

This records the native Darwin validation of the Triton-CPU 3.7.2 Arm path.
It is intentionally separate from the CIX latency tables: Apple M4 uses the
fixed-width Neon SDOT/SMMLA lowering, while CIX can also exercise 128-bit SVE2.

## Environment

- MacBook Pro `Mac16,8`, Apple M4 Pro, 10 performance plus 4 efficiency cores,
  48 GB
- macOS 15.7.3 / Darwin 24.6.0
- Python 3.11.15, PyTorch 2.13.0, Triton 3.7.2
- Triton import:
  `/Users/kevin/triton-opt-cpu/ports/triton-cpu-3.7.2/python/triton`
- Apple Clang/Clang++ 16.0.0 from Command Line Tools
- assembler flags: `-march=armv8.6-a+dotprod+i8mm+fp16+bf16`

All 116 JIT shared objects were native Mach-O arm64 objects. No Linux object
was reused. The process loaded the local `libTritonCPURuntime.dylib` and
`libsleef.3.6.1.dylib`.

## Validation status

- Darwin arm64 editable build and `make`: pass
- CPU unit tests: 20 pass, one expected skip because M4 has no SVE2
- Q4 production router and correctness: pass, maximum BF16 ULP distance 4
- W8 decode and prefill: bit-exact
- fused QKV: bit-exact
- fused MLP/SwiGLU: bit-exact
- all 65,280 finite BF16 quantizer encodings: pass
- RoPE: bit-exact
- attention: within tolerance, maximum absolute error 0.001953125; not
  bit-exact and therefore disabled by default

The monolithic runner was interrupted by the macOS watchdog. Every remaining
command passed when executed separately after the interruption.

## Q4 direct pipeline latency

Single-core production direct pipeline, N=K=1024, in microseconds:

| M | latency |
| ---: | ---: |
| 1 | 32.56 |
| 3 | 59.90 |
| 4 | 55.65 |
| 8 | 81.36 |
| 12 | 106.31 |
| 16 | 132.93 |
| 20 | 176.13 |
| 24 | 201.71 |
| 28 | 228.07 |
| 31 | 266.02 |
| 32 | 244.37 |
| 47 | 376.03 |
| 48 | 356.19 |

The legacy row-pack M16 route takes 131.51 us. Forcing M16 into two M8 blocks
takes 133.02 us at M16, 258.82 us at M32, and 370.99 us at M48.

## Q8 KAI-layout latency

The BF16-compatible result is bit-exact at N=K=1024:

| M | latency (us) | static SMMLA |
| ---: | ---: | ---: |
| 4 | 60.48 | 16 |
| 8 | 69.15 | 32 |
| 12 | 80.60 | 48 |
| 16 | 96.77 | 64 |

A longer W8-prefill M16 sample measures 80.65 us with the same 64 static
SMMLA instructions.

## Generated-code audit

| Kernel | selected instructions |
| --- | ---: |
| Q4 decode | 8 SDOT + 1 ADDP |
| Q4 prefill M4/M8/M12/M16 | 16/32/48/64 SMMLA |
| W8 decode BN64 | 32 SDOT |
| W8 vocabulary BN32 | 16 SDOT |
| Q8 prefill M4/M8/M12/M16 | 16/32/48/64 SMMLA |

The native objects contain zero SVE intrinsics, zero SVE z/p registers, zero
residual `triton_cpu.dot`, and zero external GEMV/MLP compute-runtime calls.
Q4 M4/M8/M12/M16 hot loops have no stack access. The remaining function-level
stack references (2/6/8 for the audited larger variants) are callee-save
prologue/epilogue accesses, not K-loop spills. W8 decode and vocabulary
kernels have no stack access.

## Vocabulary schedule A/B

For N=152064 and K=1024, a native Apple-Clang alternating-order test over six
runs gives 1955.35 us for BN32 (16 SDOT) and 1936.63 us for BN64 (32 SDOT).
BN64 is about 0.94% faster and every output is bit-exact; both hot loops are
spill-free. The difference is close to run-to-run variation, so production
continues to select BN32 for N>=32768. On CIX, the same comparison is
7156.51/7533.41 us in favor of BN32 by about 5%, showing why this schedule
must remain target-dependent.

No test or production kernel above introduces TLE_raw or a handwritten C
compute runtime.

## Required follow-up after the CIX M16 work

The later exact-KAI M16 pipeline changes are not included in the measurements
above. The new bundle carries eight min/max lanes through the BF16 pack K loop,
narrows the quantized intermediate to int16, and associates the matrix scale
product in KAI order. Before enabling generated M16 by default on Apple, rerun
the 18-shape bundle audit, the all-finite-BF16 pack comparison, and paired
plugin microbenchmarks. Darwin must still show Neon `SMMLA`, no SVE register,
no hot-loop stack access, and bit-identical KAI output.
