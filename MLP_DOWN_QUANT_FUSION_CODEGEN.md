# W8 MLP SwiGLU + Down-quant Fusion

Date: 2026-08-08
Device: CIX P1, Cortex-A720
Compiler: active Triton-CPU 3.7.2 port

## Result

The ordinary-JIT Qwen3 MLP used to materialize BF16 SwiGLU, return to Python,
then launch the standard W8 activation quantizer for `down_proj`.  The new
schedule keeps the required BF16 intermediate but combines those two stages:

1. compute SiLU with the audited inlined exp polynomial;
2. round SiLU to BF16, multiply by `up`, and store the BF16 SwiGLU scratch;
3. accumulate the exact BF16 magnitude maximum in the same rolled pass;
4. reread the scratch once, apply `127/absmax`, and round-to-nearest-even INT8;
5. feed the resulting activation and scale directly to the ordinary `tl.dot`
   down projection.

The complete MLP is four compiler-visible launches: input quantize, joined
gate/up SDOT, fused SwiGLU/down-quant, and down SDOT.  Previously the SwiGLU
and down quantizer were separate, making five.  No TLE_raw or external matrix
runtime is used.

## Microbenchmark and codegen

For 3072 BF16 gate/up elements on eight cores:

| Route | Latency |
| --- | ---: |
| Separate SwiGLU + W8 quantizer | 39.23 us |
| Fused ordinary-Triton kernel | 25.88 us |

The fused path is 1.52x faster for this stage.  BF16 scratch, INT8 activation
and FP32 activation scale are all bit-exact.  Validation exhausts all 65,280
finite BF16 gate encodings and also checks the full gate/up/down W8 MLP on one
and two cores.

The generated fused object has no external compute calls.  Its eight folded
stack annotations are four callee-save stores in the function prologue and
four matching restores in the epilogue; there are no hot-loop spills.

## Real decode

Qwen3-0.6B W8, fused QKV/QK norm, staged attention, 512-token prompt, eight
cores:

| Route | Isolated decode median |
| --- | ---: |
| Separate SwiGLU and down quant | 62.402 ms |
| Fused SwiGLU/down quant | 60.981 ms |

Greedy token IDs remain identical.  Profiler ranges move the 28 down
projections into the MLP range: regular W8 decode drops from 57 calls/15.060
ms to 29 calls/9.241 ms, while the MLP range grows from 9.750 to 14.263 ms.
The combined projection time therefore falls from 24.810 to 23.504 ms, a
1.31 ms saving consistent with the real-step result.

The route is enabled by default for a compatible W8 `down_proj` and can be
disabled for controlled A/B with
`FLAGGEMS_ARM_MLP_DOWN_QUANT_FUSION=0`.  Prefill, AOT and incompatible module
layouts keep their existing paths.  The down projection's existing decode
pack is retiled in place to N64, so the fusion does not retain a duplicate
matrix pack.
