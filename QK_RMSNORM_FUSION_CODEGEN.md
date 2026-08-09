# Qwen3 Q/K RMSNorm Fusion: Layout-driven Ordinary-Triton Codegen

Date: 2026-08-08
Device: CIX P1, Cortex-A720
Compiler: active Triton-CPU 3.7.2 port

## Result

Qwen3 applies independent RMSNorm operations to the Q and K projection rows
before RoPE.  In decode, this was two small Triton launches per layer.  The W8
QKV fusion already produces adjacent Q then K rows, so the production route
now uses that physical contract to normalize all Q/K heads in one program
grid and one adjacent output allocation.

The kernel contains only ordinary loads, FP32 reduction, `tl.sqrt`, BF16
rounding/multiply and stores.  It uses no TLE_raw and has no external compute
call.  Q/K weights are expanded once at setup to one 128-element row per head;
this removes a runtime Q-vs-K pointer select from the generated loop.  The
extra packed weights occupy 6 KiB per Qwen3-0.6B layer.

## Why `head_dim` stays dynamic

An early constexpr-D128 form generated 598 objdump function-body lines.  It
was spill-free after separating input/output, but LLVM fully expanded the
eight D16 tiles.  Keeping `head_dim` as a runtime loop bound produces 241
lines, zero folded spills/reloads and no external calls.  This is the same
rolled-vector shape used by the audited standalone RMSNorm and generalizes to
other tile-aligned head dimensions without a new giant object.

## Microbenchmark

BF16 Q=[16,128], K=[8,128], with allocations included:

| Threads | Two independent RMSNorms | Fused Q/K RMSNorm | Relative |
| ---: | ---: | ---: | ---: |
| 1 | 104.51 us | 71.17 us | 0.681x |
| 8 | 135.42 us | 74.86 us | 0.553x |

Outputs are bit-exact to the two standalone compiler-generated kernels.

## End-to-end and profile

Qwen3-0.6B W8, fused QKV, staged attention, 512-token prompt, eight cores:

| Route | Isolated decode median |
| --- | ---: |
| Independent Q/K norms | 65.627 ms |
| Fused Q/K norm | 62.402 ms |

The same greedy token IDs are produced.  In the profiled fused run, the 56
Q/K standalone ranges disappear: remaining standalone RMSNorm falls from 85
to 29 calls, while 28 `triton::qk_rms_norm` calls take 2.325 ms.  Standalone
plus fused norm time is 4.928 ms versus 7.877 ms before fusion, a 2.95 ms
reduction consistent with the real-step A/B.

The path is enabled by default only when the model has the fused-QKV layout.
M>1, gradients, other dtypes, non-contiguous inputs and incompatible layouts
use the saved standalone RMSNorm forwards.

## Validation

`benchmarks/flaggems_e2e/validate_w8a8.py` gates:

- bit-exact Q and K output;
- bit-exact prefill fallback;
- production default and reversible patching;
- fewer than 300 assembly lines;
- zero folded spills/reloads;
- zero external compute calls.
