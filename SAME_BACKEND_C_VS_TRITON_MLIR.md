# C and Triton Frontends with One MLIR/LLVM Backend

Re-audited: 2026-08-06
Device: CIX, Cortex-A720 CPU 11, performance governor, observed 2.5 GHz
Triton: 3.7.2 CPU port
C frontend and common backend: LLVM/MLIR 23 snapshot `87717bf9`

## Question

If equivalent C and Triton implementations are both handed to one MLIR/LLVM
backend, does the Triton frontend impose an inherent compute-performance tax?

## Controlled pipeline

```text
portable/ACLE C -> Clang 23, no LLVM optimization -> raw LLVM IR --------+
                                                                         |
Triton -> TTIR -> TTCIR -> TTTCIR -> raw LLVM IR ------------------------+
                                                                         v
             LLVM 23 import -> LLVM-dialect MLIR -> canonicalize + CSE
             -> LLVM 23 export -> identical LLVM O3 -> identical AArch64
             features -> assembly -> direct shared-object function call
```

Both paths now use the same LLVM 23 build. A follow-up Clang 16 compatibility
run produced instruction streams identical to Clang 23 for the tested C
kernels, but Clang 23 is the primary result.

The harness pins itself to CPU 11, rotates ACLE/Triton order, uses independent
page-aligned allocations and the same output address, and reports medians of
interleaved batches. Python, the Triton launcher, JIT, and dispatcher are not
inside the timed region.

This is an LLVM-dialect MLIR convergence experiment. It does not test a
C-to-SCF/Affine frontend. The frontend/lowering quality before LLVM-dialect
MLIR remains part of the comparison.

## Kernels

- RMSNorm: 1 x 1024 BF16, BF16 output and explicit intermediate BF16 rounding.
- RoPE: 16 Q heads + 8 K heads, head dimension 128, BF16 in-place output.
- W8 GEMV: KAI `qai8dxp/qsi8cxp`, K=1024, N=3072, FP32 output.

The Triton sources use ordinary `tl.load`, reshape, arithmetic, reduction, and
store operations. W8 SDOT is recognized from the algebraic multiply/reduction
graph by the compiler. There is no TLE_raw or external C compute-runtime call.

## Re-tested microbenchmarks

Median across seven process-level runs. Each process result is itself the
median of 30 interleaved ACLE/Triton batches:

| Kernel | Optimized ACLE C | Triton | Triton vs ACLE |
|---|---:|---:|---:|
| RMSNorm | 0.620652 us | 0.612115 us | 1.38% faster |
| RoPE | 0.913421 us | 0.915601 us | 0.24% slower |
| W8 GEMV | 79.4389 us | 79.1407 us | 0.38% faster |

All three optimized comparisons are within 1.4%. A representative portable-C
run was 3.428 us, 5.276 us, and 1549.6 us respectively. Portable scalar C is
not an equivalent hardware-directed implementation and is not used to decide
C-versus-Triton parity.

A separate RoPE-only 30-process stability run gives medians of 0.913413 us
for ACLE and 0.915607 us for Triton (Triton 0.24% slower). Worst process
medians were 0.942591 us and 0.931561 us respectively; neither exceeded 1 us.
This isolates occasional much larger values seen in the combined benchmark as
cross-kernel/system contamination rather than a reproducible RoPE regression.

## Alias-information A/B

The primary optimized C uses `restrict`, matching the real contract that input,
weight/cache, and output buffers do not overlap. Because Triton LLVM arguments
do not carry `noalias`, the C implementation was also compiled without
`restrict` and remeasured over seven aligned runs:

| Kernel | ACLE C without `restrict` | Triton | Triton result |
|---|---:|---:|---:|
| RMSNorm | 0.620622 us | 0.612041 us | 1.38% faster |
| RoPE | 0.965403 us | 0.915567 us | 5.16% faster |
| W8 GEMV | 79.6701 us | 79.2078 us | 0.58% faster |

Removing `restrict` changes RoPE C from 193 to 209 static instructions because
loads can no longer move across its in-place stores. It does not reveal a
Triton penalty; it makes Triton faster. The honest bound from these two C
contracts is therefore parity to a 5.2% Triton advantage, not a universal
sub-one-percent result.

## Randomized correctness audit

- RMSNorm: 64 cases / 65,536 outputs. ACLE C versus Triton is bit-exact.
  Portable C has 13 one-ULP BF16 differences and no tolerance failure.
- RoPE: 32 cases / 98,304 outputs. Both C variants are bit-exact with Triton.
- W8: 8 cases, including a `[16, 32)` block subrange and untouched-output
  sentinels. ACLE C is FP32 bit-exact with Triton. Portable C has maximum
  absolute error `9.53674e-7` and no tolerance failure.
- No implementation wrote outside the requested W8 block range.

## Hardware-counter cross-check

Explicit Cortex-A720 PMU `armv8_pmuv3_0` events were used; generic `perf`
events were rejected because they multiplexed onto the wrong cluster PMU.

| Kernel | Triton / ACLE cycles | Triton / ACLE retired instructions |
|---|---:|---:|
| RMSNorm | 0.9890 | 0.8589 |
| RoPE | 0.9897 | 1.0003 |
| W8 GEMV | 0.9989 | 1.0118 |

The counter ratios corroborate the wall-clock conclusion. RMSNorm retires
about 14.1% fewer dynamic instructions in the Triton path. RoPE and W8 are
effectively identical in cycles despite small instruction-count differences.

## Assembly and linked-object audit

| Kernel | Frontend | Static instructions | Hardware operations | Stack refs | Calls |
|---|---|---:|---|---:|---:|
| RMSNorm | ACLE C | 50 | BF16 conversions | 0 | 0 |
| RMSNorm | Triton | 64 | BF16/SVE conversions | 0 | 0 |
| RoPE | ACLE C | 193 | 32 BF16 conversion ops | 0 | 0 |
| RoPE | Triton | 193 | 32 BF16 conversion ops | 0 | 0 |
| W8 | ACLE C | 117 | 32 SDOT | 0 | 0 |
| W8 | Triton | 117 | 32 SDOT | 0 | 0 |

The final linked `.so` symbols were also disassembled. They contain no compute
calls, stack spills, or unresolved non-weak compute symbols. The small
difference between source-assembly and ELF instruction counts is alignment NOP
padding, not additional computation.

For ACLE C and Triton, the LLVM-dialect MLIR round trip leaves the operation
stream unchanged relative to direct LLVM O3. Portable W8 is different: after
the strict MLIR round trip LLVM no longer recognizes SDOT, producing zero SDOT
and about 1.55 ms instead of the roughly 243 us direct-LLVM result. Clang 16
and Clang 23 reproduce this behavior. This is a generic-loop/LLVM-dialect
import issue, not evidence that optimized C or Triton has an inherent tax.

## RMSNorm A720 alignment cliff and compiler fix

The original `std::vector` harness accidentally sampled only one address
layout. A page-offset sweep found that output offsets 16 and 32 bytes made the
Triton RMSNorm about 50% slower (`0.932 us` versus ACLE's `0.620 us`), while
offsets 0, 48, and 64 remained fast.

The final loop contained one post-increment `STP Q,Q` for the 32-byte BF16
store. At offsets 16/32, its low address bits overlap the following iteration's
32-byte input/weight loads. Replacing only that pair store with two `STR Q`
operations removes the Cortex-A720 cross-iteration memory-disambiguation false
dependency. `STNP` reproduced the regression, confirming that the pair store,
not cache allocation policy, was the trigger.

The CPU backend now runs a narrowly scoped Cortex-A720 pass after LLVM-dialect
CSE. It marks only loop-carried, independent-output `vector<16xbf16>` stores
volatile, which makes LLVM preserve two `STR Q` operations. In-place stores
such as RoPE are excluded by exact load/store address matching, and other Arm
cores retain normal store pairing.

After the fix:

- offsets 16 and 32 fall from about `0.932 us` to `0.612 us`;
- all tested offsets have Triton/ACLE ratios from `0.9863` to `0.9867`;
- RMSNorm remains 1.3-1.4% faster than ACLE instead of requiring 64-byte
  output alignment;
- code size changes from 63 to 64 instructions, with no stack access or calls;
- RoPE remains 193 instructions and W8 remains 117 instructions.

An 8-element output loop was also robust but about 1.3% slower than ACLE. A
32-element reduction tile was faster on favorable alignment but expanded to
875 instructions. Neither kernel-level workaround is needed after the targeted
compiler fix.

## W8 outer-loop address-induction A/B

The production W8 kernel still computes each N4 panel base from the block
index. A follow-up variant carried RHS and output pointers through the outer
loop to remove that multiplication. LLVM then introduced an extra
`base + inner_offset` addition in every K-loop iteration. Both variants remain
117 instructions with 32 static SDOTs and no stack access, but seven standalone
runs gave medians of about `79.42 us` for the production kernel and `79.64 us`
for the outer-pointer variant. The attempted strength reduction is therefore
not selected; the existing BN4 explicit K-pointer schedule remains the best
tested form.

## Revised conclusion

There is still no evidence of an inherent Triton frontend performance penalty
when C and Triton deliver equivalent hardware-aware structure to the same
backend. Across the complete address sweep, Triton is within 1.4% of optimized
C for all three kernels and is faster for RMSNorm and W8. Without C `restrict`,
Triton is up to 5.2% faster.

The re-audit did expose a severe Cortex-A720 pair-store corner case, but the
compiler now removes it without a kernel rewrite or alignment contract.
Portable C still loses its SDOT pattern during LLVM-dialect MLIR
import/export; that remains a separate generic-loop/MLIR issue.

The practical design target remains: ordinary Triton operations must lower to
the same explicit vector, packed-layout, reduction, and unroll structures as
expert ACLE C, while also preserving alias/alignment robustness.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu
bash benchmarks/same_backend_frontend_ab/run.sh

# Seven-run optimized comparison
SAME_BACKEND_OPTIMIZED_ONLY=1 bash -c \
  'for i in 1 2 3 4 5 6 7; do \
     taskset -c 11 artifacts/same-backend-frontend-ab/bench \
       artifacts/same-backend-frontend-ab/c_frontend.so \
       artifacts/same-backend-frontend-ab/c_acle.so \
       artifacts/same-backend-frontend-ab/triton_rms.so \
       artifacts/same-backend-frontend-ab/triton_rope.so \
       artifacts/same-backend-frontend-ab/triton_w8.so; \
   done'

# Full 64-byte page-offset sweep plus the two known bad offsets
taskset -c 11 artifacts/same-backend-frontend-ab/alignment_sweep \
  artifacts/same-backend-frontend-ab/c_acle.so \
  artifacts/same-backend-frontend-ab/triton_rms.so
```

Generated IR, MLIR, assembly, shared objects, logs, correctness runner,
alignment/stability sweeps, and PMU CSV files are under
`artifacts/same-backend-frontend-ab/`.
