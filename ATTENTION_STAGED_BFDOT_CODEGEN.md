# Arm CPU Decode Attention: Staged Ordinary-Triton BFDOT Codegen

Re-audited: 2026-08-08
Device: CIX P1, Cortex-A720
Compiler: active Triton-CPU 3.7.2 port
Precision: BF16 Q/K/V, FP32 softmax and accumulation

## Result

The production experiment now has two compiler-visible decode schedules.  It
does not use TLE_raw or call an external attention compute runtime:

- `online`: one kernel carries the online-softmax state and the complete
  `HEAD_DIM` FP32 V accumulator;
- `staged`: a QK/softmax kernel writes FP32 probabilities into caller-owned
  scratch, then a register-bounded PV kernel accumulates 64 output elements.

The staged QK source is ordinary BF16 multiply plus `tl.sum(...,
dtype=tl.float32)`.  The existing Arm CPU compiler matcher recognizes this
graph and emits BFDOT.  It halves the number of exponentials relative to the
online update and removes the per-token rescale of the complete V accumulator.

## Direct launch microbenchmark

Inputs are preallocated and the timed region contains only Triton launches or
the direct C function.  The process is pinned and uses the same random BF16
input for each implementation.

Single Cortex-A720, microseconds:

| KV length | Online Triton | Staged Triton | Handwritten C | Staged / C |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 159.53 | 84.41 | 153.06 | 0.551x |
| 512 | 562.25 | 259.88 | 592.72 | 0.438x |
| 1024 | 1100.96 | 463.15 | 1182.00 | 0.392x |
| 2048 | 2162.00 | 866.36 | 2346.98 | 0.369x |

Eight Cortex-A720 cores, microseconds:

| KV length | Online Triton | Staged Triton | Handwritten C | Staged / C |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32.75 | 32.94 | 24.57 | 1.340x |
| 512 | 96.87 | 59.85 | 86.87 | 0.689x |
| 1024 | 178.60 | 98.22 | 172.29 | 0.570x |
| 2048 | 337.97 | 178.02 | 339.32 | 0.525x |

At eight threads the second OpenMP launch is not recovered at N=128.  The
production `auto` route therefore selects staged from N=512.  A single-thread
deployment can set `FLAGGEMS_ARM_ATTN_STAGED_MIN_SEQ=0`, because staged wins
through the complete measured single-core range.

## LLVM and assembly audit

| Function | Objdump function-body lines | Stack references | Compute selection |
| --- | ---: | ---: | --- |
| Online | 616 | 184 | 80 FMLA, two scalar `expf` calls |
| Staged QK, former FP32 graph | 262 | 67 | 16 FMLA, four vector exp calls |
| Staged QK, final BF16 graph | 123 | 22 | 16 BFDOT, four `Sleef_expf4_u10` calls |
| Staged PV | 142 | 2 | 16 FMLA, no calls |

The two PV stack references are one D8/D9 callee-save pair in the prologue and
epilogue.  The N loop has no accumulator spill.  Neither staged object has an
external GEMM/attention compute call.  The QK transformation is useful only
when `tl.sum` explicitly requests FP32 reduction: omitting the dtype preserves
rounded BF16 multiply semantics, fails the BFDOT matcher, worsens accuracy,
and regresses N=128 to about 208 us.

## Correctness

Random tests cover:

- KV lengths 127, 128, 129, 257, 512, 513, 1025, and 2048;
- head dimensions 64, 128, and 256;
- GQA ratios 16/8 and 32/8, plus 8/8 MHA;
- five independent random seeds.

Against ATen SDPA, relative L2 remains below 0.003 and maximum absolute error
is at most 0.00390625.  This is the same tolerance class as the previous online
kernel.  Staged versus online maximum absolute difference is at most
0.00006104 in the tested set.  Attention remains opt-in because it is not
bit-exact to ATen.

## Real Qwen3-0.6B W8 decode

With a 512-token prompt and eight CPU threads, one profiled decode step has 28
attention calls:

| Route | 28 attention calls | Mean per call |
| --- | ---: | ---: |
| Online | 5.538 ms | 197.78 us |
| Staged BFDOT | 4.361 ms | 155.75 us |

Attention time falls 21.3%.  Seven isolated, fresh-KV-cache model decode
samples give medians of 71.893 ms online and 71.338 ms staged, a 0.77% whole
step improvement.  Greedy token IDs are identical.  The smaller whole-model
gain is expected: W8 projections and MLP dominate the remaining step.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu

OMP_NUM_THREADS=1 \
TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
PYTHONPATH="$PWD/ports/triton-cpu-3.7.2/python:$PWD/third_party/FlagGems/src" \
FLAGGEMS_VENDOR_NAME=arm \
taskset -c 11 /home/cix/venv-fep-e2e/bin/python \
  benchmarks/bench_attention_decode_codegen.py --seq-len 512

TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
TRITON_CPU_PYTHON="$PWD/ports/triton-cpu-3.7.2/python" \
OMP_NUM_THREADS=1 taskset -c 11 \
  /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/validate_w8a8.py
```

The validation gate requires 16 BFDOT instructions in the D=128 QK object,
no hot-loop PV spill, no external attention compute call, and the stated
numerical tolerances at N=128 and N=512.
