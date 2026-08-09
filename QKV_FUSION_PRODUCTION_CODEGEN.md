# Qwen3 W8 QKV Fusion: Production Ordinary-Triton Codegen

Re-audited: 2026-08-08
Device: CIX P1, eight Cortex-A720 cores
Compiler: active Triton-CPU 3.7.2 port

## Result

The curated Qwen3 W8 route now enables decode QKV fusion by default.  Q, K and
V use the same BF16 activation, so three independent linears unnecessarily ran
the activation quantizer three times and dispatched three SDOT matrices.  The
fused path:

1. quantizes the activation once with the ordinary Triton RNE kernel;
2. consumes one concatenated Q/K/V N64 compiler pack in one SDOT program grid;
3. returns Q, K and V as views of the combined BF16 output.

It uses neither TLE_raw nor an external matrix compute runtime.  The matrix
graph remains `tl.dot` through Triton/MLIR and selects the existing Arm SDOT
lowering.

## Microbenchmark

Shape: K=1024, Q/K/V outputs=2048/1024/1024, BF16 activation, W8 weights,
eight cores.  The timed region includes activation quantization, matrix
computation and Python/Triton launch for all three projections.

| Route | Latency | Relative |
| --- | ---: | ---: |
| Three independent W8 linears | 419.10 us | 1.000x |
| Shared-quantizer fused QKV | 162.06 us | 0.387x |

Fused QKV is 2.59x faster for this operator group.  Every Q/K/V output is
bit-exact to the independent route.

## Real decode

Qwen3-0.6B W8, 512-token prompt, staged attention, eight cores:

| Route | Isolated one-token decode median |
| --- | ---: |
| Independent Q/K/V | 78.254 ms |
| Fused QKV | 65.873 ms |

The same greedy token IDs are produced.  In profiler ranges, the independent
route spends 30.057 ms across 141 regular W8 decode linears.  With fusion, the
57 remaining regular linears take 16.855 ms and the 28 fused QKV calls take
7.008 ms, a 6.19 ms reduction in the projection ranges for the profiled step.

The coordinator releases the three superseded decode packs after building the
combined pack.  Each projection retains its separate KAI-layout prefill pack,
so M>1 and non-BF16 inputs keep the audited fallback.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu

TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
OMP_NUM_THREADS=8 taskset -c 0-1,6-11 \
/home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/bench_bf16_w8_qkv_flaggems_aot.py \
  --threads 8 --whole-mode auto

FLAGGEMS_PROFILE_RANGES=1 \
TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
OMP_NUM_THREADS=8 taskset -c 0-1,6-11 \
/home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/run_qwen3.py \
  --mode int8-qwen3-qkv-attn-argmax --prompt-tokens 512 \
  --threads 8 --new-tokens 1 --repeats 1 --prefill-repeats 1 \
  --profile decode
```

The validation gate checks single-core and multicore bit-exact output, shared
quantization, prefill fallback, source-pack release, unpatch pack rebuilding,
and that the curated production entry keeps fusion enabled by default.

The adjacent Q/K portion of this combined output is also the physical contract
used by the follow-on fused Q/K RMSNorm schedule documented in
`QK_RMSNORM_FUSION_CODEGEN.md`.
