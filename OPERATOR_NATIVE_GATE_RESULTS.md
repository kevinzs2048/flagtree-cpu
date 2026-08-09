# CIX ordinary-Triton native operator gate

## Conclusion

The current Arm path has a defensible operator-level value before counting
launch fusion. On CIX CPU 11, 43 of 45 tested rows pass the production gate:
correctness plus direct-call Triton/native latency no worse than 1.03x. The
only rejected rows are W8 decode at K=1024, N=1024 and N=2048. Those exact
shapes remain on KleidiAI; all twelve Qwen3 TP=1 production projection shapes
pass. No result in this report uses TLE_raw or an external compute call.

The full machine-readable result is
`artifacts/operator-native-gate/20260808-144610/results.json`; the generated
table is beside it as `results.md`. The later randomized Q/K RMSNorm and
SwiGLU verification is recorded in
`artifacts/operator-native-gate/20260808-150614/results.json`. The expanded
18-shape W8 bundle sweep is
`artifacts/operator-native-gate/20260808-151226/results.json`.

## Measurement contract

- Platform: CIX Arm64, Cortex-A720 CPU 11, fixed maximum 2.5 GHz, Linux 6.6.
- Compiler input: Triton 3.7.2 from `ports/triton-cpu-3.7.2/python/triton`.
- Timing: direct shared-object function calls, one pinned CPU, alternating
  execution order, median of batches. Python and allocation are excluded.
- W4/W8 matrix rows: both sides consume the same official KAI packed blobs;
  activation/weight packing is excluded.
- QKV rows: both sides include BF16 activation packing, matrix compute, and
  BF16 output conversion; weight packing remains excluded.
- ACLE rows use independently written Arm C intrinsics. KleidiAI rows compare
  against v1.24 microkernels whose hot loop is hand-scheduled inline AArch64
  assembly, not compiler-generated portable C.
- Acceptance threshold: accuracy passes and Triton/native <= 1.03x.

## Direct native results

| Operator family | Shapes | Triton/native range | Accuracy | Routing decision |
|---|---:|---:|---|---|
| Q4 decode vs KAI | 9 | 0.978-1.008x | bit-exact | generated |
| W8 decode vs KAI, accepted | 16 | 0.936-1.022x | max abs <=2.39e-7 | generated |
| W8 K1024, N1024/2048 | 2 | 1.070/1.034x | pass | keep KAI |
| Q/K RMSNorm vs fused ACLE | 6 | 0.953-0.977x | bit-exact | generated |
| SwiGLU+W8 quant vs fused ACLE | 6 | 0.985-1.014x | three outputs bit-exact | generated |
| Fused QKV W8 pipeline vs KAI | 3 | 0.966-0.989x | pack/output bit-exact | generated |
| RMSNorm/RoPE/same-backend W8 | 3 | 0.988-1.002x | pass | generated |

The W8 crossover is visible without fusion:

| K x N | Generated | KAI inline assembly | Ratio | Decision |
|---:|---:|---:|---:|---|
| 1024 x 1024 | 25.819 us | 24.119 us | 1.070x | KAI |
| 1024 x 2048 | 52.477 us | 50.757 us | 1.034x | KAI |
| 1024 x 3072 | 78.402 us | 79.557 us | 0.985x | generated |
| 1024 x 4096 | 103.638 us | 107.315 us | 0.966x | generated |
| 4096 x 2560 | 267.067 us | 268.525 us | 0.995x | generated |
| 2560 x 9728 | 1013.735 us | 1071.594 us | 0.946x | generated |
| 9728 x 2560 | 966.256 us | 1013.764 us | 0.953x | generated |

All TP=1 production shapes pass the direct matrix gate:

| Model | qkv | o | gate_up | down |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0.966x | 1.022x | 0.984x | 0.970x |
| Qwen3-1.7B | 0.988x | 0.964x | 0.956x | 0.982x |
| Qwen3-4B | 0.996x | 0.995x | 0.952x | 0.953x |

Q4 has no equivalent small-shape rejection in the measured set. All nine
same-blob comparisons are bit-exact and remain between 0.978x and 1.008x of
KAI.

## Precision audit

- RMSNorm, RoPE, and same-backend W8 also pass randomized differential tests
  over 64, 32, and 8 cases respectively. ACLE and Triton outputs are exact.
- Q/K RMSNorm was additionally checked over nine input/weight cases at
  Hq=16, Hkv=8, D=128: 0 mismatches in 27,648 BF16 values.
- SwiGLU+W8 quant was additionally checked over 16 bounded-random cases at
  N=3072: 0/49,152 BF16 mismatches, 0/49,152 INT8 mismatches, and bit-exact
  FP32 scale.
- Q4 KAI-layout rows are output bit-exact. W8 FP32 matrix rows use the stated
  numerical threshold; the complete BF16 QKV pipeline is bit-exact at both
  the packed-LHS and final-output boundaries.

## Code generation evidence

- Q4 decode: 8 SDOT, 1 ADDP, 6 LD1R, no stack traffic or external call.
- W8 decode: 16 SDOT, 1 ADDP, 10 LD1R, no stack traffic, lane insertion,
  residual `triton_cpu.dot`, or external call.
- BF16 activation pack: 3 FCVTNS, no stack traffic or external call.
- Q/K RMSNorm: 101 assembly lines, no stack traffic or external call.
- SwiGLU+W8 quant: no external call. Its eight stack annotations are only four
  callee-save stores in the prologue and four restores in the epilogue; both
  rolled data loops contain no stack access.

The Q4 audit found and fixed a real Triton 3.7 migration regression. The
lowered repeated eight-byte activation had changed from `LD1R` to `LDUR D`
plus a lane-duplication `MOV`, making fresh objects 13-17% slower. The Arm
dot-product conversion now recognizes the exact 2-D Triton 3.7
interleave/shape-cast/transpose graph and preserves the original eight-byte
load. A FileCheck test protects the i64 broadcast form, and the executable
benchmark requires six LD1R sites.

BN8, U1/U4, outer-pointer, flat-K8-pointer, and staged-load W8 schedules were
measured as direct A/B experiments. None produced a stable small-N gain. KAI's
remaining advantage comes from a hand-written assembly loop that explicitly
separates loads from dependent SDOT operations. The current production U2
kernel is therefore retained, and the router boundary handles the two losing
shapes instead of embedding opaque assembly in Triton.

## Reproduction

```bash
python3 benchmarks/run_operator_native_gate.py --cpu 11 --batches 9

ninja -C ports/triton-cpu-3.7.2/build-port
ports/triton-cpu-3.7.2/build-port/bin/triton-opt \
  ports/triton-cpu-3.7.2/test/TritonCPU/convert-dot-product-q4-repeat.mlir \
  --triton-cpu-convert-dot-product='enable-i8=true' | \
  /home/cix/.triton/llvm/llvm-87717bf9-ubuntu-arm64/bin/FileCheck \
  ports/triton-cpu-3.7.2/test/TritonCPU/convert-dot-product-q4-repeat.mlir
```

The runner source is `benchmarks/run_operator_native_gate.py`. Native
references are in `benchmarks/cpp_wrapper`, and all generated specialization
paths recorded in the JSON report are fresh-cache objects from the same run.
