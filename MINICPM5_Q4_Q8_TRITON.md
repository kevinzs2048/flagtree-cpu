# MiniCPM5-2.6B Q4/Q8 ordinary-Triton support on CIX

> Production vLLM update (2026-08-09): the current deployment uses a G128
> Q4 body plus a G32 `qai8dxp_f32` head, or per-channel W8 for every Q8
> linear including the head. See `VLLM_MINICPM5_LIBTRITON_JIT_RESULTS.md` for
> the same-checkpoint vLLM results and final LLVM audit. The direct-PyTorch
> history below predates that all-linear checkpoint contract.

Date: 2026-08-09

This work adds direct PyTorch inference for the three local checkpoints:

- `MiniCPM5-2.6B-W4A8-GPTQ-G128`: signed grouped Q4 weights, G128,
  dynamic token-wise asymmetric A8.
- `MiniCPM5-2.6B-W8A8-CT`: per-channel symmetric W8, dynamic token-wise
  symmetric A8.
- `MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think_512k`: BF16
  reference.

The model is a 42-layer Llama architecture with hidden size 2048,
intermediate size 6144, 16 Q heads, 2 KV heads, head dimension 128 and a
130560-entry vocabulary.  The compressed checkpoints are loaded directly
from safetensors without materializing every matrix as BF16.

The original version of this report compared the quantized Triton routes to
the BF16 checkpoint.  That only demonstrates the value of quantization; it
is not evidence that Triton matches native quantized ATen.  Every performance
result below therefore uses the same compressed checkpoint on both sides.
After the G128 ABI, decode routing and shared-RHS prefill work described here,
Q4 is now close to the joined ATen/KleidiAI baseline on one A720 core and its
full-CPU prefill is slightly faster.  Q8 retains its larger advantage over the
available native ATen controls.

## Execution and lowering

The matrix compute is ordinary Triton, not TLE_raw and not a hand-written C
runtime call:

```text
@triton.jit
  -> TTIR: tl.load / reshape / permute / tl.dot / reduction
  -> Triton-CPU IR: triton_cpu.dot plus explicit quantization/scaling graph
  -> target-aware packed-layout rewrite
  -> LLVM AArch64 NEON SDOT or I8MM SMMLA
```

The Q4 checkpoint required a native G128 asymmetric ABI and compiler matcher.
Each `[N4,K128]` weight record is 264 bytes: eight BF16 group scales followed
by 256 bytes of signed packed Q4 values.  A separate 16-byte vector per N4
tile stores the FP32 scale-weighted integer sums needed for zero-point
correction.  There is no conversion to G32 and no temporary expanded Q4
matrix.

Each activation token carries a BF16 scale and INT8 zero point.  The ordinary
Triton expression

```text
dot(q_activation, q_weight * 16) - activation_zp * sum(q_weight * 16)
```

remains visible to the compiler.  The Arm pass recognizes the complete graph,
keeps the integer accumulator across the four K32 subgroups of a G128 group,
emits the zero-point correction in native i32 rows and lowers the matrix work
to SDOT or SMMLA.  For prefill it also groups pairs of M4 dots that use the
same packed RHS, so an M8 micro-tile loads and unpacks each RHS segment once.

The prefill router is shape-adaptive.  Small inputs retain a direct M16/tail
kernel, divisible intermediate shapes choose M16, M12 or M8 to avoid an extra
tail launch, and M>=96 defaults to M12 because its lower register pressure is
consistently faster.  Real-checkpoint A/B covers M=12 through M=256; the fixed
BLOCK_M environment setting is retained only as an audit override.

Decode uses two production routes.  With enough CPUs, independent output
partitions fuse activation quantization with their SDOT work and run in
parallel.  Under a narrow affinity, the activation is quantized and compactly
packed once, then shared by all partitions.  The router selects between them
from the actual CPU affinity and thread count; an environment variable remains
only as an explicit benchmarking override.

## Covered decode operators

Q4:

- input RMSNorm + one joined Q/K/V projection;
- RoPE for Q and K;
- compiler-generated decode attention;
- Q4 output projection;
- residual add + post-attention RMSNorm + joined gate/up projection;
- decode SwiGLU + one shared compact asymmetric-A8 down pack;
- Q4 down projection + final BF16 residual store epilogue;
- final RMSNorm and vocabulary argmax;
- optional runtime-Q4 vocabulary projection.

Q8 covers the same graph with joined W8 Q/K/V, joined gate/up, fused
SwiGLU-to-down quantization and W8 down/output projections.  Its input
RMSNorm is fused into QKV activation packing, and post-attention add+RMSNorm
is fused into MLP activation packing.

Embedding/index selection and lightweight model bookkeeping remain in
PyTorch.  The profile also still sees one final residual add per layer and KV
cache `cat`; these are now a small fraction of decode time.

## Correctness and code-generation gates

The focused Q4 asymmetric validation checks M=1,3,4,8,12,16, standalone
projection, RMSNorm-to-projection and add+RMSNorm-to-projection:

- RMSNorm and add+RMSNorm routes pass their staged references.
- G128 maximum absolute differences for M=1,3,4,8,12,16 are respectively
  0.015625, 0.0234375, 0.0234375, 0.0205078, 0.01953125 and 0.0234375.
  The differences come from the FP32 group accumulation order.
- The all-zero token uses `torch.bfloat16.eps`, matching compressed-tensors.
- G128 decode contains 32 SDOT and one function-level stack reference.
- G128 M4 emits 64 static SMMLA.  The shared-RHS M8 kernel contains 32 static
  SMMLA in a four-iteration subgroup loop and has no stack access in its hot
  loop.  M12 and M16 are also direct kernels rather than multiple launches.
- All audited matrix variants contain zero residual `triton_cpu.dot` and zero
  external compute calls.
- The fused down residual epilogue remains 32 SDOT with one function-level
  stack reference and is bit exact to a staged BF16 add.

M12 and M16 still spill some FP32 accumulators around the outer G128 loop.
They remain faster than splitting the work into M4 launches, but eliminating
these spills is the clearest remaining compiler optimization.

The W8 validation covers eight decode shapes, thirteen prefill shapes and all
65280 finite BF16 encodings.  The native-ATen audit exposed and fixed a real
bug: decode used RNE activation quantization while prefill used FP-to-INT
truncation, and the old test reference repeated that truncation.  Both paths
now use RNE.  Across all prefill tests, the only differences from staged
PyTorch quantization are 10 one-count FP32 half-integer boundary cases; the
matrix output is then required to be bit exact for the activation codes
actually generated.  Decode/prefill matrix results, joined QKV,
input-RMSNorm quantization, add+RMSNorm MLP input quantization, fused down
quantization and RoPE are bit exact against their staged references.  The
attention implementation is intentionally tolerance-based rather than bit
exact; measured relative L2 error is below 0.0025 and maximum absolute error
is at most 0.001953125 in the focused test.

On the 4-token smoke prompt:

- BF16 and W8 select token 36 and generate token 38.
- The compressed-tensors Q4 PyTorch reference and Triton Q4 both select token
  608 and generate token 458.

## Corrected native-ATen comparison on CIX

Configuration: 8 PyTorch threads, one inter-op thread and a warm JIT cache.
The principal end-to-end comparison uses a 12-token prompt and quantizes the
otherwise-excluded vocabulary head identically on both sides.  Absolute times
vary with board load, so all results are alternating, same-process paired A/B
runs and their raw samples are retained in JSON.

### Q4/G128

PyTorch 2.10 on this board has KleidiAI enabled.  Its
`aten::_dyn_quant_matmul_4bit` implements per-token dynamic A8 plus groupwise
Q4, selecting dot-product GEMV at M=1 and i8mm GEMM for M>1.  This is the
native quantized baseline, not BF16 `aten::linear`.

The current standalone matrix body is not uniformly faster than KleidiAI.
On one pinned A720 core, representative M=1 projections remain about 11--19%
slower.  The joined QKV and joined gate/up kernels are different: one compact
activation pack is shared across the joined projection, so they are about
36% and 8% faster respectively than the corresponding groups of native calls.
At M=12, direct shared-RHS prefill is still about 8--24% slower for individual
matrices, while joined QKV and gate/up are about 16% and 6% faster.  This is
the useful Q4 codegen region: compiler-generated packed matrix work combined
with a Triton-level joined operator, not a claim that every standalone shape
beats KleidiAI.

End-to-end results against the joined ATen/KleidiAI model using the same Q4
checkpoint and quantized vocabulary head are:

| CIX CPU scope | Triton prefill | ATen prefill | paired ratio | Triton decode | ATen decode | paired ratio |
|---|---:|---:|---:|---:|---:|---:|
| one A720 core | 469.05 ms | 448.31 ms | 1.0467 | 138.71 ms | 137.48 ms | 1.0062 |
| CPUs 0--11, 8 threads | 137.87 ms | 144.01 ms | 0.9595 | 78.36 ms | 77.41 ms | 1.0112 |

Thus Q4 is within 4.7% for single-core prefill and 0.7% for decode.  With the
normal full CPU set, generated Triton prefill is about 4.0% faster and decode
is about 1.1% slower.  Both routes produce token 220 in the measured run.
With the shared BF16 vocabulary head instead, the single-core paired gaps are
4.4% for prefill and 0.9% for decode.  These are comparisons to native
quantized ATen/KleidiAI, not to the BF16 checkpoint.

The implementation is now a practical alternative to the native Q4 route,
but the evidence does not support calling it universally faster.

The decode MLP now fuses BF16 SwiGLU materialization with one compact
asymmetric-A8 pack for the down projection.  This removes eight redundant
activation scans/packs without adding a launch.  On the real layer-0 weights,
SwiGLU plus down improves from 286.17 to 257.95 us (1.11x), and the complete
joined gate/up + SwiGLU + down MLP improves from 628.66 to 607.46 us (1.035x).
The old and new outputs are bit exact.  LLVM audit finds no external compute
call and no TLE_raw in the new rolled kernel.

The post-change decode profile assigns 11.06 ms to 42 down projections and
5.27 ms to 42 SwiGLU-pack calls.  Together they use 16.33 ms, versus 18.93 ms
for the old down + standalone SwiGLU profile.  Total Triton CPU-time coverage
remains 93.2%; this number describes where time is spent, not a speedup by
itself.

The final residual add is now part of the down-projection BF16 store
epilogue.  On a real layer-0 tail, post-attention add/RMSNorm through final
residual improves from 703.84 to 669.03 us and remains bit exact.  A stricter
13-pair end-to-end A/B using the same Q4 model on both sides measures a more
modest 0.32% decode improvement (78.88 versus 79.09 ms); single-core improves
0.38%.  The profile no longer contains the previous 43 `aten::add` calls and
generated-compute CPU-time coverage rises to 93.9%.

Two tempting changes were measured and rejected:

- Fusing final RMSNorm into the runtime-Q4 vocabulary pack is bit exact but
  saves only 0.14 ms (4021.49 to 3883.21 us).  Changing Hugging Face final
  hidden-state semantics is not justified by that gain, so it is not routed.
- Sequential M8+M4/M8 scheduling reduced M12/M16 stack references from 33/48
  to 12/16, but M12 became 4--7% slower because the second loop expanded the
  static SMMLA body.  The source experiment was reverted; this spill problem
  needs a machine-scheduling/register-allocation solution rather than source
  duplication.
- Combining the three small decode-MLP allocations into one byte workspace
  reduced allocation count but slowed the real layer tail by 3.2% (648.02 to
  668.61 us).  CPU allocator caching made the extra slicing/views a net loss,
  so this experiment was also reverted.

### Q8

PyTorch 2.10 does not expose one fused CPU op matching token-symmetric A8 plus
per-channel W8.  Two native controls are reported:

- strict eager ATen W8A8: reduction, RNE quantization, `aten::_int_mm`, and
  scale application;
- optimized `aten::_weight_int8pack_mm` A16W8, which is more favorable to
  ATen numerically but deliberately skips A8 activation quantization.

Layer-0 q_proj results are 185.8 us for Triton at M=1, versus 21.68 ms for
strict eager ATen W8A8 and 411.6 us for the A16W8 control.  At M=4 they are
374.5 us, 89.32 ms and 1.445 ms respectively.  The same trend holds for O,
gate and down projections.  End to end at 12 prompt tokens, Triton measured
423.84/105.48 ms prefill/decode versus 2359.13/252.56 ms for the faster but
non-equivalent A16W8 control.  With the strict W8A8 graph and a one-token
prompt, native ATen decode was 12.89 seconds; Triton selected the same token.

The evidence supports both routes for different reasons: Q8 has a large
advantage over the available ATen controls, while Q4 now matches the mature
KleidiAI route closely and wins in joined/full-CPU prefill.

## Reproduction

```bash
cd /home/kevin/triton-opt-cpu

# Alternating end-to-end Q4 versus joined ATen/KleidiAI
taskset -c 0 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/ab_quantized_e2e.py --left q4 \
  --right q4_aten_fused --quantize-lm-head --prompt-tokens 12 --pairs 7

taskset -c 0-11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/ab_quantized_e2e.py --left q4 \
  --right q4_aten_fused --quantize-lm-head --prompt-tokens 12 --pairs 5

# Native Q8 control
taskset -c 4-11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/run_minicpm5.py --mode w8_aten \
  --quantize-lm-head --prompt-tokens 1

# Real-checkpoint operator microbenchmarks
taskset -c 4-11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/bench_quantized_aten_triton.py --quant both

taskset -c 0-11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/bench_q4_swiglu_down_pack.py

taskset -c 0-11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/bench_q4_final_norm_lmhead.py

# Focused correctness and LLVM-codegen gates
taskset -c 11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/minicpm5/validate_minicpm5_q4_asym.py
taskset -c 11 /home/cix/venv-fep-e2e/bin/python -S \
  benchmarks/flaggems_e2e/validate_w8a8.py
```

Primary artifacts:

- `artifacts/minicpm5/quantized_aten_triton_q4.json`
- `artifacts/minicpm5/quantized_aten_triton_w8_rne.json`
- `artifacts/minicpm5/q4_g128_pack_once_compact_ab.json`
- `artifacts/minicpm5/q4_g128_shared_rhs_pair_cap_ab.json`
- `artifacts/minicpm5/q4_g128_prefill_block_schedule_ab.json`
- `artifacts/minicpm5/q4_g128_prefill_block_schedule_extended_ab.json`
- `artifacts/minicpm5/q4_g128_prefill_block_schedule_large_ab.json`
- `artifacts/minicpm5/q4_g128_prefill_block_schedule_xlarge_ab.json`
- `artifacts/minicpm5/q4_g128_final_quant_lmhead_e2e.json`
- `artifacts/minicpm5/q4_g128_final_checkpoint_e2e.json`
- `artifacts/minicpm5/q4_g128_adaptive_default_full_cpu_e2e.json`
- `artifacts/minicpm5/q4_g128_swiglu_down_pack_ab.json`
- `artifacts/minicpm5/q4_g128_swiglu_pack_full_cpu_e2e.json`
- `artifacts/minicpm5/q4_g128_swiglu_pack_single_core_e2e.json`
- `artifacts/minicpm5/q4_g128_swiglu_pack_profile_full_cpu.json`
- `artifacts/minicpm5/q4_down_residual_micro_ab.json`
- `artifacts/minicpm5/q4_down_residual_direct_e2e_ab.json`
- `artifacts/minicpm5/q4_down_residual_direct_single_core_e2e_ab.json`
- `artifacts/minicpm5/q4_down_residual_full_cpu_stable_e2e.json`
- `artifacts/minicpm5/q4_down_residual_single_core_vs_aten_e2e.json`
- `artifacts/minicpm5/q4_down_residual_profile_full_cpu.json`
- `artifacts/minicpm5/q4_final_norm_lmhead_full_cpu_ab.json`
- `artifacts/minicpm5/q4_lmhead_partitions_full_cpu_ab.json`
- `artifacts/minicpm5/q4_g128_split_panel_prefill_ab.json`
- `artifacts/minicpm5/q4_combined_down_workspace_micro_ab.json`
- `artifacts/minicpm5/w8_aten_exact_e2e_prompt1.json`
- `artifacts/minicpm5/w8_aten_weight_only_e2e.json`
- `artifacts/minicpm5/w8_triton_rne_e2e_rerun.json`
