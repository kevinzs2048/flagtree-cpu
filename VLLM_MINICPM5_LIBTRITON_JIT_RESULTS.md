# MiniCPM5 production Q4/Q8 on vLLM with libtriton_jit

Date: 2026-08-09
Target: CIX AArch64, CPUs `0,1,6,7,8,9,10,11`, 8 PyTorch/OpenMP threads

## Result

The deployment contracts now follow `MiniCPM5_量化部署指南.md`:

- Q4 transformer body: signed G128 Q4 weights and dynamic token-wise
  asymmetric A8;
- Q4 `lm_head`: signed G32 Q4 weights and KleidiAI-compatible
  `qai8dxp_f32` dynamic asymmetric A8;
- Q8 transformer body and `lm_head`: per-output-channel signed W8 weights
  and KleidiAI `qai8dxp` dynamic per-token asymmetric A8.

Q4 uses the derived HeadG32 checkpoint. For Q8, the supplied checkpoint's
`ignore: [lm_head]` variant is first converted to an all-Linear checkpoint:
the 168 body projections are copied unchanged and `lm_head` is quantized
offline to per-channel W8. The production run therefore has no online weight
quantization and no BF16 Linear exception:

- `artifacts/models/MiniCPM5-2.6B-Q4-G128-HeadG32`;
- `artifacts/models/MiniCPM5-2.6B-W8A8-all-linear`.

The vLLM router prepares 169 checkpoint-quantized modules: four joined calls
per layer for 42 layers, plus `lm_head`. Weight packing and JIT compilation
happen at model load and are outside the measured request interval. The Q4
G32-head and Q8 all-Linear derivations are reproducible with
`benchmarks/derive_minicpm5_q4_head_g32.py` and
`benchmarks/derive_minicpm5_w8_all_linear.py` respectively.

`libtriton_jit` owns specialization, caching and C++ launch. It does not own
a matrix implementation. Compute remains ordinary Triton:

```text
tl.load / reshape / permute / tl.dot / reduction
  -> TTIR
  -> Triton-CPU IR
  -> packed-layout target rewrite
  -> LLVM AArch64
  -> SDOT for decode, SMMLA for prefill
```

There is no TLE_raw compute op, hand-written C matrix loop or external GEMM
call in these Q4/Q8 paths.

## Same-checkpoint end-to-end vLLM

Protocol: compiled vLLM, one 12-token prompt, eight greedy output tokens and
one warm-up. The launch requests the guide's compilation mode 3; vLLM 0.20.2's
CPU platform deliberately normalizes this to effective mode 2
(`DYNAMO_TRACE_ONCE` with Inductor). Q8 uses two stable measured rounds on the
supplied checkpoint.
The native Q8 row is a baseline, not the same numerical contract: it follows
the checkpoint's symmetric activation metadata and leaves ignored `lm_head`
in BF16, whereas the guide-exact Triton row uses asymmetric `qai8dxp` and W8
for all 169 Linear modules.

| Model | Backend | Prefill ms | Decode tok/s | Inference ms |
|---|---|---:|---:|---:|
| Q4 G128 + head G32 | Triton/libtriton_jit | 73.61 | 21.53 | 399.17 |
| Q4 G128 + head G32 | vLLM ATen/KleidiAI | 85.48 | 21.86 | 405.81 |
| Q8 guide-exact all-linear | Triton/libtriton_jit | 88.86 | 13.20 | 619.23 |
| Q8 all-linear checkpoint | vLLM oneDNN | 107.76 | 12.15 | 683.71 |

Q4 decode is 1.5% behind KleidiAI, but prefill is 13.9% lower and complete
inference is 1.6% lower. On the guide-aligned all-Linear Q8 checkpoint,
Triton has 8.6% higher decode throughput, 17.5% lower prefill latency and
9.4% lower complete inference time than the current oneDNN baseline. Compiled
model warm-up is a separate one-time cost and is excluded from these numbers.

On the original compatibility checkpoint (BF16 `lm_head` converted once at
model load), eager Q8 medians are 119.75 ms prefill, 10.29 tok/s and 799.86 ms
complete inference, versus native eager 147.48 ms, 9.22 tok/s and 906.50 ms.
These are not the primary guide-aligned numbers, and compiled/eager results
must not be mixed.

Q4 Triton and native now produce the same token digest. This only became true
after replacing the earlier BF16 fake-quant approximation in the G32 head
with the actual `qai8dxp_f32` min/max, zero-point and ties-away quantization
contract.

## Production-shape microbenchmarks

Timed regions include dynamic A8 preparation, output allocation, C++ custom
operator dispatch and matrix compute. Offline weight packing is excluded.
Calls alternate within one process and the reported number is the batch
median.

### Q4 G128 body, M=1

| Joined vLLM projection | N x K | Triton us | KleidiAI us | Result |
|---|---:|---:|---:|---:|
| Q/K/V | 2560 x 2048 | 52.82 | 65.95 | Triton 19.9% lower |
| attention output | 2048 x 2048 | 47.02 | 62.88 | Triton 25.2% lower |
| gate/up | 12288 x 2048 | 153.94 | 176.56 | Triton 12.8% lower |
| down | 2048 x 6144 | 89.72 | 110.01 | Triton 18.4% lower |
| four-call total | - | 343.49 | 415.40 | Triton 17.3% lower |

Q/K/V, output and gate/up are BF16 bit exact to the native route for the
measured input. The longer-K down projection has relative L2 error 0.0041
because its FP32 group accumulation order differs.

### Quantized vocabulary head

The real head shape is `N=130560, K=2048`.

| Format / M | Triton us | Native us | Result |
|---|---:|---:|---:|
| Q4 G32, M1 | 4188.78 | KleidiAI 3895.81 | Triton 7.5% higher |
| Q4 G32, M12 | 7600.02 | KleidiAI 12346.05 | Triton 38.4% lower |
| Q8 per-channel W8, M1 | 6689.99 | oneDNN 6954.34 | Triton 3.8% lower |

The refreshed guide-exact Q8 M1 result is 6746.73 us versus oneDNN 7013.16
us (3.8% lower). The asymmetric and symmetric outputs intentionally differ;
the exact correctness oracle is the official KleidiAI `qai8dxp x qsi8cxp`
pipeline, not the oneDNN output.

For the four real transformer projection shapes, multithreaded custom-op
medians for guide-exact Triton are 49.94/37.50/453.31/145.03 us for joined
QKV, output, gate/up and down. The current oneDNN baselines are
132.74/117.61/587.01/237.15 us. Direct single-program comparisons against
KleidiAI's exact pipeline range from 0.946x to 1.066x on these shapes; the
130560x2048 head is 11688.83 versus 12029.06 us (0.972x). Every direct KAI
comparison has byte-identical LHS and BF16-bit-identical output.

Q4 G32 M1 is the remaining clear decode loss. After `qai8dxp_f32`
alignment, Triton versus KleidiAI relative L2 is `2.29e-5`; only 0.0061% of
BF16 elements differ. Q8 intentionally does not use oneDNN's symmetric-A8
contract; its bit-exact oracle is the corresponding KleidiAI asymmetric
pipeline.

For Q4 G32 prefill, joined M8/M12 objects are 9.7--13.4% faster than splitting
everything into spill-free M4 tiles. M16 is the exception: one grid of two
M8 tiles is 3.3% faster than the spilling M16 object, so production uses M8
as the main block only for `M>=16`; M12 remains joined.

## LLVM/codegen audit

The final gate reads cached `.tttcir`, `.llir` and AArch64 assembly, and
requires zero residual `triton_cpu.dot` and zero external compute calls.

| Path | Required generated body |
|---|---|
| Q4 G128 decode | 32 SDOT, 4 ADDP, 8 FCVTAS |
| Q4 G32 decode | 8 SDOT, 1 ADDP, 8 FCVTAS |
| Q4 `qai8dxp` panel pack | 32 FCVTAS, no loop stack traffic |
| Q4 G128 prefill | 32/48/64 SMMLA specializations |
| Q4 G32 prefill | 16/32/48 SMMLA; no M16 object in production |
| Q8 `qai8dxp` M1 pack | 3 FCVTNS, no stack access |
| Q8 `qai8dxp` MR4 pack | 9 FCVTNS, no hot-loop stack access |
| Q8 exact-layout decode | 16 SDOT + 1 ADDP, no hot-loop stack access |
| Q8 exact-layout prefill | 16/32/48/64 SVE2 SMMLA, no hot-loop stack access |

Q4 G32 M4 is spill-free. The measured-best M8 and M12 objects retain 5 and
11 hot-loop stack references respectively; the audit records and bounds
these instead of claiming zero spill. The source-level split alternatives
were rejected by benchmark. Q8 has zero hot-loop stack references; some
MR4/M16 objects retain bounded callee-save prologue/epilogue slots.

Audit result:
`artifacts/vllm-minicpm5/minicpm_q4_q8_guide_exact_codegen_audit.json`.

## Correctness and production gates

- Q4 G128: M=1,3,4,7,8,12,16,20,24,28,31,32, eager and dynamic compiled
  calls are BF16 bit exact between the Python ordinary-Triton route and the
  C++ `libtriton_jit` operator.
- Q4 G32 `qai8dxp`: the same M set passes the same bit-exact gate.
- Q8: M=1,2,3,4,7,8,9,12,13,16,20,31,32 and dynamic compiled routing pass
  the semantic reference gate. Direct M1/M4/M7/M8/M12/M16 pipelines are
  BF16 bit exact to KleidiAI.
- The Q8 custom operator is also bit exact between eager and Inductor at the
  real joined-QKV, gate/up, down and 130560-row vocabulary shapes. Repeated
  eight-thread decode calls are deterministic, ruling out an incomplete
  output or partition race.
- Q8 LHS pack is byte exact to KleidiAI over all 65,280 finite BF16 encodings
  for both MR1 and MR4 schedules; the Python `qsi8cxp` RHS pack is also byte
  exact to the official KAI packer.
- Synthetic chunked Q4/Q8 weight packs match direct preparation exactly.
- The offline Q4 derivation preserves the sampled G128 body tensors
  byte-for-byte and reproduces every G32 `lm_head` value and scale from the
  BF16 source.
- The Q8 per-channel quantization formula reproduces the source checkpoint
  weights and scales exactly for inspected attention and MLP projections.
- The offline all-Linear derivation preserves body INT8/scales byte-for-byte
  and reproduces every `lm_head` INT8 value and scale from the BF16 source.
- The Q4 and Q8 shared library rebuild completes successfully.

The complete Q8 model does not have the same greedy token stream in eager and
compiled modes. A fixed-context top-logprob diagnostic shows why: after the
same prompt ending in token 242, eager favors token 29673, while compiled puts
29673 and 178 at an effective tie and selects 178. Since the opaque Q8 Linear
operator itself is bit exact between modes, this is an end-to-end floating
graph effect (normalization/reduction association followed by discontinuous
dynamic A8 quantization), not an SDOT/SMMLA correctness failure. Token digests
must therefore be compared within the same execution mode; direct KleidiAI
comparison remains the quantized-kernel correctness gate.

## Integration state

Three plugin layouts are supported by the patch installer:

- `integrations/vllm/vllm_plugin_fl_q4_codegen.patch` for the older plugin;
- `integrations/vllm/vllm_plugin_fl_resolved_quant_codegen.patch` for the
  refactored plugin with separate INT4 and INT8 selectors;
- `integrations/vllm/vllm_plugin_fl_current_quant_codegen.patch` for the
  current unified Arm plugin revision that previously exposed INT4 only.

The installer selects a matching patch using `git apply --check`. The active
CIX checkout was kept unmodified during measurement and therefore used the
benchmark's compatibility INT4 selector; applying the current patch exposes
the real `FL_CPU_INT8_BACKEND=libtriton_jit` selector.

The derived checkpoints use compressed-tensors safetensors so vLLM can load
them directly. This differs from the guide's torchao `.bin` container, but
the numerical Q8 contract is the same: per-channel W8 and dynamic per-row A8.
The local Q4 body starts from the existing prequantized G128 checkpoint rather
than re-quantizing the BF16 body online.

## Primary artifacts

- `artifacts/vllm-minicpm5/q4_derived_headg32_qai8dxp_libtriton_jit_compiled_3r.json`
- `artifacts/vllm-minicpm5/q4_derived_headg32_vllm_native_compiled_3r.json`
- `artifacts/vllm-minicpm5/w8_derived_all_linear_offline_libtriton_jit_compiled_3r.json`
- `artifacts/vllm-minicpm5/w8_derived_all_linear_vllm_native_compiled.json`
- `artifacts/vllm-minicpm5/lm_head_q4_g32_qai8dxp_m1_vs_kai.json`
- `artifacts/vllm-minicpm5/lm_head_q4_g32_qai8dxp_m12_vs_kai.json`
- `artifacts/vllm-minicpm5/lm_head_w8_triton_vs_onednn.json`
- `artifacts/vllm-minicpm5/q4_g128_fused_qkv_m1_vs_kai.json`
- `artifacts/vllm-minicpm5/q4_g128_o_proj_m1_vs_kai.json`
- `artifacts/vllm-minicpm5/q4_g128_fused_gate_up_m1_vs_kai.json`
- `artifacts/vllm-minicpm5/q4_g128_down_proj_m1_vs_kai.json`
- `artifacts/vllm-minicpm5/minicpm_q4_q8_production_codegen_audit.json`
- `artifacts/minicpm5-vllm-w8-kai-guide-compiled.json`
- `artifacts/minicpm5-vllm-w8-native-compiled-current.json`
- `artifacts/vllm-minicpm5/w8_all_linear_guide_exact_libtriton_jit_compiled_2r.json`
- `artifacts/vllm-minicpm5/w8_derived_all_linear_vllm_native_compiled.json`
- `artifacts/vllm-minicpm5/minicpm_q4_q8_guide_exact_codegen_audit.json`
- `artifacts/minicpm5-vllm-w8-kai-guide-eager-prompt-plus-242-logprobs.json`
- `artifacts/minicpm5-vllm-w8-kai-guide-compiled-prompt-plus-242-logprobs.json`
