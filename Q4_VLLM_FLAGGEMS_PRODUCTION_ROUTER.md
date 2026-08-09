# Q4 vLLM / FlagGems production router handoff

## What is connected

The production implementation is under
`third_party/FlagGems/src/flag_gems/runtime/backend/_arm/q4`.

| Runtime rows (`M`) | Matrix route |
| --- | --- |
| 1–3 | Ordinary Triton decode kernel lowered to NEON SDOT |
| 4 or more | Ordinary Triton activation quantization/packing, then I8MM |
| Complete 16-row blocks | M16 SMMLA object |
| Remaining 1–4 rows | Padded M4 SMMLA object |
| Remaining 5–8 rows | Padded M8 SMMLA object |
| Remaining 9–12 rows | Padded M12 SMMLA object |
| Remaining 13–15 rows | Padded M16 SMMLA object |

Weights are quantized and packed once at vLLM model load into the exact
KleidiAI `qsi4c32p` 72-byte N4/K32 ABI. Activations use the matching
`qsi8d32p` ABI. The active ordinary-Triton pack processes one M4 panel per
program: four scalar row reductions feed one vector scale division, then
bounded K8 slices are rounded with LLVM FRINTN. The packed bytes are identical
to the retired row schedule, with no runtime call or stack spill. vLLM stores
the callable in its existing `layer.cpu_linear` slot and can remove the
original BF16 weight after preparation.

The vLLM slot calls the runtime router directly in eager execution. While
Dynamo captures a graph it uses the opaque `flag_gems::arm_q4_linear`
`torch.library.custom_op`, so one compiled graph still handles changing M.
The compiled route was tested in the order M20, M1, M7, M32, M2, M12, M47,
M3 without capturing the first value of M.

Neither matrix path calls TLE, a hand-written C kernel, or KleidiAI. The KAI
ABI is used as a layout contract; the computation remains Triton IR and is
lowered by Triton-CPU.

## CIX validation on 2026-08-02

The production-size test used K=N=1024 and covered:

`M = 1, 2, 3, 4, 7, 8, 12, 15, 16, 17, 20, 24, 28, 31, 32, 47, 48`.

RHS packing round-tripped exactly. Relative to an explicit quantized PyTorch
reference, most outputs were bit-exact. The few FP32 accumulation-order cases
near BF16 rounding boundaries differed by at most 3 BF16 ULP; no tested shape
exceeded 6 non-bit-exact elements out of as many as 49,152 outputs.

Generated-code audit at K=N=1024:

| Object | Required instruction count | External calls | Stack references |
| --- | ---: | ---: | ---: |
| M4 | 16 SMMLA | 0 | 0 |
| M8 | 32 SMMLA | 0 | 6 |
| M12 | 48 SMMLA | 0 | 14 |
| M16 | 64 SMMLA | 0 | 10 |
| Decode | 8 SDOT, 1 ADDP | 0 | 0 |
| M4 activation pack | 32 FRINTN | 0 | 0 folded spill/reload |

The stack-reference count includes callee-save traffic; M12 remains the
highest-register-pressure variant and should be watched in future compiler
changes.

The focused vLLM test confirmed that one eligible layer was prepared, its BF16
weight was removed, bias semantics were preserved, and M1/M3/M4/M12/M16/M20/
M31/M32/M47 all ran through the installed `cpu_linear` callable.

Fresh CPU11 microbenchmarks include activation packing and output allocation,
but exclude one-time weight packing:

| Shape | Direct router | Opaque custom op | vLLM eager slot |
| --- | ---: | ---: | ---: |
| M1, N=K=1024 | 150.70 us | 206.86 us | 154.01 us |
| M16, N=K=1024 | 428.32 us | 477.81 us | 430.89 us |

The eager slot is within 2.6-3.3 us of direct execution. The 49-56 us opaque
dispatcher round trip is now paid only during compiled-graph execution. These
are CIX sanity measurements, not Mac performance claims.

The full-panel M4 pack measures 12.575 us for M16/K1024 via direct
shared-object calls, versus 16.139 us for the retired row schedule (1.283x).
It specializes away row predicates for complete panels and launches the
masked form only for a final partial panel. Complete-router fresh-process A/Bs
show a noisier 0.9-2.4% gain;
the smaller operator percentage is expected because the I8MM matrix dominates.
Set
`FLAGGEMS_ARM_Q4_LEGACY_ROW_PACK=1` to reproduce the old pack.

The M16 compiler schedule now consumes one packed LHS row pair at a time.
That removes the former loop-body Q-register spill and accumulator rotation,
reducing the matrix from 266.27 to 239.76 us against KleidiAI's 239.09 us.
Two generated M8 programs are now 5.6% slower on CIX. The M16 object remains
the default; `FLAGGEMS_ARM_Q4_M16_AS_M8=1` is retained only for cross-SoC A/B.

## Mac M4 Pro commands

From the repository root:

```bash
TRITON_TEST_PYTHON=/path/to/venv/bin/python \
  bash benchmarks/run_q4_router_m4pro.sh | tee q4-router-m4pro.log
```

To include the vLLM loader test:

```bash
TRITON_TEST_PYTHON=/path/to/venv/bin/python \
VLLM_SOURCE_ROOT=/path/to/vllm \
  bash benchmarks/run_q4_router_m4pro.sh | tee q4-router-m4pro.log
```

The script selects fixed-width NEON lowering (`TRITON_CPU_FIXED_I8MM=1`) and
disables SVE2 lowering, since Apple M4 provides I8MM/DotProd but not SVE.

To add the backend selector to `vllm-plugin-FL`:

```bash
bash integrations/vllm/apply_vllm_plugin_fl_q4_codegen.sh \
  /path/to/vllm-plugin-FL
export FL_CPU_INT4=1
export FL_CPU_INT4_BACKEND=triton_codegen
```

The patch has been checked with `git apply --check` against the current CIX
plugin checkout. It deliberately leaves `tleraw` as the default until the Mac
microbenchmark and vLLM end-to-end run establish the real crossover.

## Acceptance criteria on M4 Pro

1. The correctness script prints `PASS`, with no external calls in any audited
   matrix object and the expected SMMLA/SDOT counts.
2. Record `direct_pipeline_us`, `production_custom_op_us`, and
   `vllm_eager_slot_us`. Use the slot result for eager vLLM; keep the custom-op
   result as the compiled-graph dispatch diagnostic.
3. Compare identical model, prompt, thread count, batch, context, and sampling
   settings for `tleraw`, `triton_codegen`, and the native vLLM/KleidiAI route.
4. Do not switch the default based on matrix-only timing. Activation packing,
   custom-op dispatch, mixed decode/prefill row counts, and weight memory must
   all be included in the decision.
