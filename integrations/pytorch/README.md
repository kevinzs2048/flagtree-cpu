# Direct ordinary-Triton CPU operator bundle

This directory builds a small C++ dispatcher for generated Triton-CPU shared
objects. It does not implement operator arithmetic. Quantization, SDOT GEMV,
SwiGLU, RMSNorm, fused add+RMSNorm, and RoPE all execute in LLVM code generated
from ordinary Triton kernels.

Build the dispatcher and the Qwen3-0.6B bundle:

```bash
cd /home/kevin/triton-opt-cpu
bash integrations/pytorch/build.sh
bash integrations/pytorch/build_qwen3_0_6b_bf16_w8_bundle.sh
```

The cache key includes the kernel sources, the relevant 3.7 lowering sources,
and the exact `libtriton.so`. The bundle build also disassembles all 22 selected
objects and rejects unexpected calls, stack growth, missing SDOTs, or missing
round-to-even instructions.

Enable it before FlagGems replaces model linears:

```bash
export FLAGGEMS_ARM_W8_AOT_BUNDLE=/home/kevin/triton-opt-cpu/artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8
export FLAGGEMS_ARM_W8_AOT_LIBRARY=/home/kevin/triton-opt-cpu/artifacts/pytorch-triton-backend/libtriton_bf16_w8_backend.so
```

The route is optional and shape-checked. If the environment variable, shared
library, or a shape entry is absent, FlagGems keeps its previous path. The
following diagnostic switches independently disable experimental groups:

```bash
export FLAGGEMS_ARM_W8_MLP_AOT=0
export FLAGGEMS_ARM_RMS_AOT=0
export FLAGGEMS_ARM_FUSED_RMS_AOT=0
export FLAGGEMS_ARM_ROPE_AOT=0
```

Run the object audit independently with:

```bash
/home/cix/venv-fep-e2e/bin/python \
  integrations/pytorch/audit_qwen3_w8_bundle.py \
  artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8
```

Set `FLAGGEMS_PROFILE_RANGES=1` only while collecting operator coverage; its
`record_function` ranges add measurable overhead and should remain disabled
for latency runs.

The vocabulary projection can optionally partition the same generated GEMV
over persistent workers:

```bash
export FLAGGEMS_ARM_W8_AOT_VOCAB_THREADS=2
taskset -c 10-11 ...
```

This switch is disabled by default. Its threads inherit the process CPU mask,
so the mask must expose at least that many suitable cores. It changes the
core budget, not the generated arithmetic: each worker calls a disjoint output
range of the audited BN32 Triton kernel.

See `../../ORDINARY_AOT_E2E_RESULTS.md` for correctness requirements, measured
latencies, rejected configurations, and the end-to-end command.
