# vLLM / FlagGems ARM Q4 codegen router

The implementation lives in
`third_party/FlagGems/src/flag_gems/runtime/backend/_arm/q4`. It replaces the
old prefill TLE call with ordinary Triton kernels:

```text
M=1..3   -> rolled KAI-layout Triton SDOT decode
M>=4     -> Triton LHS quant/pack
          + M16 I8MM main blocks
          + M4/M8/M12 tail (M16 for a padded 13..15-row tail)
```

The runtime decision is a `torch.library.custom_op`, so vLLM's
`DYNAMO_TRACE_ONCE` graph can be reused across prefill and decode without
capturing the warmup value of `M`. The packed RHS is produced once during
model loading. No TLE/runtime compute symbol is called by either the decode or
prefill matrix kernel.

The backend is intentionally limited to AArch64 hosts that advertise both
DotProd and I8MM. On other Arm CPUs, leave this opt-in backend disabled and use
the existing vLLM fallback.

## vllm-plugin-FL hook

Apply the small opt-in backend hook to a checkout of `vllm-plugin-FL`:

```bash
bash integrations/vllm/apply_vllm_plugin_fl_q4_codegen.sh \
  /path/to/vllm-plugin-FL
```

Then select the backend before starting vLLM:

```bash
export FL_CPU_INT4=1
export FL_CPU_INT4_BACKEND=libtriton_jit
export FLAGGEMS_LIBTRITON_JIT_Q4_OP="$PWD/artifacts/vllm-libtriton-jit-q4/build/libtriton_jit_q4_op.so"
export PYTHONPATH="$PWD/third_party/FlagGems/src:$PWD/ports/triton-cpu-3.7.2/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$PWD/third_party/libtriton_jit/build-vllm/src:${LD_LIBRARY_PATH:-}"
```

Build the shared PyTorch operator first with
`bash integrations/vllm/build_libtriton_jit_q4_op.sh`. The existing `tleraw`
backend remains the plugin default; `triton_codegen` selects the Python
custom-op route and `libtriton_jit` selects the lower-overhead C++ route.
Neither generated route calls a TLE_raw or C matrix implementation.

Current CIX MiniCPM5 Q4/Q8 vLLM results, accuracy limits and codegen gates are
recorded in `VLLM_MINICPM5_LIBTRITON_JIT_RESULTS.md`.

## Tests

The portable Mac test entry point is:

```bash
TRITON_TEST_PYTHON=/path/to/venv/bin/python \
  bash benchmarks/run_q4_router_m4pro.sh
```

It forces fixed-width NEON I8MM, validates every routing boundary from M1 to
M48, reuses one dynamic `torch.compile` graph across decode and prefill, and
audits the four generated matrix objects for SDOT/SMMLA and external calls.
Set `VLLM_SOURCE_ROOT=/path/to/vllm` to include the focused vLLM loader-slot
integration test in the same run.

For a shorter smoke test:

```bash
python benchmarks/test_flaggems_q4_production_router.py --k 128 --n 64
```

For one production-shape microbenchmark (weight packing excluded, activation
packing and output allocation included), the report gives both the direct
pipeline and the actual `torch.library.custom_op` entry used by vLLM:

```bash
python benchmarks/bench_flaggems_q4_production_router.py \
  --m 16 --n 1024 --k 1024
```

Do not compare this number with a matrix-only microbenchmark. The production
number includes activation quantization/packing, output allocation, Triton
launches, and custom-op dispatch. End-to-end vLLM performance on the M4 Pro is
the acceptance test for switching the plugin default.
