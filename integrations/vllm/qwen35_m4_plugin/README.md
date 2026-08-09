# Qwen3.5 W4A8 vLLM plugin for Apple Silicon

This plugin enables the FlagGems `libtriton_jit` G32 W4A8 router and installs
a pure-PyTorch GDN recurrence on Darwin.  The latter is a deliberate safety
fallback: the packed Triton GDN decode path is quarantined after reproducible
SoC watchdog resets.

Install it into the same environment as vLLM:

```bash
uv pip install --python /path/to/vllm/.venv/bin/python --no-deps -e \
  integrations/vllm/qwen35_m4_plugin
```

At runtime set `VLLM_PLUGINS=triton_cpu_qwen35` and point
`FLAGGEMS_LIBTRITON_JIT_Q4_OP` at the bundled Q4 operator library.  Do not set
`TRITON_CPU_QWEN35_GDN_BACKEND=triton` on Darwin unless you intentionally opt
back into the unsafe experimental path.

From the handoff repository, run the text-only prefill/decode smoke test with:

```bash
VLLM_ROOT=/path/to/vllm MODEL_DIR=/path/to/Qwen3.5-9B-W4A8-G32 \
  integrations/vllm/run_qwen35_m4_w4a8.sh --max-tokens 3
```

The Darwin-safe backend keeps the quantized linear projections on
Triton/libtriton_jit, but runs GDN causal convolution and recurrent state
updates in FP32 PyTorch. Prefix caching and speculative decoding are rejected
by this safety fallback rather than silently entering an unvalidated path.
