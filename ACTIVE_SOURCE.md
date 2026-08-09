# Active Q4/Q8 Arm source tree

The production work in this workspace has two active source repositories:

- Triton-CPU 3.7.2: `ports/triton-cpu-3.7.2`
- FlagGems Arm kernels/router: `third_party/FlagGems`

The older Triton tree at the workspace root is retained only for historical
experiments and prior measurements. New Q4/Q8 builds, tests, and AOT bundles
must import `ports/triton-cpu-3.7.2/python/triton`. Relevant benchmark scripts
default to that path and also verify the imported `triton.__file__` where a
wrong import would invalidate a codegen result.

The configured CMake build for the active port is
`ports/triton-cpu-3.7.2/build-port`. Generated AOT objects and benchmark output
under `artifacts/` are test products, not source files.

Use these gates before reporting a production result:

```bash
ninja -C ports/triton-cpu-3.7.2/build-port

TRITON_CPU_PYTHON="$PWD/ports/triton-cpu-3.7.2/python" \
  /home/kevin/venv-int8-clean/bin/python -S \
  benchmarks/test_flaggems_q4_production_router.py --k 1024 --n 64

bash integrations/vllm/build_qwen3_w8_kai_codegen_bundle.sh
```

The W8 bundle builder prints the exact target-specific release directory.
Pass that directory—not its parent—to the W8 deployment and test commands.
