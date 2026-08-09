#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRITON_TEST_PYTHON:-python3}"
triton_python="${TRITON_CPU_PYTHON:-${repo_dir}/ports/triton-cpu-3.7.2/python}"
runtime="${QWEN35_TLE_RUNTIME:-${triton_python}/triton/_C/libTritonCPURuntime.dylib}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this runner is for native Darwin arm64 validation" >&2
  exit 2
fi

if [[ ! -f "${runtime}" ]]; then
  echo "ERROR: native runtime not found: ${runtime}" >&2
  echo "Build ports/triton-cpu-3.7.2 locally; do not substitute a Linux .so." >&2
  exit 2
fi

export TRITON_CPU_PYTHON="${triton_python}"
export PYTHONPATH="${triton_python}:${repo_dir}/third_party/FlagGems/src${PYTHONPATH:+:${PYTHONPATH}}"
export FLAGGEMS_VENDOR_NAME=arm
export TRITON_CPU_FIXED_I8MM=1
unset TRITON_CPU_DISABLE_SVE2_I8MM
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${python_bin}" - <<'PY'
import platform
import sys

import torch
import triton

print(f"platform={platform.platform()} machine={platform.machine()}")
print(f"python={sys.version.split()[0]} torch={torch.__version__}")
print(f"triton={triton.__version__} triton_file={triton.__file__}")
PY

# Qwen3.5-9B has 16 key heads and 32 value heads. Its joined Q/K/V causal
# convolution therefore has C = 2 * (16 * 128) + 32 * 128 = 8192.
"${python_bin}" "${repo_dir}/benchmarks/bench_qwen35_conv1d_codegen.py" \
  --channels 8192 \
  --block-size 64 \
  --tle-runtime "${runtime}"

# Decode recurrence: one token, 32 value heads, K=V=128.
FLAGGEMS_ARM_GATED_DELTA_IMPL=triton \
  "${python_bin}" "${repo_dir}/benchmarks/bench_qwen35_gated_delta_codegen.py" \
    --heads 32 \
    --k-dim 128 \
    --v-dim 128 \
    --tle-runtime "${runtime}"

# Decoder RMSNorm has D=4096. GDN applies the gated variant to 32 rows of
# head dimension 128 for each decode token.
"${python_bin}" "${repo_dir}/benchmarks/bench_qwen35_rms_codegen.py" \
  --d 4096 \
  --gated-m 32 \
  --gated-d 128 \
  --tle-runtime "${runtime}"
