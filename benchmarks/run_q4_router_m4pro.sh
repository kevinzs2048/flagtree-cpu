#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRITON_TEST_PYTHON:-python3}"
triton_python="${TRITON_CPU_PYTHON:-${repo_dir}/ports/triton-cpu-3.7.2/python}"

export TRITON_CPU_PYTHON="${triton_python}"
export PYTHONPATH="${triton_python}:${repo_dir}/third_party/FlagGems/src${PYTHONPATH:+:${PYTHONPATH}}"
export FLAGGEMS_VENDOR_NAME=arm
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Apple M4 has NEON DotProd/I8MM but no SVE. Force the architecture-appropriate
# fixed-width lowering so the run is independent of host-probe differences.
# DISABLE turns off both SVE and fixed lowering, so it must not leak in from an
# earlier strict generic-codegen A/B shell.
unset TRITON_CPU_DISABLE_SVE2_I8MM
export TRITON_CPU_FIXED_I8MM=1

"${python_bin}" "${repo_dir}/benchmarks/test_flaggems_q4_production_router.py" \
  --k 1024 --n 1024

# Compile the complete ordinary-codegen surface.  This now includes W8 decode,
# joined QKV, and the shape-specialized SwiGLU epilogue; none requires the
# development tree's TLE frontend module.
"${python_bin}" "${repo_dir}/benchmarks/audit_arm_q4_q8_codegen.py" \
  --k 1024 --n 1024

# Execute the fixed-width objects as well as inspecting them. This covers W8
# decode padding/K tails, all finite BF16 quantizer encodings, W8 prefill
# M4/M8/M12/M16 tails, joined QKV, and the three-stage W8 SwiGLU route.
"${python_bin}" "${repo_dir}/benchmarks/flaggems_e2e/validate_w8a8.py"

if [[ -n "${VLLM_SOURCE_ROOT:-}" ]]; then
  "${python_bin}" "${repo_dir}/benchmarks/test_vllm_flaggems_q4_router.py"
fi

for m in 1 3 4 8 12 16 20 24 28 31 32 47 48; do
  "${python_bin}" "${repo_dir}/benchmarks/bench_flaggems_q4_production_router.py" \
    --m "${m}" --n 1024 --k 1024 --warmup 10 --iters 20 --batches 7
done

# Focused A/B candidates. Each command is a fresh process so the module-level
# selectors are unambiguous in the emitted JSON. M16 is now the production
# default; force the older two-M8 schedule for the comparison arm.
FLAGGEMS_ARM_Q4_LEGACY_ROW_PACK=1 \
  "${python_bin}" "${repo_dir}/benchmarks/bench_flaggems_q4_production_router.py" \
    --m 16 --n 1024 --k 1024 --warmup 20 --iters 50 --batches 11

for m in 16 32 48; do
  FLAGGEMS_ARM_Q4_M16_AS_M8=1 \
    "${python_bin}" "${repo_dir}/benchmarks/bench_flaggems_q4_production_router.py" \
      --m "${m}" --n 1024 --k 1024 --warmup 20 --iters 50 --batches 11
done

# Q8 uses the same ordinary load/reshape/permute/tl.dot graph. On Apple this
# must produce fixed-width Neon SMMLA for M4/M8/M12/M16 rather than SVE.
for m in 4 8 12 16; do
  "${python_bin}" "${repo_dir}/benchmarks/bench_q8_prefill_production_codegen.py" \
    --m "${m}" --n 1024 --k 1024 --warmup 20 --iters 50 --batches 9
done

"${python_bin}" "${repo_dir}/benchmarks/bench_w8_prefill_kai_layout_codegen.py" \
  --m 16 --n 1024 --k 1024 --warmup 50 --iters 1000
