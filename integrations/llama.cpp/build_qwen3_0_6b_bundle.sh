#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="/home/cix/venv-fep-e2e/bin/python"
python_path="${repo_dir}/third_party/FlagGems/src:${repo_dir}/python:/home/cix/venv-fep-e2e/lib/python3.11/site-packages"
block_n=512
bundle_dir="${repo_dir}/artifacts/llama-triton-backend/qwen3-0.6b"
mkdir -p "${bundle_dir}"

# Profitable decode projection shapes in Qwen3-0.6B. Both the single-kernel
# fallback and the faster shared-quantization two-stage AOT kernels are kept.
# Vocabulary projection remains a policy decision because its best path
# depends on thread count.
shapes=(
  "1024 1024"
  "1024 2048"
  "2048 1024"
  "1024 3072"
  "3072 1024"
)

for shape in "${shapes[@]}"; do
  read -r k n <<<"${shape}"
  cache_dir="${repo_dir}/artifacts/aot-f32-w8-k${k}-n${n}-bn${block_n}"
  if ! find "${cache_dir}" -type f -name '_f32_w8_gemv_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      FLAGGEMS_VENDOR_NAME=arm TRITON_CACHE_DIR="${cache_dir}" \
      PYTHONPATH="${python_path}" taskset -c 0 \
      "${python_bin}" -S "${repo_dir}/benchmarks/bench_f32_decode_codegen.py" \
      --k "${k}" --n "${n}" --block-n "${block_n}" \
      --warmup 0 --iters 1 --batches 1
  fi
done

for shape in "${shapes[@]}"; do
  read -r k n <<<"${shape}"
  cache_dir="${repo_dir}/artifacts/aot-f32-w8-split-rne2-k${k}-n${n}-bn${block_n}"
  if ! find "${cache_dir}" -type f \
      -name '_f32_w8_prequant_gemv_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      FLAGGEMS_VENDOR_NAME=arm TRITON_CACHE_DIR="${cache_dir}" \
      PYTHONPATH="${python_path}" taskset -c 0 \
      "${python_bin}" -S \
      "${repo_dir}/benchmarks/bench_f32_decode_split_codegen.py" \
      --k "${k}" --n "${n}" --block-n "${block_n}" \
      --warmup 0 --iters 1 --batches 1
  fi
done

for shape in "${shapes[@]}"; do
  read -r k n <<<"${shape}"
  cache_dir="${repo_dir}/artifacts/aot-f32-w8-k${k}-n${n}-bn${block_n}"
  kernel_so="$(find "${cache_dir}" -type f \
    -name '_f32_w8_gemv_kernel.so' -print -quit)"
  shape_dir="${bundle_dir}/k${k}-n${n}-bn${block_n}"
  mkdir -p "${shape_dir}"
  ln -sfn "${kernel_so}" "${shape_dir}/_f32_w8_gemv_kernel.so"
  split_cache_dir="${repo_dir}/artifacts/aot-f32-w8-split-rne2-k${k}-n${n}-bn${block_n}"
  quant_so="$(find "${split_cache_dir}" -type f \
    -name '_quantize_f32_w8_kernel.so' -print -quit)"
  prequant_so="$(find "${split_cache_dir}" -type f \
    -name '_f32_w8_prequant_gemv_kernel.so' -print -quit)"
  ln -sfn "${quant_so}" "${shape_dir}/_quantize_f32_w8_kernel.so"
  ln -sfn "${prequant_so}" \
    "${shape_dir}/_f32_w8_prequant_gemv_kernel.so"
  printf '%s\t%s\t%s\t%s\n' \
    "${k}" "${n}" "${block_n}" "${shape_dir}"
done

printf '%s\t%s\t%s\t%s\n' \
  "1024" "151936" "-" "single_core_fallback_c; benchmark padded_n=152064 for threads>=4"
