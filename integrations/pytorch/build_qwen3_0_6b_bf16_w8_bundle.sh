#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
port_dir="${repo_dir}/ports/triton-cpu-3.7.2"
python_bin="/home/kevin/venv-int8-clean/bin/python"
python_path="${port_dir}/python:${repo_dir}/benchmarks"
bundle_dir="${repo_dir}/artifacts/pytorch-triton-backend/qwen3-0.6b-bf16-w8"
mkdir -p "${bundle_dir}"

hash_inputs() {
  sha256sum "$@" | sha256sum | cut -d ' ' -f 1
}

# A bundle cache is a compiler product, not just a shape product.  Key it by
# both kernel sources and the exact 3.7 compiler binary/source so rebuilding
# after a lowering change cannot silently reuse stale objects.
codegen_key="$(hash_inputs \
  "${repo_dir}/benchmarks/bench_bf16_w8a8_ordinary_split.py" \
  "${repo_dir}/benchmarks/bench_bf16_w8a8_wide_split.py" \
  "${repo_dir}/benchmarks/bench_w8a8_codegen.py" \
  "${repo_dir}/benchmarks/bench_bf16_w8_mlp_three_stage.py" \
  "${port_dir}/third_party/cpu/lib/TritonCPUTransforms/ConvertDotOp/ConvertDotToSVE2I8MM.cpp" \
  "${port_dir}/third_party/cpu/lib/TritonToTritonCPU/ConvertElementwiseOps.cpp" \
  "${port_dir}/third_party/cpu/language/cpu/libdevice.py" \
  "${port_dir}/python/triton/_C/libtriton.so")"

audit_quant_object() {
  local object_path="$1"
  local disassembly round_count umax_count fmax_count fmin_count
  local stack_count call_count
  disassembly="$(objdump -d --disassemble=_quantize_bf16_w8_kernel "${object_path}")"
  round_count="$(awk '$3 == "fcvtns" || $3 == "frintn" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  umax_count="$(awk '$3 == "umax" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  fmax_count="$(awk '$3 == "fmaxnm" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  fmin_count="$(awk '$3 == "fminnm" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' <<<"${disassembly}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  if [[ "${round_count}" != 4 || "${umax_count}" != 3 \
      || "${fmax_count}" != 2 || "${fmin_count}" != 0 \
      || "${stack_count}" != 0 || "${call_count}" != 0 ]]; then
    printf 'Q8 quant audit failed for %s: round=%s umax=%s fmaxnm=%s fminnm=%s stack=%s calls=%s\n' \
      "${object_path}" "${round_count}" "${umax_count}" \
      "${fmax_count}" "${fmin_count}" "${stack_count}" \
      "${call_count}" >&2
    return 1
  fi
}

audit_gemv_object() {
  local object_path="$1"
  local block_n="$2"
  local expected_sdot=$((block_n / 2))
  local disassembly sdot_count stack_count call_count
  disassembly="$(objdump -d --disassemble=_w8a8_wide_gemv_kernel "${object_path}")"
  sdot_count="$(awk '$3 == "sdot" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' <<<"${disassembly}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  if [[ "${sdot_count}" != "${expected_sdot}" || "${stack_count}" != 0 || "${call_count}" != 0 ]]; then
    printf 'Q8 GEMV audit failed for %s: sdot=%s expected=%s stack=%s calls=%s\n' \
      "${object_path}" "${sdot_count}" "${expected_sdot}" \
      "${stack_count}" "${call_count}" >&2
    return 1
  fi
}

audit_inline_swiglu_object() {
  local object_path="$1"
  local disassembly round_count stack_count call_count
  disassembly="$(objdump -d \
    --disassemble=_bf16_swiglu_inline_exp_kernel "${object_path}")"
  round_count="$(awk '$3 == "fcvtns" || $3 == "frintn" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' <<<"${disassembly}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  # The eight stack references save/restore callee-saved vector registers;
  # there are no loop-body spills.  Keep the bound explicit so regressions
  # cannot silently reintroduce SLEEF calls or unbounded spilling.
  if [[ "${round_count}" != 4 || "${call_count}" != 0 ]] \
      || (( stack_count > 8 )); then
    printf 'inline SwiGLU audit failed for %s: round=%s stack=%s calls=%s\n' \
      "${object_path}" "${round_count}" "${stack_count}" "${call_count}" >&2
    return 1
  fi
}

shapes=(
  "1024 1024 64"
  "1024 2048 64"
  "2048 1024 64"
  "1024 3072 64"
  "3072 1024 64"
  "1024 4096 64"
  "1024 152064 32"
)

for shape in "${shapes[@]}"; do
  read -r k n block_n <<<"${shape}"
  cache_dir="${repo_dir}/artifacts/aot-bf16-w8-3.7-${codegen_key}/k${k}-n${n}-bn${block_n}"
  if ! find "${cache_dir}" -type f -name '_w8a8_wide_gemv_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" \
      TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
      TRITON_CACHE_DIR="${cache_dir}" OMP_NUM_THREADS=1 \
      taskset -c 0 "${python_bin}" \
      "${repo_dir}/benchmarks/bench_bf16_w8a8_wide_split.py" \
      --k "${k}" --n "${n}" --block-n "${block_n}" \
      --warmup 0 --iters 1 --batches 1
  fi

  quant_so="$(find "${cache_dir}" -type f \
    -name '_quantize_bf16_w8_kernel.so' -print -quit)"
  gemv_so="$(find "${cache_dir}" -type f \
    -name '_w8a8_wide_gemv_kernel.so' -print -quit)"
  audit_quant_object "${quant_so}"
  audit_gemv_object "${gemv_so}" "${block_n}"
  shape_dir="${bundle_dir}/k${k}-n${n}-bn${block_n}"
  mkdir -p "${shape_dir}"
  ln -sfn "${quant_so}" "${shape_dir}/_quantize_bf16_w8_kernel.so"
  ln -sfn "${gemv_so}" "${shape_dir}/_w8a8_wide_gemv_kernel.so"
  printf '%s\t%s\t%s\t%s\n' "${k}" "${n}" "${block_n}" "${shape_dir}"
done

mlp_k=1024
mlp_n=3072
mlp_block_n=64
mlp_cache_dir="${repo_dir}/artifacts/aot-bf16-w8-3.7-${codegen_key}/mlp-k${mlp_k}-n${mlp_n}-bn${mlp_block_n}"
if ! find "${mlp_cache_dir}" -type f -name '_bf16_swiglu_kernel.so' \
    -print -quit 2>/dev/null | grep -q .; then
  env PYTHONPATH="${python_path}" \
    TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
    TRITON_CACHE_DIR="${mlp_cache_dir}" OMP_NUM_THREADS=1 \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_bf16_w8_mlp_three_stage.py" \
    --k "${mlp_k}" --n "${mlp_n}" --block-n "${mlp_block_n}" \
    --warmup 0 --iters 1 --batches 1
fi

mlp_dir="${bundle_dir}/mlp-k${mlp_k}-n${mlp_n}-bn${mlp_block_n}"
mkdir -p "${mlp_dir}"
for kernel in _quantize_bf16_w8_kernel _w8a8_wide_gemv_kernel \
    _bf16_swiglu_kernel _bf16_swiglu_inline_exp_kernel; do
  kernel_so="$(find "${mlp_cache_dir}" -type f \
    -name "${kernel}.so" -print -quit)"
  ln -sfn "${kernel_so}" "${mlp_dir}/${kernel}.so"
done
audit_quant_object "${mlp_dir}/_quantize_bf16_w8_kernel.so"
audit_gemv_object "${mlp_dir}/_w8a8_wide_gemv_kernel.so" "${mlp_block_n}"
audit_inline_swiglu_object \
  "${mlp_dir}/_bf16_swiglu_inline_exp_kernel.so"
printf 'mlp\t%s\t%s\t%s\t%s\n' \
  "${mlp_k}" "${mlp_n}" "${mlp_block_n}" "${mlp_dir}"

rms_shapes=(
  "1 1024"
  "16 128"
  "8 128"
)
for shape in "${rms_shapes[@]}"; do
  read -r rms_m rms_n <<<"${shape}"
  rms_cache_dir="${repo_dir}/artifacts/aot-rmsnorm-3.7-m${rms_m}-n${rms_n}-e1e-6"
  if ! find "${rms_cache_dir}" -type f -name '_rms_norm_aot_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env PYTHONPATH="${python_path}" \
      TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
      TRITON_CACHE_DIR="${rms_cache_dir}" OMP_NUM_THREADS=1 \
      taskset -c 0 "${python_bin}" \
      "${repo_dir}/benchmarks/bench_rmsnorm_ordinary_aot.py" \
      --m "${rms_m}" --n "${rms_n}" --eps 1e-6 --block-n 16 \
      --warmup 0 --iters 1 --batches 1
  fi
  rms_so="$(find "${rms_cache_dir}" -type f \
    -name '_rms_norm_aot_kernel.so' -print -quit)"
  rms_dir="${bundle_dir}/rms-m${rms_m}-n${rms_n}-e1e-6"
  mkdir -p "${rms_dir}"
  ln -sfn "${rms_so}" "${rms_dir}/_rms_norm_aot_kernel.so"
  printf 'rms\t%s\t%s\t%s\n' "${rms_m}" "${rms_n}" "${rms_dir}"
done

fused_rms_m=1
fused_rms_n=1024
fused_rms_cache_dir="${repo_dir}/artifacts/aot-fused-add-rms-3.7-m${fused_rms_m}-n${fused_rms_n}-e1e-6"
if ! find "${fused_rms_cache_dir}" -type f \
    -name '_fused_add_rms_aot_kernel.so' -print -quit 2>/dev/null \
    | grep -q .; then
  env PYTHONPATH="${python_path}" \
    TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
    TRITON_CACHE_DIR="${fused_rms_cache_dir}" OMP_NUM_THREADS=1 \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_fused_add_rms_ordinary_aot.py" \
    --m "${fused_rms_m}" --n "${fused_rms_n}" --eps 1e-6 --block-n 16 \
    --warmup 0 --iters 1 --batches 1
fi
fused_rms_so="$(find "${fused_rms_cache_dir}" -type f \
  -name '_fused_add_rms_aot_kernel.so' -print -quit)"
fused_rms_dir="${bundle_dir}/fused-rms-m${fused_rms_m}-n${fused_rms_n}-e1e-6"
mkdir -p "${fused_rms_dir}"
ln -sfn "${fused_rms_so}" \
  "${fused_rms_dir}/_fused_add_rms_aot_kernel.so"
printf 'fused-rms\t%s\t%s\t%s\n' \
  "${fused_rms_m}" "${fused_rms_n}" "${fused_rms_dir}"

rope_q_heads=16
rope_kv_heads=8
rope_head_dim=128
rope_cache_dir="${repo_dir}/artifacts/aot-rope-3.7-hq${rope_q_heads}-hkv${rope_kv_heads}-d${rope_head_dim}"
if ! find "${rope_cache_dir}" -type f -name '_rope_qk_aot_kernel.so' \
    -print -quit 2>/dev/null | grep -q .; then
  env PYTHONPATH="${python_path}" \
    TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
    TRITON_CACHE_DIR="${rope_cache_dir}" OMP_NUM_THREADS=1 \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_rope_ordinary_aot.py" \
    --q-heads "${rope_q_heads}" --kv-heads "${rope_kv_heads}" \
    --head-dim "${rope_head_dim}" --block-half 16 \
    --warmup 0 --iters 1 --batches 1
fi
rope_so="$(find "${rope_cache_dir}" -type f \
  -name '_rope_qk_aot_kernel.so' -print -quit)"
rope_dir="${bundle_dir}/rope-hq${rope_q_heads}-hkv${rope_kv_heads}-d${rope_head_dim}"
mkdir -p "${rope_dir}"
ln -sfn "${rope_so}" "${rope_dir}/_rope_qk_aot_kernel.so"
printf 'rope\t%s\t%s\t%s\t%s\n' \
  "${rope_q_heads}" "${rope_kv_heads}" "${rope_head_dim}" "${rope_dir}"

"${python_bin}" "${repo_dir}/integrations/pytorch/audit_qwen3_w8_bundle.py" \
  "${bundle_dir}"

printf '%s\n' \
  "codegen_key=${codegen_key}" \
  "library=${repo_dir}/artifacts/pytorch-triton-backend/libtriton_bf16_w8_backend.so" \
  "bundle=${bundle_dir}"
