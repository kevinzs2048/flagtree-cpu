#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="/home/cix/venv-fep-e2e/bin/python"
python_path="${repo_dir}/python"
port_python_path="${repo_dir}/ports/triton-cpu-3.7.2/python"
bundle_dir="${repo_dir}/artifacts/llama-triton-backend/qwen3-4b-w4"
mkdir -p "${bundle_dir}"

hash_inputs() {
  sha256sum "$@" | sha256sum | cut -d ' ' -f 1
}

find_exactly_one() {
  local search_dir="$1"
  local filename="$2"
  local matches=()
  mapfile -t matches < <(find "${search_dir}" -type f -name "${filename}" | sort)
  if (( ${#matches[@]} != 1 )); then
    printf 'expected exactly one %s under %s, found %d\n' \
      "${filename}" "${search_dir}" "${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

mapfile -t target_info < <(
  PYTHONPATH="${port_python_path}" "${python_bin}" -c '
import platform
from triton._C.libtriton import llvm
from triton.backends.cpu.target_info import get_sve_vector_bits, supplement_aarch64_features
features = sorted(supplement_aarch64_features(llvm.get_cpu_features()))
print(platform.machine())
print(llvm.get_cpu_name())
print(",".join(features))
print(get_sve_vector_bits())
'
)
if (( ${#target_info[@]} != 4 )); then
  printf 'failed to collect AOT target metadata\n' >&2
  exit 1
fi
target_arch="${target_info[0]}"
target_cpu="${target_info[1]}"
target_features="${target_info[2]}"
target_sve_vector_bits="${target_info[3]}"

write_target_manifest() {
  printf 'target_arch=%s\n' "${target_arch}"
  printf 'target_cpu=%s\n' "${target_cpu}"
  printf 'target_features=%s\n' "${target_features}"
  printf 'target_sve_vector_bits=%s\n' "${target_sve_vector_bits}"
}

audit_generated_object() {
  local object_path="$1"
  local symbol="$2"
  local disassembly
  disassembly="$(objdump -d --disassemble="${symbol}" "${object_path}")"
  local sdot_count addp_count scvtf_fixed_count stack_count call_count needed_count
  sdot_count="$(awk '$3 == "sdot" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  addp_count="$(awk '$3 == "addp" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  scvtf_fixed_count="$(awk '$3 == "scvtf" && /#4/ { count++ } END { print count + 0 }' <<<"${disassembly}")"
  stack_count="$(awk '/\[sp/ { count++ } END { print count + 0 }' <<<"${disassembly}")"
  call_count="$(awk '$3 == "bl" || $3 == "blr" { count++ } END { print count + 0 }' <<<"${disassembly}")"
  needed_count="$(readelf -d "${object_path}" | awk '/\(NEEDED\)/ { count++ } END { print count + 0 }')"
  if [[ "${sdot_count}" != 8 || "${addp_count}" != 1 ||
        "${scvtf_fixed_count}" == 0 || "${stack_count}" != 0 ||
        "${call_count}" != 0 || "${needed_count}" != 0 ]]; then
    printf 'AOT audit failed for %s: sdot=%s addp=%s scvtf_fixed=%s stack=%s calls=%s needed=%s\n' \
      "${object_path}" "${sdot_count}" "${addp_count}" \
      "${scvtf_fixed_count}" "${stack_count}" "${call_count}" \
      "${needed_count}" >&2
    return 1
  fi
}

legacy_codegen_key="$(hash_inputs \
  "${repo_dir}/benchmarks/bench_w4a8_codegen.py" \
  "${repo_dir}/third_party/cpu/backend/compiler.py" \
  "${repo_dir}/python/triton/_C/libtriton.so")"
kai_codegen_key="$(hash_inputs \
  "${repo_dir}/benchmarks/bench_w4_kleidiai_layout_codegen.py" \
  "${repo_dir}/ports/triton-cpu-3.7.2/third_party/cpu/backend/compiler.py" \
  "${repo_dir}/ports/triton-cpu-3.7.2/python/triton/_C/libtriton.so")"
q41_codegen_key="$(hash_inputs \
  "${repo_dir}/benchmarks/bench_w4a8_codegen.py" \
  "${repo_dir}/benchmarks/bench_w4a8_q41_codegen.py" \
  "${repo_dir}/third_party/cpu/backend/compiler.py" \
  "${repo_dir}/python/triton/_C/libtriton.so")"
q41_kai_codegen_key="$(hash_inputs \
  "${repo_dir}/benchmarks/bench_w4a8_q41_kai_layout_codegen.py" \
  "${repo_dir}/benchmarks/bench_w4_kleidiai_layout_codegen.py" \
  "${repo_dir}/ports/triton-cpu-3.7.2/third_party/cpu/backend/compiler.py" \
  "${repo_dir}/ports/triton-cpu-3.7.2/python/triton/_C/libtriton.so")"

# Actual Qwen3-4B Q4_0 tensor shapes observed in the GGUF:
#   Q: [2560,4096], K/V: [2560,1024], O: [4096,2560],
#   gate/up: [2560,9728].
# Keep the extra prototype/future-fusion shapes in the bundle as well.
shapes=(
  "2560 1024"
  "2560 4096"
  "4096 2560"
  "2560 640"
  "2560 2560"
  "2560 3840"
  "2560 9728"
  "2560 19456"
  "9728 2560"
)

for shape in "${shapes[@]}"; do
  read -r k n <<<"${shape}"
  cache_dir="${repo_dir}/artifacts/w4a8-dotready-range-${legacy_codegen_key}/qwen-k${k}-n${n}"
  if ! find "${cache_dir}" -type f \
      -name '_w4a8_grouped_gemv_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      TRITON_CACHE_DIR="${cache_dir}" PYTHONPATH="${python_path}" \
      taskset -c 0 "${python_bin}" \
      "${repo_dir}/benchmarks/bench_w4a8_codegen.py" \
      --k "${k}" --n "${n}" --tile-n 4 --grid whole \
      --warmup 0 --iters 1 --batches 1
  fi
  kernel_so="$(find_exactly_one \
    "${cache_dir}" '_w4a8_grouped_gemv_kernel.so')"
  static_cache_dir="${repo_dir}/artifacts/w4a8-dotready-static-${legacy_codegen_key}/qwen-k${k}-n${n}"
  if ! find "${static_cache_dir}" -type f \
      -name '_w4a8_grouped_gemv_static_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      TRITON_CACHE_DIR="${static_cache_dir}" PYTHONPATH="${python_path}" \
      taskset -c 0 "${python_bin}" \
      "${repo_dir}/benchmarks/bench_w4a8_codegen.py" \
      --k "${k}" --n "${n}" --tile-n 4 --grid whole --static-whole \
      --warmup 0 --iters 1 --batches 1
  fi
  static_kernel_so="$(find_exactly_one \
    "${static_cache_dir}" '_w4a8_grouped_gemv_static_kernel.so')"
  shape_dir="${bundle_dir}/k${k}-n${n}"
  mkdir -p "${shape_dir}"
  install -m 0755 "${kernel_so}" \
    "${shape_dir}/_w4a8_grouped_gemv_kernel.so"
  install -m 0755 "${static_kernel_so}" \
    "${shape_dir}/_w4a8_grouped_gemv_static_kernel.so"
  kai_cache_dir="${repo_dir}/artifacts/w4-kai-layout-${kai_codegen_key}/qwen-k${k}-n${n}"
  if ! find "${kai_cache_dir}" -type f \
      -name '_kai_w4_layout_split_kernel.so' \
      -print -quit 2>/dev/null | grep -q .; then
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
      TRITON_CACHE_DIR="${kai_cache_dir}" \
      PYTHONPATH="${port_python_path}" \
      taskset -c 0 "${python_bin}" \
      "${repo_dir}/benchmarks/bench_w4_kleidiai_layout_codegen.py" \
      --k "${k}" --n "${n}" --unroll 1 --compile-only
  fi
  kai_kernel_so="$(find_exactly_one \
    "${kai_cache_dir}" '_kai_w4_layout_split_kernel.so')"
  install -m 0755 "${kai_kernel_so}" \
    "${shape_dir}/_kai_w4_layout_split_kernel.so"
  audit_generated_object \
    "${shape_dir}/_kai_w4_layout_split_kernel.so" \
    _kai_w4_layout_split_kernel
  {
    printf 'K=%s\nN=%s\n' "${k}" "${n}"
    printf 'legacy_codegen_key=%s\n' "${legacy_codegen_key}"
    printf 'kai_codegen_key=%s\n' "${kai_codegen_key}"
    write_target_manifest
    (
      cd "${shape_dir}"
      sha256sum \
        _w4a8_grouped_gemv_kernel.so \
        _w4a8_grouped_gemv_static_kernel.so \
        _kai_w4_layout_split_kernel.so
    )
  } > "${shape_dir}/aot-manifest.txt"
  printf '%s\t%s\t%s\n' "${k}" "${n}" "${shape_dir}"
done

# Q4_1 feed-forward down projection.
k=9728
n=2560
cache_dir="${repo_dir}/artifacts/w4a8-dotready-range-${q41_codegen_key}/q41-k${k}-n${n}"
if ! find "${cache_dir}" -type f \
    -name '_w4a8_q4_1_gemv_kernel.so' \
    -print -quit 2>/dev/null | grep -q .; then
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TRITON_CACHE_DIR="${cache_dir}" PYTHONPATH="${python_path}" \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_w4a8_q41_codegen.py" \
    --k "${k}" --n "${n}" --warmup 0 --iters 1 --batches 1
fi
kernel_so="$(find_exactly_one \
  "${cache_dir}" '_w4a8_q4_1_gemv_kernel.so')"
static_cache_dir="${repo_dir}/artifacts/w4a8-dotready-static-${q41_codegen_key}/q41-k${k}-n${n}"
if ! find "${static_cache_dir}" -type f \
    -name '_w4a8_q4_1_gemv_static_kernel.so' \
    -print -quit 2>/dev/null | grep -q .; then
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TRITON_CACHE_DIR="${static_cache_dir}" PYTHONPATH="${python_path}" \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_w4a8_q41_codegen.py" \
    --k "${k}" --n "${n}" --static-whole \
    --warmup 0 --iters 1 --batches 1
fi
static_kernel_so="$(find_exactly_one \
  "${static_cache_dir}" '_w4a8_q4_1_gemv_static_kernel.so')"
shape_dir="${bundle_dir}/k${k}-n${n}-q41"
mkdir -p "${shape_dir}"
install -m 0755 "${kernel_so}" \
  "${shape_dir}/_w4a8_q4_1_gemv_kernel.so"
install -m 0755 "${static_kernel_so}" \
  "${shape_dir}/_w4a8_q4_1_gemv_static_kernel.so"
kai_cache_dir="${repo_dir}/artifacts/w4-q41-kai-layout-${q41_kai_codegen_key}/q41-k${k}-n${n}"
if ! find "${kai_cache_dir}" -type f \
    -name '_kai_q41_layout_split_kernel.so' \
    -print -quit 2>/dev/null | grep -q .; then
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1 \
    TRITON_CACHE_DIR="${kai_cache_dir}" \
    PYTHONPATH="${port_python_path}" \
    taskset -c 0 "${python_bin}" \
    "${repo_dir}/benchmarks/bench_w4a8_q41_kai_layout_codegen.py" \
    --k "${k}" --n "${n}" --unroll 1 --compile-only
fi
kai_kernel_so="$(find_exactly_one \
  "${kai_cache_dir}" '_kai_q41_layout_split_kernel.so')"
install -m 0755 "${kai_kernel_so}" \
  "${shape_dir}/_kai_q41_layout_split_kernel.so"
audit_generated_object \
  "${shape_dir}/_kai_q41_layout_split_kernel.so" \
  _kai_q41_layout_split_kernel
{
  printf 'K=%s\nN=%s\ntype=q4_1\n' "${k}" "${n}"
  printf 'q41_codegen_key=%s\n' "${q41_codegen_key}"
  printf 'q41_kai_codegen_key=%s\n' "${q41_kai_codegen_key}"
  write_target_manifest
  (
    cd "${shape_dir}"
    sha256sum \
      _w4a8_q4_1_gemv_kernel.so \
      _w4a8_q4_1_gemv_static_kernel.so \
      _kai_q41_layout_split_kernel.so
  )
} > "${shape_dir}/aot-manifest.txt"
printf '%s\t%s\t%s\t%s\n' "${k}" "${n}" "q4_1" "${shape_dir}"
