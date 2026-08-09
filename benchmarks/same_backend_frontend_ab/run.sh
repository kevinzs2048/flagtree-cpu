#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="${repo_dir}/benchmarks/same_backend_frontend_ab"
output_dir="${repo_dir}/artifacts/same-backend-frontend-ab"
triton_python="${repo_dir}/ports/triton-cpu-3.7.2/python"
torch_site="${TRITON_CPU_VENV_SITE:-/home/cix/venv-fep-e2e/lib/python3.11/site-packages}"
llvm_root="${TRITON_LLVM_ROOT:-/home/cix/.triton/llvm/llvm-87717bf9-ubuntu-arm64}"
clang="${SAME_BACKEND_CLANG:-${llvm_root}/bin/clang}"
mlir_translate="${llvm_root}/bin/mlir-translate"
mlir_opt="${llvm_root}/bin/mlir-opt"
export PYTHONPATH="${repo_dir}/benchmarks:${triton_python}:${torch_site}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_BACKENDS_IN_TREE=1
export TRITON_CPU_BACKEND=1
export TRITON_CACHE_DIR="${output_dir}/raw-cache-v6-a720-post-cse-store"
export TRITON_DISABLE_LINE_INFO=1
export DISABLE_LLVM_OPT=1
mkdir -p "${output_dir}"

"${clang}" --target=aarch64-linux-gnu \
  -march=armv8.6-a+sve2+i8mm+bf16 \
  -O0 -Xclang -disable-O0-optnone -ffast-math \
  -fno-discard-value-names -S -emit-llvm \
  "${source_dir}/kernels.c" -o "${output_dir}/c_frontend_raw.ll"

"${clang}" --target=aarch64-linux-gnu \
  -march=armv8.6-a+sve2+i8mm+bf16 \
  -O0 -Xclang -disable-O0-optnone -ffast-math \
  -fno-discard-value-names -S -emit-llvm \
  "${source_dir}/kernels_acle.c" -o "${output_dir}/c_acle_raw.ll"

"${clang}" --target=aarch64-linux-gnu \
  -march=armv8.6-a+sve2+i8mm+bf16 \
  -DSAME_BACKEND_NO_RESTRICT=1 \
  -O0 -Xclang -disable-O0-optnone -ffast-math \
  -fno-discard-value-names -S -emit-llvm \
  "${source_dir}/kernels_acle.c" \
  -o "${output_dir}/c_acle_no_restrict_raw.ll"

python3 "${source_dir}/generate_triton_ir.py" --output "${output_dir}"

unset DISABLE_LLVM_OPT
for input in \
  c_frontend_raw.ll \
  c_acle_raw.ll \
  c_acle_no_restrict_raw.ll \
  triton_rms_raw.llir \
  triton_rms_store8_raw.llir \
  triton_rms_store32_raw.llir \
  triton_rms_output8_raw.llir \
  triton_rope_raw.llir \
  triton_w8_raw.llir \
  triton_w8_outer_pointer_raw.llir; do
  stem="${input%.*}"
  python3 "${source_dir}/common_backend.py" \
    "${output_dir}/${input}" --output-prefix "${output_dir}/${stem}"
done

# Give both frontend products to the same LLVM-dialect MLIR backend.  This is
# deliberately below Triton's tensor/tile dialects and below Clang's C AST:
# both modules run through the same MLIR import, canonicalize/CSE, export, and
# Triton LLVM 23 O3/Arm code-generation pipeline.
for input in \
  c_frontend_raw.ll \
  c_acle_raw.ll \
  c_acle_no_restrict_raw.ll \
  triton_rms_raw.llir \
  triton_rms_store8_raw.llir \
  triton_rms_store32_raw.llir \
  triton_rms_output8_raw.llir \
  triton_rope_raw.llir \
  triton_w8_raw.llir \
  triton_w8_outer_pointer_raw.llir; do
  stem="${input%.*}"
  "${mlir_translate}" --import-llvm \
    "${output_dir}/${input}" -o "${output_dir}/${stem}.llvm-dialect.mlir"
  "${mlir_opt}" --canonicalize --cse \
    "${output_dir}/${stem}.llvm-dialect.mlir" \
    -o "${output_dir}/${stem}.common-backend.mlir"
  "${mlir_translate}" --mlir-to-llvmir \
    "${output_dir}/${stem}.common-backend.mlir" \
    -o "${output_dir}/${stem}.mlir-roundtrip.ll"
  python3 "${source_dir}/common_backend.py" \
    "${output_dir}/${stem}.mlir-roundtrip.ll" \
    --output-prefix "${output_dir}/${stem}_common_mlir"
done

g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/c_frontend_raw_common_mlir.s" \
  -o "${output_dir}/c_frontend.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/c_acle_raw_common_mlir.s" -o "${output_dir}/c_acle.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/c_acle_no_restrict_raw_common_mlir.s" \
  -o "${output_dir}/c_acle_no_restrict.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_rms_raw_common_mlir.s" \
  -o "${output_dir}/triton_rms.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_rms_store8_raw_common_mlir.s" \
  -o "${output_dir}/triton_rms_store8.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_rms_store32_raw_common_mlir.s" \
  -o "${output_dir}/triton_rms_store32.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_rms_output8_raw_common_mlir.s" \
  -o "${output_dir}/triton_rms_output8.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_rope_raw_common_mlir.s" \
  -o "${output_dir}/triton_rope.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_w8_raw_common_mlir.s" \
  -o "${output_dir}/triton_w8.so"
g++ -shared -fPIC -march=armv8.6-a+sve2+i8mm+bf16 \
  "${output_dir}/triton_w8_outer_pointer_raw_common_mlir.s" \
  -o "${output_dir}/triton_w8_outer_pointer.so"

g++ -std=c++20 -O3 -DNDEBUG "${source_dir}/bench.cpp" -ldl \
  -o "${output_dir}/bench"
g++ -std=c++20 -O3 -DNDEBUG "${source_dir}/correctness.cpp" -ldl \
  -o "${output_dir}/correctness"
g++ -std=c++20 -O3 -DNDEBUG "${source_dir}/perf_runner.cpp" -ldl \
  -o "${output_dir}/perf_runner"
g++ -std=c++20 -O3 -DNDEBUG "${source_dir}/alignment_sweep.cpp" -ldl \
  -o "${output_dir}/alignment_sweep"
g++ -std=c++20 -O3 -DNDEBUG "${source_dir}/rope_stability.cpp" -ldl \
  -o "${output_dir}/rope_stability"

taskset -c "${SAME_BACKEND_CPU:-11}" "${output_dir}/bench" \
  "${output_dir}/c_frontend.so" \
  "${output_dir}/c_acle.so" \
  "${output_dir}/triton_rms.so" \
  "${output_dir}/triton_rope.so" \
  "${output_dir}/triton_w8.so"

taskset -c "${SAME_BACKEND_CPU:-11}" "${output_dir}/correctness" \
  "${output_dir}/c_frontend.so" \
  "${output_dir}/c_acle.so" \
  "${output_dir}/triton_rms.so" \
  "${output_dir}/triton_rope.so" \
  "${output_dir}/triton_w8.so"

audit_function() {
  local assembly="$1"
  local symbol="$2"
  local body
  body="$(sed -n "/^${symbol}:/,/^[[:space:]]*\\.size[[:space:]]*${symbol},/p" \
    "${assembly}")"
  printf '%s instructions=%s sdot=%s smmla=%s stack_refs=%s calls=%s\n' \
    "${symbol}" \
    "$(grep -Ec '^[[:space:]]+[a-z]' <<<"${body}" || true)" \
    "$(grep -Ec '\bsdot\b' <<<"${body}" || true)" \
    "$(grep -Ec '\bsmmla\b' <<<"${body}" || true)" \
    "$(grep -Ec '\[sp[,#\]]' <<<"${body}" || true)" \
    "$(grep -Ec '^[[:space:]]+bl[[:space:]]' <<<"${body}" || true)"
}

audit_function "${output_dir}/c_frontend_raw_common_mlir.s" \
  _rms_same_backend_c
audit_function "${output_dir}/c_acle_raw_common_mlir.s" \
  _rms_same_backend_acle
audit_function "${output_dir}/triton_rms_raw_common_mlir.s" \
  _rms_same_backend_triton
audit_function "${output_dir}/c_frontend_raw_common_mlir.s" \
  _rope_same_backend_c
audit_function "${output_dir}/c_acle_raw_common_mlir.s" \
  _rope_same_backend_acle
audit_function "${output_dir}/triton_rope_raw_common_mlir.s" \
  _rope_same_backend_triton
audit_function "${output_dir}/c_frontend_raw_common_mlir.s" \
  _w8_same_backend_c
audit_function "${output_dir}/c_acle_raw_common_mlir.s" \
  _w8_same_backend_acle
audit_function "${output_dir}/triton_w8_raw_common_mlir.s" \
  _kai_w8_layout_pointer_kernel
