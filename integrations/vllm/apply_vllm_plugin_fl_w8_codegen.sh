#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s /path/to/vllm-plugin-FL\n' "$0" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
target_dir="$(cd "$1" && pwd)"
module_src="${repo_dir}/integrations/vllm/cpu_int8_triton_codegen.py"
module_dst="${target_dir}/vllm_fl/ops/cpu_int8_triton_codegen.py"
patch_file="${repo_dir}/integrations/vllm/vllm_plugin_fl_w8_codegen.patch"
library_src="${repo_dir}/artifacts/vllm-triton-backend/libtriton_kai_w8_decode_backend.so"
library_dst="${target_dir}/vllm_fl/ops/libtriton_kai_w8_decode_backend.so"

if [[ ! -f "${target_dir}/vllm_fl/__init__.py" ]]; then
  printf 'not a vllm-plugin-FL checkout: %s\n' "${target_dir}" >&2
  exit 1
fi
if [[ ! -f "${library_src}" ]]; then
  printf 'missing backend library; run integrations/vllm/build_w8_codegen_backend.sh\n' >&2
  exit 1
fi

if ! grep -q 'cpu_int8_triton_codegen import enable_int8' \
    "${target_dir}/vllm_fl/__init__.py"; then
  git -C "${target_dir}" apply --check "${patch_file}"
  git -C "${target_dir}" apply "${patch_file}"
fi
cp "${module_src}" "${module_dst}"
cp "${library_src}" "${library_dst}"

printf '%s\n' \
  "patched=${target_dir}" \
  "module=${module_dst}" \
  "library=${library_dst}"
